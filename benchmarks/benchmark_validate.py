#!/usr/bin/env python3

import argparse
import multiprocessing as mp
from time import perf_counter

import numpy as np
import torch
import pandas as pd
from hivemind.utils.logging import get_logger
from transformers import AutoTokenizer
from datasets import load_dataset

from bloombee import AutoDistributedModelForCausalLM
from bloombee.constants import DTYPE_MAP, PUBLIC_INITIAL_PEERS

from toploc import ProofPoly
from pathlib import Path
from statistics import mean, median

logger = get_logger()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--model_name", type=str, required=True, help="Model")
    parser.add_argument("--initial_peers", type=str, nargs="+", default=PUBLIC_INITIAL_PEERS, help="Initial peers")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--n_processes", type=str, default=1, help="Number of concurrent processes")
    parser.add_argument("--n_samples", type=int, default=4, help="Number of samples to generate.")
    parser.add_argument("--save_dir", type=str, default="outputs", help="Directory to save outputs.")
    parser.add_argument("--max_decode_tokens", type=int, default=128, help="Maximum number of decode tokens.")
    parser.add_argument("--decode_batching_size", type=int, default=8, help="Batching size for decoding.")
    parser.add_argument("--torch_dtype", type=str, default="float32")
    parser.add_argument("--job_type", type=str, default="generate", help="generate or validate")
    parser.add_argument("--dataset_name", type=str, default="stingning/ultrachat", help="Dataset to load.")

    # used in validation
    parser.add_argument("--scale_decode_mantissa", type=str, default="no", help="Scale decode mantissa.")
    
    args = parser.parse_args()

    if args.n_processes == "n_gpus":
        args.n_processes = torch.cuda.device_count()
    else:
        args.n_processes = int(args.n_processes)

    pipe_recv, pipe_send = mp.Pipe(duplex=False)
    processes = [mp.Process(target=generate if args.job_type == "generate" else validate, args=(i, args, pipe_send)) for i in range(args.n_processes)]
    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()

    speed = np.mean([pipe_recv.recv() for _ in range(args.n_processes)])
    logger.info(f"Final result: {speed=:.2f}")


K = 128


# used in validation
SCALE_DECODE_MANTISSA = "no"
TMEAN = 10
TMEDIAN = 8
TEXP = 90

def build_activation_commit(activations: list[torch.Tensor], decode_batching_size: int, K: int) -> list[str]:
    commits = []

    # Prefill
    flat_view = activations[0].view(-1)
    topk_indices = flat_view.abs().topk(K).indices.tolist()
    topk_values = [int(v) for v in flat_view[topk_indices].tolist()]
    commit = ProofPoly.from_points(topk_indices, topk_values).to_bytes()
    commits.append(commit)

    # Batched Decode
    for i in range(1, len(activations), decode_batching_size):
        flat_view = torch.cat([i.view(-1) for i in activations[i: i + decode_batching_size]])
        topk_indices = flat_view.abs().topk(K).indices.tolist()
        topk_values = [int(v) for v in flat_view[topk_indices].tolist()]
        commit = ProofPoly.from_points(topk_indices, topk_values).to_bytes()
        commits.append(commit)
        
    return commits

@torch.inference_mode()
def generate(process_idx, args, result_pipe):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use the EOS token as the padding token
    # Using use_fast=False since LlamaTokenizerFast takes a long time to start, and we decode 1 token at a time anyway

    model = AutoDistributedModelForCausalLM.from_pretrained(
        args.model_name, initial_peers=args.initial_peers, torch_dtype=DTYPE_MAP[args.torch_dtype]
    )
    logger.info(f"Created model: {process_idx=} {model.device=}")

    # add hook to save activations
    saved_activations = []
    def activation_saving_hook(module, input, output):
        saved_activations.append(output[0].detach().clone().cpu())
    saved_activations_handle = model.model.norm.register_forward_hook(activation_saving_hook)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    output_save_path = args.save_dir + "/" + f"outputs_{args.model_name.replace('/', '--')}.pt"

    ds = load_dataset(args.dataset_name, split="train")
    prompts = [i['data'][0] for _, i in zip(range(args.n_samples), ds)]

    step_times = []
    outputs = []
    saved_commits = []
    with model.transformer.h.inference_session(max_length=2048) as sess:
        for step in range(args.n_samples):
            start_time = perf_counter()
            inputs = tokenizer(prompts[step], return_tensors="pt", padding=True, truncation=True).input_ids.to(model.device)
            logger.info(f"Step {step} input: {tokenizer.decode(inputs[0])}")
            output = model.generate(inputs=inputs, max_new_tokens=args.max_decode_tokens+1, session=sess)
            logger.info(f"Step {step} output: {tokenizer.decode(output[0])}")
            # input_ids = output[0].prompt_token_ids
            # logger.info(f"Input tokens: {tokenizer.decode(input_ids)}")
            # output_ids = output[0].outputs[0].token_ids
            # output = torch.tensor([[*input_ids, *output_ids]])
            outputs.append(output)


            act_commit = build_activation_commit(saved_activations, args.decode_batching_size, K)
            saved_commits.append(act_commit)
            saved_activations = []


            step_times.append(perf_counter() - start_time)
            speed = 1 / np.mean(step_times)
            logger.info(f"{process_idx=} {step=} {speed=:.2f}")
    
    torch.save(outputs, output_save_path)
    logger.info(f"Saved outputs to {output_save_path}")

    savepath = args.save_dir + '/' + f"poly_{args.model_name.replace('/', '--')}_128.bin"
    with open(savepath, "wb") as f:
        for commit in saved_commits:
            for c in commit:
                f.write(c)
    logger.info(f"Saved to {savepath}")

    result_pipe.send(speed)


def check(activations: list[torch.Tensor], proof: list[str]) -> tuple[list[int], list[int], list[float], list[float]]:
    from toploc.C.csrc.utils import get_fp_parts
    topk_intersections: list[int] = []
    exp_intersections: list[int] = []
    mant_err_means: list[float] = []
    mant_err_medians: list[float] = []

    for act, b_poly in zip(activations, proof):
        flat_view = act.view(-1)
        prefill_topk_indices = flat_view.abs().topk(K).indices.tolist()
        prefill_topk_values = flat_view[prefill_topk_indices]
        
        poly = ProofPoly.from_bytes(b_poly)
        decode_topk_values = torch.tensor([poly(i) for i in prefill_topk_indices], dtype=torch.uint16).view(dtype=torch.bfloat16)
        decode_topk_indices = prefill_topk_indices

        prefill_exp, prefill_mants = get_fp_parts(prefill_topk_values)
        decode_exp, decode_mants = get_fp_parts(decode_topk_values)
        dec_i_2_topk_i = {index: i for i, index in enumerate(decode_topk_indices)}
        if SCALE_DECODE_MANTISSA == "down":
            decode_mants = [i // (2 ** 16) for i in decode_mants]
        elif SCALE_DECODE_MANTISSA == "up":
            decode_mants = [i * (2 ** 16) for i in decode_mants]

        topk_intersection = 0
        exp_intersection = 0
        mant_errs: list[float] = []

        for i, index in enumerate(prefill_topk_indices):
            if index in dec_i_2_topk_i:
                topk_intersection += 1
                if decode_exp[dec_i_2_topk_i[index]] == prefill_exp[i]:
                    exp_intersection += 1
                    mant_errs.append(abs(decode_mants[dec_i_2_topk_i[index]] - prefill_mants[i]))
        topk_intersections.append(topk_intersection)
        exp_intersections.append(exp_intersection)
        if len(mant_errs) == 0:
            mant_err_means.append(128.0)
            mant_err_medians.append(128.0)
        else:
            mant_err_means.append(mean(mant_errs))
            mant_err_medians.append(median(mant_errs))
      
    for mant_err_mean, mant_err_median, exp_intersection in zip(mant_err_means, mant_err_medians, exp_intersections):
        if mant_err_mean > TMEAN or mant_err_median > TMEDIAN or exp_intersection < TEXP:   
            print(f"VERIFICATION FAILED: Mantissa error mean: {mant_err_mean} above {TMEAN} or median: {mant_err_median} above {TMEDIAN} or exp intersections: {exp_intersection} below {TEXP}")
        else:
            print(f"VERIFICATION PASSED: Mantissa error mean: {mant_err_means} below {TMEAN} and median: {mant_err_medians} below {TMEDIAN} and exp intersections: {exp_intersections} above {TEXP}")
        
    return topk_intersections, exp_intersections, mant_err_means, mant_err_medians

def segment_prefill_activations(activations: torch.Tensor, max_decode_tokens: int, decode_batching_size: int) -> list[torch.Tensor]:
    ret: list[torch.Tensor] = [activations[:, :-max_decode_tokens]]
    for i in range(activations.size(1) - max_decode_tokens, activations.size(1), decode_batching_size):
        ret.append(activations[:, i:i+decode_batching_size])
    return ret

@torch.inference_mode()
def validate(process_idx, args, result_pipe):
    global SCALE_DECODE_MANTISSA
    SCALE_DECODE_MANTISSA = args.scale_decode_mantissa
    outputs_path = args.save_dir + '/' + f'outputs_{args.model_name.replace("/", "--")}.pt'
    outputs = torch.load(outputs_path)

    with open(args.save_dir + '/' + f'poly_{args.model_name.replace("/", "--")}_128.bin', 'rb') as f:
        polys = [[f.read(258) for _j in range(1 + args.max_decode_tokens // args.decode_batching_size)] for _ in range(len(outputs))]
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use the EOS token as the padding token

    model = AutoDistributedModelForCausalLM.from_pretrained(
        args.model_name, initial_peers=args.initial_peers, torch_dtype=DTYPE_MAP[args.torch_dtype]
    )

    saved_activations = []
    def activation_saving_hook(module, input, output):
        saved_activations.append(output[0].detach().clone().cpu())
    saved_activations_handle = model.model.norm.register_forward_hook(activation_saving_hook)

    names = []
    topk_intersections = []
    exp_intersections = []
    mant_err_means = []
    mant_err_medians = []

    with model.transformer.h.inference_session(max_length=2048) as sess:
        for step in range(args.n_samples):
            logger.info(f"Step {step} input: {tokenizer.decode(outputs[step][0])}")
            output = model.generate(inputs=outputs[step], max_new_tokens=args.max_decode_tokens+1, session=sess)
            logger.info(f"Step {step} output: {tokenizer.decode(output[0])}")
            # input_ids = output[0].prompt_token_ids
            # logger.info(f"Input tokens: {tokenizer.decode(input_ids)}")
            # output_ids = output[0].outputs[0].token_ids
            logger.info(f"Output tokens: {tokenizer.decode(output[0])}")
            # output = torch.tensor([[*input_ids, *output_ids]])

            activations = segment_prefill_activations(
                saved_activations[0].unsqueeze(0), args.max_decode_tokens, args.decode_batching_size
            )

            topk_res, exp_res, mant_means, mant_medians = check(activations, polys[step])

            names.extend([f"Q{step}_{j}" for j in range(len(topk_res))])
            topk_intersections.extend(topk_res)
            exp_intersections.extend(exp_res)
            mant_err_means.extend(mant_means)
            mant_err_medians.extend(mant_medians)
            saved_activations = []

    df = pd.DataFrame({
        'Name': names,
        'Topk Intersections': topk_intersections,
        'Exp Intersections': exp_intersections,
        'Mant Err Means': mant_err_means,
        'Mant Err Medians': mant_err_medians
    })

    output_file = args.save_dir + '/' + f'poly_validation_{args.model_name.replace("/", "--")}_{args.torch_dtype}_{args.tp}P100_on_{args.model_name.replace("/", "--")}.parquet'
    df.to_parquet(output_file, index=False)
    print(df)


if __name__ == "__main__":
    main()
