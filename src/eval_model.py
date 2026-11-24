"""Generate benchmark answers and score the three reasoning tasks."""

import argparse
from itertools import islice
import json
import os
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, set_seed
from tqdm import tqdm
from recurrent_dllm.benchmarks import code_solution, records, score
from recurrent_dllm.refinement import load_for_evaluation
from recurrent_dllm.sampling import generate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--task",
        required=True,
        choices=["countdown", "gsm8k", "svamp", "humaneval", "mbpp"],
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=Path, help="Directory written by train.py")
    group.add_argument("--baseline", action="store_true", help="Unmodified LLaDA")
    p.add_argument(
        "--model",
        help="Baseline model ID, or local override of the checkpoint's backbone",
    )
    p.add_argument("--output", type=Path, required=True, help="New JSONL file")
    p.add_argument("--length", type=int, default=128)
    p.add_argument(
        "--steps", type=int, default=128, help="Number of function evaluations"
    )
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--limit", type=int, help="Evaluate only the first N tasks")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if a.output.exists() or (a.limit is not None and a.limit < 1):
        p.error("Specify a new output file and a positive example limit.")
    if a.block_size <= 0 or a.length < 1 or a.length % a.block_size:
        p.error("The generation length must be a positive multiple of the block size.")
    blocks = a.length // a.block_size
    if not blocks <= a.steps <= a.length or a.steps % blocks or a.temperature < 0:
        p.error(
            "Use a nonnegative temperature and a step count divisible by the block count, between block count and generation length."
        )
    set_seed(a.seed)
    torch.backends.cudnn.deterministic = True
    world = int(os.environ.get("WORLD_SIZE", 1))
    if torch.cuda.is_available():
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    rank = dist.get_rank() if world > 1 else 0
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model, backbone, meta = load_for_evaluation(a.model, a.checkpoint, dtype)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(backbone, trust_remote_code=True)
    mask = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else 126336
    code_task = a.task in {"humaneval", "mbpp"}
    kwargs = (
        dict(
            last_layer_id=meta["source_layer"],
            current_layer_id=meta["target_layer"],
            cross_attention_scale=a.scale,
        )
        if meta
        else {}
    )
    rows = []
    for index, (task_id, question, target) in tqdm(
        enumerate(islice(records(a.task), a.limit)), disable=rank != 0
    ):
        if index % world != rank:
            continue
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if not code_task:
            prompt += "<reasoning>"
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        out, nfe = generate(
            model,
            ids,
            gen_length=a.length,
            steps=a.steps,
            block_length=a.block_size,
            temperature=a.temperature,
            mask_id=mask,
            **kwargs,
        )
        generated = tokenizer.decode(out[0, -a.length :], skip_special_tokens=False)
        if code_task:
            for marker in filter(None, ("<|eot_id|>", tokenizer.eos_token)):
                generated = generated.split(marker, 1)[0]
            row = {
                "task_id": task_id,
                "solution": code_solution(a.task, generated, target),
            }
        else:
            row = dict(
                task_id=task_id,
                prompt=prompt,
                generation=generated,
                target=target,
                nfe=nfe,
                correct=score(a.task, generated, target),
            )
        rows.append((index, row))
    if world > 1:
        shards = [None] * world if rank == 0 else None
        dist.gather_object(rows, shards, dst=0)
        dist.destroy_process_group()
        if rank != 0:
            return
        rows = [entry for shard in shards for entry in shard]
    if not rows:
        raise RuntimeError("Benchmark contains no examples.")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=a.output.parent) as tmp:
        pending = Path(tmp) / "samples.jsonl"
        with pending.open("w", encoding="utf-8") as f:
            for _, row in sorted(rows):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        pending.replace(a.output)
    print(f"Saved {len(rows)} samples to {a.output}")
    if not code_task:
        correct = sum(row["correct"] for _, row in rows)
        print(f"Accuracy: {correct}/{len(rows)} ({100 * correct / len(rows):.2f}%)")


if __name__ == "__main__":
    main()
