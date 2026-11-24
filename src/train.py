"""Supervised recurrent denoising with the LLaDA training collator."""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import get_dataset_config_names, load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed

from recurrent_dllm import LLaDAModelLM
from recurrent_dllm.refinement import attach_refinement, restore_weights, save_adapter
from recurrent_dllm.sampling import reset_latest_layer_states
from recurrent_dllm.training_data import (
    dLLMDataCollator,
    dLLMSFTDataset,
    preprocess_dataset,
)


class RecurrentTrainer(Trainer):
    def __init__(self, *args, refinement, **kwargs):
        super().__init__(*args, **kwargs)
        self.refinement = refinement
        self.model_accepts_loss_kwargs = False

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        tokens = inputs["input_ids"]
        positions = inputs["next_unmask_positions"]
        indices = torch.arange(tokens.shape[1], device=tokens.device)
        inside = (indices >= inputs["start_block"][:, None]) & (
            indices <= inputs["end_block"][:, None]
        )
        labels = inputs["GT"].masked_fill(~inside, -100)
        scale = random.choice((0.5, 1.0, 1.5, 2.0))
        losses = []
        reset_latest_layer_states(model)
        for step in range(positions.shape[1]):
            output = model(
                input_ids=tokens,
                last_layer_id=self.refinement["source_layer"],
                current_layer_id=self.refinement["target_layer"],
                cross_attention_scale=scale,
                use_cache=False,
            )
            losses.append(
                F.cross_entropy(output.logits.flatten(0, 1), labels.flatten())
            )
            reveal = positions[:, step]
            prediction = output.logits.detach().argmax(-1).gather(1, reveal)
            # Earlier forward passes still need their original token indices during backward.
            tokens = tokens.scatter(1, reveal, prediction)
        reset_latest_layer_states(model)
        loss = torch.stack(losses).mean()
        return (loss, output) if return_outputs else loss

    def _save(self, output_dir=None, state_dict=None):
        model = self.accelerator.unwrap_model(self.model)
        save_adapter(
            model, self.refinement, Path(output_dir or self.args.output_dir) / "adapter"
        )

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        directory = Path(resume_from_checkpoint) / "adapter"
        saved = json.loads((directory / "refinement.json").read_text(encoding="utf-8"))
        if saved != self.refinement:
            raise ValueError(
                "Resume with the same backbone, layer choices and training mode."
            )
        restore_weights(
            self.accelerator.unwrap_model(model if model is not None else self.model),
            directory,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--mode", choices=["joint", "module-only"], default="joint")
    p.add_argument("--train-data", default="simplescaling/s1K")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--steps",
        type=int,
        help="Default 10,000 joint or 20,000 module-only optimizer updates",
    )
    p.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    p.add_argument("--accumulation", type=int, default=1)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--source-layer", type=int, help="Default last transformer block")
    p.add_argument("--target-layer", type=int, default=1)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--resume-from-checkpoint", type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    a = p.parse_args()
    steps = a.steps if a.steps is not None else (10000 if a.mode == "joint" else 20000)
    if (
        min(steps, a.batch_size, a.accumulation, a.save_every) <= 0
        or min(a.block_size, a.max_length) < 3
    ):
        p.error("Use positive step and batch counts and lengths of at least three.")
    if (
        not a.resume_from_checkpoint
        and a.output.is_dir()
        and any(a.output.glob("checkpoint-*"))
    ):
        p.error(
            "This run already has checkpoints. Use a fresh output directory or --resume-from-checkpoint."
        )
    set_seed(a.seed)
    torch.backends.cudnn.deterministic = True
    tokenizer = AutoTokenizer.from_pretrained(
        a.model, padding_side="right", trust_remote_code=True
    )
    dtype = torch.bfloat16 if a.precision == "bf16" else torch.float32
    model = LLaDAModelLM.from_pretrained(a.model, torch_dtype=dtype)
    source = model.config.n_layers if a.source_layer is None else a.source_layer
    if not 0 <= source <= model.config.n_layers:
        p.error("The source layer is outside the backbone.")
    meta = dict(
        backbone=a.model, mode=a.mode, source_layer=source, target_layer=a.target_layer
    )
    model = attach_refinement(model, a.mode, a.target_layer).to(dtype)
    configs = (
        ["main"]
        if a.train_data == "openai/gsm8k"
        else get_dataset_config_names(a.train_data)
    )
    datasets = [load_dataset(a.train_data, name=cfg, split="train") for cfg in configs]
    train_data, _ = preprocess_dataset(datasets, a.train_data, tokenizer, a.max_length)
    if not train_data:
        raise ValueError(
            "No training examples survived the original length and answer filters."
        )
    args = TrainingArguments(
        output_dir=str(a.output),
        max_steps=steps,
        per_device_train_batch_size=a.batch_size,
        gradient_accumulation_steps=a.accumulation,
        learning_rate=a.learning_rate,
        weight_decay=0.1,
        max_grad_norm=1.0,
        logging_steps=16,
        save_steps=a.save_every,
        save_total_limit=40,
        bf16=a.precision == "bf16",
        report_to="none",
        remove_unused_columns=False,
        seed=a.seed,
        ddp_find_unused_parameters=False,
    )
    trainer = RecurrentTrainer(
        model=model,
        args=args,
        refinement=meta,
        train_dataset=dLLMSFTDataset(train_data, tokenizer, a.max_length),
        data_collator=dLLMDataCollator(
            tokenizer=tokenizer,
            mask_token_id=126336,
            max_length=a.max_length,
            K=2,
            per_deNums=[1],
            block_sizes=[a.block_size],
        ),
    )
    trainer.train(
        resume_from_checkpoint=str(a.resume_from_checkpoint)
        if a.resume_from_checkpoint
        else None
    )


if __name__ == "__main__":
    main()
