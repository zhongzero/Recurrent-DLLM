"""Cross-step memory, trainable parameters and portable adapter checkpoints."""

from copy import deepcopy
from pathlib import Path
import json
import os
import tempfile

from torch import nn
from safetensors.torch import load_file, save_file
from peft import LoraConfig, TaskType, get_peft_model

from .modeling_llada import LLaDAModelLM


class CrossStepAttention(nn.Module):
    def __init__(self, block):
        super().__init__()
        # Start from the backbone projections, then optimize independent parameters.
        self.query = deepcopy(block.q_proj)
        self.key = deepcopy(block.k_proj)
        self.value = deepcopy(block.v_proj)
        self.output = deepcopy(block.attn_out)


def attach_refinement(model, mode, target):
    if mode not in {"joint", "module-only"}:
        raise ValueError(f"Unknown training mode: {mode}")
    if model.config.block_type != "llama" or model.config.block_group_size != 1:
        raise ValueError(
            "This release supports LLaDA backbones with ungrouped Llama blocks."
        )
    if not 1 <= target <= model.config.n_layers:
        raise ValueError(f"Injection block must be in [1, {model.config.n_layers}]")
    model.requires_grad_(False)
    block = model.model.transformer.blocks[target - 1]
    block.csrr = CrossStepAttention(block)
    if mode == "joint":
        config = LoraConfig(
            r=128,
            lora_alpha=256,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, config)
    block.csrr.requires_grad_(True)
    return model


def save_adapter(model, metadata, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    weights = {
        n: p.detach().cpu().contiguous()
        for n, p in model.named_parameters()
        if p.requires_grad
    }
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        stage = Path(tmp) / "adapter"
        stage.mkdir()
        save_file(weights, str(stage / "refinement.safetensors"))
        (stage / "refinement.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        os.rename(stage, path)


def restore_weights(model, checkpoint):
    state = load_file(str(checkpoint / "refinement.safetensors"))
    expected = {name for name, param in model.named_parameters() if param.requires_grad}
    if set(state) != expected:
        raise ValueError(
            "Checkpoint parameters do not match the selected architecture."
        )
    model.load_state_dict(state, strict=False)


def load_for_evaluation(base, checkpoint, dtype):
    metadata = None
    if checkpoint:
        checkpoint = Path(checkpoint)
        metadata = json.loads(
            (checkpoint / "refinement.json").read_text(encoding="utf-8")
        )
        if base is None:
            base = metadata["backbone"]
    base = base or "GSAI-ML/LLaDA-8B-Instruct"
    model = LLaDAModelLM.from_pretrained(base, torch_dtype=dtype)
    if checkpoint:
        model = attach_refinement(model, metadata["mode"], metadata["target_layer"]).to(
            dtype
        )
        restore_weights(model, checkpoint)
    return model.eval(), base, metadata
