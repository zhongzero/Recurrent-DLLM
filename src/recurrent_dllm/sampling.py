# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA


import numpy as np
import torch
import torch.nn.functional as F


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = (
        torch.zeros(
            mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
        )
        + base
    )

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1

    return num_transfer_tokens


def reset_latest_layer_states(model):
    seen = set()
    stack = [model]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))

        if hasattr(current, "latest_layer_hidden_states"):
            current.latest_layer_hidden_states = None
        if hasattr(current, "latest_layer_similarity"):
            current.latest_layer_similarity = None

        for attr_name in ("model", "base_model", "module"):
            child = getattr(current, attr_name, None)
            if child is not None:
                stack.append(child)

        get_base_model = getattr(current, "get_base_model", None)
        if callable(get_base_model):
            try:
                child = get_base_model()
            except TypeError:
                child = None
            if child is not None:
                stack.append(child)


@torch.no_grad()
def generate(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
    threshold=None,
    is_dream=False,
    tokenizer=None,
    record_layer_states=False,
    print_layer_similarity=False,
    layer_similarity_pairs=None,
    last_layer_id=None,
    current_layer_id=None,
    cross_attention_scale=1.0,
):
    """
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    """

    reset_latest_layer_states(model)
    if is_dream:
        mask_id = 151666

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(
        model.device
    )
    x[:, : prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    trace_kwargs = dict(
        record_layer_states=record_layer_states,
        print_layer_similarity=print_layer_similarity,
        layer_similarity_pairs=layer_similarity_pairs,
        last_layer_id=last_layer_id,
        current_layer_id=current_layer_id,
        cross_attention_scale=cross_attention_scale,
    )

    nfe = 0
    for num_block in range(num_blocks):
        block_mask_index = (
            x[
                :,
                prompt.shape[1] + num_block * block_length : prompt.shape[1]
                + (num_block + 1) * block_length,
            ]
            == mask_id
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            nfe += 1
            mask_index = x == mask_id
            logits = model(x, **trace_kwargs).logits
            if is_dream:
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length :] = 0
            x0, transfer_index = get_transfer_index(
                logits,
                temperature,
                remasking,
                mask_index,
                x,
                num_transfer_tokens[:, i] if threshold is None else None,
                threshold,
                mask_id=mask_id,
            )
            x[transfer_index] = x0[transfer_index]
            i += 1
            if (
                x[
                    :,
                    prompt.shape[1] + num_block * block_length : prompt.shape[1]
                    + (num_block + 1) * block_length,
                ]
                == mask_id
            ).sum() == 0:
                break
    return x, nfe


def get_transfer_index(
    logits,
    temperature,
    remasking,
    mask_index,
    x,
    num_transfer_tokens,
    threshold=None,
    mask_id=126336,
):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    logits_with_noise[:, :, mask_id] = -torch.inf
    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
        )  # b, l
    elif remasking == "random":
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index
