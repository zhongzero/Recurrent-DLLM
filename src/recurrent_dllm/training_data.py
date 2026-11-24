"""Original SFT preprocessing and consecutive-step mask construction."""

import random
import torch
from transformers import DefaultDataCollator, DataCollatorWithPadding
from tqdm import tqdm


class dLLMSFTDataset(torch.utils.data.Dataset):
    """
    Similar to AR datasets, except in inference, we keep the timsteps fixed
    """

    def __init__(self, data, tokenizer, max_length, eval=False):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eval = eval

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        out = self.data[idx]
        return out


class dLLMDataCollator(DefaultDataCollator):
    """
    Adds the forward noising process to the batch.
    Modify forward_process to change the noise schedule
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mask_token_id = kwargs["tokenizer"].mask_token_id
        self.tokenizer = kwargs["tokenizer"]
        if "max_length" in kwargs:
            self.max_length = kwargs["max_length"]
        if kwargs["tokenizer"].mask_token_id is None:
            assert "mask_token_id" in kwargs, (
                "For dLLM models, pass a mask_token_id or set it equal to tokenizer.mask_token_id"
            )
            self.mask_token_id = kwargs["mask_token_id"]

        self.K = kwargs.get("K", 4)  # 连续forward/backward次数
        self.per_deNums = kwargs.get("per_deNums", [1, 2, 4, 8])  # 每次的de数量
        self.block_sizes = kwargs.get(
            "block_sizes", [32, 64, 128, 256, 512]
        )  # block大小

        self.pad_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

    def forward_process(self, batch, eps=1e-3):
        input_ids = batch["input_ids"]
        prompt_lengths = batch["prompt_lengths"]
        B, N = input_ids.shape

        per_deNum = random.choice(self.per_deNums)
        L = random.choice(self.block_sizes)
        if L > N - prompt_lengths.max().item():
            L = N - prompt_lengths.max().item()
        if self.K * per_deNum >= L:
            L = self.K * per_deNum + 1
        assert L <= N, "Block size larger than sequence length!"

        start_block = torch.zeros((B,), dtype=torch.long, device=input_ids.device)
        end_block = torch.zeros((B,), dtype=torch.long, device=input_ids.device)
        mask_indices = torch.zeros((B, N), dtype=torch.bool, device=input_ids.device)
        next_unmask_positions = torch.zeros(
            (B, self.K, per_deNum), dtype=torch.long, device=input_ids.device
        )

        for b in range(B):
            start_min = prompt_lengths[b].item()
            start_max = N - L
            if start_min >= start_max:
                start_block[b] = start_max
            else:
                start_block[b] = random.randint(start_min, start_max)
            end_block[b] = start_block[b] + L - 1

            P = torch.randperm(L)
            mask_num = random.randint(0, L - self.K * per_deNum)

            mask_indices[b, : start_block[b]] = False
            mask_indices[b, end_block[b] + 1 :] = True
            for i in range(mask_num, L):
                mask_pos = start_block[b] + P[i].item()
                mask_indices[b, mask_pos] = True

            for k in range(self.K):
                for d in range(per_deNum):
                    mask_pos = start_block[b] + P[mask_num + k * per_deNum + d].item()
                    next_unmask_positions[b, k, d] = mask_pos

        t = None

        noisy_batch_init = torch.where(mask_indices, self.mask_token_id, input_ids)
        return (
            noisy_batch_init,
            t,
            mask_indices,
            start_block,
            end_block,
            next_unmask_positions,
        )

    def __call__(self, batch):
        batch = self.pad_collator(batch)
        if "attention_mask" in batch:
            batch.pop("attention_mask")

        batch["GT"] = batch["input_ids"].clone()
        (
            noisy_batch_init,
            batch["t"],
            mask_indices,
            start_block,
            end_block,
            next_unmask_positions,
        ) = self.forward_process(batch)
        batch["input_ids"] = noisy_batch_init.long()
        batch["start_block"] = start_block
        batch["end_block"] = end_block
        batch["next_unmask_positions"] = next_unmask_positions
        return batch


SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>
"""


def preprocess_dataset(
    data, train_data_path, tokenizer, max_length, max_dataNum=2000, test_split=0.01
):
    min_response_length = 32

    data = [item for dataset in data for item in dataset]

    print(f"Original dataset length: {len(data)}")
    if "simplescaling/s1K" in train_data_path:
        print("Using simplescaling/s1K dataset preprocessing rules.")
        data = [item for item in data if item.get("cot_type", "") == "math"]
        print(f"Filtered dataset length (cot_type='math'): {len(data)}")
        filter_data = []
        for i in range(len(data)):
            answer_start = data[i]["attempt"].rfind("\\boxed{")
            answer_end = data[i]["attempt"].rfind("}")
            if answer_start != -1 and answer_end != -1 and answer_end > answer_start:
                answer_content = data[i]["attempt"][
                    answer_start + len("\\boxed{") : answer_end
                ]
                trajectory = f"<reasoning>{data[i]['attempt']}</reasoning>\n<answer>\\boxed{{{answer_content}}}</answer>"
            else:
                continue
            data[i]["answer"] = trajectory
            keys_to_remove = [
                key for key in data[i].keys() if key not in ["question", "answer"]
            ]
            for key in keys_to_remove:
                del data[i][key]
            filter_data.append(data[i])
        data = filter_data
        print(f"Final dataset length after filtering invalid answers: {len(data)}")
    elif "openai/gsm8k" in train_data_path:
        print("Using openai/gsm8k dataset preprocessing rules.")
        filter_data = []
        for i in range(len(data)):
            answer_start = data[i]["answer"].rfind("#### ")
            if answer_start != -1:
                answer_content = data[i]["answer"][
                    answer_start + len("#### ") :
                ].strip()
                trajectory = f"<reasoning>{data[i]['answer'][:answer_start]}</reasoning>\n<answer>\\boxed{{{answer_content}}}</answer>"
            else:
                continue
            data[i]["answer"] = trajectory
            keys_to_remove = [
                key for key in data[i].keys() if key not in ["question", "answer"]
            ]
            for key in keys_to_remove:
                del data[i][key]
            filter_data.append(data[i])
        data = filter_data
        print(f"Final dataset length after filtering invalid answers: {len(data)}")
    elif "EleutherAI/hendrycks_math" in train_data_path:
        print("Using EleutherAI/hendrycks_math dataset preprocessing rules.")
        filter_data = []
        for i in range(len(data)):
            answer_start = data[i]["solution"].rfind("\\boxed{")
            answer_end = data[i]["solution"].rfind("}")
            if answer_start != -1 and answer_end != -1 and answer_end > answer_start:
                answer_content = data[i]["solution"][
                    answer_start + len("\\boxed{") : answer_end
                ]
                trajectory = f"<reasoning>{data[i]['solution']}</reasoning>\n<answer>\\boxed{{{answer_content}}}</answer>"
            else:
                continue
            data[i]["question"] = data[i]["problem"]
            data[i]["answer"] = trajectory
            keys_to_remove = [
                key for key in data[i].keys() if key not in ["question", "answer"]
            ]
            for key in keys_to_remove:
                del data[i][key]
            filter_data.append(data[i])
        data = filter_data
        print(f"Final dataset length after filtering invalid answers: {len(data)}")
    else:
        raise NotImplementedError(
            f"Preprocessing rules for this dataset in '{train_data_path}' are not implemented."
        )

    preprocessed_data = []
    for i in tqdm(range(len(data)), desc="Preprocessing dataset"):
        question = SYSTEM_PROMPT + "\n\n" + data[i]["question"]

        answer = data[i]["answer"]

        prompt = [{"role": "user", "content": question}]
        inputs = (
            tokenizer.apply_chat_template(prompt, tokenize=False)
            + answer
            + "<|eot_id|>"
        )
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False) + "<reasoning>"

        length = tokenizer(
            inputs, return_tensors="pt", truncation=False
        ).input_ids.shape[1]
        if length > max_length:
            continue

        tokenized_input = tokenizer(
            inputs, return_tensors="pt", truncation=True, padding=False
        ).input_ids.squeeze(0)

        tokenized_prompt = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_length
        )
        prompt_lengths = tokenized_prompt.attention_mask.sum(-1)

        if length - prompt_lengths.item() < min_response_length:
            continue
        preprocessed_data.append(
            {
                "input_ids": tokenized_input,
                "prompt_lengths": prompt_lengths.item(),
            }
        )
    print(f"Preprocessing completed. Total samples: {len(preprocessed_data)}")
    random.shuffle(preprocessed_data)
    if len(preprocessed_data) > max_dataNum:
        preprocessed_data = preprocessed_data[:max_dataNum]
        print(f"Truncated preprocessed data number to max_dataNum: {max_dataNum}")

    test_data = preprocessed_data[: int(len(preprocessed_data) * test_split)]
    train_data = preprocessed_data[int(len(preprocessed_data) * test_split) :]
    return train_data, test_data
