import re
import torch
from torch.utils.data import Dataset

from src.tokenizer.bpe import ByteLevelBPETokenizer

_CALC_TAG_RE = re.compile(r"<<[^<>]*>>")


def clean_gsm8k_reasoning_and_answer(raw_answer: str) -> tuple[str, str]:
    if "####" in raw_answer:
        body, ans = raw_answer.rsplit("####", 1)
    else:
        body, ans = raw_answer, ""
    cleaned_body = _CALC_TAG_RE.sub("", body).strip()
    return cleaned_body, ans.strip()


def format_cot_prompt(question: str) -> str:
    return f"<|bos|><|user|>\n{question.strip()}\n<|assistant|>\n"


def format_cot_response(reasoning: str, final_answer: str) -> str:
    return f"<think>\n{reasoning.strip()}\n</think>\n#### {final_answer.strip()}<|eos|>"


class CoTSFTDataset(Dataset):
    """Formats CoT examples with assistant-only loss masking (-100 on prompt tokens)."""

    def __init__(self, examples: list[dict], tokenizer: ByteLevelBPETokenizer, max_seq_len: int = 2048) -> None:
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        for ex in examples:
            prompt_str = format_cot_prompt(ex["question"])
            if "reasoning" in ex and "final_answer" in ex:
                reasoning, final_ans = ex["reasoning"], ex["final_answer"]
            else:
                reasoning, final_ans = clean_gsm8k_reasoning_and_answer(ex["answer"])

            resp_str = format_cot_response(reasoning, final_ans)
            prompt_ids = tokenizer.encode(prompt_str)
            resp_ids = tokenizer.encode(resp_str)
            full_ids = prompt_ids + resp_ids
            if len(full_ids) > max_seq_len:
                continue
            labels = [-100] * len(prompt_ids) + resp_ids
            self.samples.append((torch.tensor(full_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def cot_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    pad_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(ids.size(0) for ids, _ in batch)
    bsz = len(batch)
    out_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long)
    out_labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    for i, (ids, labels) in enumerate(batch):
        length = ids.size(0)
        out_ids[i, :length] = ids
        out_labels[i, :length] = labels
    return out_ids, out_labels
