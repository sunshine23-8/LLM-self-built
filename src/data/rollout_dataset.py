import random
import torch
from torch.utils.data import Dataset

from src.data.cot_dataset import format_cot_prompt
from src.tokenizer.bpe import ByteLevelBPETokenizer


def split_questions_train_val(
    records: list[dict],
    val_ratio: float = 0.1,
    seed: int = 42,
    question_key: str = "question",
) -> tuple[list[dict], list[dict]]:
    unique_qs = sorted({r[question_key] for r in records})
    rng = random.Random(seed)
    rng.shuffle(unique_qs)
    val_count = max(1, int(len(unique_qs) * val_ratio)) if len(unique_qs) > 1 else 0
    val_qs = set(unique_qs[:val_count])
    train_recs = [r for r in records if r[question_key] not in val_qs]
    val_recs = [r for r in records if r[question_key] in val_qs]
    return train_recs, val_recs


class RolloutVerifierDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer: ByteLevelBPETokenizer, max_seq_len: int = 2048) -> None:
        self.samples: list[tuple[torch.Tensor, float]] = []
        eos_tok = "<|eos|>"
        for rec in records:
            text = format_cot_prompt(rec["question"]) + rec["rollout_text"].strip()
            if not text.endswith(eos_tok):
                text += eos_tok
            ids = tokenizer.encode(text)[:max_seq_len]
            self.samples.append((torch.tensor(ids, dtype=torch.long), float(rec["label"])))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float]:
        return self.samples[idx]


def rollout_collate_fn(
    batch: list[tuple[torch.Tensor, float]],
    pad_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(ids.size(0) for ids, _ in batch)
    out_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    out_labels = torch.tensor([lbl for _, lbl in batch], dtype=torch.float32)
    for i, (ids, _) in enumerate(batch):
        out_ids[i, : ids.size(0)] = ids
    return out_ids, out_labels
