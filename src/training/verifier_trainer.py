from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.rollout_dataset import RolloutVerifierDataset, rollout_collate_fn
from src.inference.scorer import OutcomeVerifierModel


def train_orm_verifier(
    model: OutcomeVerifierModel,
    train_dataset: RolloutVerifierDataset,
    pad_token_id: int = 0,
    eos_token_id: int = 2,
    epochs: int = 2,
    batch_size: int = 8,
    lr: float = 2e-5,
    save_path: Path | None = None,
) -> dict[str, list[float]]:
    device = next(model.parameters()).device
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: rollout_collate_fn(b, pad_token_id=pad_token_id),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    history: dict[str, list[float]] = {"loss": []}

    for _ in range(epochs):
        for input_ids, labels in loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(input_ids, pad_token_id=pad_token_id, eos_token_id=eos_token_id)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            history["loss"].append(float(loss.item()))

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict()}, save_path)
    return history
