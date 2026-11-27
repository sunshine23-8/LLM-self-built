import argparse
import json
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.data.rollout_dataset import RolloutVerifierDataset, split_questions_train_val
from src.inference.scorer import OutcomeVerifierModel
from src.model.config import ModelConfig
from src.model.transformer import DecoderOnlyLM
from src.tokenizer.bpe import ByteLevelBPETokenizer
from src.training.verifier_trainer import train_orm_verifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Learned ORM Verifier")
    parser.add_argument("--config", type=Path, default=Path("configs/compact_12gb.yaml"))
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("artifacts/tokenizer"))
    parser.add_argument("--sft-ckpt", type=Path, required=True)
    parser.add_argument("--rollouts-json", type=Path, required=True)
    parser.add_argument("--out-ckpt", type=Path, default=Path("artifacts/orm_verifier/orm.pt"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    cfg = ModelConfig.from_yaml(args.config)
    tok = ByteLevelBPETokenizer.load(args.tokenizer_dir)
    with args.rollouts_json.open("r", encoding="utf-8") as f:
        records = json.load(f)

    train_recs, val_recs = split_questions_train_val(records, val_ratio=0.1, seed=42)
    print(f"ORM question-split counts: train={len(train_recs)}, val={len(val_recs)}")
    train_ds = RolloutVerifierDataset(train_recs, tok, max_seq_len=cfg.block_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = DecoderOnlyLM(cfg).to(device)
    ckpt = torch.load(args.sft_ckpt, map_location=device, weights_only=True)
    backbone.load_state_dict(ckpt["model_state_dict"])
    orm = OutcomeVerifierModel(backbone).to(device)

    train_orm_verifier(
        orm,
        train_ds,
        pad_token_id=tok.token_to_id("<|pad|>"),
        eos_token_id=tok.token_to_id("<|eos|>"),
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_path=args.out_ckpt,
    )


if __name__ == "__main__":
    main()
