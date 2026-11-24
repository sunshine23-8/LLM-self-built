import argparse
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from src.data.cot_dataset import CoTSFTDataset, cot_collate_fn
from src.data.leakage_filter import LeakageFilter
from src.model.config import ModelConfig
from src.model.transformer import DecoderOnlyLM
from src.tokenizer.bpe import ByteLevelBPETokenizer
from src.training.trainer import TrainConfig, Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Chain-of-Thought (CoT) SFT training on GSM8K")
    parser.add_argument("--config", type=Path, default=Path("configs/compact_12gb.yaml"))
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("artifacts/tokenizer"))
    parser.add_argument("--init-ckpt", type=Path, required=True, help="Path to pre-trained base model checkpoint")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/cot_sft"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()

    cfg = ModelConfig.from_yaml(args.config)
    tok = ByteLevelBPETokenizer.load(args.tokenizer_dir)
    print("Downloading GSM8K dataset...")
    gsm_train = list(load_dataset("openai/gsm8k", "main", split="train"))
    gsm_test_qs = [row["question"] for row in load_dataset("openai/gsm8k", "main", split="test")]

    flt = LeakageFilter(gsm_test_qs, jaccard_threshold=0.80, ngram_n=5)
    clean_train, stats = flt.filter_records(gsm_train, question_key="question")
    print(f"CoT SFT anti-leakage stats: {stats}")

    ds = CoTSFTDataset(clean_train, tok, max_seq_len=cfg.block_size)
    pad_id = tok.token_to_id("<|pad|>")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: cot_collate_fn(b, pad_id))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device} from {args.init_ckpt}...")
    model = DecoderOnlyLM(cfg).to(device)
    ckpt = torch.load(args.init_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])

    tcfg = TrainConfig(
        lr=args.lr,
        min_lr=args.lr * 0.1,
        warmup_steps=max(20, int(0.05 * args.max_steps)),
        max_steps=args.max_steps,
        grad_accum_steps=args.grad_accum,
        amp_dtype=cfg.dtype if device == "cuda" else "float32",
        gradient_checkpointing=True,
        out_dir=args.out_dir,
    )

    trainer = Trainer(model, tcfg, loader, causal_shift=True)
    trainer.train()
    print("CoT SFT training complete!")


if __name__ == "__main__":
    main()
