import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.data.pretrain_dataset import PackedTokenDataset
from src.model.config import ModelConfig
from src.model.transformer import DecoderOnlyLM
from src.training.trainer import TrainConfig, Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Pre-training for Self-Built LLM")
    parser.add_argument("--config", type=Path, default=Path("configs/compact_12gb.yaml"))
    parser.add_argument("--train-bin", type=Path, required=True, help="Path to train.bin")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/pretrain"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    cfg = ModelConfig.from_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing {cfg.count_parameters():,} param model on {device}...")
    model = DecoderOnlyLM(cfg).to(device)

    print(f"Loading packed dataset from {args.train_bin}...")
    ds = PackedTokenDataset(args.train_bin, block_size=cfg.block_size)
    print(f"Total training windows: {len(ds):,} (each of length {cfg.block_size})")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    tcfg = TrainConfig(
        lr=args.lr,
        min_lr=args.lr * 0.1,
        warmup_steps=max(50, int(0.05 * args.max_steps)),
        max_steps=args.max_steps,
        grad_accum_steps=args.grad_accum,
        amp_dtype=cfg.dtype if device == "cuda" else "float32",
        gradient_checkpointing=True,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        out_dir=args.out_dir,
    )

    print("=" * 60)
    print(f"Starting Pre-training: max_steps={args.max_steps}, batch_size={args.batch_size}, grad_accum={args.grad_accum}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum * cfg.block_size:,} tokens/step")
    print("=" * 60)

    trainer = Trainer(model, tcfg, loader, causal_shift=False)
    start_step = trainer.load_checkpoint(args.resume) if args.resume is not None else 0
    if start_step > 0:
        print(f"Resuming training from step {start_step}...")
    trainer.train(start_step=start_step)
    print("Pre-training complete!")


if __name__ == "__main__":
    main()
