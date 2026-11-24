import argparse
from pathlib import Path
import torch
from src.data.pretrain_dataset import benchmark_training_throughput
from src.model.config import ModelConfig
from src.model.transformer import DecoderOnlyLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LLM training throughput and calibrate token budget")
    parser.add_argument("--config", type=Path, default=Path("configs/compact_12gb.yaml"))
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--gpu-hours-budget", type=float, default=6.0)
    args = parser.parse_args()

    cfg = ModelConfig.from_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, cfg.dtype) if hasattr(torch, cfg.dtype) else torch.float32
    if device == "cpu":
        dtype = torch.float32  # CPU doesn't accelerate FP16 AMP

    print(f"Loading Model ({cfg.count_parameters():,} parameters) on {device} ({dtype})...")
    model = DecoderOnlyLM(cfg).to(device)
    model.gradient_checkpointing = True

    print("Running throughput benchmark...")
    stats = benchmark_training_throughput(model, batch_size=args.micro_batch_size, seq_len=args.seq_len, dtype=dtype)
    calibrated_tokens = int(stats["tokens_per_sec"] * args.gpu_hours_budget * 3600)
    print("=" * 60)
    print(f"Training Throughput : {stats['tokens_per_sec']:.1f} tokens/sec")
    print(f"Peak VRAM           : {stats['peak_vram_mb']:.1f} MB")
    print(f"Projected Budget    : {calibrated_tokens:,} tokens in {args.gpu_hours_budget} GPU-hours")
    print("=" * 60)


if __name__ == "__main__":
    main()
