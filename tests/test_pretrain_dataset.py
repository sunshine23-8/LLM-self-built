from pathlib import Path
import torch
from src.data.pretrain_dataset import PackedTokenDataset, benchmark_training_throughput, pack_texts_to_bin
from src.model.config import ModelConfig
from src.model.transformer import DecoderOnlyLM
from src.tokenizer.bpe import ByteLevelBPETokenizer


def test_pack_texts_and_memmap_dataset_windows(tmp_path: Path):
    tok = ByteLevelBPETokenizer()
    tok.train(["hello world 123\n", "math is fun 456\n"] * 5, vocab_size=300)

    bin_path = tmp_path / "train.bin"
    total_tokens = pack_texts_to_bin(["hello world 123\n", "math is fun 456\n"] * 10, tok, bin_path)
    assert total_tokens > 32

    ds = PackedTokenDataset(bin_path, block_size=16)
    assert len(ds) == (total_tokens - 1) // 16
    x, y = ds[0]
    assert x.shape == (16,)
    assert y.shape == (16,)
    assert x.dtype == torch.long
    # Next-token shift alignment
    x1, _ = ds[1]
    assert y[-1].item() == x1[0].item()


def test_benchmark_training_throughput_returns_valid_metrics():
    cfg = ModelConfig(vocab_size=300, block_size=64, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=128)
    model = DecoderOnlyLM(cfg)
    stats = benchmark_training_throughput(model, batch_size=2, seq_len=32, num_steps=3, warmup_steps=1, dtype=torch.float32)
    assert stats["tokens_per_sec"] > 0.0
    assert stats["steps_per_hour"] > 0.0
    assert "peak_vram_mb" in stats
