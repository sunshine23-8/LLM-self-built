from pathlib import Path
import pytest
from src.model.config import ModelConfig


def test_compact_and_standard_yaml_parameter_counts():
    compact = ModelConfig.from_yaml(Path("configs/compact_12gb.yaml"))
    standard = ModelConfig.from_yaml(Path("configs/standard_24gb.yaml"))

    assert compact.d_head == 64
    assert compact.block_size == 2048
    assert compact.vocab_size == 16384
    assert compact.count_parameters() == 43_790_848
    assert compact.kv_cache_bytes(batch_size=1, seq_len=2048) == 25_165_824  # 24.0 MiB
    assert compact.kv_cache_bytes(batch_size=16, seq_len=2048) == 402_653_184  # 384.0 MiB

    assert standard.d_head == 64
    assert standard.block_size == 2048
    assert standard.vocab_size == 16384
    assert standard.count_parameters() == 197_166_080
    assert standard.kv_cache_bytes(batch_size=1, seq_len=2048) == 33_554_432  # 32.0 MiB
    assert standard.kv_cache_bytes(batch_size=16, seq_len=2048) == 536_870_912  # 512.0 MiB


def test_invalid_gqa_heads_or_backend_raises():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(d_model=512, n_heads=8, n_kv_heads=3)
    with pytest.raises(ValueError, match="attn_backend"):
        ModelConfig(attn_backend="unknown")
