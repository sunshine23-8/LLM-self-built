import pytest
import torch
from src.cache.dynamic_kv_cache import DynamicKVCache


def test_dynamic_kv_cache_grows_by_chunk_and_returns_exact_views():
    cache = DynamicKVCache(
        n_layers=2,
        batch_size=1,
        n_kv_heads=4,
        d_head=64,
        max_seq_len=512,
        chunk_size=128,
        dtype=torch.float32,
        device="cpu",
    )
    assert cache.get_seq_len(0) == 0
    initial_alloc = cache.allocated_bytes()
    assert initial_alloc == 2 * 2 * 1 * 4 * 128 * 64 * 4

    # Prefill 100 tokens (fits inside initial chunk 128)
    k1 = torch.randn(1, 4, 100, 64)
    v1 = torch.randn(1, 4, 100, 64)
    k_view, v_view = cache.update(0, k1, v1)
    assert k_view.shape == (1, 4, 100, 64)
    assert torch.allclose(k_view, k1)
    assert cache.allocated_bytes() == initial_alloc

    # Append 40 tokens -> triggers growth to 256
    k2 = torch.randn(1, 4, 40, 64)
    v2 = torch.randn(1, 4, 40, 64)
    k_view2, v_view2 = cache.update(0, k2, v2)
    assert k_view2.shape == (1, 4, 140, 64)
    assert torch.allclose(k_view2[:, :, :100, :], k1)
    assert torch.allclose(k_view2[:, :, 100:, :], k2)
    assert cache.get_seq_len(0) == 140
    assert cache.allocated_bytes() > initial_alloc


def test_copy_batch_isolates_prefix_memory_and_overflow_raises():
    cache = DynamicKVCache(
        n_layers=1, batch_size=1, n_kv_heads=2, d_head=16, max_seq_len=32, chunk_size=16, dtype=torch.float32
    )
    k_prompt = torch.ones(1, 2, 10, 16)
    v_prompt = torch.full((1, 2, 10, 16), 2.0)
    cache.update(0, k_prompt, v_prompt)

    copied = cache.copy_batch(4)
    assert copied.batch_size == 4
    assert copied.get_seq_len(0) == 10

    k_next = torch.arange(4, dtype=torch.float32).view(4, 1, 1, 1).expand(4, 2, 1, 16)
    k_out, _ = copied.update(0, k_next, k_next)
    assert k_out.shape == (4, 2, 11, 16)
    assert torch.allclose(k_out[0, :, 10, :], torch.zeros(2, 16))
    assert torch.allclose(k_out[3, :, 10, :], torch.full((2, 16), 3.0))

    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        copied.update(0, torch.zeros(4, 2, 25, 16), torch.zeros(4, 2, 25, 16))
