import torch
from src.cache.dynamic_kv_cache import DynamicKVCache
from src.model.attention import GroupedQueryAttention
from src.model.config import ModelConfig
from src.model.ffn import SwiGLUFFN
from src.model.norm import RMSNorm
from src.model.rope import apply_rope, precompute_rope_freqs


def test_rmsnorm_and_swiglu_shapes():
    norm = RMSNorm(64)
    x = torch.randn(2, 10, 64)
    y = norm(x)
    assert y.shape == x.shape
    rms = y.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)

    ffn = SwiGLUFFN(64, 128)
    assert ffn(x).shape == x.shape


def test_rope_prefill_vs_step_equivalence():
    cos, sin = precompute_rope_freqs(d_head=16, max_seq_len=64)
    q = torch.randn(1, 4, 8, 16)
    k = torch.randn(1, 2, 8, 16)

    q_full, k_full = apply_rope(q, k, cos, sin, start_pos=0)
    q_step, k_step = apply_rope(q[:, :, 5:6, :], k[:, :, 5:6, :], cos, sin, start_pos=5)
    assert torch.allclose(q_full[:, :, 5:6, :], q_step, atol=1e-6)
    assert torch.allclose(k_full[:, :, 5:6, :], k_step, atol=1e-6)


def test_gqa_cached_step_matches_uncached_full_forward():
    torch.manual_seed(42)
    cfg = ModelConfig(vocab_size=256, block_size=64, d_model=64, n_layers=1, n_heads=4, n_kv_heads=2, d_ff=128)
    attn = GroupedQueryAttention(cfg, layer_idx=0).eval()
    cos, sin = precompute_rope_freqs(cfg.d_head, cfg.block_size, cfg.rope_theta)

    x = torch.randn(1, 12, 64)
    with torch.no_grad():
        full_out = attn(x, cos, sin, start_pos=0, kv_cache=None)

        cache = DynamicKVCache(1, 1, cfg.n_kv_heads, cfg.d_head, cfg.block_size, dtype=torch.float32)
        prefill_out = attn(x[:, :8, :], cos, sin, start_pos=0, kv_cache=cache)
        step_outs = [prefill_out]
        for pos in range(8, 12):
            step_outs.append(attn(x[:, pos : pos + 1, :], cos, sin, start_pos=pos, kv_cache=cache))
        cached_out = torch.cat(step_outs, dim=1)

    assert torch.allclose(full_out, cached_out, atol=1e-5)


def test_gqa_manual_matches_sdpa_backend():
    torch.manual_seed(11)
    cfg_manual = ModelConfig(
        vocab_size=256, block_size=64, d_model=64, n_layers=1, n_heads=4, n_kv_heads=2, d_ff=128, attn_backend="manual"
    )
    cfg_sdpa = ModelConfig(
        vocab_size=256, block_size=64, d_model=64, n_layers=1, n_heads=4, n_kv_heads=2, d_ff=128, attn_backend="sdpa"
    )
    attn_manual = GroupedQueryAttention(cfg_manual, layer_idx=0).eval()
    attn_sdpa = GroupedQueryAttention(cfg_sdpa, layer_idx=0).eval()
    attn_sdpa.load_state_dict(attn_manual.state_dict())

    cos, sin = precompute_rope_freqs(cfg_manual.d_head, cfg_manual.block_size, cfg_manual.rope_theta)
    x = torch.randn(2, 10, 64)
    with torch.no_grad():
        y_manual = attn_manual(x, cos, sin)
        y_sdpa = attn_sdpa(x, cos, sin)
    assert torch.allclose(y_manual, y_sdpa, atol=1e-5)
