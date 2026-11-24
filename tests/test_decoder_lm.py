import torch
from src.model.config import ModelConfig
from src.model.transformer import DecoderOnlyLM


def test_decoder_lm_parameter_count_and_weight_tying():
    cfg = ModelConfig.from_yaml("configs/compact_12gb.yaml")
    model = DecoderOnlyLM(cfg)
    actual_params = sum(p.numel() for p in model.parameters())
    assert actual_params == cfg.count_parameters()
    assert model.embed_tokens.weight is model.lm_head.weight


def test_decoder_lm_kv_cache_logits_match_uncached():
    torch.manual_seed(7)
    cfg = ModelConfig(vocab_size=512, block_size=64, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=128)
    model = DecoderOnlyLM(cfg).eval()
    tokens = torch.randint(0, 512, (1, 10))

    with torch.no_grad():
        logits_full, hidden_full = model.forward_with_hidden(tokens)
        cache = model.create_kv_cache(batch_size=1, dtype=torch.float32, device="cpu")
        logits_pre = model(tokens[:, :6], kv_cache=cache, start_pos=0)
        step_logits = [logits_pre]
        for pos in range(6, 10):
            step_logits.append(model(tokens[:, pos : pos + 1], kv_cache=cache, start_pos=pos))
        logits_cached = torch.cat(step_logits, dim=1)

    assert hidden_full.shape == (1, 10, 64)
    diff = (logits_full - logits_cached).abs()
    assert diff.max().item() < 1e-5
    assert diff.mean().item() < 1e-6
