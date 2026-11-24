import torch
from src.inference.generator import AutoregressiveGenerator
from src.model.config import ModelConfig
from src.model.transformer import DecoderOnlyLM


def test_greedy_generation_kv_cache_on_vs_off_exact_match():
    torch.manual_seed(123)
    cfg = ModelConfig(vocab_size=256, block_size=128, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=128)
    model = DecoderOnlyLM(cfg).eval()
    gen = AutoregressiveGenerator(model, pad_token_id=0, eos_token_id=255)  # avoid early stop on id 2

    prompt = torch.randint(1, 200, (1, 12))
    res_on = gen.generate(prompt, max_new_tokens=32, temperature=0.0, use_cache=True)
    res_off = gen.generate(prompt, max_new_tokens=32, temperature=0.0, use_cache=False)

    assert torch.equal(res_on.sequences, res_off.sequences)
    assert res_on.sequences.shape == (1, 44)
    assert res_on.token_logprobs.shape == (1, 32)
    assert len(res_on.next_token_latencies_ms) == 31


def test_batched_copy_kv_cache_generates_k_distinct_sequences():
    torch.manual_seed(99)
    cfg = ModelConfig(vocab_size=256, block_size=128, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=128)
    model = DecoderOnlyLM(cfg).eval()
    gen = AutoregressiveGenerator(model, pad_token_id=0, eos_token_id=255)

    prompt = torch.randint(1, 200, (1, 8))
    res = gen.generate(
        prompt, max_new_tokens=16, temperature=0.9, top_p=0.95, use_cache=True, stop_token_ids=[], num_return_sequences=4
    )
    assert res.sequences.shape == (4, 24)
    assert res.token_logprobs.shape == (4, 16)
    assert res.forward_passes == 16
    assert res.generated_tokens == 64
    assert torch.equal(res.sequences[:, :8], prompt.expand(4, -1))
