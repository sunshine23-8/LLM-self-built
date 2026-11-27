import torch
from src.cache.shared_prefix_kv_cache import SharedPrefixKVCache
from src.inference.generator import AutoregressiveGenerator
from src.inference.scaling import CandidateChain, TestTimeScaler, majority_vote, weighted_majority_vote
from src.model.config import ModelConfig
from src.model.transformer import DecoderOnlyLM
from src.tokenizer.bpe import ByteLevelBPETokenizer


def test_majority_vote_and_weighted_vote_on_shared_pool():
    answers = ["42", "42", "42", "36", "36", "50", "50", "100"]
    cands = [
        CandidateChain(
            text=f"<think>ok</think>\n#### {ans}",
            think_text="ok",
            extracted_answer=ans,
            strict_format_valid=True,
            extraction_source="hash",
            think_tokens=5,
            generated_tokens=10,
            mean_logprob=-0.5,
            heuristic_score=0.9 if ans == "42" else 0.3,
            orm_score=0.1 if ans == "42" else (0.95 if ans == "36" else 0.05),
        )
        for ans in answers
    ]
    assert majority_vote(cands) == "42"
    assert weighted_majority_vote(cands, score_attr="orm_score") == "36"


def test_shared_pool_scaler_and_shared_prefix_kv_cache():
    tok = ByteLevelBPETokenizer()
    tok.train(["<|user|>\nWhat is 2+2?\n<|assistant|>\n<think>\n2+2=4\n</think>\n#### 4"], vocab_size=300)
    cfg = ModelConfig(vocab_size=tok.vocab_size, block_size=128, d_model=32, n_layers=1, n_heads=4, n_kv_heads=2, d_ff=64)
    model = DecoderOnlyLM(cfg).eval()
    gen = AutoregressiveGenerator(model, pad_token_id=tok.token_to_id("<|pad|>"), eos_token_id=tok.token_to_id("<|eos|>"))

    scaler = TestTimeScaler(gen, tok, max_batch_k=4)
    rec = scaler.evaluate_question("What is 2+2?", "4", k_pool=8, max_new_tokens=12, temperature=0.8)
    assert len(rec.pool) == 8
    assert rec.pool_generated_tokens > 0
    assert rec.pool_forward_passes > 0

    sp_cache = SharedPrefixKVCache(n_layers=1, n_kv_heads=2, d_head=16, max_seq_len=32, dtype=torch.float32)
    sp_cache.set_prefix(0, torch.ones(1, 2, 6, 16), torch.ones(1, 2, 6, 16), num_branches=4)
    k_all, _ = sp_cache.append_decode(0, torch.zeros(4, 2, 1, 16), torch.zeros(4, 2, 1, 16))
    assert k_all.shape == (4, 2, 7, 16)
    assert sp_cache.prefix_bytes() < sp_cache.decode_bytes() * 6 + sp_cache.prefix_bytes()
