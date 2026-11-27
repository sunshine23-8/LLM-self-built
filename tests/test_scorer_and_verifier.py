import torch
from src.data.rollout_dataset import RolloutVerifierDataset, rollout_collate_fn, split_questions_train_val
from src.inference.scorer import HeuristicCoTScorer, OutcomeVerifierModel, get_final_answer_token_indices, safe_check_arithmetic_steps
from src.model.config import ModelConfig
from src.model.transformer import DecoderOnlyLM
from src.tokenizer.bpe import ByteLevelBPETokenizer
from src.training.verifier_trainer import train_orm_verifier


def test_heuristic_scorer_rewards_valid_format_and_consistent_arithmetic():
    valid_eqs, total_eqs = safe_check_arithmetic_steps("Step 1: 12 + 8 = 20\nStep 2: 20 * 3 = 60\nStep 3: 5 + 5 = 99")
    assert total_eqs == 3
    assert valid_eqs == 2

    scorer = HeuristicCoTScorer()
    good = "<think>\n12 + 8 = 20\n20 * 3 = 60\n</think>\n#### 60"
    bad = "<think>\n12 + 8 = 99\n"
    assert scorer.score(good, mean_logprob=-0.5) > scorer.score(bad, mean_logprob=-0.5)


def test_orm_pools_final_answer_token_before_eos_and_trains():
    tok = ByteLevelBPETokenizer()
    pad_id = tok.token_to_id("<|pad|>")
    eos_id = tok.token_to_id("<|eos|>")

    ids = torch.tensor(
        [
            [1, 10, 11, 12, eos_id, pad_id, pad_id],
            [1, 10, 11, 12, 13, 14, eos_id],
        ],
        dtype=torch.long,
    )
    pool_idx = get_final_answer_token_indices(ids, pad_token_id=pad_id, eos_token_id=eos_id)
    assert pool_idx.tolist() == [3, 5]

    records = [
        {"question": f"Q{i}?", "rollout_text": f"<think>\n1+1=2\n</think>\n#### {2 if i % 2 == 0 else 3}", "label": float(i % 2 == 0)}
        for i in range(10)
    ]
    train_recs, val_recs = split_questions_train_val(records, val_ratio=0.2, seed=42)
    assert {r["question"] for r in train_recs}.isdisjoint({r["question"] for r in val_recs})

    cfg = ModelConfig(vocab_size=tok.vocab_size, block_size=64, d_model=32, n_layers=1, n_heads=4, n_kv_heads=2, d_ff=64)
    orm = OutcomeVerifierModel(DecoderOnlyLM(cfg))
    ds = RolloutVerifierDataset(train_recs, tok, max_seq_len=64)
    batch_ids, _ = rollout_collate_fn([ds[0], ds[1]], pad_token_id=pad_id)
    probs = orm.score_sequences(batch_ids, pad_token_id=pad_id, eos_token_id=eos_id)
    assert probs.shape == (2,)
    assert torch.all((probs >= 0.0) & (probs <= 1.0))

    history = train_orm_verifier(orm, ds, pad_token_id=pad_id, eos_token_id=eos_id, epochs=1, batch_size=2, lr=1e-3)
    assert len(history["loss"]) > 0
