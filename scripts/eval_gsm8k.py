import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
import torch

from src.eval.gsm8k_eval import GSM8KMetricSummary, is_numeric_equal
from src.eval.reasoning_analysis import (
    analyze_length_buckets,
    classify_error_category,
    compute_correlations,
    plot_reasoning_and_compute_charts,
)
from src.inference.generator import AutoregressiveGenerator
from src.inference.scaling import TestTimeScaler
from src.inference.scorer import HeuristicCoTScorer, OutcomeVerifierModel
from src.model.config import ModelConfig
from src.model.transformer import DecoderOnlyLM
from src.tokenizer.bpe import ByteLevelBPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CoT SFT + Test-Time Scaling on GSM8K Test")
    parser.add_argument("--config", type=Path, default=Path("configs/compact_12gb.yaml"))
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("artifacts/tokenizer"))
    parser.add_argument("--sft-ckpt", type=Path, required=True)
    parser.add_argument("--orm-ckpt", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/gsm8k_eval"))
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates full 1,319 GSM8K test split")
    parser.add_argument("--k-pool", type=int, default=16)
    parser.add_argument("--max-batch-k", type=int, default=8)
    args = parser.parse_args()

    cfg = ModelConfig.from_yaml(args.config)
    tok = ByteLevelBPETokenizer.load(args.tokenizer_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, cfg.dtype) if device == "cuda" else torch.float32

    model = DecoderOnlyLM(cfg).to(device=device, dtype=dtype)
    ckpt = torch.load(args.sft_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    orm: OutcomeVerifierModel | None = None
    if args.orm_ckpt is not None:
        orm = OutcomeVerifierModel(model).to(device=device, dtype=dtype)
        orm_ckpt = torch.load(args.orm_ckpt, map_location=device, weights_only=True)
        orm.load_state_dict(orm_ckpt["model_state_dict"])
        orm.eval()

    gen = AutoregressiveGenerator(model, pad_token_id=tok.token_to_id("<|pad|>"), eos_token_id=tok.token_to_id("<|eos|>"))
    scaler = TestTimeScaler(gen, tok, HeuristicCoTScorer(), orm, max_batch_k=args.max_batch_k)

    test_ds = list(load_dataset("openai/gsm8k", "main", split="test"))
    if args.limit > 0:
        test_ds = test_ds[: args.limit]

    records = []
    error_samples = []
    first_cands = []
    golds = []

    for row in test_ds:
        gold_num = row["answer"].rsplit("####", 1)[-1].strip()
        rec = scaler.evaluate_question(row["question"], gold_num, k_pool=args.k_pool)
        records.append(rec)
        first_cands.append(rec.pool[0])
        golds.append(gold_num)
        if not rec.maj_at_16 and len(error_samples) < 20:
            err_cat = classify_error_category(rec.pool[0], gold_num, oracle_had_correct=rec.oracle_pass_at_16)
            error_samples.append({"question": row["question"], "gold": gold_num, "category": err_cat, "trace": rec.pool[0].text})

    n = len(records)
    summary = GSM8KMetricSummary(
        total=n,
        pass_at_1_greedy=sum(r.greedy_correct for r in records) / n,
        maj_at_4=sum(r.maj_at_4 for r in records) / n,
        maj_at_8=sum(r.maj_at_8 for r in records) / n,
        maj_at_16=sum(r.maj_at_16 for r in records) / n,
        bon_at_8_heuristic=sum(r.bon_at_8_heuristic for r in records) / n,
        bon_at_16_heuristic=sum(r.bon_at_16_heuristic for r in records) / n,
        bon_at_8_orm=(sum(bool(r.bon_at_8_orm) for r in records) / n) if orm else None,
        bon_at_16_orm=(sum(bool(r.bon_at_16_orm) for r in records) / n) if orm else None,
        weighted_maj_at_16=(sum(bool(r.weighted_maj_at_16) for r in records) / n) if orm else None,
        oracle_pass_at_16=sum(r.oracle_pass_at_16 for r in records) / n,
        strict_format_rate=sum(c.strict_format_valid for c in first_cands) / n,
    )

    corrs = compute_correlations([c.think_tokens for c in first_cands], [is_numeric_equal(c.extracted_answer, g) for c, g in zip(first_cands, golds)])
    bucket_report = analyze_length_buckets(first_cands, golds)
    avg_greedy_toks = sum(r.greedy_tokens for r in records) / n
    avg_pool_toks = sum(r.pool_generated_tokens for r in records) / n
    compute_points = [
        {"name": "E1 Greedy (K=1)", "avg_tokens": avg_greedy_toks, "accuracy": summary.pass_at_1_greedy},
        {"name": "E2 Maj@4", "avg_tokens": avg_pool_toks * 0.25, "accuracy": summary.maj_at_4},
        {"name": "E3 Maj@8", "avg_tokens": avg_pool_toks * 0.50, "accuracy": summary.maj_at_8},
        {"name": "E4 Maj@16", "avg_tokens": avg_pool_toks, "accuracy": summary.maj_at_16},
        {"name": "E5 BoN@16 (Heur)", "avg_tokens": avg_pool_toks, "accuracy": summary.bon_at_16_heuristic},
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"metrics": asdict(summary), "correlations": corrs, "length_analysis": bucket_report}, f, indent=2)
    with (args.out_dir / "error_cases_sample.json").open("w", encoding="utf-8") as f:
        json.dump(error_samples, f, indent=2, ensure_ascii=False)
    plot_reasoning_and_compute_charts(bucket_report["buckets"], compute_points, args.out_dir)
    print(f"Saved GSM8K evaluation report to {args.out_dir}")


if __name__ == "__main__":
    main()
