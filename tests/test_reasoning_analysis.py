from pathlib import Path
from src.eval.reasoning_analysis import (
    analyze_length_buckets,
    classify_error_category,
    compute_correlations,
    detect_repetition_loop,
    plot_reasoning_and_compute_charts,
)
from src.inference.scaling import CandidateChain


def test_repetition_loop_correlation_buckets_and_error_taxonomy(tmp_path: Path):
    loop_text = " ".join(["step repeat loop check now"] * 6)
    assert detect_repetition_loop(loop_text, ngram_n=4, min_repeats=3) is True
    assert detect_repetition_loop("Step 1 add 2. Step 2 multiply by 3.", ngram_n=4, min_repeats=3) is False

    cands = [
        CandidateChain("<think>\n2+2=4\n</think>\n#### 4", "2+2=4", "4", True, "hash", 40, 50, -0.3, 1.0, 0.9),
        CandidateChain("<think>\n5+5=10\n</think>\n#### 10", "5+5=10", "10", True, "hash", 80, 95, -0.4, 1.0, 0.8),
        CandidateChain(f"<think>\n{loop_text}", loop_text, None, False, "none", 300, 320, -1.2, 0.0, 0.1),
        CandidateChain("<think>\n2*3=99\n</think>\n#### 99", "2*3=99", "99", True, "hash", 150, 170, -0.7, 0.5, 0.2),
    ]
    golds = ["4", "10", "20", "6"]

    corrs = compute_correlations([c.think_tokens for c in cands], [True, True, False, False])
    assert -1.0 <= corrs["spearman_rho"] <= 1.0

    bucket_report = analyze_length_buckets(cands, golds)
    assert len(bucket_report["buckets"]) == 5
    assert bucket_report["mean_len_correct"] == 60.0
    assert bucket_report["mean_len_incorrect"] == 225.0

    assert classify_error_category(cands[2], "20", oracle_had_correct=False) == "reasoning_loop"
    assert classify_error_category(cands[3], "6", oracle_had_correct=False) == "arithmetic_error"
    assert classify_error_category(cands[3], "6", oracle_had_correct=True) == "verifier_error"

    compute_pts = [
        {"name": "E1 Greedy", "avg_tokens": 50.0, "accuracy": 0.4},
        {"name": "E4 Maj@16", "avg_tokens": 800.0, "accuracy": 0.65},
    ]
    charts = plot_reasoning_and_compute_charts(bucket_report["buckets"], compute_pts, tmp_path)
    assert len(charts) == 2
    assert all(p.exists() for p in charts)
