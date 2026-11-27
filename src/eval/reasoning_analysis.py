from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.eval.gsm8k_eval import is_numeric_equal
from src.inference.scaling import CandidateChain
from src.inference.scorer import safe_check_arithmetic_steps


def detect_repetition_loop(text: str, ngram_n: int = 8, min_repeats: int = 3) -> bool:
    words = text.split()
    if len(words) < ngram_n * min_repeats:
        return False
    grams = [tuple(words[i : i + ngram_n]) for i in range(len(words) - ngram_n + 1)]
    counts = Counter(grams)
    return any(cnt >= min_repeats for cnt in counts.values())


def _rank_array(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_x = x[order]
    n = len(x)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def compute_correlations(think_lengths: list[int], correct_flags: list[bool]) -> dict[str, float]:
    x = np.asarray(think_lengths, dtype=np.float64)
    y = np.asarray([1.0 if f else 0.0 for f in correct_flags], dtype=np.float64)
    return {
        "pearson_point_biserial": _pearson(x, y),
        "spearman_rho": _pearson(_rank_array(x), _rank_array(y)),
    }


def analyze_length_buckets(candidates: list[CandidateChain], gold_answers: list[str]) -> dict:
    bounds = [("<64", 0, 64), ("64-128", 64, 128), ("128-256", 128, 256), ("256-512", 256, 512), (">512", 512, 10**9)]
    buckets: list[dict] = []
    correct_lens: list[int] = []
    incorrect_lens: list[int] = []

    for c, gold in zip(candidates, gold_answers):
        if is_numeric_equal(c.extracted_answer, gold):
            correct_lens.append(c.think_tokens)
        else:
            incorrect_lens.append(c.think_tokens)

    for label, lo, hi in bounds:
        items = [(c, g) for c, g in zip(candidates, gold_answers) if lo <= c.think_tokens < hi]
        n = len(items)
        if n == 0:
            buckets.append({"bucket": label, "count": 0, "valid_format_rate": 0.0, "accuracy": 0.0, "repetition_loop_rate": 0.0})
            continue
        fmt_ok = sum(1 for c, _ in items if c.strict_format_valid) / n
        acc = sum(1 for c, g in items if is_numeric_equal(c.extracted_answer, g)) / n
        loops = sum(1 for c, _ in items if detect_repetition_loop(c.think_text)) / n
        buckets.append(
            {
                "bucket": label,
                "count": n,
                "valid_format_rate": fmt_ok,
                "accuracy": acc,
                "repetition_loop_rate": loops,
            }
        )

    return {
        "buckets": buckets,
        "mean_len_correct": float(np.mean(correct_lens)) if correct_lens else 0.0,
        "mean_len_incorrect": float(np.mean(incorrect_lens)) if incorrect_lens else 0.0,
        "note": "Correlation metrics measure statistical association only, not causal effect of CoT length on accuracy.",
    }


def classify_error_category(cand: CandidateChain, gold_answer: str, oracle_had_correct: bool = False) -> str:
    if oracle_had_correct and not is_numeric_equal(cand.extracted_answer, gold_answer):
        return "verifier_error"
    if detect_repetition_loop(cand.think_text):
        return "reasoning_loop"
    if not cand.strict_format_valid and cand.extracted_answer is None:
        return "answer_extraction_error"
    if "\ufffd" in cand.text:
        return "tokenization_error"
    valid_eqs, total_eqs = safe_check_arithmetic_steps(cand.think_text)
    if total_eqs > 0 and valid_eqs < total_eqs:
        return "arithmetic_error"
    return "logical_error"


def plot_reasoning_and_compute_charts(
    bucket_stats: list[dict],
    compute_points: list[dict],
    out_dir: str | Path,
) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Accuracy & Format & Loop Rate vs <think> Length Buckets
    labels = [b["bucket"] for b in bucket_stats]
    accs = [b["accuracy"] * 100.0 for b in bucket_stats]
    loops = [b["repetition_loop_rate"] * 100.0 for b in bucket_stats]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(labels))
    ax.bar(x - 0.18, accs, width=0.36, label="Accuracy (%)")
    ax.bar(x + 0.18, loops, width=0.36, label="Repetition Loop Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("GSM8K Accuracy & Repetition Loops vs <think> Length Bucket")
    ax.set_xlabel("<think> Token Count Bucket")
    ax.set_ylabel("Percentage (%)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    p1 = out / "accuracy_vs_think_length.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Compute-Budget-Matched Frontier (Accuracy vs Total Generated Tokens)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    xs = [pt["avg_tokens"] for pt in compute_points]
    ys = [pt["accuracy"] * 100.0 for pt in compute_points]
    ax.plot(xs, ys, marker="o", linestyle="-")
    for pt in compute_points:
        ax.annotate(pt["name"], (pt["avg_tokens"], pt["accuracy"] * 100.0), textcoords="offset points", xytext=(5, 5))
    ax.set_title("Compute-Accuracy Frontier (Test-Time Scaling)")
    ax.set_xlabel("Avg Generated Tokens / Question")
    ax.set_ylabel("GSM8K Accuracy (%)")
    ax.grid(True, alpha=0.3)
    p2 = out / "compute_accuracy_frontier.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return [p1, p2]
