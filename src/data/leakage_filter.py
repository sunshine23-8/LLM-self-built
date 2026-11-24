import re
from typing import Iterable

_PUNCT_RE = re.compile(r"[^\w\s]")
_SPACE_RE = re.compile(r"\s+")
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")


def normalize_question(text: str) -> str:
    cleaned = _THOUSANDS_RE.sub("", text.lower())
    cleaned = _PUNCT_RE.sub(" ", cleaned)
    return _SPACE_RE.sub(" ", cleaned).strip()


def char_ngram_set(text: str, n: int = 5) -> set[str]:
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


class LeakageFilter:
    """3-tier duplicate detector (exact, normalized, n-gram Jaccard) against benchmark test questions."""

    def __init__(
        self,
        reference_questions: Iterable[str],
        jaccard_threshold: float = 0.80,
        ngram_n: int = 5,
    ) -> None:
        self.jaccard_threshold = jaccard_threshold
        self.ngram_n = ngram_n
        self._exact_set: set[str] = set()
        self._norm_set: set[str] = set()
        self._ref_ngrams: list[set[str]] = []

        for q in reference_questions:
            self._exact_set.add(q)
            norm = normalize_question(q)
            self._norm_set.add(norm)
            self._ref_ngrams.append(char_ngram_set(norm, n=self.ngram_n))

    def classify(self, candidate: str) -> str:
        if candidate in self._exact_set:
            return "exact"
        norm = normalize_question(candidate)
        if norm in self._norm_set:
            return "normalized"
        cand_grams = char_ngram_set(norm, n=self.ngram_n)
        for ref_grams in self._ref_ngrams:
            if jaccard_similarity(cand_grams, ref_grams) >= self.jaccard_threshold:
                return "near_duplicate"
        return "clean"

    def filter_records(self, records: Iterable[dict], question_key: str = "question") -> tuple[list[dict], dict[str, int]]:
        kept: list[dict] = []
        stats = {"clean": 0, "exact": 0, "normalized": 0, "near_duplicate": 0}
        for rec in records:
            label = self.classify(rec[question_key])
            stats[label] += 1
            if label == "clean":
                kept.append(rec)
        return kept, stats
