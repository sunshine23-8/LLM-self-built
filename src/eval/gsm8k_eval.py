from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import re

_HASH_RE = re.compile(r"####\s*([-+]?\$?\d[\d,]*(?:\.\d+)?(?:/\d+)?)")
_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")


def normalize_numeric_str(raw: str) -> Decimal | None:
    if raw is None:
        return None
    cleaned = raw.strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    if not cleaned:
        return None
    try:
        if "/" in cleaned and not cleaned.startswith("/"):
            frac = Fraction(cleaned)
            return Decimal(frac.numerator) / Decimal(frac.denominator)
        return Decimal(cleaned)
    except (InvalidOperation, ZeroDivisionError, ValueError):
        return None


def is_numeric_equal(pred_str: str | None, gold_str: str) -> bool:
    if pred_str is None:
        return False
    pred_dec = normalize_numeric_str(pred_str)
    gold_dec = normalize_numeric_str(gold_str)
    if pred_dec is None or gold_dec is None:
        return False
    return abs(pred_dec - gold_dec) <= Decimal("1e-6")


def extract_gsm8k_prediction(text: str) -> tuple[str | None, bool, str]:
    has_think_open = "<think>" in text
    has_think_close = "</think>" in text
    after_think = text.split("</think>", 1)[1] if has_think_close else text

    hash_match = _HASH_RE.search(after_think)
    if hash_match:
        strict_ok = has_think_open and has_think_close
        return hash_match.group(1), strict_ok, "hash"

    boxed_matches = _BOXED_RE.findall(after_think)
    if boxed_matches:
        num_match = _NUM_RE.search(boxed_matches[-1])
        if num_match:
            return num_match.group(0), False, "boxed"

    all_nums = _NUM_RE.findall(after_think)
    if all_nums:
        return all_nums[-1], False, "fallback_last_num"

    return None, False, "none"


@dataclass
class GSM8KMetricSummary:
    total: int
    pass_at_1_greedy: float
    maj_at_4: float
    maj_at_8: float
    maj_at_16: float
    bon_at_8_heuristic: float
    bon_at_16_heuristic: float
    bon_at_8_orm: float | None
    bon_at_16_orm: float | None
    weighted_maj_at_16: float | None
    oracle_pass_at_16: float
    strict_format_rate: float
