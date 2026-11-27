from src.eval.gsm8k_eval import extract_gsm8k_prediction, is_numeric_equal, normalize_numeric_str


def test_normalize_numeric_str_handles_currency_commas_decimals_and_fractions():
    assert normalize_numeric_str("$1,250.00") == normalize_numeric_str("1250")
    assert normalize_numeric_str("-42.5%") == normalize_numeric_str("-42.5")
    assert normalize_numeric_str("3/4") == normalize_numeric_str("0.75")
    assert normalize_numeric_str("no_number") is None


def test_extract_gsm8k_prediction_prioritizes_hash_then_boxed_then_fallback():
    strict_text = "<think>\nStep 1: 20 + 22 = 42\n</think>\n#### $42.00"
    pred, strict_ok, src = extract_gsm8k_prediction(strict_text)
    assert strict_ok is True
    assert src == "hash"
    assert is_numeric_equal(pred, "42")

    boxed_text = "<think>\nCalculation done\n</think>\nTherefore \\boxed{1,200}."
    pred_b, strict_b, src_b = extract_gsm8k_prediction(boxed_text)
    assert strict_b is False
    assert src_b == "boxed"
    assert is_numeric_equal(pred_b, "1200")

    unclosed_text = "<think>\n10 + 5 = 15"
    pred_u, strict_u, src_u = extract_gsm8k_prediction(unclosed_text)
    assert strict_u is False
    assert src_u == "fallback_last_num"
    assert is_numeric_equal(pred_u, "15")
