from src.data.leakage_filter import LeakageFilter, normalize_question


def test_normalize_question_standardizes_case_numbers_and_punctuation():
    assert normalize_question("How many apples does Jane have? (1,200 total!)") == "how many apples does jane have 1200 total"


def test_leakage_filter_detects_exact_normalized_and_near_duplicates():
    test_qs = [
        "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast.",
        "A robe takes 2 bolts of blue fiber and half that much white fiber.",
    ]
    flt = LeakageFilter(test_qs, jaccard_threshold=0.80, ngram_n=5)

    assert flt.classify(test_qs[0]) == "exact"
    assert flt.classify("  JANET'S ducks lay 16 eggs per day! She eats 3 for breakfast. ") == "normalized"
    assert flt.classify("Janet's ducks lay 16 eggs per day. She eats 3 eggs for breakfast.") == "near_duplicate"
    assert flt.classify("Compute the area of a circle with radius 7 meters.") == "clean"

    records = [
        {"question": test_qs[0], "id": 1},
        {"question": "Solve 15 * 4 + 10.", "id": 2},
    ]
    kept, stats = flt.filter_records(records)
    assert [r["id"] for r in kept] == [2]
    assert stats["exact"] == 1
    assert stats["clean"] == 1
