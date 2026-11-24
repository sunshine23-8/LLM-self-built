from pathlib import Path
from src.tokenizer.pretokenizer import pretokenize
from src.tokenizer.bpe import ByteLevelBPETokenizer, SPECIAL_TOKENS


def test_pretokenizer_splits_digits_individually():
    tokens = pretokenize("Sum 123 + 45 = 168\n")
    assert "123" not in tokens
    assert "45" not in tokens
    assert ["1", "2", "3"] == [t for t in tokens if t in {"1", "2", "3"}][:3]


def test_bpe_train_encode_decode_roundtrip_and_special_tokens(tmp_path: Path):
    corpus = [
        "Alice has 12 apples and buys 34 more apples.\n",
        "Total: 12 + 34 = 46 apples. #### 46\n",
        "Bob calculates 98 - 17 = 81 in his notebook.\n",
        "Unicode math check: 5 × 6 ÷ 2 = 15, √9 = 3.\n",
    ] * 10

    tok = ByteLevelBPETokenizer()
    tok.train(corpus, vocab_size=320)

    for idx, sp in enumerate(SPECIAL_TOKENS):
        assert tok.token_to_id(sp) == idx
        assert tok.encode(sp) == [idx]

    sample = "<|user|>\nWhat is 123 + 45?\n<|assistant|>\n<think>\n123 + 45 = 168\n</think>\n#### 168"
    ids = tok.encode(sample, add_bos=True, add_eos=True)
    assert ids[0] == tok.token_to_id("<|bos|>")
    assert ids[-1] == tok.token_to_id("<|eos|>")
    assert tok.token_to_id("<think>") in ids
    assert tok.token_to_id("</think>") in ids

    decoded = tok.decode(ids, skip_special_tokens=False)
    assert decoded == f"<|bos|>{sample}<|eos|>"

    save_dir = tmp_path / "tok"
    tok.save(save_dir)
    reloaded = ByteLevelBPETokenizer.load(save_dir)
    assert reloaded.encode(sample, add_bos=True, add_eos=True) == ids
    assert reloaded.decode(ids, skip_special_tokens=True) == "\nWhat is 123 + 45?\n\n\n123 + 45 = 168\n\n#### 168"
