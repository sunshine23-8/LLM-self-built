import torch
from src.data.cot_dataset import (
    CoTSFTDataset,
    clean_gsm8k_reasoning_and_answer,
    cot_collate_fn,
    format_cot_prompt,
    format_cot_response,
)
from src.tokenizer.bpe import ByteLevelBPETokenizer


def test_clean_gsm8k_strips_calculator_tags_and_splits_answer():
    raw = "She sold 48/2 = <<48/2=24>>24 clips.\nTotal = 48+24 = <<48+24=72>>72.\n#### 72"
    reasoning, final_ans = clean_gsm8k_reasoning_and_answer(raw)
    assert "<<" not in reasoning
    assert "48/2 = 24 clips." in reasoning
    assert final_ans == "72"


def test_cot_sft_dataset_masks_prompt_tokens_only():
    tok = ByteLevelBPETokenizer()
    examples = [
        {"question": "What is 2 + 3?", "reasoning": "2 + 3 = 5", "final_answer": "5"},
        {"question": "What is 10 - 4?", "reasoning": "10 - 4 = 6", "final_answer": "6"},
    ]
    tok.train([format_cot_prompt(e["question"]) + format_cot_response(e["reasoning"], e["final_answer"]) for e in examples], vocab_size=300)

    ds = CoTSFTDataset(examples, tok, max_seq_len=512)
    assert len(ds) == 2

    prompt_len = len(tok.encode(format_cot_prompt(examples[0]["question"])))
    ids0, labels0 = ds[0]
    assert torch.all(labels0[:prompt_len] == -100)
    assert torch.equal(labels0[prompt_len:], ids0[prompt_len:])
    assert ids0[-1].item() == tok.token_to_id("<|eos|>")

    batch_ids, batch_labels = cot_collate_fn([ds[0], ds[1]], pad_token_id=tok.token_to_id("<|pad|>"))
    assert batch_ids.shape == batch_labels.shape
    assert batch_labels[1, -1].item() == -100 or batch_ids[1, -1].item() == tok.token_to_id("<|eos|>")
