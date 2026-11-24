# Training Pipeline & Track C (Reasoning & Test-Time Compute) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify the staged training pipeline (Anti-Leakage Filter, Packed Pre-training + Throughput Calibration, Math Mid-training, CoT SFT with `<think>...</think>`) and the complete Track C Reasoning evaluation stack (Greedy, Shared-Pool Majority Voting, Heuristic & Learned ORM Best-of-N, Compute-Budget Comparison, CoT Length Telemetry, and 6-Category Error Analysis on 1,319 GSM8K Test Split).

**Architecture:** Consumes the Core LLM stack built in Plan 1 (`ModelConfig`, `ByteLevelBPETokenizer`, `DecoderOnlyLM`, `DynamicKVCache`, `AutoregressiveGenerator`). Implements tasks in strict priority order (**P0 → P1 → P2 → P3**) with stage checkpoints so Core Foundation (`P0`) and Majority Voting (`P1`) are never blocked by optional extensions (`P3`: ORM Verifier, Weighted Voting, `SharedPrefixKVCache`).

**Tech Stack:** Python 3.10+, PyTorch 2.2+, HuggingFace `datasets` (raw data download only), NumPy, Matplotlib, Pytest.

**Spec:** [spec.md](file:///d:/DUT.AI/LLM%20self-built/LLM-self-built/specs/001-llm-reasoning-system/spec.md)  
**Prerequisite Plan:** [2026-11-10-core-llm-kv-cache.md](file:///d:/DUT.AI/LLM%20self-built/LLM-self-built/docs/superpowers/plans/2026-11-10-core-llm-kv-cache.md)

## Global Constraints

- **Priority order**: `P0` (Throughput calibration & Pre-train checkpoint) → `P1` (Anti-leakage, Mid-train, CoT SFT, Greedy `Pass@1`, Majority Voting `Maj@K`, GSM8K 1,319 eval) → `P2` (Heuristic Scorer `BoN@K`, Compute-budget tracking, CoT length 4-metric analysis, ≥5 Error Analysis cases) → `P3` (Learned ORM Verifier, Verifier-Weighted Voting, `SharedPrefixKVCache`).
- **Pre-training budget policy**: Measure actual GPU `tokens/sec` via `scripts/benchmark_throughput.py` before locking training steps; save reproducible checkpoints after each stage (`Pre-train` → `Mid-train` → `CoT SFT` → `ORM`).
- **Anti-leakage**: Filter `Open-R1` and `GSM8K train` against all 1,319 GSM8K test questions via Exact + Normalized + Near-Duplicate (`5-gram Jaccard >= 0.80`).
- **Fair Test-Time Scaling**: Generate one shared pool of $K=16$ rollouts ($T=0.7, \text{top\_p}=0.95$) per test question and evaluate `Maj@4`, `Maj@8`, `Maj@16`, `BoN@8`, `BoN@16`, and `Pass@16` (reported explicitly as **Oracle Upper Bound**) on that same pool alongside deterministic `Greedy Pass@1`.
- **ORM Verifier**: Question-level train/val rollout split (no shared question across splits), pool hidden state at the **final answer token** (last non-pad token before `<|eos|>`), train with `BCEWithLogitsLoss`.
- **Follow TDD strictly**: Write failing test → confirm failure → implement minimal code → confirm pass → commit.

---

### Task 1 [P1]: 3-Tier Anti-Leakage Filter (`src/data/leakage_filter.py`)

**Files:**
- Create: `src/data/__init__.py`
- Create: `src/data/leakage_filter.py`
- Test: `tests/test_leakage_filter.py`

**Interfaces:**
- Produces:
  - `normalize_question(text: str) -> str`
  - `char_ngram_set(text: str, n: int = 5) -> set[str]`
  - `jaccard_similarity(a: set[str], b: set[str]) -> float`
  - `LeakageFilter(reference_questions: Iterable[str], jaccard_threshold: float = 0.80, ngram_n: int = 5)`
  - `LeakageFilter.classify(candidate: str) -> str` (`"clean"`, `"exact"`, `"normalized"`, `"near_duplicate"`)
  - `LeakageFilter.filter_records(records: Iterable[dict], question_key: str = "question") -> tuple[list[dict], dict[str, int]]`

- [ ] **Step 1: Write the failing test in `tests/test_leakage_filter.py`**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_leakage_filter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data'`

- [ ] **Step 3: Implement `src/data/leakage_filter.py`**

Create empty `src/data/__init__.py`.

```python
# src/data/leakage_filter.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_leakage_filter.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/data/__init__.py src/data/leakage_filter.py tests/test_leakage_filter.py
git commit -m "feat: add 3-tier anti-leakage filter for GSM8K test protection"
```

---

### Task 2 [P0]: Packed Token Dataset & Throughput Benchmark (`src/data/pretrain_dataset.py`, `scripts/benchmark_throughput.py`)

**Files:**
- Create: `src/data/pretrain_dataset.py`
- Create: `scripts/benchmark_throughput.py`
- Test: `tests/test_pretrain_dataset.py`

**Interfaces:**
- Consumes: `ByteLevelBPETokenizer` (`src/tokenizer/bpe.py`), `DecoderOnlyLM` (`src/model/transformer.py`)
- Produces:
  - `pack_texts_to_bin(texts: Iterable[str], tokenizer: ByteLevelBPETokenizer, out_path: str | Path) -> int`
  - `PackedTokenDataset(bin_path: str | Path, block_size: int)` returning `(x, y)` tensors of shape `(block_size,)`
  - `benchmark_training_throughput(model: DecoderOnlyLM, batch_size: int, seq_len: int, num_steps: int = 20, warmup_steps: int = 5, dtype: torch.dtype = torch.float32) -> dict[str, float]`

- [ ] **Step 1: Write the failing test in `tests/test_pretrain_dataset.py`**

```python
from pathlib import Path
import torch
from src.data.pretrain_dataset import PackedTokenDataset, benchmark_training_throughput, pack_texts_to_bin
from src.model.config import ModelConfig
from src.model.transformer import DecoderOnlyLM
from src.tokenizer.bpe import ByteLevelBPETokenizer


def test_pack_texts_and_memmap_dataset_windows(tmp_path: Path):
    tok = ByteLevelBPETokenizer()
    tok.train(["hello world 123\n", "math is fun 456\n"] * 5, vocab_size=300)

    bin_path = tmp_path / "train.bin"
    total_tokens = pack_texts_to_bin(["hello world 123\n", "math is fun 456\n"] * 10, tok, bin_path)
    assert total_tokens > 32

    ds = PackedTokenDataset(bin_path, block_size=16)
    assert len(ds) == (total_tokens - 1) // 16
    x, y = ds[0]
    assert x.shape == (16,)
    assert y.shape == (16,)
    assert x.dtype == torch.long
    # Next-token shift alignment
    x1, _ = ds[1]
    assert y[-1].item() == x1[0].item()


def test_benchmark_training_throughput_returns_valid_metrics():
    cfg = ModelConfig(vocab_size=300, block_size=64, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=128)
    model = DecoderOnlyLM(cfg)
    stats = benchmark_training_throughput(model, batch_size=2, seq_len=32, num_steps=3, warmup_steps=1, dtype=torch.float32)
    assert stats["tokens_per_sec"] > 0.0
    assert stats["steps_per_hour"] > 0.0
    assert "peak_vram_mb" in stats
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pretrain_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.pretrain_dataset'`

- [ ] **Step 3: Implement `src/data/pretrain_dataset.py` and `scripts/benchmark_throughput.py`**

```python
# src/data/pretrain_dataset.py
from pathlib import Path
import time
from typing import Iterable
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.model.transformer import DecoderOnlyLM
from src.tokenizer.bpe import ByteLevelBPETokenizer


def pack_texts_to_bin(texts: Iterable[str], tokenizer: ByteLevelBPETokenizer, out_path: str | Path) -> int:
    if tokenizer.vocab_size > 65535:
        raise ValueError("uint16 packing requires vocab_size <= 65535")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    buf: list[int] = []
    total = 0
    with out.open("wb") as f:
        for text in texts:
            ids = tokenizer.encode(text, add_bos=False, add_eos=True)
            buf.extend(ids)
            if len(buf) >= 1_000_000:
                arr = np.array(buf, dtype=np.uint16)
                arr.tofile(f)
                total += len(arr)
                buf.clear()
        if buf:
            arr = np.array(buf, dtype=np.uint16)
            arr.tofile(f)
            total += len(arr)
    return total


class PackedTokenDataset(Dataset):
    """Non-overlapping block_size windows backed by uint16 memmap."""

    def __init__(self, bin_path: str | Path, block_size: int) -> None:
        self.bin_path = Path(bin_path)
        self.block_size = block_size
        self.data = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        self.num_windows = max(0, (len(self.data) - 1) // block_size)

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.block_size
        chunk = np.asarray(self.data[start : start + self.block_size + 1], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def benchmark_training_throughput(
    model: DecoderOnlyLM,
    batch_size: int,
    seq_len: int,
    num_steps: int = 20,
    warmup_steps: int = 5,
    dtype: torch.dtype = torch.float16,
) -> dict[str, float]:
    device = next(model.parameters()).device
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    use_amp = device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and dtype == torch.float16))

    x = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)

    def step_fn() -> None:
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(warmup_steps):
        step_fn()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    for _ in range(num_steps):
        step_fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    tokens_total = num_steps * batch_size * seq_len
    tok_per_sec = tokens_total / elapsed if elapsed > 0 else 0.0
    steps_per_sec = num_steps / elapsed if elapsed > 0 else 0.0
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0

    return {
        "tokens_per_sec": tok_per_sec,
        "steps_per_hour": steps_per_sec * 3600.0,
        "peak_vram_mb": peak_mb,
    }
```

```python
# scripts/benchmark_throughput.py
import argparse
from pathlib import Path
import torch
from src.data.pretrain_dataset import benchmark_training_throughput
from src.model.config import ModelConfig
from src.model.transformer import DecoderOnlyLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/compact_12gb.yaml"))
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--gpu-hours-budget", type=float, default=6.0)
    args = parser.parse_args()

    cfg = ModelConfig.from_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, cfg.dtype)
    model = DecoderOnlyLM(cfg).to(device)
    model.gradient_checkpointing = True

    stats = benchmark_training_throughput(model, batch_size=args.micro_batch_size, seq_len=args.seq_len, dtype=dtype)
    calibrated_tokens = int(stats["tokens_per_sec"] * args.gpu_hours_budget * 3600)
    print(f"Throughput: {stats['tokens_per_sec']:.1f} tok/s | Peak VRAM: {stats['peak_vram_mb']:.1f} MB")
    print(f"For {args.gpu_hours_budget} GPU-hours -> Calibrated Token Budget: {calibrated_tokens:,} tokens")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pretrain_dataset.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/data/pretrain_dataset.py scripts/benchmark_throughput.py tests/test_pretrain_dataset.py
git commit -m "feat: add uint16 packed dataset and GPU throughput calibrator"
```

---

### Task 3 [P1]: CoT SFT Dataset (`<think>...</think>`) & Assistant-Only Loss Masking (`src/data/cot_dataset.py`)

**Files:**
- Create: `src/data/cot_dataset.py`
- Test: `tests/test_cot_dataset.py`

**Interfaces:**
- Consumes: `ByteLevelBPETokenizer` (`src/tokenizer/bpe.py`)
- Produces:
  - `clean_gsm8k_reasoning_and_answer(raw_answer: str) -> tuple[str, str]`
  - `format_cot_prompt(question: str) -> str`
  - `format_cot_response(reasoning: str, final_answer: str) -> str`
  - `CoTSFTDataset(examples: list[dict], tokenizer: ByteLevelBPETokenizer, max_seq_len: int = 2048)`
  - `cot_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]], pad_token_id: int = 0) -> tuple[torch.Tensor, torch.Tensor]`

- [ ] **Step 1: Write the failing test in `tests/test_cot_dataset.py`**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cot_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.cot_dataset'`

- [ ] **Step 3: Implement `src/data/cot_dataset.py`**

```python
# src/data/cot_dataset.py
import re
import torch
from torch.utils.data import Dataset

from src.tokenizer.bpe import ByteLevelBPETokenizer

_CALC_TAG_RE = re.compile(r"<<[^<>]*>>")


def clean_gsm8k_reasoning_and_answer(raw_answer: str) -> tuple[str, str]:
    if "####" in raw_answer:
        body, ans = raw_answer.rsplit("####", 1)
    else:
        body, ans = raw_answer, ""
    cleaned_body = _CALC_TAG_RE.sub("", body).strip()
    return cleaned_body, ans.strip()


def format_cot_prompt(question: str) -> str:
    return f"<|bos|><|user|>\n{question.strip()}\n<|assistant|>\n"


def format_cot_response(reasoning: str, final_answer: str) -> str:
    return f"<think>\n{reasoning.strip()}\n</think>\n#### {final_answer.strip()}<|eos|>"


class CoTSFTDataset(Dataset):
    """Formats CoT examples with assistant-only loss masking (-100 on prompt tokens)."""

    def __init__(self, examples: list[dict], tokenizer: ByteLevelBPETokenizer, max_seq_len: int = 2048) -> None:
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        for ex in examples:
            prompt_str = format_cot_prompt(ex["question"])
            if "reasoning" in ex and "final_answer" in ex:
                reasoning, final_ans = ex["reasoning"], ex["final_answer"]
            else:
                reasoning, final_ans = clean_gsm8k_reasoning_and_answer(ex["answer"])

            resp_str = format_cot_response(reasoning, final_ans)
            prompt_ids = tokenizer.encode(prompt_str)
            resp_ids = tokenizer.encode(resp_str)
            full_ids = prompt_ids + resp_ids
            if len(full_ids) > max_seq_len:
                continue
            labels = [-100] * len(prompt_ids) + resp_ids
            self.samples.append((torch.tensor(full_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def cot_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    pad_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(ids.size(0) for ids, _ in batch)
    bsz = len(batch)
    out_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long)
    out_labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    for i, (ids, labels) in enumerate(batch):
        length = ids.size(0)
        out_ids[i, :length] = ids
        out_labels[i, :length] = labels
    return out_ids, out_labels
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cot_dataset.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/data/cot_dataset.py tests/test_cot_dataset.py
git commit -m "feat: add CoT SFT dataset with <think> template and assistant loss masking"
```

---

### Task 4 [P0/P1]: Staged Trainer & CLI Scripts (`src/training/trainer.py`, `scripts/train_core.py`, `scripts/train_cot_sft.py`)

**Files:**
- Create: `src/training/__init__.py`
- Create: `src/training/trainer.py`
- Create: `scripts/train_core.py`
- Create: `scripts/train_cot_sft.py`
- Test: `tests/test_trainer.py`

**Interfaces:**
- Consumes: `DecoderOnlyLM` (`src/model/transformer.py`), `PackedTokenDataset` (`src/data/pretrain_dataset.py`), `CoTSFTDataset` (`src/data/cot_dataset.py`), `LeakageFilter` (`src/data/leakage_filter.py`)
- Produces:
  - `TrainConfig` dataclass
  - `cosine_lr_with_warmup(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float`
  - `Trainer(model: DecoderOnlyLM, train_config: TrainConfig, train_loader: DataLoader, val_loader: DataLoader | None = None, causal_shift: bool = False)` with `train() -> dict[str, list[float]]`, `save_checkpoint(path: Path, step: int) -> None`, `load_checkpoint(path: Path) -> int`

- [ ] **Step 1: Write the failing test in `tests/test_trainer.py`**

```python
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.model.config import ModelConfig
from src.model.transformer import DecoderOnlyLM
from src.training.trainer import TrainConfig, Trainer, cosine_lr_with_warmup


def test_cosine_lr_schedule():
    assert cosine_lr_with_warmup(0, warmup_steps=10, max_steps=100, max_lr=1e-3, min_lr=1e-4) == 0.0
    assert cosine_lr_with_warmup(10, warmup_steps=10, max_steps=100, max_lr=1e-3, min_lr=1e-4) == 1e-3
    assert cosine_lr_with_warmup(100, warmup_steps=10, max_steps=100, max_lr=1e-3, min_lr=1e-4) == 1e-4


def test_trainer_decreases_loss_and_roundtrips_checkpoint(tmp_path: Path):
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, block_size=32, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=64)
    model = DecoderOnlyLM(cfg)

    x = torch.randint(0, 64, (8, 16))
    y = torch.roll(x, shifts=-1, dims=1)
    y[:, -1] = -100
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)

    tcfg = TrainConfig(
        lr=3e-3,
        min_lr=3e-4,
        warmup_steps=2,
        max_steps=8,
        grad_accum_steps=1,
        amp_dtype="float32",
        eval_interval=4,
        save_interval=4,
        out_dir=tmp_path / "ckpt",
    )
    trainer = Trainer(model, tcfg, loader, val_loader=loader, causal_shift=False)
    history = trainer.train()
    assert history["train_loss"][-1] < history["train_loss"][0]
    ckpt_file = tmp_path / "ckpt" / "step_000008.pt"
    assert ckpt_file.exists()

    model2 = DecoderOnlyLM(cfg)
    trainer2 = Trainer(model2, tcfg, loader, causal_shift=False)
    loaded_step = trainer2.load_checkpoint(ckpt_file)
    assert loaded_step == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_trainer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.training'`

- [ ] **Step 3: Implement `src/training/trainer.py`, `scripts/train_core.py`, and `scripts/train_cot_sft.py`**

Create empty `src/training/__init__.py`.

```python
# src/training/trainer.py
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.model.transformer import DecoderOnlyLM


@dataclass
class TrainConfig:
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 100
    max_steps: int = 1000
    grad_accum_steps: int = 1
    grad_clip: float = 1.0
    amp_dtype: str = "float16"
    gradient_checkpointing: bool = False
    eval_interval: int = 100
    save_interval: int = 200
    out_dir: Path = Path("artifacts/checkpoints")


def cosine_lr_with_warmup(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * (step / warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


def _compute_loss(logits: torch.Tensor, labels: torch.Tensor, causal_shift: bool) -> torch.Tensor:
    if causal_shift:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)


class Trainer:
    def __init__(
        self,
        model: DecoderOnlyLM,
        train_config: TrainConfig,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        causal_shift: bool = False,
    ) -> None:
        self.model = model
        self.cfg = train_config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.causal_shift = causal_shift
        self.device = next(model.parameters()).device
        self.model.gradient_checkpointing = train_config.gradient_checkpointing

        decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
        nodecay_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": train_config.weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=train_config.lr,
            betas=train_config.betas,
        )

        self.amp_torch_dtype = getattr(torch, train_config.amp_dtype)
        self.use_amp = self.device.type == "cuda" and self.amp_torch_dtype in {torch.float16, torch.bfloat16}
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.use_amp and self.amp_torch_dtype == torch.float16))

    def save_checkpoint(self, path: Path, step: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "model_config": asdict(self.model.config),
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return int(ckpt["step"])

    @torch.inference_mode()
    def evaluate(self) -> float:
        if self.val_loader is None:
            return 0.0
        self.model.eval()
        losses: list[float] = []
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_torch_dtype, enabled=self.use_amp):
                logits = self.model(x)
                loss = _compute_loss(logits, y, causal_shift=self.causal_shift)
            losses.append(float(loss.item()))
        self.model.train()
        return sum(losses) / max(1, len(losses))

    def train(self, start_step: int = 0) -> dict[str, list[float]]:
        self.model.train()
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        data_iter = iter(self.train_loader)

        for step in range(start_step + 1, self.cfg.max_steps + 1):
            lr = cosine_lr_with_warmup(step, self.cfg.warmup_steps, self.cfg.max_steps, self.cfg.lr, self.cfg.min_lr)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for _ in range(self.cfg.grad_accum_steps):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    x, y = next(data_iter)
                x, y = x.to(self.device), y.to(self.device)
                with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_torch_dtype, enabled=self.use_amp):
                    logits = self.model(x)
                    loss = _compute_loss(logits, y, causal_shift=self.causal_shift) / self.cfg.grad_accum_steps
                self.scaler.scale(loss).backward()
                accum_loss += float(loss.item())

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            history["train_loss"].append(accum_loss)

            if self.val_loader is not None and step % self.cfg.eval_interval == 0:
                history["val_loss"].append(self.evaluate())

            if step % self.cfg.save_interval == 0 or step == self.cfg.max_steps:
                self.save_checkpoint(self.cfg.out_dir / f"step_{step:06d}.pt", step=step)

        return history
```

```python
# scripts/train_core.py
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.data.pretrain_dataset import PackedTokenDataset
from src.model.config import ModelConfig
from src.model.transformer import DecoderOnlyLM
from src.training.trainer import TrainConfig, Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/compact_12gb.yaml"))
    parser.add_argument("--train-bin", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/pretrain"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    cfg = ModelConfig.from_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DecoderOnlyLM(cfg).to(device)
    ds = PackedTokenDataset(args.train_bin, block_size=cfg.block_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    tcfg = TrainConfig(
        lr=args.lr,
        min_lr=args.lr * 0.1,
        warmup_steps=max(50, int(0.05 * args.max_steps)),
        max_steps=args.max_steps,
        grad_accum_steps=args.grad_accum,
        amp_dtype=cfg.dtype,
        gradient_checkpointing=True,
        out_dir=args.out_dir,
    )
    trainer = Trainer(model, tcfg, loader, causal_shift=False)
    start_step = trainer.load_checkpoint(args.resume) if args.resume is not None else 0
    trainer.train(start_step=start_step)


if __name__ == "__main__":
    main()
```

```python
# scripts/train_cot_sft.py
import argparse
from pathlib import Path
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from src.data.cot_dataset import CoTSFTDataset, cot_collate_fn
from src.data.leakage_filter import LeakageFilter
from src.model.config import ModelConfig
from src.model.transformer import DecoderOnlyLM
from src.tokenizer.bpe import ByteLevelBPETokenizer
from src.training.trainer import TrainConfig, Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/compact_12gb.yaml"))
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("artifacts/tokenizer"))
    parser.add_argument("--init-ckpt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/cot_sft"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()

    cfg = ModelConfig.from_yaml(args.config)
    tok = ByteLevelBPETokenizer.load(args.tokenizer_dir)
    gsm_train = list(load_dataset("openai/gsm8k", "main", split="train"))
    gsm_test_qs = [row["question"] for row in load_dataset("openai/gsm8k", "main", split="test")]

    flt = LeakageFilter(gsm_test_qs, jaccard_threshold=0.80, ngram_n=5)
    clean_train, stats = flt.filter_records(gsm_train, question_key="question")
    print(f"CoT SFT anti-leakage stats: {stats}")

    ds = CoTSFTDataset(clean_train, tok, max_seq_len=cfg.block_size)
    pad_id = tok.token_to_id("<|pad|>")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: cot_collate_fn(b, pad_id))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DecoderOnlyLM(cfg).to(device)
    ckpt = torch.load(args.init_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])

    tcfg = TrainConfig(
        lr=args.lr,
        min_lr=args.lr * 0.1,
        warmup_steps=max(20, int(0.05 * args.max_steps)),
        max_steps=args.max_steps,
        grad_accum_steps=args.grad_accum,
        amp_dtype=cfg.dtype,
        gradient_checkpointing=True,
        out_dir=args.out_dir,
    )
    trainer = Trainer(model, tcfg, loader, causal_shift=True)
    trainer.train()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_trainer.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/training/__init__.py src/training/trainer.py scripts/train_core.py scripts/train_cot_sft.py tests/test_trainer.py
git commit -m "feat: add staged Trainer and CLI scripts for Core pre-train and CoT SFT"
```

---

### Task 5 [P1]: GSM8K Regex Extractor & Automated Evaluator (`src/eval/gsm8k_eval.py`)

**Files:**
- Create: `src/eval/gsm8k_eval.py`
- Test: `tests/test_gsm8k_eval.py`

**Interfaces:**
- Produces:
  - `normalize_numeric_str(raw: str) -> Decimal | None`
  - `extract_gsm8k_prediction(text: str) -> tuple[str | None, bool, str]` (`(raw_number, strict_format_valid, source)`)
  - `is_numeric_equal(pred_str: str | None, gold_str: str) -> bool`
  - `GSM8KMetricSummary` dataclass (`total`, `pass_at_1`, `maj_at_4`, `maj_at_8`, `maj_at_16`, `bon_at_8_heuristic`, `bon_at_16_heuristic`, `bon_at_8_orm`, `bon_at_16_orm`, `weighted_maj_at_16`, `oracle_pass_at_16`, `strict_format_rate`)

- [ ] **Step 1: Write the failing test in `tests/test_gsm8k_eval.py`**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gsm8k_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eval.gsm8k_eval'`

- [ ] **Step 3: Implement `src/eval/gsm8k_eval.py`**

```python
# src/eval/gsm8k_eval.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gsm8k_eval.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/eval/gsm8k_eval.py tests/test_gsm8k_eval.py
git commit -m "feat: add GSM8K regex extractor and Decimal equivalence evaluator"
```

---

### Task 6 [P2 & P3]: Heuristic/Arithmetic Scorer (`src/inference/scorer.py`) & Learned ORM Verifier (`src/data/rollout_dataset.py`, `src/training/verifier_trainer.py`, `scripts/train_verifier.py`)

**Files:**
- Create: `src/inference/scorer.py`
- Create: `src/data/rollout_dataset.py`
- Create: `src/training/verifier_trainer.py`
- Create: `scripts/train_verifier.py`
- Test: `tests/test_scorer_and_verifier.py`

**Interfaces:**
- Consumes: `DecoderOnlyLM` (`forward_with_hidden`), `extract_gsm8k_prediction`, `is_numeric_equal`
- Produces:
  - `safe_check_arithmetic_steps(think_text: str) -> tuple[int, int]` (`(valid_steps, total_steps)`, zero `eval()`)
  - `HeuristicCoTScorer.score(full_text: str, mean_logprob: float) -> float`
  - `get_final_answer_token_indices(input_ids: torch.Tensor, pad_token_id: int = 0, eos_token_id: int = 2) -> torch.Tensor`
  - `OutcomeVerifierModel(backbone: DecoderOnlyLM)` with `forward(input_ids: torch.Tensor, pad_token_id: int = 0, eos_token_id: int = 2) -> torch.Tensor` (logits `(B,)`) and `score_sequences(...) -> torch.Tensor` (`sigmoid` probabilities `(B,)`)
  - `split_questions_train_val(records: list[dict], val_ratio: float = 0.1, seed: int = 42) -> tuple[list[dict], list[dict]]`
  - `RolloutVerifierDataset` & `rollout_collate_fn`
  - `train_orm_verifier(...) -> dict[str, list[float]]`

- [ ] **Step 1: Write the failing test in `tests/test_scorer_and_verifier.py`**

```python
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
    assert pool_idx.tolist() == [3, 5]  # points to token 12 and token 14 right before EOS

    records = [{"question": f"Q{i}?", "rollout_text": f"<think>\n1+1=2\n</think>\n#### {2 if i % 2 == 0 else 3}", "label": float(i % 2 == 0)} for i in range(10)]
    train_recs, val_recs = split_questions_train_val(records, val_ratio=0.2, seed=42)
    assert {r["question"] for r in train_recs}.isdisjoint({r["question"] for r in val_recs})

    cfg = ModelConfig(vocab_size=tok.vocab_size, block_size=64, d_model=32, n_layers=1, n_heads=4, n_kv_heads=2, d_ff=64)
    orm = OutcomeVerifierModel(DecoderOnlyLM(cfg))
    ds = RolloutVerifierDataset(train_recs, tok, max_seq_len=64)
    batch_ids, batch_labels = rollout_collate_fn([ds[0], ds[1]], pad_token_id=pad_id)
    probs = orm.score_sequences(batch_ids, pad_token_id=pad_id, eos_token_id=eos_id)
    assert probs.shape == (2,)
    assert torch.all((probs >= 0.0) & (probs <= 1.0))

    history = train_orm_verifier(orm, ds, pad_token_id=pad_id, eos_token_id=eos_id, epochs=1, batch_size=2, lr=1e-3)
    assert len(history["loss"]) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scorer_and_verifier.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.inference.scorer'`

- [ ] **Step 3: Implement `scorer.py`, `rollout_dataset.py`, `verifier_trainer.py`, and `scripts/train_verifier.py`**

```python
# src/inference/scorer.py
from decimal import Decimal, InvalidOperation
import operator
import re
import torch
import torch.nn as nn

from src.eval.gsm8k_eval import extract_gsm8k_prediction
from src.model.transformer import DecoderOnlyLM

_EQ_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)")
_OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
}


def safe_check_arithmetic_steps(think_text: str) -> tuple[int, int]:
    matches = _EQ_RE.findall(think_text)
    valid = 0
    total = 0
    for lhs_s, op_s, rhs_s, res_s in matches:
        try:
            a, b, expected = Decimal(lhs_s), Decimal(rhs_s), Decimal(res_s)
            if op_s == "/" and b == 0:
                total += 1
                continue
            computed = _OPS[op_s](a, b)
            total += 1
            if abs(computed - expected) <= Decimal("1e-3"):
                valid += 1
        except (InvalidOperation, ZeroDivisionError):
            total += 1
    return valid, total


class HeuristicCoTScorer:
    """Self-written P2 scorer combining format compliance, arithmetic step consistency, and length-norm logprob."""

    def __init__(self, format_weight: float = 1.5, arith_weight: float = 1.0, logprob_weight: float = 0.5) -> None:
        self.format_weight = format_weight
        self.arith_weight = arith_weight
        self.logprob_weight = logprob_weight

    def score(self, full_text: str, mean_logprob: float) -> float:
        pred, strict_ok, src = extract_gsm8k_prediction(full_text)
        fmt_score = 1.0 if strict_ok else (0.4 if pred is not None and src != "none" else 0.0)
        think_body = full_text.split("<think>", 1)[1].split("</think>", 1)[0] if "<think>" in full_text else full_text
        valid_eqs, total_eqs = safe_check_arithmetic_steps(think_body)
        arith_score = (valid_eqs / total_eqs) if total_eqs > 0 else 0.5
        return (
            self.format_weight * fmt_score
            + self.arith_weight * arith_score
            + self.logprob_weight * float(mean_logprob)
        )


def get_final_answer_token_indices(
    input_ids: torch.Tensor,
    pad_token_id: int = 0,
    eos_token_id: int = 2,
) -> torch.Tensor:
    non_pad = input_ids != pad_token_id
    last_non_pad = non_pad.long().sum(dim=1).clamp(min=1) - 1
    batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
    is_eos = input_ids[batch_idx, last_non_pad] == eos_token_id
    return torch.where(is_eos & (last_non_pad > 0), last_non_pad - 1, last_non_pad)


class OutcomeVerifierModel(nn.Module):
    """P3 Learned Outcome Reward Model pooling the final answer token hidden state."""

    def __init__(self, backbone: DecoderOnlyLM) -> None:
        super().__init__()
        self.backbone = backbone
        self.score_head = nn.Linear(backbone.config.d_model, 1)

    def forward(self, input_ids: torch.Tensor, pad_token_id: int = 0, eos_token_id: int = 2) -> torch.Tensor:
        _, hidden = self.backbone.forward_with_hidden(input_ids)
        pool_idx = get_final_answer_token_indices(input_ids, pad_token_id=pad_token_id, eos_token_id=eos_token_id)
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        final_hidden = hidden[batch_idx, pool_idx, :]
        return self.score_head(final_hidden).squeeze(-1)

    @torch.inference_mode()
    def score_sequences(self, input_ids: torch.Tensor, pad_token_id: int = 0, eos_token_id: int = 2) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self.forward(input_ids, pad_token_id=pad_token_id, eos_token_id=eos_token_id))
```

```python
# src/data/rollout_dataset.py
import random
import torch
from torch.utils.data import Dataset

from src.data.cot_dataset import format_cot_prompt
from src.tokenizer.bpe import ByteLevelBPETokenizer


def split_questions_train_val(
    records: list[dict],
    val_ratio: float = 0.1,
    seed: int = 42,
    question_key: str = "question",
) -> tuple[list[dict], list[dict]]:
    unique_qs = sorted({r[question_key] for r in records})
    rng = random.Random(seed)
    rng.shuffle(unique_qs)
    val_count = max(1, int(len(unique_qs) * val_ratio)) if len(unique_qs) > 1 else 0
    val_qs = set(unique_qs[:val_count])
    train_recs = [r for r in records if r[question_key] not in val_qs]
    val_recs = [r for r in records if r[question_key] in val_qs]
    return train_recs, val_recs


class RolloutVerifierDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer: ByteLevelBPETokenizer, max_seq_len: int = 2048) -> None:
        self.samples: list[tuple[torch.Tensor, float]] = []
        eos_tok = "<|eos|>"
        for rec in records:
            text = format_cot_prompt(rec["question"]) + rec["rollout_text"].strip()
            if not text.endswith(eos_tok):
                text += eos_tok
            ids = tokenizer.encode(text)[:max_seq_len]
            self.samples.append((torch.tensor(ids, dtype=torch.long), float(rec["label"])))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float]:
        return self.samples[idx]


def rollout_collate_fn(
    batch: list[tuple[torch.Tensor, float]],
    pad_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(ids.size(0) for ids, _ in batch)
    out_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    out_labels = torch.tensor([lbl for _, lbl in batch], dtype=torch.float32)
    for i, (ids, _) in enumerate(batch):
        out_ids[i, : ids.size(0)] = ids
    return out_ids, out_labels
```

```python
# src/training/verifier_trainer.py
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.rollout_dataset import RolloutVerifierDataset, rollout_collate_fn
from src.inference.scorer import OutcomeVerifierModel


def train_orm_verifier(
    model: OutcomeVerifierModel,
    train_dataset: RolloutVerifierDataset,
    pad_token_id: int = 0,
    eos_token_id: int = 2,
    epochs: int = 2,
    batch_size: int = 8,
    lr: float = 2e-5,
    save_path: Path | None = None,
) -> dict[str, list[float]]:
    device = next(model.parameters()).device
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: rollout_collate_fn(b, pad_token_id=pad_token_id),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    history: dict[str, list[float]] = {"loss": []}

    for _ in range(epochs):
        for input_ids, labels in loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(input_ids, pad_token_id=pad_token_id, eos_token_id=eos_token_id)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            history["loss"].append(float(loss.item()))

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict()}, save_path)
    return history
```

```python
# scripts/train_verifier.py
import argparse
import json
from pathlib import Path
import torch
from src.data.rollout_dataset import RolloutVerifierDataset, split_questions_train_val
from src.inference.scorer import OutcomeVerifierModel
from src.model.config import ModelConfig
from src.model.transformer import DecoderOnlyLM
from src.tokenizer.bpe import ByteLevelBPETokenizer
from src.training.verifier_trainer import train_orm_verifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/compact_12gb.yaml"))
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("artifacts/tokenizer"))
    parser.add_argument("--sft-ckpt", type=Path, required=True)
    parser.add_argument("--rollouts-json", type=Path, required=True)
    parser.add_argument("--out-ckpt", type=Path, default=Path("artifacts/orm_verifier/orm.pt"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    cfg = ModelConfig.from_yaml(args.config)
    tok = ByteLevelBPETokenizer.load(args.tokenizer_dir)
    with args.rollouts_json.open("r", encoding="utf-8") as f:
        records = json.load(f)

    train_recs, val_recs = split_questions_train_val(records, val_ratio=0.1, seed=42)
    print(f"ORM question-split counts: train={len(train_recs)}, val={len(val_recs)}")
    train_ds = RolloutVerifierDataset(train_recs, tok, max_seq_len=cfg.block_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = DecoderOnlyLM(cfg).to(device)
    ckpt = torch.load(args.sft_ckpt, map_location=device, weights_only=True)
    backbone.load_state_dict(ckpt["model_state_dict"])
    orm = OutcomeVerifierModel(backbone).to(device)

    train_orm_verifier(
        orm,
        train_ds,
        pad_token_id=tok.token_to_id("<|pad|>"),
        eos_token_id=tok.token_to_id("<|eos|>"),
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_path=args.out_ckpt,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_scorer_and_verifier.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/inference/scorer.py src/data/rollout_dataset.py src/training/verifier_trainer.py scripts/train_verifier.py tests/test_scorer_and_verifier.py
git commit -m "feat: add HeuristicCoTScorer and Learned ORM Verifier on final answer token"
```

---

### Task 7 [P1–P3]: Shared-Pool Test-Time Scaling Engine (`src/inference/scaling.py`) & Optional P3 `SharedPrefixKVCache` (`src/cache/shared_prefix_kv_cache.py`)

**Files:**
- Create: `src/inference/scaling.py`
- Create: `src/cache/shared_prefix_kv_cache.py` (P3 optional memory optimization)
- Test: `tests/test_test_time_scaling.py`

**Interfaces:**
- Consumes: `AutoregressiveGenerator`, `ByteLevelBPETokenizer`, `HeuristicCoTScorer`, `OutcomeVerifierModel`, `extract_gsm8k_prediction`, `is_numeric_equal`
- Produces:
  - `CandidateChain` dataclass (`text`, `think_text`, `extracted_answer`, `strict_format_valid`, `extraction_source`, `think_tokens`, `generated_tokens`, `mean_logprob`, `heuristic_score`, `orm_score`)
  - `PoolEvalRecord` dataclass (`question`, `gold_answer`, `greedy_answer`, `greedy_correct`, `greedy_tokens`, `pool`, `maj_at_4`, `maj_at_8`, `maj_at_16`, `bon_at_8_heuristic`, `bon_at_16_heuristic`, `bon_at_8_orm`, `bon_at_16_orm`, `weighted_maj_at_16`, `oracle_pass_at_16`, `pool_generated_tokens`, `pool_forward_passes`, `pool_wall_ms`)
  - `majority_vote(candidates: list[CandidateChain]) -> str | None`
  - `weighted_majority_vote(candidates: list[CandidateChain], score_attr: str = "orm_score") -> str | None`
  - `TestTimeScaler(generator, tokenizer, heuristic_scorer=None, orm_model=None, max_batch_k=8)`
  - `SharedPrefixKVCache(n_layers, n_kv_heads, d_head, max_seq_len, dtype, device)` (P3)

- [ ] **Step 1: Write the failing test in `tests/test_test_time_scaling.py`**

```python
import torch
from src.cache.shared_prefix_kv_cache import SharedPrefixKVCache
from src.inference.generator import AutoregressiveGenerator
from src.inference.scaling import CandidateChain, TestTimeScaler, majority_vote, weighted_majority_vote
from src.model.config import ModelConfig
from src.model.transformer import DecoderOnlyLM
from src.tokenizer.bpe import ByteLevelBPETokenizer


def test_majority_vote_and_weighted_vote_on_shared_pool():
    # Matches review example: 42, 42, 42, 36, 36, 50, 50, 100
    answers = ["42", "42", "42", "36", "36", "50", "50", "100"]
    cands = [
        CandidateChain(
            text=f"<think>ok</think>\n#### {ans}",
            think_text="ok",
            extracted_answer=ans,
            strict_format_valid=True,
            extraction_source="hash",
            think_tokens=5,
            generated_tokens=10,
            mean_logprob=-0.5,
            heuristic_score=0.9 if ans == "42" else 0.3,
            orm_score=0.1 if ans == "42" else (0.95 if ans == "36" else 0.05),
        )
        for ans in answers
    ]
    assert majority_vote(cands) == "42"
    # Weighted by orm_score: 36 gets 2*0.95=1.90 > 42 gets 3*0.1=0.30
    assert weighted_majority_vote(cands, score_attr="orm_score") == "36"


def test_shared_pool_scaler_and_shared_prefix_kv_cache():
    tok = ByteLevelBPETokenizer()
    tok.train(["<|user|>\nWhat is 2+2?\n<|assistant|>\n<think>\n2+2=4\n</think>\n#### 4"], vocab_size=300)
    cfg = ModelConfig(vocab_size=tok.vocab_size, block_size=128, d_model=32, n_layers=1, n_heads=4, n_kv_heads=2, d_ff=64)
    model = DecoderOnlyLM(cfg).eval()
    gen = AutoregressiveGenerator(model, pad_token_id=tok.token_to_id("<|pad|>"), eos_token_id=tok.token_to_id("<|eos|>"))

    scaler = TestTimeScaler(gen, tok, max_batch_k=4)
    rec = scaler.evaluate_question("What is 2+2?", "4", k_pool=8, max_new_tokens=12, temperature=0.8)
    assert len(rec.pool) == 8
    assert rec.pool_generated_tokens > 0
    assert rec.pool_forward_passes > 0

    # SharedPrefixKVCache stores prompt once + K decode buffers
    sp_cache = SharedPrefixKVCache(n_layers=1, n_kv_heads=2, d_head=16, max_seq_len=32, dtype=torch.float32)
    sp_cache.set_prefix(0, torch.ones(1, 2, 6, 16), torch.ones(1, 2, 6, 16), num_branches=4)
    k_all, v_all = sp_cache.append_decode(0, torch.zeros(4, 2, 1, 16), torch.zeros(4, 2, 1, 16))
    assert k_all.shape == (4, 2, 7, 16)
    assert sp_cache.prefix_bytes() < sp_cache.decode_bytes() * 6 + sp_cache.prefix_bytes()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_test_time_scaling.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.inference.scaling'`

- [ ] **Step 3: Implement `src/inference/scaling.py` and `src/cache/shared_prefix_kv_cache.py`**

```python
# src/cache/shared_prefix_kv_cache.py
import torch


class SharedPrefixKVCache:
    """P3 optional cache: stores 1 read-only prompt KV tensor shared across K decode branch buffers."""

    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        d_head: int,
        max_seq_len: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cpu",
    ) -> None:
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.device = torch.device(device)
        self._prefix_k: list[torch.Tensor | None] = [None] * n_layers
        self._prefix_v: list[torch.Tensor | None] = [None] * n_layers
        self._decode_k: list[torch.Tensor | None] = [None] * n_layers
        self._decode_v: list[torch.Tensor | None] = [None] * n_layers
        self._decode_len: list[int] = [0] * n_layers
        self.num_branches = 1

    def set_prefix(self, layer_idx: int, k_prefix: torch.Tensor, v_prefix: torch.Tensor, num_branches: int) -> None:
        if k_prefix.size(0) != 1:
            raise ValueError("SharedPrefixKVCache expects single-prompt prefix (B=1)")
        self.num_branches = num_branches
        self._prefix_k[layer_idx] = k_prefix.detach()
        self._prefix_v[layer_idx] = v_prefix.detach()
        rem_cap = max(1, self.max_seq_len - k_prefix.size(2))
        self._decode_k[layer_idx] = torch.empty(
            (num_branches, self.n_kv_heads, rem_cap, self.d_head), dtype=self.dtype, device=self.device
        )
        self._decode_v[layer_idx] = torch.empty(
            (num_branches, self.n_kv_heads, rem_cap, self.d_head), dtype=self.dtype, device=self.device
        )
        self._decode_len[layer_idx] = 0

    def append_decode(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cur = self._decode_len[layer_idx]
        add = k_new.size(2)
        self._decode_k[layer_idx][:, :, cur : cur + add, :] = k_new
        self._decode_v[layer_idx][:, :, cur : cur + add, :] = v_new
        self._decode_len[layer_idx] = cur + add

        pk = self._prefix_k[layer_idx].expand(self.num_branches, -1, -1, -1)
        pv = self._prefix_v[layer_idx].expand(self.num_branches, -1, -1, -1)
        dk = self._decode_k[layer_idx][:, :, : cur + add, :]
        dv = self._decode_v[layer_idx][:, :, : cur + add, :]
        return torch.cat([pk, dk], dim=2), torch.cat([pv, dv], dim=2)

    def prefix_bytes(self) -> int:
        return sum(
            (k.numel() * k.element_size() + v.numel() * v.element_size())
            for k, v in zip(self._prefix_k, self._prefix_v)
            if k is not None and v is not None
        )

    def decode_bytes(self) -> int:
        return sum(
            2 * self.num_branches * self.n_kv_heads * dlen * self.d_head * self._decode_k[0].element_size()
            for dlen in self._decode_len
            if self._decode_k[0] is not None
        )
```

```python
# src/inference/scaling.py
from collections import defaultdict
from dataclasses import dataclass
import time
import torch

from src.data.cot_dataset import format_cot_prompt
from src.eval.gsm8k_eval import extract_gsm8k_prediction, is_numeric_equal, normalize_numeric_str
from src.inference.generator import AutoregressiveGenerator
from src.inference.scorer import HeuristicCoTScorer, OutcomeVerifierModel
from src.tokenizer.bpe import ByteLevelBPETokenizer


@dataclass
class CandidateChain:
    text: str
    think_text: str
    extracted_answer: str | None
    strict_format_valid: bool
    extraction_source: str
    think_tokens: int
    generated_tokens: int
    mean_logprob: float
    heuristic_score: float
    orm_score: float | None = None


@dataclass
class PoolEvalRecord:
    question: str
    gold_answer: str
    greedy_answer: str | None
    greedy_correct: bool
    greedy_tokens: int
    greedy_forward_passes: int
    pool: list[CandidateChain]
    maj_at_4: bool
    maj_at_8: bool
    maj_at_16: bool
    bon_at_8_heuristic: bool
    bon_at_16_heuristic: bool
    bon_at_8_orm: bool | None
    bon_at_16_orm: bool | None
    weighted_maj_at_16: bool | None
    oracle_pass_at_16: bool
    pool_generated_tokens: int
    pool_forward_passes: int
    pool_wall_ms: float


def _canonical_vote_key(ans: str | None) -> str | None:
    if ans is None:
        return None
    dec = normalize_numeric_str(ans)
    return str(dec.normalize()) if dec is not None else ans


def majority_vote(candidates: list[CandidateChain]) -> str | None:
    counts: dict[str, int] = defaultdict(int)
    best_lp: dict[str, float] = defaultdict(lambda: float("-inf"))
    rep_str: dict[str, str] = {}
    for c in candidates:
        key = _canonical_vote_key(c.extracted_answer)
        if key is None:
            continue
        counts[key] += 1
        if c.mean_logprob > best_lp[key]:
            best_lp[key] = c.mean_logprob
            rep_str[key] = c.extracted_answer  # type: ignore[assignment]
    if not counts:
        return None
    winner = max(counts.keys(), key=lambda k: (counts[k], best_lp[k]))
    return rep_str[winner]


def weighted_majority_vote(candidates: list[CandidateChain], score_attr: str = "orm_score") -> str | None:
    weights: dict[str, float] = defaultdict(float)
    rep_str: dict[str, str] = {}
    for c in candidates:
        key = _canonical_vote_key(c.extracted_answer)
        if key is None:
            continue
        score = getattr(c, score_attr)
        weights[key] += float(score) if score is not None else 0.0
        if key not in rep_str:
            rep_str[key] = c.extracted_answer  # type: ignore[assignment]
    if not weights:
        return None
    winner = max(weights.keys(), key=lambda k: weights[k])
    return rep_str[winner]


class TestTimeScaler:
    def __init__(
        self,
        generator: AutoregressiveGenerator,
        tokenizer: ByteLevelBPETokenizer,
        heuristic_scorer: HeuristicCoTScorer | None = None,
        orm_model: OutcomeVerifierModel | None = None,
        max_batch_k: int = 8,
    ) -> None:
        self.generator = generator
        self.tokenizer = tokenizer
        self.heuristic = heuristic_scorer or HeuristicCoTScorer()
        self.orm = orm_model
        self.max_batch_k = max_batch_k
        self.pad_id = tokenizer.token_to_id("<|pad|>")
        self.eos_id = tokenizer.token_to_id("<|eos|>")

    def _to_candidate(self, full_ids: torch.Tensor, prompt_len: int, logprobs: torch.Tensor) -> CandidateChain:
        gen_ids = [i for i in full_ids[prompt_len:].tolist() if i != self.pad_id]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
        pred, strict_ok, src = extract_gsm8k_prediction(text)

        think_text = ""
        think_tokens = 0
        if "<think>" in text:
            after_open = text.split("<think>", 1)[1]
            think_text = after_open.split("</think>", 1)[0] if "</think>" in after_open else after_open
            think_tokens = len(self.tokenizer.encode(think_text))

        active_lp = logprobs[: len(gen_ids)]
        mean_lp = float(active_lp.mean().item()) if active_lp.numel() > 0 else -10.0
        h_score = self.heuristic.score(text, mean_lp)

        orm_score: float | None = None
        if self.orm is not None:
            seq_tensor = full_ids.unsqueeze(0)
            orm_score = float(self.orm.score_sequences(seq_tensor, pad_token_id=self.pad_id, eos_token_id=self.eos_id).item())

        return CandidateChain(
            text=text,
            think_text=think_text,
            extracted_answer=pred,
            strict_format_valid=strict_ok,
            extraction_source=src,
            think_tokens=think_tokens,
            generated_tokens=len(gen_ids),
            mean_logprob=mean_lp,
            heuristic_score=h_score,
            orm_score=orm_score,
        )

    def evaluate_question(
        self,
        question: str,
        gold_answer: str,
        k_pool: int = 16,
        max_new_tokens: int = 768,
        temperature: float = 0.7,
        top_p: float = 0.95,
        repetition_penalty: float = 1.05,
    ) -> PoolEvalRecord:
        prompt_str = format_cot_prompt(question)
        prompt_ids = torch.tensor([self.tokenizer.encode(prompt_str)], dtype=torch.long)

        # E1: Deterministic Greedy baseline (1x compute)
        greedy_res = self.generator.generate(
            prompt_ids, max_new_tokens=max_new_tokens, temperature=0.0, use_cache=True, stop_token_ids=[self.eos_id]
        )
        greedy_cand = self._to_candidate(greedy_res.sequences[0], greedy_res.prompt_len, greedy_res.token_logprobs[0])
        greedy_ok = is_numeric_equal(greedy_cand.extracted_answer, gold_answer)

        # Shared pool of K=16 rollouts via chunked Copy KV Cache
        pool: list[CandidateChain] = []
        pool_tokens = 0
        pool_forwards = 0
        t0 = time.perf_counter()
        remaining = k_pool
        while remaining > 0:
            cur_k = min(remaining, self.max_batch_k)
            res = self.generator.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                use_cache=True,
                stop_token_ids=[self.eos_id],
                num_return_sequences=cur_k,
            )
            pool_tokens += res.generated_tokens
            pool_forwards += res.forward_passes
            for row in range(cur_k):
                pool.append(self._to_candidate(res.sequences[row], res.prompt_len, res.token_logprobs[row]))
            remaining -= cur_k
        wall_ms = (time.perf_counter() - t0) * 1000.0

        maj4 = is_numeric_equal(majority_vote(pool[:4]), gold_answer)
        maj8 = is_numeric_equal(majority_vote(pool[:8]), gold_answer)
        maj16 = is_numeric_equal(majority_vote(pool[:16]), gold_answer)

        bon8_h = is_numeric_equal(max(pool[:8], key=lambda c: c.heuristic_score).extracted_answer, gold_answer)
        bon16_h = is_numeric_equal(max(pool[:16], key=lambda c: c.heuristic_score).extracted_answer, gold_answer)

        if self.orm is not None:
            bon8_orm = is_numeric_equal(max(pool[:8], key=lambda c: c.orm_score or 0.0).extracted_answer, gold_answer)
            bon16_orm = is_numeric_equal(max(pool[:16], key=lambda c: c.orm_score or 0.0).extracted_answer, gold_answer)
            wmaj16 = is_numeric_equal(weighted_majority_vote(pool[:16], score_attr="orm_score"), gold_answer)
        else:
            bon8_orm, bon16_orm, wmaj16 = None, None, None

        oracle16 = any(is_numeric_equal(c.extracted_answer, gold_answer) for c in pool[:16])

        return PoolEvalRecord(
            question=question,
            gold_answer=gold_answer,
            greedy_answer=greedy_cand.extracted_answer,
            greedy_correct=greedy_ok,
            greedy_tokens=greedy_cand.generated_tokens,
            greedy_forward_passes=greedy_res.forward_passes,
            pool=pool,
            maj_at_4=maj4,
            maj_at_8=maj8,
            maj_at_16=maj16,
            bon_at_8_heuristic=bon8_h,
            bon_at_16_heuristic=bon16_h,
            bon_at_8_orm=bon8_orm,
            bon_at_16_orm=bon16_orm,
            weighted_maj_at_16=wmaj16,
            oracle_pass_at_16=oracle16,
            pool_generated_tokens=pool_tokens,
            pool_forward_passes=pool_forwards,
            pool_wall_ms=wall_ms,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_test_time_scaling.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/inference/scaling.py src/cache/shared_prefix_kv_cache.py tests/test_test_time_scaling.py
git commit -m "feat: add shared-pool TestTimeScaler and optional P3 SharedPrefixKVCache"
```

---

### Task 8 [P2]: Reasoning Telemetry, Compute-Budget Frontier, 6-Category Error Analysis & GSM8K Eval CLI (`src/eval/reasoning_analysis.py`, `scripts/eval_gsm8k.py`)

**Files:**
- Create: `src/eval/reasoning_analysis.py`
- Create: `scripts/eval_gsm8k.py`
- Test: `tests/test_reasoning_analysis.py`

**Interfaces:**
- Consumes: `PoolEvalRecord`, `CandidateChain`, `TestTimeScaler`, `safe_check_arithmetic_steps`, `is_numeric_equal`
- Produces:
  - `detect_repetition_loop(text: str, ngram_n: int = 8, min_repeats: int = 3) -> bool`
  - `compute_correlations(think_lengths: list[int], correct_flags: list[bool]) -> dict[str, float]` (`pearson_point_biserial`, `spearman_rho`)
  - `analyze_length_buckets(candidates: list[CandidateChain], gold_answers: list[str]) -> list[dict]` (returns per-bucket `count`, `valid_format_rate`, `accuracy`, `repetition_loop_rate` + overall `mean_len_correct`, `mean_len_incorrect`)
  - `classify_error_category(cand: CandidateChain, gold_answer: str, oracle_had_correct: bool = False) -> str` (6 expected categories)
  - `plot_reasoning_and_compute_charts(bucket_stats: list[dict], compute_points: list[dict], out_dir: str | Path) -> list[Path]`

- [ ] **Step 1: Write the failing test in `tests/test_reasoning_analysis.py`**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_reasoning_analysis.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eval.reasoning_analysis'`

- [ ] **Step 3: Implement `src/eval/reasoning_analysis.py` and `scripts/eval_gsm8k.py`**

```python
# src/eval/reasoning_analysis.py
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
```

```python
# scripts/eval_gsm8k.py
import argparse
from dataclasses import asdict
import json
from pathlib import Path
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
    parser = argparse.ArgumentParser()
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
    dtype = getattr(torch, cfg.dtype)

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_reasoning_analysis.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Run full test suite and commit**

Run: `pytest tests/ -v`
Expected: PASS (all Core + Track C unit tests green)

```bash
git add src/eval/reasoning_analysis.py scripts/eval_gsm8k.py tests/test_reasoning_analysis.py
git commit -m "feat: add reasoning length correlation, compute frontier, and GSM8K eval CLI"
```

