# Core LLM Foundation & KV Cache Profiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify the mandatory from-scratch Core LLM stack — Math-Aware Byte-Level BPE Tokenizer, Modern Decoder-Only Transformer (`RMSNorm`, `RoPE`, `GQA`, `SwiGLU`, `Weight Tying`), `DynamicKVCache`, Autoregressive Generator, and the KV Cache Profiling Harness across $L \in \{128, 512, 1024, 2048\}$.

**Architecture:** Clean PyTorch modular architecture separating tokenizer (`src/tokenizer`), model layers (`src/model`), dynamic KV cache buffer (`src/cache`), autoregressive decoding (`src/inference`), and empirical CUDA profiling (`src/eval`). All components support dual hardware profiles (`Compact 43.8M` for 12–16GB VRAM and `Standard 197.2M` for 24GB VRAM) with unified `block_size = 2048` and `vocab_size = 16,384`.

**Tech Stack:** Python 3.10+, PyTorch 2.2+, PyYAML, Matplotlib, NumPy, Pytest.

**Spec:** [spec.md](file:///d:/DUT.AI/LLM%20self-built/LLM-self-built/specs/001-llm-reasoning-system/spec.md)

## Global Constraints

- Implement Tokenizer, Transformer layers (including explicit manual `Q @ K^T / sqrt(d_head)` causal attention by default), and Dynamic KV Cache from scratch using PyTorch tensor primitives (do not wrap `transformers.AutoModel` or `tokenizers` / `tiktoken`).
- Separate `attn_backend="manual"` (default from-scratch core) from `attn_backend="sdpa"` (opt-in optimization comparison).
- `vocab_size = 16,384`, `block_size = 2,048`, `rope_theta = 10,000.0`, `tie_word_embeddings = True`.
- Fixed 7 special tokens: `0: <|pad|>`, `1: <|bos|>`, `2: <|eos|>`, `3: <|user|>`, `4: <|assistant|>`, `5: <think>`, `6: </think>`.
- Exact parameter counts: Compact `43,790,848` (~43.8M); Standard `197,166,080` (~197.2M).
- 2-tier KV Cache equivalence: Tier 1 Numerical in FP32 (`max_abs_err < 1e-5`, log `mean_abs_err`; empirically measure FP16/BF16 error without brittle hard bounds); Tier 2 Functional (100% greedy token match over 64 tokens).
- KV Cache profiling must measure TTFT, TPOT, Tokens/s, and Peak/Allocated VRAM for $L \in \{128, 512, 1024, 2048\}$ with both `kv_cache=True` and `kv_cache=False`.
- Follow TDD strictly: write failing test → confirm failure → implement minimal code → confirm pass → commit.

---

### Task 1: Project Scaffolding, `ModelConfig`, and Hardware Profiles

**Files:**
- Create: `pyproject.toml`
- Create: `src/__init__.py`
- Create: `src/model/__init__.py`
- Create: `src/model/config.py`
- Create: `configs/compact_12gb.yaml`
- Create: `configs/standard_24gb.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `ModelConfig` dataclass with fields `vocab_size`, `block_size`, `d_model`, `n_layers`, `n_heads`, `n_kv_heads`, `d_ff`, `rope_theta`, `norm_eps`, `dropout`, `tie_word_embeddings`, `dtype`, `attn_backend`
  - `ModelConfig.d_head` property (`int`)
  - `ModelConfig.count_parameters() -> int`
  - `ModelConfig.kv_cache_bytes(batch_size: int, seq_len: int, bytes_per_elem: int = 2) -> int`
  - `ModelConfig.from_yaml(path: str | Path) -> ModelConfig`
  - `ModelConfig.to_yaml(path: str | Path) -> None`

- [ ] **Step 1: Create `pyproject.toml` and package markers**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "llm-self-built"
version = "0.1.0"
description = "Self-built Reasoning LLM with Dynamic KV Cache and Test-Time Compute"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.2.0",
    "pyyaml>=6.0",
    "numpy>=1.26.0",
    "matplotlib>=3.8.0",
    "datasets>=2.18.0",
    "tqdm>=4.66.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0.0"]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

Create empty `src/__init__.py` and `src/model/__init__.py`.

- [ ] **Step 2: Write the failing test in `tests/test_config.py`**

```python
from pathlib import Path
import pytest
from src.model.config import ModelConfig


def test_compact_and_standard_yaml_parameter_counts():
    compact = ModelConfig.from_yaml(Path("configs/compact_12gb.yaml"))
    standard = ModelConfig.from_yaml(Path("configs/standard_24gb.yaml"))

    assert compact.d_head == 64
    assert compact.block_size == 2048
    assert compact.vocab_size == 16384
    assert compact.count_parameters() == 43_790_848
    assert compact.kv_cache_bytes(batch_size=1, seq_len=2048) == 25_165_824  # 24.0 MiB
    assert compact.kv_cache_bytes(batch_size=16, seq_len=2048) == 402_653_184  # 384.0 MiB

    assert standard.d_head == 64
    assert standard.block_size == 2048
    assert standard.vocab_size == 16384
    assert standard.count_parameters() == 197_166_080
    assert standard.kv_cache_bytes(batch_size=1, seq_len=2048) == 33_554_432  # 32.0 MiB
    assert standard.kv_cache_bytes(batch_size=16, seq_len=2048) == 536_870_912  # 512.0 MiB


def test_invalid_gqa_heads_or_backend_raises():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(d_model=512, n_heads=8, n_kv_heads=3)
    with pytest.raises(ValueError, match="attn_backend"):
        ModelConfig(attn_backend="unknown")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.model.config'`

- [ ] **Step 4: Create `src/model/config.py` and YAML configs**

```python
# src/model/config.py
from dataclasses import asdict, dataclass
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 16384
    block_size: int = 2048
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    n_kv_heads: int = 4
    d_ff: int = 1408
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    dtype: str = "float16"
    attn_backend: str = "manual"

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")
        if self.d_head % 2 != 0:
            raise ValueError(f"d_head ({self.d_head}) must be even for RoPE")
        if self.attn_backend not in {"manual", "sdpa"}:
            raise ValueError(f"attn_backend must be 'manual' or 'sdpa', got {self.attn_backend}")

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    def count_parameters(self) -> int:
        emb = self.vocab_size * self.d_model
        q_proj = self.d_model * (self.n_heads * self.d_head)
        k_proj = self.d_model * (self.n_kv_heads * self.d_head)
        v_proj = self.d_model * (self.n_kv_heads * self.d_head)
        o_proj = (self.n_heads * self.d_head) * self.d_model
        ffn = 3 * self.d_model * self.d_ff
        norms = 2 * self.d_model
        per_layer = q_proj + k_proj + v_proj + o_proj + ffn + norms
        lm_head = 0 if self.tie_word_embeddings else emb
        return emb + self.n_layers * per_layer + self.d_model + lm_head

    def kv_cache_bytes(self, batch_size: int, seq_len: int, bytes_per_elem: int = 2) -> int:
        return 2 * self.n_layers * batch_size * self.n_kv_heads * seq_len * self.d_head * bytes_per_elem

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        model_dict = raw.get("model", raw)
        return cls(**model_dict)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("w", encoding="utf-8") as f:
            yaml.safe_dump({"model": asdict(self)}, f, sort_keys=False)
```

Create `configs/compact_12gb.yaml`:
```yaml
model:
  vocab_size: 16384
  block_size: 2048
  d_model: 512
  n_layers: 12
  n_heads: 8
  n_kv_heads: 4
  d_ff: 1408
  rope_theta: 10000.0
  norm_eps: 1.0e-6
  dropout: 0.0
  tie_word_embeddings: true
  dtype: float16
  attn_backend: manual
```

Create `configs/standard_24gb.yaml`:
```yaml
model:
  vocab_size: 16384
  block_size: 2048
  d_model: 1024
  n_layers: 16
  n_heads: 16
  n_kv_heads: 4
  d_ff: 2816
  rope_theta: 10000.0
  norm_eps: 1.0e-6
  dropout: 0.0
  tie_word_embeddings: true
  dtype: bfloat16
  attn_backend: manual
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/__init__.py src/model/__init__.py src/model/config.py configs/compact_12gb.yaml configs/standard_24gb.yaml tests/test_config.py
git commit -m "feat: add ModelConfig and compact/standard hardware profiles"
```

---

### Task 2: Math-Aware Pre-tokenizer & Custom Byte-Level BPE Tokenizer

**Files:**
- Create: `src/tokenizer/__init__.py`
- Create: `src/tokenizer/pretokenizer.py`
- Create: `src/tokenizer/bpe.py`
- Create: `scripts/train_tokenizer.py`
- Test: `tests/test_tokenizer.py`

**Interfaces:**
- Produces:
  - `SPECIAL_TOKENS: list[str] = ["<|pad|>", "<|bos|>", "<|eos|>", "<|user|>", "<|assistant|>", "<think>", "</think>"]`
  - `bytes_to_unicode() -> dict[int, str]`
  - `unicode_to_bytes() -> dict[str, int]`
  - `pretokenize(text: str) -> list[str]`
  - `ByteLevelBPETokenizer` with `train(texts: Iterable[str], vocab_size: int = 16384) -> None`, `encode(text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]`, `decode(ids: list[int], skip_special_tokens: bool = False) -> str`, `save(dir_path: str | Path) -> None`, `load(dir_path: str | Path) -> ByteLevelBPETokenizer`, `token_to_id(token: str) -> int`, `vocab_size` property

- [ ] **Step 1: Write the failing test in `tests/test_tokenizer.py`**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tokenizer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.tokenizer'`

- [ ] **Step 3: Implement `src/tokenizer/pretokenizer.py` and `src/tokenizer/bpe.py`**

Create empty `src/tokenizer/__init__.py`.

```python
# src/tokenizer/pretokenizer.py
import re
from functools import lru_cache

# Splits each digit 0-9 individually while keeping word prefixes and math symbols clean.
PRETOKENIZE_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+|[0-9]| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
)


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


@lru_cache(maxsize=1)
def unicode_to_bytes() -> dict[str, int]:
    return {v: k for k, v in bytes_to_unicode().items()}


def pretokenize(text: str) -> list[str]:
    return PRETOKENIZE_PATTERN.findall(text)
```

```python
# src/tokenizer/bpe.py
from collections import Counter
import json
from pathlib import Path
import re
from typing import Iterable

from src.tokenizer.pretokenizer import bytes_to_unicode, pretokenize, unicode_to_bytes

SPECIAL_TOKENS: list[str] = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|user|>",
    "<|assistant|>",
    "<think>",
    "</think>",
]


def _get_pair_stats(word_freqs: dict[tuple[str, ...], int]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for word, freq in word_freqs.items():
        if len(word) < 2:
            continue
        for i in range(len(word) - 1):
            counts[(word[i], word[i + 1])] += freq
    return counts


def _merge_word(word: tuple[str, ...], pair: tuple[str, str], merged: str) -> tuple[str, ...]:
    new_word: list[str] = []
    i = 0
    n = len(word)
    a, b = pair
    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            new_word.append(merged)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


class ByteLevelBPETokenizer:
    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        merges: list[tuple[str, str]] | None = None,
        special_tokens: list[str] | None = None,
    ) -> None:
        self.special_tokens = list(special_tokens) if special_tokens is not None else list(SPECIAL_TOKENS)
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = unicode_to_bytes()
        self.merges: list[tuple[str, str]] = list(merges) if merges is not None else []
        self.bpe_ranks: dict[tuple[str, str], int] = {p: i for i, p in enumerate(self.merges)}
        self._bpe_cache: dict[str, tuple[str, ...]] = {}

        if vocab is not None:
            self.vocab = dict(vocab)
        else:
            self.vocab = {}
            for tok in self.special_tokens:
                self.vocab[tok] = len(self.vocab)
            for b in range(256):
                ch = self.byte_encoder[b]
                if ch not in self.vocab:
                    self.vocab[ch] = len(self.vocab)

        self.id_to_token: dict[int, str] = {i: t for t, i in self.vocab.items()}
        self.special_token_ids: set[int] = {self.vocab[t] for t in self.special_tokens if t in self.vocab}
        escaped = [re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True)]
        self._special_re = re.compile("(" + "|".join(escaped) + ")") if escaped else None

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def token_to_id(self, token: str) -> int:
        return self.vocab[token]

    def train(self, texts: Iterable[str], vocab_size: int = 16384) -> None:
        base_size = len(self.special_tokens) + 256
        if vocab_size < base_size:
            raise ValueError(f"vocab_size ({vocab_size}) must be >= {base_size}")

        # Reset to base vocabulary
        self.vocab = {}
        for tok in self.special_tokens:
            self.vocab[tok] = len(self.vocab)
        for b in range(256):
            self.vocab[self.byte_encoder[b]] = len(self.vocab)
        self.merges = []
        self._bpe_cache.clear()

        word_freqs: Counter[tuple[str, ...]] = Counter()
        for text in texts:
            parts = self._special_re.split(text) if self._special_re else [text]
            for part in parts:
                if not part or part in self.special_tokens:
                    continue
                for piece in pretokenize(part):
                    encoded_piece = tuple(self.byte_encoder[b] for b in piece.encode("utf-8"))
                    # Never merge digits with non-digits or multiple digits together
                    if len(encoded_piece) >= 2 and not any(ch.isdigit() for ch in piece):
                        word_freqs[encoded_piece] += 1

        word_dict = dict(word_freqs)
        num_merges = vocab_size - len(self.vocab)
        for _ in range(num_merges):
            stats = _get_pair_stats(word_dict)
            if not stats:
                break
            best_pair, best_freq = stats.most_common(1)[0]
            if best_freq < 1:
                break
            merged_tok = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)
            self.vocab[merged_tok] = len(self.vocab)
            word_dict = {
                _merge_word(word, best_pair, merged_tok): freq for word, freq in word_dict.items()
            }

        self.bpe_ranks = {p: i for i, p in enumerate(self.merges)}
        self.id_to_token = {i: t for t, i in self.vocab.items()}
        self.special_token_ids = {self.vocab[t] for t in self.special_tokens}

    def _bpe(self, token: str) -> tuple[str, ...]:
        if token in self._bpe_cache:
            return self._bpe_cache[token]
        word = tuple(token)
        if len(word) < 2:
            self._bpe_cache[token] = word
            return word

        while True:
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
            ranked = [(self.bpe_ranks[p], p) for p in pairs if p in self.bpe_ranks]
            if not ranked:
                break
            _, best_pair = min(ranked)
            word = _merge_word(word, best_pair, best_pair[0] + best_pair[1])
            if len(word) == 1:
                break

        self._bpe_cache[token] = word
        return word

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.vocab["<|bos|>"])

        parts = self._special_re.split(text) if self._special_re else [text]
        for part in parts:
            if not part:
                continue
            if part in self.vocab and part in self.special_tokens:
                ids.append(self.vocab[part])
                continue
            for piece in pretokenize(part):
                byte_str = "".join(self.byte_encoder[b] for b in piece.encode("utf-8"))
                for tok in self._bpe(byte_str):
                    ids.append(self.vocab[tok])

        if add_eos:
            ids.append(self.vocab["<|eos|>"])
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        out_chunks: list[str] = []
        byte_buf = bytearray()

        def flush_bytes() -> None:
            if byte_buf:
                out_chunks.append(bytes(byte_buf).decode("utf-8", errors="replace"))
                byte_buf.clear()

        for idx in ids:
            tok = self.id_to_token[idx]
            if idx in self.special_token_ids:
                flush_bytes()
                if not skip_special_tokens:
                    out_chunks.append(tok)
            else:
                for ch in tok:
                    byte_buf.append(self.byte_decoder[ch])

        flush_bytes()
        return "".join(out_chunks)

    def save(self, dir_path: str | Path) -> None:
        out = Path(dir_path)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "vocab.json").open("w", encoding="utf-8") as f:
            json.dump({"special_tokens": self.special_tokens, "vocab": self.vocab}, f, ensure_ascii=False, indent=2)
        with (out / "merges.txt").open("w", encoding="utf-8") as f:
            for a, b in self.merges:
                f.write(f"{a} {b}\n")

    @classmethod
    def load(cls, dir_path: str | Path) -> "ByteLevelBPETokenizer":
        src = Path(dir_path)
        with (src / "vocab.json").open("r", encoding="utf-8") as f:
            data = json.load(f)
        merges: list[tuple[str, str]] = []
        with (src / "merges.txt").open("r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                a, b = line.split(" ")
                merges.append((a, b))
        return cls(vocab=data["vocab"], merges=merges, special_tokens=data["special_tokens"])
```

- [ ] **Step 4: Create `scripts/train_tokenizer.py`**

```python
# scripts/train_tokenizer.py
import argparse
from pathlib import Path
from datasets import load_dataset
from src.tokenizer.bpe import ByteLevelBPETokenizer


def iter_corpus(tinystories_samples: int, gsm8k_samples: int):
    ts = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    for i, row in enumerate(ts):
        if i >= tinystories_samples:
            break
        yield row["text"]

    gsm = load_dataset("openai/gsm8k", "main", split="train")
    for i, row in enumerate(gsm):
        if i >= gsm8k_samples:
            break
        yield f"{row['question']}\n{row['answer']}\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--tinystories-samples", type=int, default=50000)
    parser.add_argument("--gsm8k-samples", type=int, default=7473)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tokenizer"))
    args = parser.parse_args()

    tok = ByteLevelBPETokenizer()
    tok.train(iter_corpus(args.tinystories_samples, args.gsm8k_samples), vocab_size=args.vocab_size)
    tok.save(args.output_dir)
    print(f"Saved tokenizer with vocab_size={tok.vocab_size} to {args.output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_tokenizer.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add src/tokenizer/__init__.py src/tokenizer/pretokenizer.py src/tokenizer/bpe.py scripts/train_tokenizer.py tests/test_tokenizer.py
git commit -m "feat: implement math-aware byte-level BPE tokenizer from scratch"
```

---

### Task 3: Dynamic Tensor KV Cache (`DynamicKVCache`)

**Files:**
- Create: `src/cache/__init__.py`
- Create: `src/cache/dynamic_kv_cache.py`
- Test: `tests/test_dynamic_kv_cache.py`

**Interfaces:**
- Produces:
  - `DynamicKVCache(n_layers, batch_size, n_kv_heads, d_head, max_seq_len, chunk_size=128, dtype=torch.float16, device="cpu")`
  - `update(layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]`
  - `get_seq_len(layer_idx: int = 0) -> int`
  - `reset() -> None`
  - `copy_batch(k_factor: int) -> DynamicKVCache` (Copy KV Cache: duplicates prefilled prompt cache into K independent branch buffers)
  - `allocated_bytes() -> int`
  - `active_bytes() -> int`

- [ ] **Step 1: Write the failing test in `tests/test_dynamic_kv_cache.py`**

```python
import pytest
import torch
from src.cache.dynamic_kv_cache import DynamicKVCache


def test_dynamic_kv_cache_grows_by_chunk_and_returns_exact_views():
    cache = DynamicKVCache(
        n_layers=2,
        batch_size=1,
        n_kv_heads=4,
        d_head=64,
        max_seq_len=512,
        chunk_size=128,
        dtype=torch.float32,
        device="cpu",
    )
    assert cache.get_seq_len(0) == 0
    initial_alloc = cache.allocated_bytes()
    assert initial_alloc == 2 * 2 * 1 * 4 * 128 * 64 * 4

    # Prefill 100 tokens (fits inside initial chunk 128)
    k1 = torch.randn(1, 4, 100, 64)
    v1 = torch.randn(1, 4, 100, 64)
    k_view, v_view = cache.update(0, k1, v1)
    assert k_view.shape == (1, 4, 100, 64)
    assert torch.allclose(k_view, k1)
    assert cache.allocated_bytes() == initial_alloc

    # Append 40 tokens -> triggers growth to 256
    k2 = torch.randn(1, 4, 40, 64)
    v2 = torch.randn(1, 4, 40, 64)
    k_view2, v_view2 = cache.update(0, k2, v2)
    assert k_view2.shape == (1, 4, 140, 64)
    assert torch.allclose(k_view2[:, :, :100, :], k1)
    assert torch.allclose(k_view2[:, :, 100:, :], k2)
    assert cache.get_seq_len(0) == 140
    assert cache.allocated_bytes() > initial_alloc


def test_copy_batch_isolates_prefix_memory_and_overflow_raises():
    cache = DynamicKVCache(
        n_layers=1, batch_size=1, n_kv_heads=2, d_head=16, max_seq_len=32, chunk_size=16, dtype=torch.float32
    )
    k_prompt = torch.ones(1, 2, 10, 16)
    v_prompt = torch.full((1, 2, 10, 16), 2.0)
    cache.update(0, k_prompt, v_prompt)

    copied = cache.copy_batch(4)
    assert copied.batch_size == 4
    assert copied.get_seq_len(0) == 10

    k_next = torch.arange(4, dtype=torch.float32).view(4, 1, 1, 1).expand(4, 2, 1, 16)
    k_out, _ = copied.update(0, k_next, k_next)
    assert k_out.shape == (4, 2, 11, 16)
    assert torch.allclose(k_out[0, :, 10, :], torch.zeros(2, 16))
    assert torch.allclose(k_out[3, :, 10, :], torch.full((2, 16), 3.0))

    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        copied.update(0, torch.zeros(4, 2, 25, 16), torch.zeros(4, 2, 25, 16))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dynamic_kv_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.cache'`

- [ ] **Step 3: Implement `src/cache/dynamic_kv_cache.py`**

Create empty `src/cache/__init__.py`.

```python
# src/cache/dynamic_kv_cache.py
import math
import torch


class DynamicKVCache:
    """Chunk-growing dynamic tensor buffer for autoregressive KV caching."""

    def __init__(
        self,
        n_layers: int,
        batch_size: int,
        n_kv_heads: int,
        d_head: int,
        max_seq_len: int,
        chunk_size: int = 128,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cpu",
        initial_capacity: int | None = None,
    ) -> None:
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.max_seq_len = max_seq_len
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.device = torch.device(device)

        init_cap = initial_capacity if initial_capacity is not None else min(chunk_size, max_seq_len)
        self._capacities: list[int] = [init_cap for _ in range(n_layers)]
        self._seq_lens: list[int] = [0 for _ in range(n_layers)]
        self._k_cache: list[torch.Tensor] = [
            torch.empty((batch_size, n_kv_heads, init_cap, d_head), dtype=dtype, device=self.device)
            for _ in range(n_layers)
        ]
        self._v_cache: list[torch.Tensor] = [
            torch.empty((batch_size, n_kv_heads, init_cap, d_head), dtype=dtype, device=self.device)
            for _ in range(n_layers)
        ]

    def get_seq_len(self, layer_idx: int = 0) -> int:
        return self._seq_lens[layer_idx]

    def _grow_layer(self, layer_idx: int, required_len: int) -> None:
        chunks_needed = math.ceil(required_len / self.chunk_size)
        new_cap = min(self.max_seq_len, max(self._capacities[layer_idx], chunks_needed * self.chunk_size))
        cur_len = self._seq_lens[layer_idx]

        new_k = torch.empty(
            (self.batch_size, self.n_kv_heads, new_cap, self.d_head), dtype=self.dtype, device=self.device
        )
        new_v = torch.empty(
            (self.batch_size, self.n_kv_heads, new_cap, self.d_head), dtype=self.dtype, device=self.device
        )
        if cur_len > 0:
            new_k[:, :, :cur_len, :] = self._k_cache[layer_idx][:, :, :cur_len, :]
            new_v[:, :, :cur_len, :] = self._v_cache[layer_idx][:, :, :cur_len, :]

        self._k_cache[layer_idx] = new_k
        self._v_cache[layer_idx] = new_v
        self._capacities[layer_idx] = new_cap

    def update(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cur_len = self._seq_lens[layer_idx]
        add_len = k_new.size(2)
        new_len = cur_len + add_len
        if new_len > self.max_seq_len:
            raise ValueError(f"Requested sequence length {new_len} exceeds max_seq_len {self.max_seq_len}")

        if new_len > self._capacities[layer_idx]:
            self._grow_layer(layer_idx, new_len)

        self._k_cache[layer_idx][:, :, cur_len:new_len, :] = k_new
        self._v_cache[layer_idx][:, :, cur_len:new_len, :] = v_new
        self._seq_lens[layer_idx] = new_len

        return (
            self._k_cache[layer_idx][:, :, :new_len, :],
            self._v_cache[layer_idx][:, :, :new_len, :],
        )

    def reset(self) -> None:
        for i in range(self.n_layers):
            self._seq_lens[i] = 0

    def copy_batch(self, k_factor: int) -> "DynamicKVCache":
        """Copy KV Cache baseline: clones prefilled prompt cache into k_factor independent branch buffers."""
        if k_factor < 1:
            raise ValueError("k_factor must be >= 1")
        max_cur_cap = max(self._capacities)
        cloned = DynamicKVCache(
            n_layers=self.n_layers,
            batch_size=self.batch_size * k_factor,
            n_kv_heads=self.n_kv_heads,
            d_head=self.d_head,
            max_seq_len=self.max_seq_len,
            chunk_size=self.chunk_size,
            dtype=self.dtype,
            device=self.device,
            initial_capacity=max_cur_cap,
        )
        for i in range(self.n_layers):
            cur_len = self._seq_lens[i]
            cloned._seq_lens[i] = cur_len
            if cur_len > 0:
                k_rep = self._k_cache[i][:, :, :cur_len, :].repeat_interleave(k_factor, dim=0)
                v_rep = self._v_cache[i][:, :, :cur_len, :].repeat_interleave(k_factor, dim=0)
                cloned._k_cache[i][:, :, :cur_len, :] = k_rep
                cloned._v_cache[i][:, :, :cur_len, :] = v_rep
        return cloned

    def allocated_bytes(self) -> int:
        return sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in zip(self._k_cache, self._v_cache))

    def active_bytes(self) -> int:
        elem_size = self._k_cache[0].element_size()
        return sum(
            2 * self.batch_size * self.n_kv_heads * seq_len * self.d_head * elem_size
            for seq_len in self._seq_lens
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dynamic_kv_cache.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/cache/__init__.py src/cache/dynamic_kv_cache.py tests/test_dynamic_kv_cache.py
git commit -m "feat: implement chunk-growing DynamicKVCache with copy_batch"
```

---

### Task 4: Modern Transformer Building Blocks (`RMSNorm`, `RoPE`, `SwiGLUFFN`, `GroupedQueryAttention`)

**Files:**
- Create: `src/model/norm.py`
- Create: `src/model/rope.py`
- Create: `src/model/ffn.py`
- Create: `src/model/attention.py`
- Test: `tests/test_transformer_blocks.py`

**Interfaces:**
- Produces:
  - `RMSNorm(dim: int, eps: float = 1e-6)`
  - `precompute_rope_freqs(d_head: int, max_seq_len: int, theta: float = 10000.0, device=None) -> tuple[torch.Tensor, torch.Tensor]` (shapes `(1, 1, max_seq_len, d_head)`)
  - `apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, start_pos: int = 0) -> tuple[torch.Tensor, torch.Tensor]`
  - `SwiGLUFFN(d_model: int, d_ff: int, dropout: float = 0.0)`
  - `GroupedQueryAttention(config: ModelConfig, layer_idx: int)` with `forward(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, start_pos: int = 0, kv_cache: DynamicKVCache | None = None) -> torch.Tensor`

- [ ] **Step 1: Write the failing test in `tests/test_transformer_blocks.py`**

```python
import torch
from src.cache.dynamic_kv_cache import DynamicKVCache
from src.model.attention import GroupedQueryAttention
from src.model.config import ModelConfig
from src.model.ffn import SwiGLUFFN
from src.model.norm import RMSNorm
from src.model.rope import apply_rope, precompute_rope_freqs


def test_rmsnorm_and_swiglu_shapes():
    norm = RMSNorm(64)
    x = torch.randn(2, 10, 64)
    y = norm(x)
    assert y.shape == x.shape
    rms = y.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)

    ffn = SwiGLUFFN(64, 128)
    assert ffn(x).shape == x.shape


def test_rope_prefill_vs_step_equivalence():
    cos, sin = precompute_rope_freqs(d_head=16, max_seq_len=64)
    q = torch.randn(1, 4, 8, 16)
    k = torch.randn(1, 2, 8, 16)

    q_full, k_full = apply_rope(q, k, cos, sin, start_pos=0)
    q_step, k_step = apply_rope(q[:, :, 5:6, :], k[:, :, 5:6, :], cos, sin, start_pos=5)
    assert torch.allclose(q_full[:, :, 5:6, :], q_step, atol=1e-6)
    assert torch.allclose(k_full[:, :, 5:6, :], k_step, atol=1e-6)


def test_gqa_cached_step_matches_uncached_full_forward():
    torch.manual_seed(42)
    cfg = ModelConfig(vocab_size=256, block_size=64, d_model=64, n_layers=1, n_heads=4, n_kv_heads=2, d_ff=128)
    attn = GroupedQueryAttention(cfg, layer_idx=0).eval()
    cos, sin = precompute_rope_freqs(cfg.d_head, cfg.block_size, cfg.rope_theta)

    x = torch.randn(1, 12, 64)
    with torch.no_grad():
        full_out = attn(x, cos, sin, start_pos=0, kv_cache=None)

        cache = DynamicKVCache(1, 1, cfg.n_kv_heads, cfg.d_head, cfg.block_size, dtype=torch.float32)
        prefill_out = attn(x[:, :8, :], cos, sin, start_pos=0, kv_cache=cache)
        step_outs = [prefill_out]
        for pos in range(8, 12):
            step_outs.append(attn(x[:, pos : pos + 1, :], cos, sin, start_pos=pos, kv_cache=cache))
        cached_out = torch.cat(step_outs, dim=1)

    assert torch.allclose(full_out, cached_out, atol=1e-5)


def test_gqa_manual_matches_sdpa_backend():
    torch.manual_seed(11)
    cfg_manual = ModelConfig(
        vocab_size=256, block_size=64, d_model=64, n_layers=1, n_heads=4, n_kv_heads=2, d_ff=128, attn_backend="manual"
    )
    cfg_sdpa = ModelConfig(
        vocab_size=256, block_size=64, d_model=64, n_layers=1, n_heads=4, n_kv_heads=2, d_ff=128, attn_backend="sdpa"
    )
    attn_manual = GroupedQueryAttention(cfg_manual, layer_idx=0).eval()
    attn_sdpa = GroupedQueryAttention(cfg_sdpa, layer_idx=0).eval()
    attn_sdpa.load_state_dict(attn_manual.state_dict())

    cos, sin = precompute_rope_freqs(cfg_manual.d_head, cfg_manual.block_size, cfg_manual.rope_theta)
    x = torch.randn(2, 10, 64)
    with torch.no_grad():
        y_manual = attn_manual(x, cos, sin)
        y_sdpa = attn_sdpa(x, cos, sin)
    assert torch.allclose(y_manual, y_sdpa, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transformer_blocks.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.model.norm'`

- [ ] **Step 3: Implement `norm.py`, `rope.py`, `ffn.py`, and `attention.py`**

```python
# src/model/norm.py
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.float()
        normed = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normed.to(orig_dtype)) * self.weight
```

```python
# src/model/rope.py
import torch


def precompute_rope_freqs(
    d_head: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    half_dim = d_head // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half_dim, dtype=torch.float32, device=device) * 2.0 / d_head))
    positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)  # (max_seq_len, d_head // 2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, d_head)
    cos = emb.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, max_seq_len, d_head)
    sin = emb.sin().unsqueeze(0).unsqueeze(0)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    start_pos: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = q.size(2)
    c = cos[:, :, start_pos : start_pos + seq_len, :].to(dtype=q.dtype, device=q.device)
    s = sin[:, :, start_pos : start_pos + seq_len, :].to(dtype=q.dtype, device=q.device)
    q_rot = q * c + _rotate_half(q) * s
    k_rot = k * c + _rotate_half(k) * s
    return q_rot, k_rot
```

```python
# src/model/ffn.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.w_gate(x)) * self.w_up(x)
        return self.dropout(self.w_down(hidden))
```

```python
# src/model/attention.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.cache.dynamic_kv_cache import DynamicKVCache
from src.model.config import ModelConfig
from src.model.rope import apply_rope


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.d_head = config.d_head
        self.n_rep = self.n_heads // self.n_kv_heads
        self.dropout = config.dropout
        self.attn_backend = config.attn_backend

        self.w_q = nn.Linear(config.d_model, self.n_heads * self.d_head, bias=False)
        self.w_k = nn.Linear(config.d_model, self.n_kv_heads * self.d_head, bias=False)
        self.w_v = nn.Linear(config.d_model, self.n_kv_heads * self.d_head, bias=False)
        self.w_o = nn.Linear(self.n_heads * self.d_head, config.d_model, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def _manual_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if causal_mask is not None:
            scores = scores.masked_fill(~causal_mask, float("-inf"))
        probs = F.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
        if self.training and self.dropout > 0.0:
            probs = self.attn_dropout(probs)
        return torch.matmul(probs, v)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
        kv_cache: DynamicKVCache | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.w_q(x).view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x).view(bsz, seq_len, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(bsz, seq_len, self.n_kv_heads, self.d_head).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin, start_pos=start_pos)

        if kv_cache is not None:
            k, v = kv_cache.update(self.layer_idx, k, v)

        total_len = k.size(2)
        if self.n_rep > 1:
            k = (
                k[:, :, None, :, :]
                .expand(bsz, self.n_kv_heads, self.n_rep, total_len, self.d_head)
                .reshape(bsz, self.n_heads, total_len, self.d_head)
            )
            v = (
                v[:, :, None, :, :]
                .expand(bsz, self.n_kv_heads, self.n_rep, total_len, self.d_head)
                .reshape(bsz, self.n_heads, total_len, self.d_head)
            )

        attn_mask: torch.Tensor | None = None
        if seq_len > 1:
            row_idx = torch.arange(seq_len, device=x.device).unsqueeze(1)
            col_idx = torch.arange(total_len, device=x.device).unsqueeze(0)
            attn_mask = col_idx <= (start_pos + row_idx)

        if self.attn_backend == "manual":
            out = self._manual_attention(q, k, v, attn_mask)
        else:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.d_head)
        return self.resid_dropout(self.w_o(out))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_transformer_blocks.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/model/norm.py src/model/rope.py src/model/ffn.py src/model/attention.py tests/test_transformer_blocks.py
git commit -m "feat: implement RMSNorm, RoPE, SwiGLU FFN, and GQA with KV cache"
```

---

### Task 5: Full `DecoderOnlyLM` Architecture & Weight Tying

**Files:**
- Create: `src/model/transformer.py`
- Test: `tests/test_decoder_lm.py`

**Interfaces:**
- Produces:
  - `TransformerBlock(config: ModelConfig, layer_idx: int)`
  - `DecoderOnlyLM(config: ModelConfig)` with:
    - `gradient_checkpointing: bool` attribute (default `False`)
    - `forward(input_ids: torch.Tensor, kv_cache: DynamicKVCache | None = None, start_pos: int = 0) -> torch.Tensor` (returns logits `(B, T, vocab_size)`)
    - `forward_with_hidden(input_ids: torch.Tensor, kv_cache: DynamicKVCache | None = None, start_pos: int = 0) -> tuple[torch.Tensor, torch.Tensor]` (returns `(logits, hidden_states)`)
    - `create_kv_cache(batch_size: int, dtype: torch.dtype | None = None, device: torch.device | str | None = None, chunk_size: int = 128) -> DynamicKVCache`

- [ ] **Step 1: Write the failing test in `tests/test_decoder_lm.py`**

```python
import torch
from src.model.config import ModelConfig
from src.model.transformer import DecoderOnlyLM


def test_decoder_lm_parameter_count_and_weight_tying():
    cfg = ModelConfig.from_yaml("configs/compact_12gb.yaml")
    model = DecoderOnlyLM(cfg)
    actual_params = sum(p.numel() for p in model.parameters())
    assert actual_params == cfg.count_parameters()
    assert model.embed_tokens.weight is model.lm_head.weight


def test_decoder_lm_kv_cache_logits_match_uncached():
    torch.manual_seed(7)
    cfg = ModelConfig(vocab_size=512, block_size=64, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=128)
    model = DecoderOnlyLM(cfg).eval()
    tokens = torch.randint(0, 512, (1, 10))

    with torch.no_grad():
        logits_full, hidden_full = model.forward_with_hidden(tokens)
        cache = model.create_kv_cache(batch_size=1, dtype=torch.float32, device="cpu")
        logits_pre = model(tokens[:, :6], kv_cache=cache, start_pos=0)
        step_logits = [logits_pre]
        for pos in range(6, 10):
            step_logits.append(model(tokens[:, pos : pos + 1], kv_cache=cache, start_pos=pos))
        logits_cached = torch.cat(step_logits, dim=1)

    assert hidden_full.shape == (1, 10, 64)
    diff = (logits_full - logits_cached).abs()
    assert diff.max().item() < 1e-5
    assert diff.mean().item() < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_decoder_lm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.model.transformer'`

- [ ] **Step 3: Implement `src/model/transformer.py`**

```python
# src/model/transformer.py
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.cache.dynamic_kv_cache import DynamicKVCache
from src.model.attention import GroupedQueryAttention
from src.model.config import ModelConfig
from src.model.ffn import SwiGLUFFN
from src.model.norm import RMSNorm
from src.model.rope import precompute_rope_freqs


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = GroupedQueryAttention(config, layer_idx=layer_idx)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn = SwiGLUFFN(config.d_model, config.d_ff, dropout=config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
        kv_cache: DynamicKVCache | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, start_pos=start_pos, kv_cache=kv_cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class DecoderOnlyLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([TransformerBlock(config, layer_idx=i) for i in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        cos, sin = precompute_rope_freqs(config.d_head, config.block_size, theta=config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scale residual output projections per GPT-2/Llama convention
        res_std = 0.02 / math.sqrt(2 * config.n_layers)
        for block in self.layers:
            nn.init.normal_(block.attn.w_o.weight, mean=0.0, std=res_std)
            nn.init.normal_(block.ffn.w_down.weight, mean=0.0, std=res_std)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def create_kv_cache(
        self,
        batch_size: int,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        chunk_size: int = 128,
    ) -> DynamicKVCache:
        param = self.embed_tokens.weight
        resolved_dtype = dtype if dtype is not None else param.dtype
        resolved_device = device if device is not None else param.device
        return DynamicKVCache(
            n_layers=self.config.n_layers,
            batch_size=batch_size,
            n_kv_heads=self.config.n_kv_heads,
            d_head=self.config.d_head,
            max_seq_len=self.config.block_size,
            chunk_size=chunk_size,
            dtype=resolved_dtype,
            device=resolved_device,
        )

    def forward_with_hidden(
        self,
        input_ids: torch.Tensor,
        kv_cache: DynamicKVCache | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = input_ids.size(1)
        if start_pos + seq_len > self.config.block_size:
            raise ValueError(
                f"Sequence position {start_pos + seq_len} exceeds block_size {self.config.block_size}"
            )

        x = self.drop(self.embed_tokens(input_ids))
        for block in self.layers:
            if self.training and self.gradient_checkpointing and kv_cache is None:
                x = checkpoint(
                    lambda h, b=block: b(h, self.rope_cos, self.rope_sin, start_pos=start_pos, kv_cache=None),
                    x,
                    use_reentrant=False,
                )
            else:
                x = block(x, self.rope_cos, self.rope_sin, start_pos=start_pos, kv_cache=kv_cache)

        hidden = self.norm(x)
        logits = self.lm_head(hidden)
        return logits, hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: DynamicKVCache | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        logits, _ = self.forward_with_hidden(input_ids, kv_cache=kv_cache, start_pos=start_pos)
        return logits
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_decoder_lm.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/model/transformer.py tests/test_decoder_lm.py
git commit -m "feat: implement DecoderOnlyLM with tied embeddings and checkpointing"
```

---

### Task 6: Autoregressive Generation Engine (`AutoregressiveGenerator`)

**Files:**
- Create: `src/inference/__init__.py`
- Create: `src/inference/generator.py`
- Test: `tests/test_generator.py`

**Interfaces:**
- Produces:
  - `GenerationResult` dataclass (`sequences`, `prompt_len`, `token_logprobs`, `ttft_ms`, `decode_ms`, `tokens_per_sec`, `next_token_latencies_ms`, `peak_vram_mb`, `kv_cache_allocated_mb`, `forward_passes`, `generated_tokens`)
  - `AutoregressiveGenerator(model: DecoderOnlyLM, pad_token_id: int = 0, eos_token_id: int = 2)`
  - `AutoregressiveGenerator.generate(prompt_ids, max_new_tokens, temperature=0.0, top_p=1.0, repetition_penalty=1.0, use_cache=True, stop_token_ids=None, num_return_sequences=1) -> GenerationResult`

- [ ] **Step 1: Write the failing test in `tests/test_generator.py`**

```python
import torch
from src.inference.generator import AutoregressiveGenerator
from src.model.config import ModelConfig
from src.model.transformer import DecoderOnlyLM


def test_greedy_generation_kv_cache_on_vs_off_exact_match():
    torch.manual_seed(123)
    cfg = ModelConfig(vocab_size=256, block_size=128, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=128)
    model = DecoderOnlyLM(cfg).eval()
    gen = AutoregressiveGenerator(model, pad_token_id=0, eos_token_id=255)  # avoid early stop on id 2

    prompt = torch.randint(1, 200, (1, 12))
    res_on = gen.generate(prompt, max_new_tokens=32, temperature=0.0, use_cache=True)
    res_off = gen.generate(prompt, max_new_tokens=32, temperature=0.0, use_cache=False)

    assert torch.equal(res_on.sequences, res_off.sequences)
    assert res_on.sequences.shape == (1, 44)
    assert res_on.token_logprobs.shape == (1, 32)
    assert len(res_on.next_token_latencies_ms) == 31


def test_batched_copy_kv_cache_generates_k_distinct_sequences():
    torch.manual_seed(99)
    cfg = ModelConfig(vocab_size=256, block_size=128, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=128)
    model = DecoderOnlyLM(cfg).eval()
    gen = AutoregressiveGenerator(model, pad_token_id=0, eos_token_id=255)

    prompt = torch.randint(1, 200, (1, 8))
    res = gen.generate(
        prompt, max_new_tokens=16, temperature=0.9, top_p=0.95, use_cache=True, stop_token_ids=[], num_return_sequences=4
    )
    assert res.sequences.shape == (4, 24)
    assert res.token_logprobs.shape == (4, 16)
    assert res.forward_passes == 16
    assert res.generated_tokens == 64
    assert torch.equal(res.sequences[:, :8], prompt.expand(4, -1))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_generator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.inference'`

- [ ] **Step 3: Implement `src/inference/generator.py`**

Create empty `src/inference/__init__.py`.

```python
# src/inference/generator.py
from dataclasses import dataclass
import time
import torch
import torch.nn.functional as F

from src.model.transformer import DecoderOnlyLM


@dataclass
class GenerationResult:
    sequences: torch.Tensor
    prompt_len: int
    token_logprobs: torch.Tensor
    ttft_ms: float
    decode_ms: float
    tokens_per_sec: float
    next_token_latencies_ms: list[float]
    peak_vram_mb: float
    kv_cache_allocated_mb: float
    forward_passes: int
    generated_tokens: int


def _apply_repetition_penalty(logits: torch.Tensor, generated: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty == 1.0 or generated.numel() == 0:
        return logits
    out = logits.clone()
    for row in range(out.size(0)):
        unique_ids = torch.unique(generated[row])
        scores = out[row, unique_ids]
        out[row, unique_ids] = torch.where(scores < 0, scores * penalty, scores / penalty)
    return out


def _sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> tuple[torch.Tensor, torch.Tensor]:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    if temperature <= 0.0:
        next_tok = torch.argmax(log_probs, dim=-1, keepdim=True)
        tok_lp = torch.gather(log_probs, dim=-1, index=next_tok)
        return next_tok, tok_lp

    scaled = logits.float() / temperature
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(probs, dim=-1)
        mask = cum_probs - probs > top_p
        sorted_logits[mask] = float("-inf")
        filtered = torch.full_like(scaled, float("-inf"))
        filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
        scaled = filtered

    probs = F.softmax(scaled, dim=-1)
    next_tok = torch.multinomial(probs, num_samples=1)
    tok_lp = torch.gather(log_probs, dim=-1, index=next_tok)
    return next_tok, tok_lp


class AutoregressiveGenerator:
    def __init__(self, model: DecoderOnlyLM, pad_token_id: int = 0, eos_token_id: int = 2) -> None:
        self.model = model
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    @staticmethod
    def _sync(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        use_cache: bool = True,
        stop_token_ids: list[int] | None = None,
        num_return_sequences: int = 1,
    ) -> GenerationResult:
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        prompt_ids = prompt_ids.to(device)
        bsz, prompt_len = prompt_ids.shape
        if prompt_len + max_new_tokens > self.model.config.block_size:
            max_new_tokens = max(0, self.model.config.block_size - prompt_len)

        stop_ids = set(stop_token_ids) if stop_token_ids is not None else {self.eos_token_id}
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        kv_cache = self.model.create_kv_cache(batch_size=bsz, dtype=dtype, device=device) if use_cache else None

        self._sync(device)
        t_start = time.perf_counter()

        # Prefill phase
        logits = self.model(prompt_ids, kv_cache=kv_cache, start_pos=0)
        first_logits = logits[:, -1, :]

        if num_return_sequences > 1:
            if bsz != 1:
                raise ValueError("num_return_sequences > 1 requires batch_size == 1")
            first_logits = first_logits.expand(num_return_sequences, -1).contiguous()
            sequences = prompt_ids.expand(num_return_sequences, -1).clone()
            if kv_cache is not None:
                kv_cache = kv_cache.copy_batch(num_return_sequences)
            out_bsz = num_return_sequences
        else:
            sequences = prompt_ids.clone()
            out_bsz = bsz

        if max_new_tokens == 0:
            self._sync(device)
            ttft_ms = (time.perf_counter() - t_start) * 1000.0
            peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
            cache_mb = kv_cache.allocated_bytes() / (1024**2) if kv_cache is not None else 0.0
            return GenerationResult(
                sequences=sequences,
                prompt_len=prompt_len,
                token_logprobs=torch.empty((out_bsz, 0), device=device),
                ttft_ms=ttft_ms,
                decode_ms=0.0,
                tokens_per_sec=0.0,
                next_token_latencies_ms=[],
                peak_vram_mb=peak_mb,
                kv_cache_allocated_mb=cache_mb,
                forward_passes=1,
                generated_tokens=0,
            )

        # First token sampling
        next_tok, next_lp = _sample_next_token(first_logits, temperature=temperature, top_p=top_p)
        self._sync(device)
        ttft_ms = (time.perf_counter() - t_start) * 1000.0

        sequences = torch.cat([sequences, next_tok], dim=1)
        logprobs_list = [next_lp]
        finished = torch.zeros(out_bsz, dtype=torch.bool, device=device)
        for sid in stop_ids:
            finished |= next_tok.squeeze(1) == sid

        step_latencies: list[float] = []
        decode_t0 = time.perf_counter()

        for step in range(1, max_new_tokens):
            if finished.all():
                break
            cur_pos = prompt_len + step - 1
            step_t0 = time.perf_counter()

            if use_cache:
                step_logits = self.model(next_tok, kv_cache=kv_cache, start_pos=cur_pos)[:, -1, :]
            else:
                step_logits = self.model(sequences, kv_cache=None, start_pos=0)[:, -1, :]

            step_logits = _apply_repetition_penalty(step_logits, sequences[:, prompt_len:], repetition_penalty)
            next_tok, next_lp = _sample_next_token(step_logits, temperature=temperature, top_p=top_p)

            next_tok = torch.where(finished.unsqueeze(1), torch.full_like(next_tok, self.pad_token_id), next_tok)
            next_lp = torch.where(finished.unsqueeze(1), torch.zeros_like(next_lp), next_lp)

            self._sync(device)
            step_latencies.append((time.perf_counter() - step_t0) * 1000.0)

            sequences = torch.cat([sequences, next_tok], dim=1)
            logprobs_list.append(next_lp)
            for sid in stop_ids:
                finished |= next_tok.squeeze(1) == sid

        self._sync(device)
        decode_ms = (time.perf_counter() - decode_t0) * 1000.0
        gen_steps = len(step_latencies)
        tokens_per_sec = (gen_steps * out_bsz) / (decode_ms / 1000.0) if decode_ms > 0 and gen_steps > 0 else 0.0
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
        cache_mb = kv_cache.allocated_bytes() / (1024**2) if kv_cache is not None else 0.0
        total_gen_tokens = int((sequences[:, prompt_len:] != self.pad_token_id).sum().item())

        return GenerationResult(
            sequences=sequences,
            prompt_len=prompt_len,
            token_logprobs=torch.cat(logprobs_list, dim=1),
            ttft_ms=ttft_ms,
            decode_ms=decode_ms,
            tokens_per_sec=tokens_per_sec,
            next_token_latencies_ms=step_latencies,
            peak_vram_mb=peak_mb,
            kv_cache_allocated_mb=cache_mb,
            forward_passes=1 + gen_steps,
            generated_tokens=total_gen_tokens,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_generator.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/inference/__init__.py src/inference/generator.py tests/test_generator.py
git commit -m "feat: add AutoregressiveGenerator with KV cache and prefix expansion"
```

---

### Task 7: KV Cache Profiling Harness & Visualization ($L \in \{128, 512, 1024, 2048\}$)

**Files:**
- Create: `src/eval/__init__.py`
- Create: `src/eval/kv_profiler.py`
- Create: `scripts/run_kv_profiling.py`
- Test: `tests/test_kv_profiler.py`

**Interfaces:**
- Produces:
  - `ProfileRecord` dataclass (`context_length`, `use_cache`, `attn_backend`, `ttft_ms`, `tpot_mean_ms`, `tpot_p95_ms`, `tokens_per_sec`, `peak_vram_mb`, `kv_cache_allocated_mb`, `theoretical_kv_mb`)
  - `run_kv_cache_benchmark(model: DecoderOnlyLM, context_lengths=(128, 512, 1024, 2048), prompt_len=64, warmup_runs=1, repeat_runs=3) -> list[ProfileRecord]`
  - `save_profile_results(records: list[ProfileRecord], out_dir: str | Path) -> None`
  - `plot_kv_profile(records: list[ProfileRecord], out_dir: str | Path) -> list[Path]`

- [ ] **Step 1: Write the failing test in `tests/test_kv_profiler.py`**

```python
from pathlib import Path
from src.eval.kv_profiler import plot_kv_profile, run_kv_cache_benchmark, save_profile_results
from src.model.config import ModelConfig
from src.model.transformer import DecoderOnlyLM


def test_kv_profiler_benchmark_and_artifacts(tmp_path: Path):
    cfg = ModelConfig(vocab_size=256, block_size=256, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=128)
    model = DecoderOnlyLM(cfg).eval()

    records = run_kv_cache_benchmark(
        model,
        context_lengths=(64, 128),
        prompt_len=16,
        warmup_runs=1,
        repeat_runs=1,
    )
    assert len(records) == 4  # 2 lengths x 2 cache modes
    assert {r.context_length for r in records} == {64, 128}
    assert {r.use_cache for r in records} == {True, False}

    save_profile_results(records, tmp_path)
    assert (tmp_path / "kv_profile.csv").exists()
    assert (tmp_path / "kv_profile.json").exists()

    charts = plot_kv_profile(records, tmp_path)
    assert len(charts) == 3
    for chart in charts:
        assert chart.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_kv_profiler.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eval'`

- [ ] **Step 3: Implement `src/eval/kv_profiler.py` and `scripts/run_kv_profiling.py`**

Create empty `src/eval/__init__.py`.

```python
# src/eval/kv_profiler.py
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.inference.generator import AutoregressiveGenerator
from src.model.transformer import DecoderOnlyLM


@dataclass
class ProfileRecord:
    context_length: int
    use_cache: bool
    attn_backend: str
    ttft_ms: float
    tpot_mean_ms: float
    tpot_p95_ms: float
    tokens_per_sec: float
    peak_vram_mb: float
    kv_cache_allocated_mb: float
    theoretical_kv_mb: float


def run_kv_cache_benchmark(
    model: DecoderOnlyLM,
    context_lengths: Sequence[int] = (128, 512, 1024, 2048),
    prompt_len: int = 64,
    warmup_runs: int = 2,
    repeat_runs: int = 5,
) -> list[ProfileRecord]:
    device = next(model.parameters()).device
    elem_bytes = next(model.parameters()).element_size()
    # Use unreachable stop token so generation reaches exact target context_length
    gen = AutoregressiveGenerator(model, pad_token_id=0, eos_token_id=-1)
    records: list[ProfileRecord] = []

    for total_len in context_lengths:
        p_len = min(prompt_len, total_len // 2)
        gen_len = total_len - p_len
        prompt_ids = torch.randint(1, model.config.vocab_size, (1, p_len), device=device)

        for use_cache in (True, False):
            for _ in range(warmup_runs):
                gen.generate(prompt_ids, max_new_tokens=gen_len, temperature=0.0, use_cache=use_cache, stop_token_ids=[])

            ttfts, tpots_mean, tpots_p95, tps_list, vrams, allocs = [], [], [], [], [], []
            for _ in range(repeat_runs):
                res = gen.generate(
                    prompt_ids, max_new_tokens=gen_len, temperature=0.0, use_cache=use_cache, stop_token_ids=[]
                )
                ttfts.append(res.ttft_ms)
                lats = np.array(res.next_token_latencies_ms, dtype=np.float64)
                tpots_mean.append(float(lats.mean()) if lats.size > 0 else 0.0)
                tpots_p95.append(float(np.percentile(lats, 95)) if lats.size > 0 else 0.0)
                tps_list.append(res.tokens_per_sec)
                vrams.append(res.peak_vram_mb)
                allocs.append(res.kv_cache_allocated_mb)

            theo_mb = (
                model.config.kv_cache_bytes(batch_size=1, seq_len=total_len, bytes_per_elem=elem_bytes) / (1024**2)
                if use_cache
                else 0.0
            )
            records.append(
                ProfileRecord(
                    context_length=total_len,
                    use_cache=use_cache,
                    attn_backend=model.config.attn_backend,
                    ttft_ms=float(np.mean(ttfts)),
                    tpot_mean_ms=float(np.mean(tpots_mean)),
                    tpot_p95_ms=float(np.mean(tpots_p95)),
                    tokens_per_sec=float(np.mean(tps_list)),
                    peak_vram_mb=float(np.max(vrams)),
                    kv_cache_allocated_mb=float(np.mean(allocs)),
                    theoretical_kv_mb=theo_mb,
                )
            )
    return records


def save_profile_results(records: list[ProfileRecord], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in records]
    with (out / "kv_profile.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    if rows:
        with (out / "kv_profile.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def plot_kv_profile(records: list[ProfileRecord], out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    on_recs = sorted([r for r in records if r.use_cache], key=lambda r: r.context_length)
    off_recs = sorted([r for r in records if not r.use_cache], key=lambda r: r.context_length)
    xs_on = [r.context_length for r in on_recs]
    xs_off = [r.context_length for r in off_recs]

    paths: list[Path] = []

    # 1. Throughput (Tokens/sec)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs_on, [r.tokens_per_sec for r in on_recs], marker="o", label="KV Cache ON")
    ax.plot(xs_off, [r.tokens_per_sec for r in off_recs], marker="s", label="KV Cache OFF")
    ax.set_title("Decode Throughput vs Context Length")
    ax.set_xlabel("Context Length (L)")
    ax.set_ylabel("Tokens / sec")
    ax.grid(True, alpha=0.3)
    ax.legend()
    p1 = out / "tokens_per_sec_vs_L.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p1)

    # 2. TTFT vs Next Token Latency (TPOT)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs_on, [r.tpot_mean_ms for r in on_recs], marker="o", label="TPOT (KV Cache ON)")
    ax.plot(xs_off, [r.tpot_mean_ms for r in off_recs], marker="s", label="TPOT (KV Cache OFF)")
    ax.plot(xs_on, [r.ttft_ms for r in on_recs], marker="^", linestyle="--", label="TTFT (KV Cache ON)")
    ax.set_title("First Token Latency (TTFT) vs Next Token Latency (TPOT)")
    ax.set_xlabel("Context Length (L)")
    ax.set_ylabel("Latency (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    p2 = out / "latency_ttft_tpot_vs_L.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p2)

    # 3. Peak VRAM Usage
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs_on, [r.peak_vram_mb for r in on_recs], marker="o", label="Peak VRAM (KV Cache ON)")
    ax.plot(xs_off, [r.peak_vram_mb for r in off_recs], marker="s", label="Peak VRAM (KV Cache OFF)")
    ax.set_title("Peak VRAM Usage vs Context Length")
    ax.set_xlabel("Context Length (L)")
    ax.set_ylabel("VRAM (MB)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    p3 = out / "vram_vs_L.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p3)

    return paths
```

```python
# scripts/run_kv_profiling.py
import argparse
from pathlib import Path
import torch
from src.eval.kv_profiler import plot_kv_profile, run_kv_cache_benchmark, save_profile_results
from src.model.config import ModelConfig
from src.model.transformer import DecoderOnlyLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/compact_12gb.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/kv_profile"))
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 1024, 2048])
    parser.add_argument("--backends", type=str, nargs="+", default=["manual", "sdpa"])
    args = parser.parse_args()

    base_cfg = ModelConfig.from_yaml(args.config)
    dtype = getattr(torch, base_cfg.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for backend in args.backends:
        cfg = ModelConfig(**{**base_cfg.__dict__, "attn_backend": backend})
        model = DecoderOnlyLM(cfg).to(device=device, dtype=dtype).eval()
        out_sub = args.out_dir / backend
        records = run_kv_cache_benchmark(model, context_lengths=args.lengths)
        save_profile_results(records, out_sub)
        plot_kv_profile(records, out_sub)
        print(f"Saved KV profiling results ({backend}) to {out_sub}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_kv_profiler.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Run full Core test suite and commit**

Run: `pytest tests/ -v`
Expected: PASS (all Core unit tests green)

```bash
git add src/eval/__init__.py src/eval/kv_profiler.py scripts/run_kv_profiling.py tests/test_kv_profiler.py
git commit -m "feat: add KV cache profiling benchmark and plotting harness"
```
