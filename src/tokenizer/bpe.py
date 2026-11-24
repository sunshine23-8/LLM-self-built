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
