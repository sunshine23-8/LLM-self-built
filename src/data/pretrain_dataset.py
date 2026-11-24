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
    """Encode texts stream into a packed flat uint16 binary file."""
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
    """Measure forward + backward + optimizer step throughput on available device."""
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
