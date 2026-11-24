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
