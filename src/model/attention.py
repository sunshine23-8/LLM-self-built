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
