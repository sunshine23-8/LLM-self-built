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
