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
        """Calculate exact parameter count including embeddings, blocks, norms and tied head."""
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
        """Calculate total KV Cache memory consumption in bytes."""
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
