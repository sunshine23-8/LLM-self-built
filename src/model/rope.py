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
