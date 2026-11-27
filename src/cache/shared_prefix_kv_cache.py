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
