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
