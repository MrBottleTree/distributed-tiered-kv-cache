"""Exact, small-scale Q/K reference for mass over logical token windows."""

import torch
import torch.nn.functional as F


def gather_visible_keys(
    kv_cache: torch.Tensor, block_table: torch.Tensor, seq_len: int
) -> torch.Tensor:
    """Return logical K positions [seq_len, H_kv, D] from vLLM's paged cache."""
    if kv_cache.ndim != 5 or kv_cache.shape[0] != 2:
        raise ValueError("expected vLLM KV cache [2, pages, page_size, H_kv, D]")
    page_size = kv_cache.shape[2]
    page_count = (seq_len + page_size - 1) // page_size
    if seq_len <= 0 or page_count > block_table.numel():
        raise ValueError("invalid sequence length or incomplete block table")
    page_ids = block_table[:page_count].to(dtype=torch.long)
    keys = kv_cache[0].index_select(0, page_ids)
    return keys.reshape(-1, *keys.shape[-2:])[:seq_len]


def compute_window_mass(
    query: torch.Tensor,
    keys: torch.Tensor,
    *,
    window_size: int,
    scale: float,
    causal: bool = True,
    sliding_window: tuple[int, int] | None = None,
    softcap: float = 0.0,
    query_rows_per_chunk: int = 16,
) -> torch.Tensor:
    """Post-softmax mass [query_row, query_head, logical_window].

    Query rows are the final rows of ``keys`` (decode or chunked prefill). Grouped
    query heads share a KV head, but retain separate mass measurements.
    """
    if query.ndim != 3 or keys.ndim != 3:
        raise ValueError("expected query [Q, H_q, D] and keys [S, H_kv, D]")
    q_len, q_heads, dim = query.shape
    seq_len, kv_heads, key_dim = keys.shape
    if (
        not 0 < q_len <= seq_len
        or q_heads % kv_heads
        or dim != key_dim
        or window_size <= 0
        or query_rows_per_chunk <= 0
    ):
        raise ValueError("incompatible query/key shapes or window size")

    groups = q_heads // kv_heads
    key_positions = torch.arange(seq_len, device=query.device)
    windows = (seq_len + window_size - 1) // window_size
    masses = []
    key_fp32 = keys.float()

    # Chunk query rows so prefill never keeps a full Q-by-K matrix in memory.
    for start in range(0, q_len, query_rows_per_chunk):
        q = query[start : start + query_rows_per_chunk].float()
        rows = q.shape[0]
        q_grouped = q.reshape(rows, kv_heads, groups, dim)
        logits = torch.einsum("rhgd,shd->rhgs", q_grouped, key_fp32)
        logits = logits.reshape(rows, q_heads, seq_len) * scale
        if softcap > 0:
            logits = softcap * torch.tanh(logits / softcap)

        q_positions = seq_len - q_len + torch.arange(
            start, start + rows, device=query.device
        )
        visible = torch.ones((rows, seq_len), dtype=torch.bool, device=query.device)
        if causal:
            visible &= key_positions[None, :] <= q_positions[:, None]
        if sliding_window is not None:
            left, right = sliding_window
            if left >= 0:
                visible &= key_positions[None, :] >= q_positions[:, None] - left
            if right >= 0:
                visible &= key_positions[None, :] <= q_positions[:, None] + right
        probabilities = F.softmax(logits.masked_fill(~visible[:, None], -torch.inf), dim=-1)
        padded = F.pad(probabilities, (0, windows * window_size - seq_len))
        masses.append(padded.reshape(rows, q_heads, windows, window_size).sum(-1))

    return torch.cat(masses, dim=0)
