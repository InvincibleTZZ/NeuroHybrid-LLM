import math

import torch
import torch.nn as nn


class LocalWindowAttention(nn.Module):
    def __init__(self, window_size: int = 256, query_chunk_size: int = 128):
        super().__init__()
        self.window_size = window_size
        self.query_chunk_size = query_chunk_size

    def _build_local_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        token_idx = torch.arange(seq_len, device=device)
        query_idx = token_idx[:, None]
        key_idx = token_idx[None, :]
        causal_mask = key_idx <= query_idx
        window_mask = key_idx >= (query_idx - self.window_size + 1)
        return causal_mask & window_mask

    def _prepare_additive_mask(
        self,
        attention_mask: torch.Tensor | None,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None

        if attention_mask.dim() == 4:
            return attention_mask[:, :, :seq_len, :seq_len].to(device=device, dtype=dtype)

        if attention_mask.dim() == 2:
            key_mask = attention_mask[:, None, None, :seq_len].to(device=device, dtype=dtype)
            return (1.0 - key_mask) * torch.finfo(dtype).min

        raise ValueError(f"Unsupported attention_mask shape: {tuple(attention_mask.shape)}")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # q: [B, H, L, D]
        # k: [B, H, L, D]
        # v: [B, H, L, D]
        assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4
        assert q.shape == k.shape == v.shape

        _, _, seq_len, head_dim = q.shape
        scale = 1.0 / math.sqrt(head_dim)
        additive_mask = self._prepare_additive_mask(attention_mask, seq_len, q.dtype, q.device)

        local_mask = self._build_local_mask(seq_len, q.device)
        output_chunks = []

        for start_idx in range(0, seq_len, self.query_chunk_size):
            end_idx = min(start_idx + self.query_chunk_size, seq_len)

            # q_chunk: [B, H, C, D]
            q_chunk = q[:, :, start_idx:end_idx, :]

            # scores: [B, H, C, L]
            scores = torch.matmul(q_chunk, k.transpose(-1, -2)) * scale

            local_mask_chunk = local_mask[start_idx:end_idx, :]
            scores = scores.masked_fill(
                ~local_mask_chunk[None, None, :, :],
                torch.finfo(scores.dtype).min,
            )

            if additive_mask is not None:
                scores = scores + additive_mask[:, :, start_idx:end_idx, :]

            # attn_weights: [B, H, C, L]
            attn_weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

            # out_chunk: [B, H, C, D]
            out_chunk = torch.matmul(attn_weights, v)
            output_chunks.append(out_chunk)

        # out: [B, H, L, D]
        out = torch.cat(output_chunks, dim=2)
        assert out.shape == q.shape
        return out
