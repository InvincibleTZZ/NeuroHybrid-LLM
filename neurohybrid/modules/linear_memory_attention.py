import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearMemoryAttention(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def _prepare_token_mask(
        self,
        attention_mask: torch.Tensor | None,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None

        if attention_mask.dim() == 2:
            token_mask = attention_mask[:, :seq_len].to(device=device, dtype=dtype)
            return token_mask[:, None, :, None]

        if attention_mask.dim() == 4:
            key_visibility = attention_mask[:, 0, seq_len - 1, :seq_len].to(device=device)
            token_mask = (key_visibility == 0).to(dtype)
            return token_mask[:, None, :, None]

        raise ValueError(f"Unsupported attention_mask shape: {tuple(attention_mask.shape)}")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        # q: [B, H, L, D]
        # k: [B, H, L, D]
        # v: [B, H, L, D]
        assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4
        assert q.shape == k.shape == v.shape

        token_mask = self._prepare_token_mask(attention_mask, q.shape[2], q.dtype, q.device)

        # phi_q: [B, H, L, D]
        # phi_k: [B, H, L, D]
        phi_q = F.elu(q) + 1.0
        phi_k = F.elu(k) + 1.0

        if token_mask is not None:
            phi_q = phi_q * token_mask
            phi_k = phi_k * token_mask
            v = v * token_mask

        if gate is not None:
            assert gate.shape == (*q.shape[:3], 1)
            phi_k = phi_k * gate
            v = v * gate

        batch_size, num_heads, seq_len, head_dim = q.shape
        kv_state = torch.zeros(
            batch_size,
            num_heads,
            head_dim,
            head_dim,
            device=q.device,
            dtype=q.dtype,
        )
        k_state = torch.zeros(
            batch_size,
            num_heads,
            head_dim,
            device=q.device,
            dtype=q.dtype,
        )
        outputs = []

        for token_idx in range(seq_len):
            # phi_q_t: [B, H, D]
            # phi_k_t: [B, H, D]
            # v_t:     [B, H, D]
            phi_q_t = phi_q[:, :, token_idx, :]
            phi_k_t = phi_k[:, :, token_idx, :]
            v_t = v[:, :, token_idx, :]

            # kv_state: [B, H, D, D]
            kv_state = kv_state + phi_k_t.unsqueeze(-1) * v_t.unsqueeze(-2)

            # k_state: [B, H, D]
            k_state = k_state + phi_k_t

            # numerator_t: [B, H, D]
            numerator_t = torch.matmul(phi_q_t.unsqueeze(-2), kv_state).squeeze(-2)

            # denominator_t: [B, H, 1]
            denominator_t = (phi_q_t * k_state).sum(dim=-1, keepdim=True)

            # out_t: [B, H, D]
            out_t = numerator_t / denominator_t.clamp_min(self.eps)
            outputs.append(out_t)

        # out: [B, H, L, D]
        out = torch.stack(outputs, dim=2)
        assert out.shape == q.shape
        stats = {}
        if gate is not None:
            stats["gate_mean"] = float(gate.mean().detach().cpu())
        return out, stats
