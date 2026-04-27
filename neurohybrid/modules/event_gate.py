import torch
import torch.nn as nn


class EventGate(nn.Module):
    def __init__(self, beta: float = 0.0, temperature: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.beta = beta
        self.temperature = temperature
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
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        # q: [B, H, L, D]
        assert q.ndim == 4

        seq_len = q.shape[2]
        token_mask = self._prepare_token_mask(attention_mask, seq_len, q.dtype, q.device)

        # saliency: [B, H, L, 1]
        saliency = q.pow(2).mean(dim=-1, keepdim=True).sqrt()

        if token_mask is None:
            mean_saliency = saliency.mean(dim=2, keepdim=True)
            std_saliency = saliency.std(dim=2, unbiased=False, keepdim=True)
            valid_token_count = torch.full(
                (q.shape[0], 1, 1, 1),
                fill_value=seq_len,
                device=q.device,
                dtype=q.dtype,
            )
        else:
            masked_saliency = saliency * token_mask
            valid_token_count = token_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
            mean_saliency = masked_saliency.sum(dim=2, keepdim=True) / valid_token_count
            centered = (saliency - mean_saliency) * token_mask
            var_saliency = centered.pow(2).sum(dim=2, keepdim=True) / valid_token_count
            std_saliency = var_saliency.sqrt()

        threshold = mean_saliency + self.beta * std_saliency

        # logits: [B, H, L, 1]
        logits = (saliency - threshold) / max(self.temperature, self.eps)
        soft_gate = torch.sigmoid(logits)
        hard_gate = (soft_gate >= 0.5).to(q.dtype)
        gate = hard_gate.detach() - soft_gate.detach() + soft_gate

        if token_mask is not None:
            gate = gate * token_mask
            soft_gate = soft_gate * token_mask
            hard_gate = hard_gate * token_mask

        stats = {
            "gate_beta": float(self.beta),
            "gate_temperature": float(self.temperature),
            "saliency_mean": float(mean_saliency.mean().detach().cpu()),
            "saliency_std": float(std_saliency.mean().detach().cpu()),
            "threshold_mean": float(threshold.mean().detach().cpu()),
            "soft_gate_mean": float(soft_gate.mean().detach().cpu()),
            "hard_gate_mean": float(hard_gate.mean().detach().cpu()),
            "active_ratio": float(hard_gate.mean().detach().cpu()),
        }
        return gate, stats
