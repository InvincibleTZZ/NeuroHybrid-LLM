import torch
import torch.nn as nn


class DendriticFusion(nn.Module):
    """
    Dendritic bilinear integration:
    y = 0.5 * (y_local + y_memory) + fusion_scale * sigmoid(gamma) * (y_local * y_memory)
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        fusion_scale: float = 0.5,
    ):
        super().__init__()
        self.fusion_scale = fusion_scale
        self.gamma = nn.Parameter(torch.zeros(1, num_heads, 1, head_dim))

    def forward(
        self,
        y_local: torch.Tensor,
        y_memory: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        # y_local: [B, H, L, D]
        # y_memory: [B, H, L, D]
        assert y_local.ndim == 4 and y_memory.ndim == 4
        assert y_local.shape == y_memory.shape

        # k: [1, H, 1, D]
        k = torch.sigmoid(self.gamma).to(dtype=y_local.dtype, device=y_local.device)

        # bilinear: [B, H, L, D]
        bilinear = y_local * y_memory

        # fused: [B, H, L, D]
        fused = 0.5 * (y_local + y_memory) + self.fusion_scale * k * bilinear

        stats = {
            "fusion_scale": float(self.fusion_scale),
            "dendritic_k_mean": float(k.mean().detach().cpu()),
            "dendritic_k_min": float(k.min().detach().cpu()),
            "dendritic_k_max": float(k.max().detach().cpu()),
        }
        return fused, stats
