import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from .dendritic_fusion import DendriticFusion
from .event_gate import EventGate
from .linear_memory_attention import LinearMemoryAttention
from .local_window_attention import LocalWindowAttention


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    hidden_states: [B, H_kv, L, D]
    return: [B, H_q, L, D]
    """
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch,
        num_kv_heads,
        n_rep,
        seq_len,
        head_dim,
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class NeuroHybridAttention(nn.Module):
    def __init__(
        self,
        old_attn: nn.Module,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        window_size: int = 256,
        use_event_gate: bool = False,
        gate_beta: float = 0.0,
        gate_temperature: float = 1.0,
        use_dendritic_fusion: bool = False,
        fusion_scale: float = 0.5,
        layer_idx: int | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.use_event_gate = use_event_gate
        self.gate_beta = gate_beta
        self.gate_temperature = gate_temperature
        self.use_dendritic_fusion = use_dendritic_fusion
        self.fusion_scale = fusion_scale
        self.layer_idx = layer_idx
        self.num_key_value_groups = num_heads // num_kv_heads
        self.is_causal = True
        self.last_stats = {
            "layer_idx": layer_idx,
            "use_event_gate": use_event_gate,
            "use_dendritic_fusion": use_dendritic_fusion,
        }

        # Reuse the pretrained Q/K/V/O projections directly.
        self.q_proj = old_attn.q_proj
        self.k_proj = old_attn.k_proj
        self.v_proj = old_attn.v_proj
        self.o_proj = old_attn.o_proj

        self.local_window_attention = LocalWindowAttention(window_size=window_size)
        self.linear_memory_attention = LinearMemoryAttention()
        self.event_gate = EventGate(beta=gate_beta, temperature=gate_temperature) if use_event_gate else None
        self.dendritic_fusion = (
            DendriticFusion(num_heads=num_heads, head_dim=head_dim, fusion_scale=fusion_scale)
            if use_dendritic_fusion
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # hidden_states: [B, L, hidden_size]
        assert hidden_states.ndim == 3

        if use_cache or past_key_values is not None or kwargs.get("past_key_value") is not None:
            raise NotImplementedError("NeuroHybridAttention Day 2 only supports use_cache=False.")

        batch_size, seq_len, _ = hidden_states.shape

        # q_states: [B, L, H_q * D] -> [B, H_q, L, D]
        q_states = self.q_proj(hidden_states)
        q_states = q_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # k_states: [B, L, H_kv * D] -> [B, H_kv, L, D]
        k_states = self.k_proj(hidden_states)
        k_states = k_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # v_states: [B, L, H_kv * D] -> [B, H_kv, L, D]
        v_states = self.v_proj(hidden_states)
        v_states = v_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        assert q_states.shape == (batch_size, self.num_heads, seq_len, self.head_dim)
        assert k_states.shape == (batch_size, self.num_kv_heads, seq_len, self.head_dim)
        assert v_states.shape == (batch_size, self.num_kv_heads, seq_len, self.head_dim)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q_states, k_states = apply_rotary_pos_emb(q_states, k_states, cos, sin)

        gate = None
        gate_stats = None
        if self.event_gate is not None:
            gate, gate_stats = self.event_gate(q_states, attention_mask=attention_mask)
            assert gate.shape == (batch_size, self.num_heads, seq_len, 1)

        # Repeat KV heads to match grouped-query attention layout.
        k_states = repeat_kv(k_states, self.num_key_value_groups)
        v_states = repeat_kv(v_states, self.num_key_value_groups)

        assert k_states.shape == q_states.shape
        assert v_states.shape == q_states.shape

        # local_out: [B, H_q, L, D]
        local_out = self.local_window_attention(
            q=q_states,
            k=k_states,
            v=v_states,
            attention_mask=attention_mask,
        )

        # linear_out: [B, H_q, L, D]
        linear_out, linear_stats = self.linear_memory_attention(
            q=q_states,
            k=k_states,
            v=v_states,
            attention_mask=attention_mask,
            gate=gate,
        )

        fusion_stats = None
        if self.dendritic_fusion is not None:
            # hybrid_out: [B, H_q, L, D]
            hybrid_out, fusion_stats = self.dendritic_fusion(local_out, linear_out)
        else:
            # hybrid_out: [B, H_q, L, D]
            hybrid_out = 0.5 * local_out + 0.5 * linear_out

        # hybrid_out: [B, L, H_q * D]
        hybrid_out = hybrid_out.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.hidden_size)

        # attn_output: [B, L, hidden_size]
        attn_output = self.o_proj(hybrid_out)
        assert attn_output.shape == hidden_states.shape

        self.last_stats = {
            "layer_idx": self.layer_idx,
            "use_event_gate": self.use_event_gate,
            "use_dendritic_fusion": self.use_dendritic_fusion,
            "window_size": self.window_size,
            "gate_beta": self.gate_beta,
            "gate_temperature": self.gate_temperature,
            "fusion_scale": self.fusion_scale,
            "q_abs_mean": float(q_states.abs().mean().detach().cpu()),
            "k_abs_mean": float(k_states.abs().mean().detach().cpu()),
            "v_abs_mean": float(v_states.abs().mean().detach().cpu()),
            "local_abs_mean": float(local_out.abs().mean().detach().cpu()),
            "linear_abs_mean": float(linear_out.abs().mean().detach().cpu()),
            "hybrid_abs_mean": float(hybrid_out.abs().mean().detach().cpu()),
            "write_ratio_mean": gate_stats["active_ratio"] if gate_stats is not None else None,
            "dendritic_k_mean": fusion_stats["dendritic_k_mean"] if fusion_stats is not None else None,
            "dendritic_k_min": fusion_stats["dendritic_k_min"] if fusion_stats is not None else None,
            "dendritic_k_max": fusion_stats["dendritic_k_max"] if fusion_stats is not None else None,
            "linear_stats": linear_stats,
            "gate_stats": gate_stats,
            "fusion_stats": fusion_stats,
        }

        if output_attentions:
            return attn_output, None
        return attn_output, None

