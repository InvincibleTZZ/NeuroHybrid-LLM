from neurohybrid.modules.hybrid_attention import NeuroHybridAttention


def patch_qwen_attention(
    model,
    replace_last_n_layers: int = 4,
    window_size: int = 256,
    use_event_gate: bool = False,
    gate_beta: float = 0.0,
    gate_temperature: float = 1.0,
    use_dendritic_fusion: bool = False,
    fusion_scale: float = 0.5,
):
    layers = model.model.layers
    total_layers = len(layers)

    if replace_last_n_layers <= 0:
        return model

    if replace_last_n_layers > total_layers:
        raise ValueError(
            f"replace_last_n_layers={replace_last_n_layers} exceeds total layers={total_layers}"
        )

    start_idx = total_layers - replace_last_n_layers

    for layer_idx in range(start_idx, total_layers):
        old_attn = layers[layer_idx].self_attn
        new_attn = NeuroHybridAttention(
            old_attn=old_attn,
            hidden_size=model.config.hidden_size,
            num_heads=model.config.num_attention_heads,
            num_kv_heads=model.config.num_key_value_heads,
            head_dim=model.config.hidden_size // model.config.num_attention_heads,
            window_size=window_size,
            use_event_gate=use_event_gate,
            gate_beta=gate_beta,
            gate_temperature=gate_temperature,
            use_dendritic_fusion=use_dendritic_fusion,
            fusion_scale=fusion_scale,
            layer_idx=layer_idx,
        )
        layers[layer_idx].self_attn = new_attn
        print(f"[NeuroHybrid] Replaced layer {layer_idx} self_attn with NeuroHybridAttention")

    return model
