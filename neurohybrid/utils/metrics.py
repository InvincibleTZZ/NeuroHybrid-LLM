from typing import Any


def _sanitize_value(value: Any):
    if isinstance(value, dict):
        return {key: _sanitize_value(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def collect_neurohybrid_stats(model):
    layers = getattr(getattr(model, "model", None), "layers", [])
    collected_layers = []
    write_ratio_values = []
    dendritic_k_mean_values = []
    dendritic_k_min_values = []
    dendritic_k_max_values = []

    for layer_idx, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        last_stats = getattr(attn, "last_stats", None)
        if last_stats is None:
            continue

        sanitized = _sanitize_value(last_stats)
        sanitized.setdefault("layer_idx", layer_idx)
        collected_layers.append(sanitized)

        write_ratio = sanitized.get("write_ratio_mean")
        if write_ratio is not None:
            write_ratio_values.append(write_ratio)

        dendritic_k_mean = sanitized.get("dendritic_k_mean")
        dendritic_k_min = sanitized.get("dendritic_k_min")
        dendritic_k_max = sanitized.get("dendritic_k_max")
        if dendritic_k_mean is not None:
            dendritic_k_mean_values.append(dendritic_k_mean)
        if dendritic_k_min is not None:
            dendritic_k_min_values.append(dendritic_k_min)
        if dendritic_k_max is not None:
            dendritic_k_max_values.append(dendritic_k_max)

    result = {
        "num_patched_layers": len(collected_layers),
        "layers": collected_layers,
    }

    if write_ratio_values:
        result["write_ratio_mean_avg"] = sum(write_ratio_values) / len(write_ratio_values)
    if dendritic_k_mean_values:
        result["dendritic_k_mean_avg"] = sum(dendritic_k_mean_values) / len(dendritic_k_mean_values)
    if dendritic_k_min_values:
        result["dendritic_k_min_avg"] = sum(dendritic_k_min_values) / len(dendritic_k_min_values)
    if dendritic_k_max_values:
        result["dendritic_k_max_avg"] = sum(dendritic_k_max_values) / len(dendritic_k_max_values)

    return result
