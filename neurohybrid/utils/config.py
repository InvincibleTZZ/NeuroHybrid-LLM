import json
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_MODEL_FALLBACKS = {
    "Qwen/Qwen2.5-0.5B": REPO_ROOT / "LLM_model" / "Qwen2.5-0.5B",
}


def merge_dicts(base: dict[str, Any] | None, override: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path, defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise ImportError("PyYAML is required. Please run `pip install pyyaml`.") from error

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return merge_dicts(defaults, loaded)


def save_config(path: str | Path, config: dict[str, Any]) -> None:
    try:
        import yaml
    except ImportError as error:
        raise ImportError("PyYAML is required. Please run `pip install pyyaml`.") from error

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


def resolve_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def resolve_torch_dtype(dtype_name: str | None, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32

    normalized = (dtype_name or "bf16").lower()
    if normalized in {"bf16", "bfloat16"}:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def resolve_model_source(model_name_or_path: str) -> tuple[str, bool]:
    model_path = Path(model_name_or_path)
    if model_path.exists():
        return str(model_path), True

    fallback = LOCAL_MODEL_FALLBACKS.get(model_name_or_path)
    if fallback is not None and fallback.exists():
        return str(fallback), True

    return model_name_or_path, False


def resolve_checkpoint_dir(checkpoint_path: str | Path | None) -> Path | None:
    if not checkpoint_path:
        return None

    path = Path(checkpoint_path)
    if not path.exists():
        return None

    if path.is_file():
        return path.parent

    direct_file = path / "trainable_params.pt"
    if direct_file.exists():
        return path

    final_dir = path / "final"
    if (final_dir / "trainable_params.pt").exists():
        return final_dir

    step_dirs = sorted(
        candidate
        for candidate in path.glob("step_*")
        if candidate.is_dir() and (candidate / "trainable_params.pt").exists()
    )
    if step_dirs:
        return step_dirs[-1]

    return path


def load_checkpoint_config(checkpoint_path: str | Path | None) -> dict[str, Any]:
    checkpoint_dir = resolve_checkpoint_dir(checkpoint_path)
    if checkpoint_dir is None:
        return {}

    candidate_paths = [
        checkpoint_dir / "run_config.yaml",
        checkpoint_dir / "run_config.json",
        checkpoint_dir.parent / "run_config.yaml",
        checkpoint_dir.parent / "run_config.json",
    ]

    for candidate_path in candidate_paths:
        if not candidate_path.exists():
            continue
        if candidate_path.suffix == ".yaml":
            return load_config(candidate_path)
        with candidate_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    return {}


def load_trainable_state(
    model: torch.nn.Module,
    checkpoint_path: str | Path | None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint_dir = resolve_checkpoint_dir(checkpoint_path)
    if checkpoint_dir is None:
        return {"loaded": False, "checkpoint_dir": None, "trainable_params_path": None}

    state_path = checkpoint_dir / "trainable_params.pt"
    if not state_path.exists():
        return {
            "loaded": False,
            "checkpoint_dir": str(checkpoint_dir),
            "trainable_params_path": None,
        }

    state_dict = torch.load(state_path, map_location=map_location)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    return {
        "loaded": True,
        "checkpoint_dir": str(checkpoint_dir),
        "trainable_params_path": str(state_path),
        "missing_keys_count": len(missing_keys),
        "unexpected_keys_count": len(unexpected_keys),
        "missing_keys_preview": sorted(missing_keys)[:10],
        "unexpected_keys_preview": sorted(unexpected_keys)[:10],
    }
