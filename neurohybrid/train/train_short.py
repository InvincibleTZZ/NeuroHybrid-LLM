import argparse
import gc
import json
from itertools import cycle
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from neurohybrid.models.patch_qwen import patch_qwen_attention
from neurohybrid.modules.hybrid_attention import NeuroHybridAttention
from neurohybrid.training.datasets import build_tiny_lm_dataset, build_wikitext_lm_dataset
from neurohybrid.utils.config import (
    load_config,
    resolve_device,
    resolve_model_source,
    resolve_torch_dtype,
    save_config,
)
from neurohybrid.utils.logging_utils import log_metrics, setup_logger
from neurohybrid.utils.metrics import collect_neurohybrid_stats
from neurohybrid.utils.result_table import write_default_reports


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = REPO_ROOT / "results" / "train_short.json"


def _cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _warmup_lambda(current_step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, float(current_step + 1) / float(warmup_steps))


def _mark_only_neurohybrid_trainable(model: torch.nn.Module) -> list[str]:
    for param in model.parameters():
        param.requires_grad = False

    trainable_names = []
    for layer in getattr(getattr(model, "model", None), "layers", []):
        attn = getattr(layer, "self_attn", None)
        if not isinstance(attn, NeuroHybridAttention):
            continue

        for module_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            module = getattr(attn, module_name)
            for param_name, param in module.named_parameters(recurse=False):
                param.requires_grad = True
                trainable_names.append(f"layers.{attn.layer_idx}.self_attn.{module_name}.{param_name}")

        if attn.dendritic_fusion is not None:
            for param_name, param in attn.dendritic_fusion.named_parameters(recurse=True):
                param.requires_grad = True
                trainable_names.append(
                    f"layers.{attn.layer_idx}.self_attn.dendritic_fusion.{param_name}"
                )

        if attn.event_gate is not None:
            for param_name, param in attn.event_gate.named_parameters(recurse=True):
                param.requires_grad = True
                trainable_names.append(f"layers.{attn.layer_idx}.self_attn.event_gate.{param_name}")

    return trainable_names


def _count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def _get_trainable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def _save_checkpoint(
    checkpoint_dir: Path,
    model: torch.nn.Module,
    run_config: dict,
    step: int,
    stats: dict,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(_get_trainable_state_dict(model), checkpoint_dir / "trainable_params.pt")
    save_config(checkpoint_dir / "run_config.yaml", run_config)
    _save_json(
        checkpoint_dir / "checkpoint_meta.json",
        {
            "step": step,
            "checkpoint_format": "trainable_params_only_state_dict",
            "num_trainable_tensors": len(_get_trainable_state_dict(model)),
            "stats": stats,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--result-path", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--gradient-checkpointing", choices=["true", "false"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("neurohybrid.train_short")
    _cleanup()
    result_path = Path(args.result_path) if args.result_path else RESULT_PATH

    defaults = {
        "use_neurohybrid": True,
        "replace_last_n_layers": 4,
        "window_size": 256,
        "use_event_gate": True,
        "gate_beta": 0.5,
        "gate_temperature": 1.0,
        "use_dendritic_fusion": True,
        "fusion_scale": 0.5,
        "gradient_accumulation_steps": 8,
        "max_steps": 100,
        "learning_rate": 2e-5,
        "weight_decay": 0.0,
        "warmup_steps": 10,
        "log_every": 5,
        "save_every": 50,
        "dtype": "bf16",
        "gradient_checkpointing": True,
        "train_only_neurohybrid": True,
        "dataset_mode": "tiny_text",
        "dataset_path": str(REPO_ROOT / "data" / "wikitext-2-raw-v1"),
        "train_split": "train",
        "max_train_samples": None,
    }
    config = load_config(args.config, defaults=defaults)

    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    if args.max_seq_length is not None:
        config["max_seq_length"] = args.max_seq_length
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.gradient_accumulation_steps is not None:
        config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.gradient_checkpointing is not None:
        config["gradient_checkpointing"] = args.gradient_checkpointing == "true"

    device = resolve_device(args.device)
    dtype = resolve_torch_dtype(config.get("dtype"), device)
    model_source, local_files_only = resolve_model_source(config["model_name"])
    output_dir = REPO_ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = dict(config)
    run_config["resolved_model_source"] = model_source
    run_config["device"] = str(device)
    run_config["resolved_dtype"] = str(dtype).replace("torch.", "")
    save_config(output_dir / "run_config.yaml", run_config)

    logger.info(
        f"[train_short] device={device} dtype={run_config['resolved_dtype']} model={model_source}"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        local_files_only=local_files_only,
        dtype=dtype,
    )
    model.config.use_cache = False

    if config.get("use_neurohybrid", False):
        model = patch_qwen_attention(
            model,
            replace_last_n_layers=config["replace_last_n_layers"],
            window_size=config["window_size"],
            use_event_gate=config.get("use_event_gate", False),
            gate_beta=config.get("gate_beta", 0.5),
            gate_temperature=config.get("gate_temperature", 1.0),
            use_dendritic_fusion=config.get("use_dendritic_fusion", False),
            fusion_scale=config.get("fusion_scale", 0.5),
        )

    model.to(device)

    if config.get("gradient_checkpointing", False) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    if config.get("train_only_neurohybrid", False):
        trainable_names = _mark_only_neurohybrid_trainable(model)
    else:
        for param in model.parameters():
            param.requires_grad = True
        trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]

    total_params, trainable_params = _count_parameters(model)
    if trainable_params == 0:
        raise RuntimeError("No trainable parameters were enabled.")

    logger.info(
        f"[train_short] total_params={total_params} trainable_params={trainable_params}"
    )

    dataset_mode = config.get("dataset_mode")
    if dataset_mode == "tiny_text":
        dataset = build_tiny_lm_dataset(
            tokenizer,
            max_seq_length=config["max_seq_length"],
            num_samples=max(config["max_steps"] * config["batch_size"], 256),
        )
    elif dataset_mode == "wikitext2":
        dataset = build_wikitext_lm_dataset(
            tokenizer,
            dataset_path=config["dataset_path"],
            split=config.get("train_split", "train"),
            max_seq_length=int(config["max_seq_length"]),
            num_samples=config.get("max_train_samples"),
        )
    else:
        raise ValueError(f"Unsupported dataset_mode={dataset_mode}")
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
    dataloader_iter = cycle(dataloader)

    optimizer = AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: _warmup_lambda(step, int(config["warmup_steps"])),
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    completed_steps = 0
    micro_steps = 0
    last_loss = None
    loss_trace = []
    latest_stats = {}
    grad_accum_steps = int(config["gradient_accumulation_steps"])

    while completed_steps < int(config["max_steps"]):
        batch = next(dataloader_iter)
        batch = {key: value.to(device) for key, value in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        if not torch.isfinite(loss):
            raise RuntimeError("Encountered a non-finite loss during short training.")

        (loss / grad_accum_steps).backward()
        micro_steps += 1

        if micro_steps % grad_accum_steps != 0:
            continue

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        completed_steps += 1

        last_loss = float(loss.detach().item())
        loss_trace.append(last_loss)
        latest_stats = collect_neurohybrid_stats(model)
        peak_allocated_gb = (
            torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
        )

        metrics = {
            "loss": last_loss,
            "write_ratio": latest_stats.get("write_ratio_mean_avg"),
            "dendritic_k": latest_stats.get("dendritic_k_mean_avg"),
            "memory_gb": peak_allocated_gb,
            "learning_rate": scheduler.get_last_lr()[0],
        }

        if completed_steps == 1 or completed_steps % int(config["log_every"]) == 0:
            log_metrics(completed_steps, metrics, logger=logger)

        if completed_steps % int(config["save_every"]) == 0:
            _save_checkpoint(
                output_dir / f"step_{completed_steps:04d}",
                model,
                run_config,
                completed_steps,
                latest_stats,
            )

    final_dir = output_dir / "final"
    _save_checkpoint(final_dir, model, run_config, completed_steps, latest_stats)

    peak_allocated_gb = (
        torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    )
    peak_reserved_gb = (
        torch.cuda.max_memory_reserved(device) / 1024**3 if device.type == "cuda" else 0.0
    )

    result = {
        "model_variant": "neurohybrid" if config.get("use_neurohybrid", False) else "baseline",
        "resolved_model_source": model_source,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "output_dir": str(output_dir),
        "completed_steps": completed_steps,
        "requested_max_steps": int(config["max_steps"]),
        "batch_size": int(config["batch_size"]),
        "max_seq_length": int(config["max_seq_length"]),
        "gradient_accumulation_steps": grad_accum_steps,
        "last_loss": last_loss,
        "loss_trace_tail": loss_trace[-10:],
        "peak_allocated_gb": peak_allocated_gb,
        "peak_reserved_gb": peak_reserved_gb,
        "write_ratio_mean_avg": latest_stats.get("write_ratio_mean_avg"),
        "dendritic_k_mean_avg": latest_stats.get("dendritic_k_mean_avg"),
        "num_patched_layers": latest_stats.get("num_patched_layers"),
        "num_trainable_params": trainable_params,
        "num_total_params": total_params,
        "checkpoint_format": "trainable_params_only_state_dict",
        "final_checkpoint_dir": str(final_dir),
        "trainable_param_names": trainable_names,
        "smoke_test": completed_steps < defaults["max_steps"],
    }

    _save_json(result_path, result)
    write_default_reports(REPO_ROOT)
    logger.info(f"[train_short] saved result to {result_path}")

    del model
    del tokenizer
    _cleanup()


if __name__ == "__main__":
    main()
