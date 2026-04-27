import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from neurohybrid.models.patch_qwen import patch_qwen_attention
from neurohybrid.training.datasets import build_tiny_lm_dataset, build_wikitext_lm_dataset
from neurohybrid.utils.config import (
    load_checkpoint_config,
    load_config,
    load_trainable_state,
    merge_dicts,
    resolve_device,
    resolve_model_source,
    resolve_torch_dtype,
)
from neurohybrid.utils.metrics import collect_neurohybrid_stats
from neurohybrid.utils.result_table import write_default_reports


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = REPO_ROOT / "results" / "eval_ppl.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--result-path", default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    result_path = Path(args.result_path) if args.result_path else RESULT_PATH
    eval_config = load_config(
        args.config,
        defaults={
            "batch_size": 1,
            "max_seq_length": 512,
            "eval_steps": 20,
            "dataset_mode": "tiny_text",
            "dataset_path": str(REPO_ROOT / "data" / "wikitext-2-raw-v1"),
            "eval_split": "validation",
            "max_eval_samples": None,
        },
    )
    checkpoint_config = load_checkpoint_config(eval_config.get("checkpoint_path"))
    config = merge_dicts(checkpoint_config, eval_config)
    if args.max_seq_length is not None:
        config["max_seq_length"] = args.max_seq_length
    if args.eval_steps is not None:
        config["eval_steps"] = args.eval_steps
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

    device = resolve_device(args.device)
    dtype = resolve_torch_dtype(config.get("dtype"), device)
    model_source, local_files_only = resolve_model_source(config["model_name"])

    tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        local_files_only=local_files_only,
        dtype=dtype,
    )
    model.config.use_cache = False

    use_neurohybrid = bool(config.get("use_neurohybrid", False))
    checkpoint_info = {"loaded": False, "checkpoint_dir": None, "trainable_params_path": None}
    if use_neurohybrid:
        model = patch_qwen_attention(
            model,
            replace_last_n_layers=config.get("replace_last_n_layers", 4),
            window_size=config.get("window_size", 256),
            use_event_gate=config.get("use_event_gate", False),
            gate_beta=config.get("gate_beta", 0.5),
            gate_temperature=config.get("gate_temperature", 1.0),
            use_dendritic_fusion=config.get("use_dendritic_fusion", False),
            fusion_scale=config.get("fusion_scale", 0.5),
        )
    if config.get("checkpoint_path"):
        checkpoint_info = load_trainable_state(model, config.get("checkpoint_path"))

    model.to(device)
    model.eval()

    dataset_mode = config.get("dataset_mode")
    if dataset_mode == "tiny_text":
        requested_samples = None
        if config.get("eval_steps") is not None:
            requested_samples = int(config["eval_steps"]) * int(config["batch_size"])
        dataset = build_tiny_lm_dataset(
            tokenizer,
            max_seq_length=int(config["max_seq_length"]),
            num_samples=requested_samples or 256,
        )
    elif dataset_mode == "wikitext2":
        dataset = build_wikitext_lm_dataset(
            tokenizer,
            dataset_path=config["dataset_path"],
            split=config.get("eval_split", "validation"),
            max_seq_length=int(config["max_seq_length"]),
            num_samples=config.get("max_eval_samples"),
        )
    else:
        raise ValueError(f"Unsupported dataset_mode={dataset_mode}")
    dataloader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False)

    total_loss = 0.0
    num_batches = 0
    latest_stats = {}
    max_eval_steps = config.get("eval_steps")
    with torch.no_grad():
        for batch in dataloader:
            if max_eval_steps is not None and num_batches >= int(max_eval_steps):
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            total_loss += float(outputs.loss.item())
            num_batches += 1
            if use_neurohybrid:
                latest_stats = collect_neurohybrid_stats(model)

    avg_loss = total_loss / max(num_batches, 1)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")

    result = {
        "model_variant": "neurohybrid" if use_neurohybrid else "baseline",
        "resolved_model_source": model_source,
        "avg_loss": avg_loss,
        "ppl": ppl,
        "num_batches": num_batches,
        "max_seq_length": int(config["max_seq_length"]),
        "batch_size": int(config["batch_size"]),
        "dataset_mode": dataset_mode,
        "eval_split": config.get("eval_split"),
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "write_ratio_mean_avg": latest_stats.get("write_ratio_mean_avg"),
        "dendritic_k_mean_avg": latest_stats.get("dendritic_k_mean_avg"),
        "checkpoint_info": checkpoint_info,
    }
    _save_json(result_path, result)
    write_default_reports(REPO_ROOT)
    print(f"[eval_ppl] saved result to {result_path}")


if __name__ == "__main__":
    main()
