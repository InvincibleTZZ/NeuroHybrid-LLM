import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neurohybrid.models.patch_qwen import patch_qwen_attention
from neurohybrid.utils.config import (
    load_checkpoint_config,
    load_config,
    load_trainable_state,
    merge_dicts,
    resolve_device,
    resolve_model_source,
    resolve_torch_dtype,
)
from neurohybrid.utils.result_table import write_default_reports


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = REPO_ROOT / "results" / "profile_decode.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--result-path", default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    result_path = Path(args.result_path) if args.result_path else RESULT_PATH
    eval_config = load_config(args.config, defaults={"max_seq_length": 512})
    checkpoint_config = load_checkpoint_config(eval_config.get("checkpoint_path"))
    config = merge_dicts(checkpoint_config, eval_config)
    if args.max_seq_length is not None:
        config["max_seq_length"] = args.max_seq_length

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

    use_neurohybrid = bool(config.get("use_neurohybrid", False))
    checkpoint_info = {"loaded": False, "checkpoint_dir": None, "trainable_params_path": None}
    if use_neurohybrid:
        model.config.use_cache = False
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

    prompt = (
        "NeuroHybrid attention combines a local branch and a memory branch. "
        "Summarize why this can be useful for a small language model."
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=config["max_seq_length"])
    inputs = {key: value.to(device) for key, value in inputs.items()}

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if use_neurohybrid:
        generation_kwargs["use_cache"] = False

    start_time = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)
    elapsed_sec = time.perf_counter() - start_time

    prompt_len = int(inputs["input_ids"].shape[1])
    generated_tokens = int(output_ids.shape[1] - prompt_len)
    prediction = tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True).strip()
    peak_allocated_gb = (
        torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    )
    peak_reserved_gb = (
        torch.cuda.max_memory_reserved(device) / 1024**3 if device.type == "cuda" else 0.0
    )

    result = {
        "model_variant": "neurohybrid" if use_neurohybrid else "baseline",
        "resolved_model_source": model_source,
        "prompt_len": prompt_len,
        "generated_tokens": generated_tokens,
        "elapsed_sec": elapsed_sec,
        "tokens_per_second": generated_tokens / max(elapsed_sec, 1e-6),
        "peak_allocated_gb": peak_allocated_gb,
        "peak_reserved_gb": peak_reserved_gb,
        "prediction": prediction,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "checkpoint_info": checkpoint_info,
    }
    _save_json(result_path, result)
    write_default_reports(REPO_ROOT)
    print(f"[profile_decode] saved result to {result_path}")


if __name__ == "__main__":
    main()
