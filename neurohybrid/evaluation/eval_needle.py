import argparse
import json
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
RESULT_PATH = REPO_ROOT / "results" / "eval_needle.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--result-path", default=None)
    parser.add_argument("--context-lens", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    return parser.parse_args()


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _build_needle_inputs(tokenizer, context_len: int, target: str) -> dict[str, torch.Tensor]:
    prefix_ids = tokenizer(
        "Here is a long document. ",
        add_special_tokens=False,
    )["input_ids"]
    filler_ids = tokenizer(
        "This document discusses science, engineering, history, and many unrelated details. ",
        add_special_tokens=False,
    )["input_ids"]
    needle_ids = tokenizer(
        f"Important note: The secret code is {target}. Remember it exactly. ",
        add_special_tokens=False,
    )["input_ids"]
    question_ids = tokenizer(
        "What is the secret code? Answer:",
        add_special_tokens=False,
    )["input_ids"]

    body_budget = max(context_len - len(question_ids), len(prefix_ids) + len(needle_ids))
    body_ids = list(prefix_ids)
    insertion_point = max(body_budget // 2, len(prefix_ids))

    while len(body_ids) < insertion_point:
        body_ids.extend(filler_ids)
    body_ids.extend(needle_ids)
    while len(body_ids) < body_budget:
        body_ids.extend(filler_ids)
    body_ids = body_ids[:body_budget]

    if len(needle_ids) < body_budget:
        center = max(0, min(body_budget - len(needle_ids), body_budget // 2))
        body_ids[center:center + len(needle_ids)] = needle_ids

    input_ids = body_ids + question_ids
    input_ids = input_ids[:context_len]
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
    }


def main() -> None:
    args = parse_args()
    result_path = Path(args.result_path) if args.result_path else RESULT_PATH
    eval_config = load_config(args.config, defaults={"max_seq_length": 512})
    checkpoint_config = load_checkpoint_config(eval_config.get("checkpoint_path"))
    config = merge_dicts(checkpoint_config, eval_config)

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

    if args.context_lens:
        context_lens = [int(item) for item in args.context_lens.split(",") if item.strip()]
    else:
        context_lens = [512, 1024, 2048]
    target = "73915"
    results = []

    for context_len in context_lens:
        case = {
            "context_len": context_len,
            "success": False,
            "prediction": "",
            "target": target,
            "peak_allocated_gb": 0.0,
            "peak_reserved_gb": 0.0,
            "error": None,
        }
        try:
            inputs = _build_needle_inputs(tokenizer, context_len, target)
            inputs = {key: value.to(device) for key, value in inputs.items()}

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            generate_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": False,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if use_neurohybrid:
                generate_kwargs["use_cache"] = False

            with torch.inference_mode():
                output_ids = model.generate(**inputs, **generate_kwargs)

            prompt_len = int(inputs["input_ids"].shape[1])
            prediction = tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True).strip()
            case["prediction"] = prediction
            case["success"] = target in prediction
            if device.type == "cuda":
                case["peak_allocated_gb"] = torch.cuda.max_memory_allocated(device) / 1024**3
                case["peak_reserved_gb"] = torch.cuda.max_memory_reserved(device) / 1024**3
        except Exception as error:
            case["error"] = str(error)
        results.append(case)

    summary = {
        "model_variant": "neurohybrid" if use_neurohybrid else "baseline",
        "resolved_model_source": model_source,
        "results": results,
        "checkpoint_info": checkpoint_info,
    }
    _save_json(result_path, summary)
    write_default_reports(REPO_ROOT)
    print(f"[eval_needle] saved result to {result_path}")


if __name__ == "__main__":
    main()
