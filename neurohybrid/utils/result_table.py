import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def _build_day4_table(day4_results: dict[str, Any] | None) -> list[str]:
    lines = ["## Day 4 Ablation", ""]
    if not day4_results:
        lines.append("No `results/day4_ablation.json` found.")
        lines.append("")
        return lines

    lines.extend(
        [
            "| Variant | Layers | Seq Len | Loss | Write Ratio | Dendritic k | Peak Memory (GB) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in day4_results.get("results", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("name", "-")),
                    str(item.get("replace_last_n_layers", "-")),
                    str(item.get("seq_len", "-")),
                    _fmt(item.get("loss")),
                    _fmt(item.get("write_ratio_mean")),
                    _fmt(item.get("dendritic_k_mean_avg")),
                    _fmt(item.get("peak_reserved_gb")),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _build_train_table(train_results: dict[str, Any] | None) -> list[str]:
    lines = ["## Short Training", ""]
    if not train_results:
        lines.append("No `results/train_short.json` found.")
        lines.append("")
        return lines

    lines.extend(
        [
            "| Variant | Steps | Last Loss | Write Ratio | Dendritic k | Peak Memory (GB) | Checkpoint Format |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            "| "
            + " | ".join(
                [
                    str(train_results.get("model_variant", "-")),
                    str(train_results.get("completed_steps", "-")),
                    _fmt(train_results.get("last_loss")),
                    _fmt(train_results.get("write_ratio_mean_avg")),
                    _fmt(train_results.get("dendritic_k_mean_avg")),
                    _fmt(train_results.get("peak_allocated_gb")),
                    str(train_results.get("checkpoint_format", "-")),
                ]
            )
            + " |",
        ]
    )
    lines.append("")
    return lines


def _build_eval_table(eval_results: dict[str, Any] | None) -> list[str]:
    lines = ["## Eval PPL", ""]
    if not eval_results:
        lines.append("No `results/eval_ppl.json` found.")
        lines.append("")
        return lines

    lines.extend(
        [
            "| Variant | Avg Loss | PPL | Num Batches | Max Seq Len |",
            "| --- | ---: | ---: | ---: | ---: |",
            "| "
            + " | ".join(
                [
                    str(eval_results.get("model_variant", "-")),
                    _fmt(eval_results.get("avg_loss")),
                    _fmt(eval_results.get("ppl")),
                    str(eval_results.get("num_batches", "-")),
                    str(eval_results.get("max_seq_length", "-")),
                ]
            )
            + " |",
        ]
    )
    lines.append("")
    return lines


def _build_profile_table(profile_results: dict[str, Any] | None) -> list[str]:
    lines = ["## Decode Profile", ""]
    if not profile_results:
        lines.append("No `results/profile_decode.json` found.")
        lines.append("")
        return lines

    lines.extend(
        [
            "| Variant | Prompt Len | Generated | Elapsed (s) | Tokens/s | Peak Memory (GB) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            "| "
            + " | ".join(
                [
                    str(profile_results.get("model_variant", "-")),
                    str(profile_results.get("prompt_len", "-")),
                    str(profile_results.get("generated_tokens", "-")),
                    _fmt(profile_results.get("elapsed_sec")),
                    _fmt(profile_results.get("tokens_per_second")),
                    _fmt(profile_results.get("peak_reserved_gb")),
                ]
            )
            + " |",
        ]
    )
    lines.append("")
    return lines


def _build_needle_table(needle_results: dict[str, Any] | None) -> list[str]:
    lines = ["## Needle-in-a-Haystack", ""]
    if not needle_results:
        lines.append("No `results/eval_needle.json` found.")
        lines.append("")
        return lines

    lines.extend(
        [
            "| Context Len | Success | Prediction | Target | Peak Memory (GB) |",
            "| ---: | --- | --- | --- | ---: |",
        ]
    )
    for item in needle_results.get("results", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("context_len", "-")),
                    _fmt(item.get("success")),
                    str(item.get("prediction", "-")).replace("\n", " "),
                    str(item.get("target", "-")),
                    _fmt(item.get("peak_reserved_gb")),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def generate_results_markdown(repo_root: str | Path | None = None) -> str:
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    results_dir = root / "results"

    day4_results = _load_json(results_dir / "day4_ablation.json")
    train_results = _load_json(results_dir / "train_short.json")
    eval_results = _load_json(results_dir / "eval_ppl.json")
    profile_results = _load_json(results_dir / "profile_decode.json")
    needle_results = _load_json(results_dir / "eval_needle.json")

    lines = [
        "# NeuroHybrid Experiments",
        "",
        "Auto-generated from JSON artifacts in `results/`.",
        "",
    ]
    lines.extend(_build_day4_table(day4_results))
    lines.extend(_build_train_table(train_results))
    lines.extend(_build_eval_table(eval_results))
    lines.extend(_build_profile_table(profile_results))
    lines.extend(_build_needle_table(needle_results))
    return "\n".join(lines).strip() + "\n"


def write_default_reports(repo_root: str | Path | None = None) -> list[Path]:
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    markdown = generate_results_markdown(root)
    output_paths = [
        root / "docs" / "experiments.md",
        root / "results" / "summary.md",
    ]
    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    markdown = generate_results_markdown(args.repo_root)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
        print(f"[result_table] wrote {output_path}")
        return

    output_paths = write_default_reports(args.repo_root)
    for output_path in output_paths:
        print(f"[result_table] wrote {output_path}")


if __name__ == "__main__":
    main()
