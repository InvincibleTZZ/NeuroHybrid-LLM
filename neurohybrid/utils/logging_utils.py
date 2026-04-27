import logging
import sys
from typing import Any


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def log_metrics(step: int, metrics: dict[str, Any], logger: logging.Logger | None = None) -> str:
    ordered_keys = [
        "loss",
        "avg_loss",
        "write_ratio",
        "dendritic_k",
        "memory_gb",
        "peak_allocated_gb",
        "learning_rate",
        "ppl",
    ]
    parts = []
    seen = set()

    for key in ordered_keys:
        if key not in metrics:
            continue
        seen.add(key)
        value = metrics[key]
        if key in {"memory_gb", "peak_allocated_gb"} and value is not None:
            parts.append(f"{key}={float(value):.2f}GB")
        else:
            parts.append(f"{key}={_format_value(value)}")

    for key, value in metrics.items():
        if key in seen:
            continue
        parts.append(f"{key}={_format_value(value)}")

    line = f"[step {step}] " + " ".join(parts)
    if logger is None:
        logger = setup_logger("neurohybrid")
    logger.info(line)
    return line
