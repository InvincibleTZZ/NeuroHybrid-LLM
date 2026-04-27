from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


TINY_TEXT_BANK = [
    "Paris is the capital of France. Berlin is the capital of Germany. Tokyo is the capital of Japan.",
    "A transformer language model predicts the next token from left to right and improves with better data and optimization.",
    "The quick brown fox jumps over the lazy dog while a short benchmark sentence keeps repeating for debugging.",
    "Python example: def add(x, y): return x + y. A clean training script should log loss and save lightweight checkpoints.",
    "A local window branch preserves nearby token interactions, while a linear memory branch offers an efficient long-range summary.",
    "Neuroscience-inspired computation often emphasizes sparse communication, event-driven updates, and nonlinear integration.",
    "A Wikipedia-style paragraph can describe science, history, and engineering in concise sentences that remain easy to tokenize offline.",
    "Long repeated text helps stress sequence length handling. Long repeated text helps stress sequence length handling.",
]


@dataclass
class TinyLMDataset(Dataset):
    samples: list[dict[str, torch.Tensor]]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]


def _load_split_from_disk(dataset_path: str | Path, split: str):
    from datasets import load_from_disk

    dataset_obj = load_from_disk(str(dataset_path))
    if hasattr(dataset_obj, "keys"):
        if split not in dataset_obj:
            raise ValueError(f"Split '{split}' not found in dataset at {dataset_path}")
        return dataset_obj[split]
    return dataset_obj


def _build_chunked_lm_samples(
    tokenizer,
    token_sequences: list[list[int]],
    max_seq_length: int,
    num_samples: int | None = None,
) -> list[dict[str, torch.Tensor]]:
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must provide eos_token_id for chunked LM datasets.")

    flat_token_ids = []
    for token_ids in token_sequences:
        if not token_ids:
            continue
        flat_token_ids.extend(token_ids)
        flat_token_ids.append(eos_token_id)

    total_chunks = len(flat_token_ids) // max_seq_length
    if num_samples is not None:
        total_chunks = min(total_chunks, num_samples)

    samples = []
    for chunk_idx in range(total_chunks):
        start = chunk_idx * max_seq_length
        end = start + max_seq_length
        input_ids = torch.tensor(flat_token_ids[start:end], dtype=torch.long)
        attention_mask = torch.ones(max_seq_length, dtype=torch.long)
        labels = input_ids.clone()
        samples.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )
    return samples


def _expand_text(tokenizer, base_text: str, max_seq_length: int) -> str:
    text = base_text.strip()
    while len(tokenizer(text, add_special_tokens=False)["input_ids"]) < max_seq_length:
        text = f"{text} {base_text.strip()}"
    return text


def build_tiny_lm_dataset(tokenizer, max_seq_length: int, num_samples: int = 256) -> Dataset:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    samples = []
    for sample_idx in range(num_samples):
        base_text = TINY_TEXT_BANK[sample_idx % len(TINY_TEXT_BANK)]
        text = _expand_text(tokenizer, base_text, max_seq_length)
        encoded = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        samples.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )

    return TinyLMDataset(samples)


def build_wikitext_lm_dataset(
    tokenizer,
    dataset_path: str | Path,
    split: str,
    max_seq_length: int,
    num_samples: int | None = None,
    text_column: str = "text",
) -> Dataset:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = _load_split_from_disk(dataset_path, split)
    texts = []
    for text in dataset[text_column]:
        if text is None:
            continue
        normalized = text.strip()
        if not normalized:
            continue
        texts.append(normalized)

    token_sequences = tokenizer(texts, add_special_tokens=False)["input_ids"]
    samples = _build_chunked_lm_samples(
        tokenizer=tokenizer,
        token_sequences=token_sequences,
        max_seq_length=max_seq_length,
        num_samples=num_samples,
    )
    return TinyLMDataset(samples)
