# Copyright 2026 GCPO Authors
# Licensed under the Apache License, Version 2.0 (the "License").

"""Prepare DAPO-Math-17K and AIME-2024 for VERL training."""

import argparse
import json
from pathlib import Path
from typing import Any

import datasets
import pyarrow.parquet as pq


TRAIN_DATASET = "open-r1/DAPO-Math-17k-Processed"
TRAIN_CONFIG = "all"
VALIDATION_DATASET = "BytedTsinghua-SIA/AIME-2024"
VALIDATION_CONFIG = "default"
EXPECTED_TRAIN_SIZE = 17_398
EXPECTED_VALIDATION_SIZE = 30


def _normalize_messages(example: dict[str, Any], row_index: int):
    messages = example.get("source_prompt")
    if not isinstance(messages, list) or not messages:
        messages = example.get("prompt")
    if not isinstance(messages, list) or not messages:
        raise ValueError(
            f"row {row_index}: source_prompt or prompt must be a non-empty "
            "list of chat messages"
        )

    normalized = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(
                f"row {row_index}: source_prompt[{message_index}] must be a mapping"
            )
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role:
            raise ValueError(
                f"row {row_index}: source_prompt[{message_index}].role must be "
                "a non-empty string"
            )
        if not isinstance(content, str) or not content:
            raise ValueError(
                f"row {row_index}: source_prompt[{message_index}].content must be "
                "a non-empty string"
            )
        normalized.append({"role": role, "content": content})
    return normalized


def _normalize_reward_model(example: dict[str, Any], row_index: int):
    reward_model = example.get("reward_model")
    if not isinstance(reward_model, dict):
        raise ValueError(f"row {row_index}: reward_model must be a mapping")
    ground_truth = reward_model.get("ground_truth")
    if ground_truth is None or not str(ground_truth).strip():
        raise ValueError(
            f"row {row_index}: reward_model.ground_truth must be non-empty"
        )
    return {
        "style": str(reward_model.get("style", "rule-lighteval/MATH_v2")),
        "ground_truth": str(ground_truth),
    }


def _normalize_data_source(example: dict[str, Any], split: str, row_index: int):
    data_source = str(example.get("data_source") or "")
    if split == "train":
        if data_source != "math_dapo":
            raise ValueError(
                f"row {row_index}: train data_source must be 'math_dapo', "
                f"got {data_source!r}"
            )
    elif split == "validation":
        if data_source != "math_dapo" and not data_source.startswith("aime"):
            raise ValueError(
                f"row {row_index}: validation data_source must be 'math_dapo' "
                f"or start with 'aime', got {data_source!r}"
            )
    else:
        raise ValueError(f"Unsupported split: {split}")
    return data_source


def normalize_dataset(
    source: datasets.Dataset,
    split: str,
    expected_size: int | None = None,
) -> datasets.Dataset:
    """Validate, deduplicate, and convert a source split to VERL schema."""
    normalized_rows = []
    seen_answers: dict[str, str] = {}

    for row_index, example in enumerate(source):
        messages = _normalize_messages(example, row_index)
        reward_model = _normalize_reward_model(example, row_index)
        data_source = _normalize_data_source(example, split, row_index)
        prompt_key = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        ground_truth = reward_model["ground_truth"]

        previous_answer = seen_answers.get(prompt_key)
        if previous_answer is not None:
            if previous_answer != ground_truth:
                raise ValueError(
                    f"row {row_index}: duplicate prompt has conflicting ground truths "
                    f"({previous_answer!r} != {ground_truth!r})"
                )
            continue
        seen_answers[prompt_key] = ground_truth

        extra_info = example.get("extra_info")
        if extra_info is None:
            extra_info = {}
        if not isinstance(extra_info, dict):
            raise ValueError(f"row {row_index}: extra_info must be a mapping")
        normalized_extra_info = dict(extra_info)
        normalized_extra_info["index"] = str(
            normalized_extra_info.get("index", row_index)
        )
        normalized_extra_info["split"] = split

        normalized_rows.append(
            {
                "data_source": data_source,
                "prompt": messages,
                "ability": str(example.get("ability") or "MATH"),
                "reward_model": reward_model,
                "extra_info": normalized_extra_info,
            }
        )

    if expected_size is not None and len(normalized_rows) != expected_size:
        raise ValueError(
            f"{split} dataset contains {len(normalized_rows)} unique prompts; "
            f"expected {expected_size}"
        )
    return datasets.Dataset.from_list(normalized_rows)


def write_dataset(dataset: datasets.Dataset, output_path: str | Path) -> Path:
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(output_path))
    return output_path


def _local_parquet_files(local_source: str | Path) -> list[Path]:
    path = Path(local_source).expanduser()
    if path.is_file():
        if path.suffix != ".parquet":
            raise ValueError(f"Local dataset file must be parquet: {path}")
        return [path]
    if path.is_dir():
        files = sorted(path.rglob("*.parquet"), key=lambda item: item.as_posix())
        if files:
            return files
        raise FileNotFoundError(f"No parquet shards found under: {path}")
    raise FileNotFoundError(f"Local dataset source does not exist: {path}")


def load_source(
    dataset_name: str,
    config_name: str,
    local_source: str | Path | None = None,
) -> datasets.Dataset:
    """Load a Hugging Face dataset or deterministic local parquet shard set."""
    if local_source is None:
        return datasets.load_dataset(dataset_name, config_name, split="train")

    shards = _local_parquet_files(local_source)
    loaded = [datasets.Dataset(pq.read_table(shard)) for shard in shards]
    if len(loaded) == 1:
        return loaded[0]
    return datasets.concatenate_datasets(loaded)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare DAPO-Math-17k-Processed and AIME-2024 for VERL."
    )
    parser.add_argument("--output-dir", default="data/dapo_math_17k")
    parser.add_argument("--train-source", default=TRAIN_DATASET)
    parser.add_argument("--train-config", default=TRAIN_CONFIG)
    parser.add_argument("--validation-source", default=VALIDATION_DATASET)
    parser.add_argument("--validation-config", default=VALIDATION_CONFIG)
    parser.add_argument(
        "--train-local", help="Local train parquet file or directory of shards."
    )
    parser.add_argument(
        "--validation-local",
        help="Local validation parquet file or directory of shards.",
    )
    parser.add_argument(
        "--expected-train-size", type=int, default=EXPECTED_TRAIN_SIZE
    )
    parser.add_argument(
        "--expected-validation-size", type=int, default=EXPECTED_VALIDATION_SIZE
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    train_source = load_source(
        args.train_source, args.train_config, args.train_local
    )
    validation_source = load_source(
        args.validation_source, args.validation_config, args.validation_local
    )
    train = normalize_dataset(
        train_source, split="train", expected_size=args.expected_train_size
    )
    validation = normalize_dataset(
        validation_source,
        split="validation",
        expected_size=args.expected_validation_size,
    )

    output_dir = Path(args.output_dir).expanduser()
    train_path = write_dataset(train, output_dir / "train.parquet")
    validation_path = write_dataset(validation, output_dir / "test.parquet")
    print(f"Wrote {len(train)} train rows to {train_path}", flush=True)
    print(
        f"Wrote {len(validation)} validation rows to {validation_path}", flush=True
    )


if __name__ == "__main__":
    main()
