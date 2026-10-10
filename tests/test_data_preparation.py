import importlib.util
from pathlib import Path

import datasets
import pyarrow.parquet as pq
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "data"
    / "prepare_dapo_math_17k.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_dapo_math_17k", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _row(problem="What is 6 * 7?", answer="42", index="row-0", source="math_dapo"):
    prompt = [
        {
            "role": "user",
            "content": f"Solve the problem.\n\n{problem}\n\nAnswer on the last line.",
        }
    ]
    return {
        "prompt": problem,
        "solution": answer,
        "data_source": source,
        "source_prompt": prompt,
        "ability": "MATH",
        "reward_model": {
            "style": "rule-lighteval/MATH_v2",
            "ground_truth": answer,
        },
        "extra_info": {"index": index},
    }


def _aime_row(problem="What is 6 * 7?", answer="42", index=0):
    row = _row(problem, answer, index)
    row["prompt"] = row.pop("source_prompt")
    row.pop("solution")
    row["extra_info"] = {"index": index, "raw_problem": problem, "split": None}
    return row


def test_normalize_preserves_structured_prompt_reward_and_stable_index():
    source = datasets.Dataset.from_list([_row()])

    normalized = MODULE.normalize_dataset(source, split="train", expected_size=1)

    assert normalized.column_names == [
        "data_source",
        "prompt",
        "ability",
        "reward_model",
        "extra_info",
    ]
    assert normalized[0]["prompt"] == _row()["source_prompt"]
    assert normalized[0]["reward_model"] == {
        "style": "rule-lighteval/MATH_v2",
        "ground_truth": "42",
    }
    assert normalized[0]["extra_info"] == {"index": "row-0", "split": "train"}


def test_normalize_deduplicates_exact_prompts_in_source_order():
    first = _aime_row()
    source = datasets.Dataset.from_list(
        [first, first, _aime_row("What is 7 * 8?", "56", 1)]
    )

    normalized = MODULE.normalize_dataset(
        source, split="validation", expected_size=2
    )

    assert [row["reward_model"]["ground_truth"] for row in normalized] == [
        "42",
        "56",
    ]


def test_normalize_rejects_duplicate_prompt_with_conflicting_answers():
    source = datasets.Dataset.from_list(
        [_row(answer="42"), _row(answer="41", index="row-1")]
    )

    with pytest.raises(ValueError, match="conflicting ground truths"):
        MODULE.normalize_dataset(source, split="train")


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda row: row.pop("source_prompt"), "source_prompt"),
        (lambda row: row.update(data_source="openai/gsm8k"), "data_source"),
        (lambda row: row.update(reward_model={"style": "x"}), "ground_truth"),
    ],
)
def test_normalize_rejects_invalid_training_rows(mutation, error):
    row = _row()
    mutation(row)

    with pytest.raises(ValueError, match=error):
        MODULE.normalize_dataset(datasets.Dataset.from_list([row]), split="train")


def test_expected_size_is_checked_after_deduplication():
    row = _row()
    source = datasets.Dataset.from_list([row, row])

    with pytest.raises(ValueError, match="1 unique prompts; expected 2"):
        MODULE.normalize_dataset(source, split="train", expected_size=2)


def test_local_directory_loads_all_parquet_shards_in_sorted_order(tmp_path):
    source_dir = tmp_path / "source" / "data"
    source_dir.mkdir(parents=True)
    datasets.Dataset.from_list([_row("second", "2", "b")]).to_parquet(
        str(source_dir / "train-00001.parquet")
    )
    datasets.Dataset.from_list([_row("first", "1", "a")]).to_parquet(
        str(source_dir / "train-00000.parquet")
    )

    loaded = MODULE.load_source("unused", "unused", tmp_path / "source")

    assert [row["extra_info"]["index"] for row in loaded] == ["a", "b"]


def test_local_source_rejects_directory_without_parquet(tmp_path):
    with pytest.raises(FileNotFoundError, match="No parquet shards"):
        MODULE.load_source("unused", "unused", tmp_path)


def test_write_dataset_produces_verl_compatible_parquet(tmp_path):
    normalized = MODULE.normalize_dataset(
        datasets.Dataset.from_list([_row()]), split="train", expected_size=1
    )

    output = MODULE.write_dataset(normalized, tmp_path / "train.parquet")
    reloaded = pq.read_table(output).to_pylist()

    assert output == tmp_path / "train.parquet"
    assert reloaded[0]["data_source"] == "math_dapo"
    assert reloaded[0]["prompt"][0]["role"] == "user"


def test_main_converts_local_train_and_aime_sources(tmp_path):
    train_path = tmp_path / "source-train.parquet"
    validation_path = tmp_path / "source-val.parquet"
    output_dir = tmp_path / "output"
    datasets.Dataset.from_list([_row()]).to_parquet(str(train_path))
    aime = _aime_row()
    datasets.Dataset.from_list([aime, aime]).to_parquet(str(validation_path))

    MODULE.main(
        [
            "--train-source",
            "unused",
            "--train-local",
            str(train_path),
            "--validation-source",
            "unused",
            "--validation-local",
            str(validation_path),
            "--output-dir",
            str(output_dir),
            "--expected-train-size",
            "1",
            "--expected-validation-size",
            "1",
        ]
    )

    assert pq.read_table(output_dir / "train.parquet").num_rows == 1
    assert pq.read_table(output_dir / "test.parquet").num_rows == 1
