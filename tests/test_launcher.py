import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "train" / "run_qwen3_4b_gcpo.sh"


def _runtime_tree(tmp_path):
    model_root = tmp_path / "model"
    (model_root / "qwen3_4B").mkdir(parents=True)
    (model_root / "all-MiniLM-L6-v2").mkdir()
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "train.parquet").touch()
    (data_root / "test.parquet").touch()
    verl_root = tmp_path / "verl"
    trainer = verl_root / "verl" / "trainer" / "ppo"
    trainer.mkdir(parents=True)
    (trainer / "ray_trainer.py").write_text(
        "from gcpo import estimator as _gcpo_estimator\n"
    )
    return model_root, data_root, verl_root


def _fake_python(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python3"
    python.write_text(
        """#!/usr/bin/env bash
printf '%s\n' "$@" > "$ARGUMENT_LOG"
printf '%s\n' "$PWD" > "$WORKDIR_LOG"
printf '%s\n' "${HF_HUB_OFFLINE:-}" "${TRANSFORMERS_OFFLINE:-}" "${HF_DATASETS_OFFLINE:-}" "${CUDA_VISIBLE_DEVICES:-}" > "$ENV_LOG"
"""
    )
    python.chmod(0o755)
    return bin_dir


def test_launcher_uses_paper_recipe_local_models_and_binary_reward(tmp_path):
    model_root, data_root, verl_root = _runtime_tree(tmp_path)
    bin_dir = _fake_python(tmp_path)
    argument_log = tmp_path / "arguments.log"
    workdir_log = tmp_path / "workdir.log"
    env_log = tmp_path / "env.log"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "MODEL_ROOT": str(model_root),
            "TRAIN_FILE": str(data_root / "train.parquet"),
            "VAL_FILE": str(data_root / "test.parquet"),
            "VERL_ROOT": str(verl_root),
            "ARGUMENT_LOG": str(argument_log),
            "WORKDIR_LOG": str(workdir_log),
            "ENV_LOG": str(env_log),
        }
    )

    result = subprocess.run(
        ["bash", str(LAUNCHER), "trainer.total_training_steps=2"],
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    arguments = argument_log.read_text().splitlines()
    expected = {
        "algorithm.adv_estimator=gcpo",
        "+algorithm.gcpo_eta=1.0",
        "+algorithm.gcpo_exact_threshold=8",
        "algorithm.norm_adv_by_std_in_grpo=True",
        "algorithm.use_kl_in_reward=False",
        "data.max_prompt_length=1024",
        "data.max_response_length=8192",
        "actor_rollout_ref.actor.loss_agg_mode=token-mean",
        "actor_rollout_ref.rollout.max_num_batched_tokens=9216",
        "actor_rollout_ref.rollout.n=8",
        "actor_rollout_ref.rollout.temperature=1.0",
        "trainer.n_gpus_per_node=8",
        "trainer.total_training_steps=2",
    }
    assert expected.issubset(arguments)
    assert f"actor_rollout_ref.model.path={model_root / 'qwen3_4B'}" in arguments
    assert (
        f"+algorithm.sentence_transformer_model="
        f"{model_root / 'all-MiniLM-L6-v2'}"
    ) in arguments
    assert f"data.train_files={data_root / 'train.parquet'}" in arguments
    assert f"data.val_files={data_root / 'test.parquet'}" in arguments
    assert f"custom_reward_function.path={ROOT / 'src' / 'gcpo' / 'reward.py'}" in arguments
    assert workdir_log.read_text().strip() == str(verl_root)
    assert env_log.read_text().splitlines() == [
        "1",
        "1",
        "1",
        "0,1,2,3,4,5,6,7",
    ]


def test_launcher_has_no_eos_or_terminal_answer_gate():
    text = LAUNCHER.read_text()

    assert "require_eos" not in text
    assert "require_terminal_answer" not in text


def test_launcher_fails_before_python_when_inputs_are_missing(tmp_path):
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        text=True,
        capture_output=True,
        env={**os.environ, "VERL_ROOT": str(tmp_path / "missing")},
    )

    assert result.returncode != 0
    assert "Run scripts/setup_verl.sh first" in result.stderr
