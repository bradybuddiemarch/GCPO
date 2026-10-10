#!/usr/bin/env bash

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
verl_root=${VERL_ROOT:-"$repo_dir/third_party/verl"}
train_file=${TRAIN_FILE:-"$repo_dir/data/dapo_math_17k/train.parquet"}
val_file=${VAL_FILE:-"$repo_dir/data/dapo_math_17k/test.parquet"}
model_root=${MODEL_ROOT:-"$repo_dir/model"}
model_path=${MODEL_PATH:-"$model_root/qwen3_4B"}
semantic_model=${SENTENCE_TRANSFORMER_MODEL:-"$model_root/all-MiniLM-L6-v2"}
semantic_device=${SENTENCE_TRANSFORMER_DEVICE:-cuda}
reward_function="$repo_dir/src/gcpo/reward.py"

if [[ ! -f "$verl_root/verl/trainer/ppo/ray_trainer.py" ]] || \
   ! grep -q "from gcpo import estimator as _gcpo_estimator" \
        "$verl_root/verl/trainer/ppo/ray_trainer.py"; then
    echo "Patched VERL checkout is missing. Run scripts/setup_verl.sh first." >&2
    exit 1
fi
if [[ ! -f "$train_file" || ! -f "$val_file" ]]; then
    echo "Prepared DAPO/AIME parquet files are missing." >&2
    echo "Run: python3 scripts/data/prepare_dapo_math_17k.py" >&2
    exit 1
fi
if [[ ! -d "$model_path" || ! -d "$semantic_model" ]]; then
    echo "Local model directories are missing under $model_root." >&2
    echo "See model/README.md for the required layout." >&2
    exit 1
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTHONPATH="$repo_dir/src:$verl_root${PYTHONPATH:+:$PYTHONPATH}"

cd "$verl_root"
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gcpo \
    +algorithm.use_sentence_transformer=True \
    +algorithm.sentence_transformer_model="$semantic_model" \
    +algorithm.sentence_transformer_device="$semantic_device" \
    +algorithm.sentence_transformer_batch_size=128 \
    +algorithm.gcpo_eta=1.0 \
    +algorithm.gcpo_exact_threshold=8 \
    +algorithm.gcpo_n_samples=500 \
    +algorithm.gcpo_epsilon=1.0e-6 \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    data.train_files="$train_file" \
    data.val_files="$val_file" \
    data.prompt_key=prompt \
    data.train_batch_size=256 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="$model_path" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_model_len=9216 \
    actor_rollout_ref.rollout.max_num_batched_tokens=9216 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    custom_reward_function.path="$reward_function" \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=gcpo \
    trainer.experiment_name=qwen3_4b_gcpo \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    "$@"
