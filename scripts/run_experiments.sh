#!/usr/bin/env bash
# Full fine-tuning runs on a single 16 GB GPU (RTX 2000 Ada), run sequentially.
# Usage (from repo root, venv with llm-rl-train on PATH):
#   setsid nohup scripts/run_experiments.sh > runs/run_experiments.log 2>&1 &
# Each run writes runs/<name>/{config.json,metrics.jsonl,checkpoints/}, plus
# runs/<name>.exitcode when it finishes.
set -u
cd "$(dirname "$0")/.."
mkdir -p runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run() {
  local name="$1"; shift
  llm-rl-train --output_dir "runs/$name" "$@" > "runs/$name.log" 2>&1
  echo $? > "runs/$name.exitcode"
}

# 1. format_copy + GRPO (~10 min)
run format_copy_grpo \
  --task format_copy --algo grpo \
  --steps 51 --batch_size 8 --group_size 6 \
  --min_new_tokens 1 --max_new_tokens 24 \
  --lr 3e-5 --ppo_epochs 2 --minibatch_size 8 --grad_accum_steps 6 \
  --clip_eps 0.2 --kl_coef 0.05 --max_grad_norm 0.5 --warmup_steps 10 \
  --format_copy_eval_n 64 --eval_interval 50 --save_interval 50

# 2. math_hard + GRPO (~1 day)
run math_hard_grpo \
  --task math_hard --algo grpo \
  --steps 501 --batch_size 8 --group_size 8 \
  --min_new_tokens 8 --max_new_tokens 512 --max_prompt_tokens 512 \
  --temperature 0.8 --top_p 0.95 \
  --lr 3e-5 --ppo_epochs 2 --minibatch_size 4 --grad_accum_steps 16 \
  --clip_eps 0.2 --max_grad_norm 0.5 --kl_coef 0.05 \
  --cuda_empty_cache_interval 50 \
  --math_hard_eval_n 512 --eval_interval 100 --save_interval 100
