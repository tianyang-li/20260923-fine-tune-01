# LLM RL fine-tuning (GRPO / REINFORCE + LoRA), single local GPU

Standalone port of the CS 285 (Spring 2026) HW4 LLM-RL setup. The original ran on
Modal H100s; this repo runs locally on one consumer GPU and has no Modal, Gradescope,
or W&B-account dependency.

What it does: samples `group_size` completions per prompt from a LoRA-wrapped causal
LM, scores them with a programmatic reward, computes group-relative advantages, and
updates only the LoRA adapter with either

- **REINFORCE** — single-pass, sequence-level `-A · mean_t log π(a_t)` loss, or
- **GRPO** — PPO-style clipped token ratio vs. the rollout policy, `ppo_epochs` passes,

plus a sampled KL penalty (k3 estimator) against the frozen base model (the same
weights with the adapter disabled, so no second model copy is kept in memory).

Tasks:

| `--task` | Prompt | Reward |
|---|---|---|
| `format_copy` | "Copy this integer exactly: N" | +0.2 has `<answer>` tag, +0.1 strict XML only, +1.0 exact number. Fast sanity check. |
| `math_hard` | MATH level-5 problems with numeric answers (`the-jb/hendrycks-math`, downloaded on first use) | +0.1 contains `\boxed{`, +1.0 boxed answer exact, +0.1 exact via fallback last-number parse |

## Layout

```
src/llm_rl/
  train.py            # training loop + CLI (llm-rl-train)
  eval.py             # evaluate a saved adapter (llm-rl-eval)
  config.py           # TrainConfig defaults
  models/load.py      # base model + LoRA (peft) loading, chat tokenization
  models/logprobs.py  # per-token logprobs, completion mask, KL estimator
  rl/reinforce.py     # REINFORCE update
  rl/grpo.py          # GRPO update
  rollout/            # HF generate() sampler, rollout batch / minibatching
  tasks/              # format_copy, math_hard (+ answer parsing in utils/)
```

## Setup

Requires Python 3.13, an NVIDIA GPU with bf16 support (Ampere or newer), and
[uv](https://docs.astral.sh/uv/).

**Option A: install into an existing uv venv that already has CUDA torch** (how this
machine is set up: the shared venv at `../.venv` has `torch 2.14.0+cu130`):

```bash
source ../.venv/bin/activate
uv pip install -e ".[wandb]"      # keeps the existing torch since it satisfies torch>=2.7
```

`llm-rl-train` and `llm-rl-eval` are then on `PATH` while the venv is active. Don't use
`uv sync --active` for this: it makes the venv exactly match `uv.lock` and removes
unrelated packages (Jupyter, etc.) from it.

**Option B: a project-local venv**:

```bash
uv sync --extra wandb             # creates ./.venv from uv.lock
uv run llm-rl-train ...           # (deactivate any other venv first, or uv warns and ignores it)
```

Model weights and datasets download from the Hugging Face Hub on first use into
`~/.cache/huggingface` (set `HF_HOME` to put them elsewhere). The default models are
not gated, so no HF login is needed.

### Logging: no W&B account required

W&B is **off by default**. Every run always writes, under `--output_dir`:

- `config.json` — full run config
- `metrics.jsonl` — one line per logged step (rewards, KL, loss, GPU memory, eval scores)
- `checkpoints/step_XXXXXX/adapter/` — LoRA adapter + tokenizer (plus `optimizer.pt`)

Optional: `--wandb_enabled` logs to W&B too (needs `wandb login`), or run with
`WANDB_MODE=offline --wandb_enabled` to get W&B-format logs locally without an account.

## Running on this machine (RTX 2000 Ada, 16 GB)

The hyperparameters below are the original HW4 ones, adjusted to fit in 16 GB of VRAM.
The effective batch per optimizer step is unchanged; only the per-forward-pass
minibatch is smaller:

- `--minibatch_size 4 --grad_accum_steps 16` instead of `8 × 8` for math_hard.
- `--logprob_batch_size 4` (default) scores the rollout's old/ref logprobs 4 sequences
  at a time. Scoring all 64 at once needs ~17 GB for the logits alone at 1k tokens.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces fragmentation OOMs.

Run from the repo root with the venv active.

### 1. Sanity check: format_copy + GRPO (~8 s/step, ~8.4 GB peak)

```bash
llm-rl-train \
  --task format_copy --algo grpo \
  --output_dir runs/format_copy_grpo \
  --steps 51 --batch_size 8 --group_size 6 \
  --min_new_tokens 1 --max_new_tokens 24 \
  --lr 3e-5 --ppo_epochs 2 --minibatch_size 8 --grad_accum_steps 6 \
  --clip_eps 0.2 --kl_coef 0.05 --max_grad_norm 0.5 --warmup_steps 10 \
  --eval_interval 50 --save_interval 50
```

About 7 minutes. Watch `rollout/mean_total_reward...` in `metrics.jsonl` rise toward
~1.3 (the maximum reward).

### 2. format_copy + REINFORCE

```bash
llm-rl-train \
  --task format_copy --algo reinforce \
  --output_dir runs/format_copy_reinforce \
  --steps 51 --batch_size 8 --group_size 6 \
  --min_new_tokens 1 --max_new_tokens 24 \
  --lr 3e-5 --minibatch_size 8 --grad_accum_steps 6 \
  --kl_coef 0.05 --max_grad_norm 0.5 --warmup_steps 10 \
  --eval_interval 50 --save_interval 50
```

### 3. math_hard + GRPO (~150–200 s/step, ~9.8 GB peak)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True llm-rl-train \
  --task math_hard --algo grpo \
  --output_dir runs/math_hard_grpo \
  --steps 501 --batch_size 8 --group_size 8 \
  --min_new_tokens 8 --max_new_tokens 512 --max_prompt_tokens 512 \
  --temperature 0.8 --top_p 0.95 \
  --lr 3e-5 --ppo_epochs 2 --minibatch_size 4 --grad_accum_steps 16 \
  --clip_eps 0.2 --max_grad_norm 0.5 --kl_coef 0.05 \
  --cuda_empty_cache_interval 50 \
  --eval_interval 100 --save_interval 100 --math_hard_eval_n 128
```

Most of each step is `generate()` for 64 completions of up to 512 tokens, so the full
501 steps take roughly **1 day** on this card. REINFORCE (single pass) is somewhat
faster per step. To iterate faster, try `--group_size 4`, `--max_new_tokens 256`, or a
0.5B model.

For REINFORCE, use `--algo reinforce`, drop `--ppo_epochs`/`--clip_eps`, and use
`--steps 201`.

Long runs: start them under `nohup ... > train.log 2>&1 &` or in `tmux` so they survive
the terminal closing. Each eval over `N` math problems generates up to 512 tokens per
problem, so `--math_hard_eval_n 128` (instead of 512) keeps periodic evals to a few
minutes.

### Evaluate a saved adapter

```bash
llm-rl-eval --task math_hard \
  --adapter_path runs/math_hard_grpo/checkpoints/step_000501/adapter \
  --math_hard_eval_n 512 --eval_batch_size 32
```

### Useful flags

| Flag | Default | Notes |
|---|---|---|
| `--model_name` | `Qwen/Qwen2.5-Math-1.5B-Instruct` | Any HF causal LM with a chat template |
| `--lora_r` / `--lora_alpha` | 16 / 32 | Adapter size |
| `--lora_target_modules` | all attn + MLP projections | Comma-separated Linear-layer suffixes |
| `--minibatch_size` | 8 | Sequences per training forward/backward. The main VRAM knob. |
| `--logprob_batch_size` | 4 | Sequences per forward when scoring rollouts |
| `--grad_checkpointing` | on | `--no-grad_checkpointing` is faster but uses much more memory |
| `--rollout_on_cpu` | on | Keeps rollout tensors on CPU between sampling and update |
| `--normalize_advantages` | off | Extra batch-level z-score after group normalization |

If you hit CUDA OOM, lower these in order: `--minibatch_size` (raise
`--grad_accum_steps` to compensate), `--max_new_tokens`, `--group_size`/`--batch_size`.

## Which models can this machine fine-tune?

Hardware: NVIDIA RTX 2000 Ada (16 GB GDDR6, ~224 GB/s, bf16 tensor cores), 20 CPU
threads, 62 GB RAM.

The limiting factor is VRAM during the training forward/backward. That is the bf16 base
weights (~2 GB per 1B params), plus activations with gradient checkpointing, plus the
full-vocabulary logits for every token in the minibatch (fp32 for the loss). Vocabulary
size matters a lot: Qwen's is 152k, Llama 3's is 128k. The adapter and its AdamW state
are tiny (tens of MB). Speed is limited by `generate()` on a low-bandwidth card, so
rollout time grows roughly with parameter count × completion length.

| Model (HF id) | Params | bf16 weights | Fit on 16 GB with this code | Notes |
|---|---|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | ~1 GB | Easily | Fastest iteration, good for debugging reward/algo changes |
| `Qwen/Qwen3-0.6B` | 0.6B | ~1.2 GB | Easily | Hybrid thinking model; `<think>` blocks are stripped before answer parsing |
| `Qwen/Qwen2.5-Math-1.5B-Instruct` | 1.5B | ~3 GB | **Yes, the default. Tested (see above).** | Strong math prior, the HW4 reference model |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | ~3 GB | Yes | General-purpose counterpart |
| `Qwen/Qwen3-1.7B` | 1.7B | ~3.4 GB | Yes | Thinking outputs are long, so raise `--max_new_tokens` and expect slower rollouts |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.7B | ~3.4 GB | Yes | Small 49k vocab, so logits are cheap and larger minibatches fit |
| `meta-llama/Llama-3.2-1B-Instruct` | 1.2B | ~2.5 GB | Yes | Gated: accept the license and `hf auth login` first |
| `Qwen/Qwen2.5-3B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct`, `Qwen/Qwen3-4B` | 3–4B | 6–8 GB | Tight | Use `--minibatch_size 1-2`, `--logprob_batch_size 2`, and `--max_new_tokens ≤ 256` for math. Several times slower per step. |
| 7B+ (Qwen2.5-7B, Llama-3.1-8B, …) | 7–8B | 15–16 GB | **No** | bf16 weights alone fill the card. Would need 4-bit QLoRA (bitsandbytes) plus a smaller rollout, which this code doesn't do. |

Only the default Qwen2.5-Math-1.5B-Instruct has been run end to end here. The other rows
are estimates from parameter count and vocabulary size, so do a short smoke run
(`--steps 2 --eval_interval 0 --format_copy_eval_n 16`) and check
`train/gpu_peak_memory_allocated_gigabytes_since_step_start` in `metrics.jsonl` before a
long run.
