# Lab notebook — porting CS 285 HW4 LLM-RL to a standalone local repo

Date: 2026-09-23. Author: Tianyang Li (with Claude Code).

Goal: take the LoRA LLM-RL fine-tuning setup from CS 285 Spring 2026 HW4 (designed for
Modal H100s, with student `TODO` stubs) and turn it into a standalone repo that runs by
itself on this machine. This notebook records every step so the result can be
reproduced from scratch without the original session.

---

## 0. Machine and starting state

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 2000 Ada Generation, 16 GB (16380 MiB), compute capability 8.9, bf16 capable |
| Driver / CUDA (driver) | 580.178.04 / CUDA 13.0 |
| CPU / RAM | 20 threads / 62 GB |
| OS | Ubuntu 22.04.5 LTS, kernel 6.8.0 |
| uv | 0.12.18 |
| Python | CPython 3.13.6 (uv-managed) |

Directory layout (all under `/media/lty/hdd-20241206/code/20260923-fine-tune/`):

```
20260923-fine-tune/
  .venv/                                 # shared uv venv (Python 3.13.6) with torch 2.14.0+cu130, Jupyter, black, …
  berkeleydeeprlcourse-spring2026/hw4/   # source: git@github.com:berkeleydeeprlcourse/homework_spring2026.git @ 59e40fc
  code-20260923-fine-tune/               # this repo (target); remote git@github.com:tianyang-li/20260923-fine-tune-01.git
```

Starting state of this repo: a `uv init --package` scaffold (`pyproject.toml` using the
`uv_build` backend, `requires-python >=3.13`, `src/code_20260923_fine_tune/__init__.py`
printing hello, empty `README.md`, Python `.gitignore`).

Source hw4 contents: package `hw4/` (train, eval, config, models, rl, rollout, tasks,
utils, gradescope_bundle), `scripts/modal_train.py`, and a pyproject pinned to Python
3.12, `torch<2.7`, `transformers<4.57`, `numpy<2.0`, with Modal as the only base dependency.

---

## 1. Port the package

```bash
SRC=/media/lty/hdd-20241206/code/20260923-fine-tune/berkeleydeeprlcourse-spring2026/hw4/hw4
cd /media/lty/hdd-20241206/code/20260923-fine-tune/code-20260923-fine-tune
mkdir -p src/llm_rl
cp -r $SRC/. src/llm_rl/
rm -rf src/llm_rl/gradescope_bundle.py src/llm_rl/__pycache__ src/llm_rl/*/__pycache__ src/code_20260923_fine_tune
grep -rl "hw4" src/llm_rl | xargs sed -i 's/\bhw4\./llm_rl./g; s/from hw4 /from llm_rl /g'
```

Decisions:
- Package renamed `hw4` → `llm_rl` under `src/`. The uv scaffold package was removed.
- **Dropped** `scripts/modal_train.py` (Modal remote runner) and `gradescope_bundle.py`
  (course submission). Nothing else imports them.
- `config.py`: `wandb_project` `"llm-rl-hw4"` → `"llm-rl"`, and **`wandb_enabled` default
  `True` → `False`** so runs need no W&B account. Metrics always go to
  `<output_dir>/metrics.jsonl` and `config.json` via the existing `WandBLogger` local writer.

## 2. Implement the student TODOs

The handout left these as `raise NotImplementedError("student TODO: …")`. Each
implementation follows the spec in the handout's comments. Final code:

### `src/llm_rl/models/logprobs.py`

```python
def compute_per_token_logprobs(model, input_ids, attention_mask, *, enable_grad=True):
    """Returns log p(x_t | x_<t) for t in [1, L-1]. input_ids/attention_mask are [B, L]; output is [B, L-1]."""
    # Uses fused cross-entropy (log_softmax + gather) on the existing logits to avoid
    # materializing a second dense [B, L-1, V] tensor.
    with torch.set_grad_enabled(enable_grad):
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = out.logits[:, :-1, :]
        targets = input_ids[:, 1:]
        B, Lm1, V = logits.shape
        nll = F.cross_entropy(
            logits.reshape(B * Lm1, V).float(),
            targets.reshape(B * Lm1),
            reduction="none",
        )
        return -nll.view(B, Lm1)


def build_completion_mask(input_ids, attention_mask, prompt_input_len, pad_token_id):
    """Mask over per-token positions [B, L-1], selecting completion tokens only."""
    # mask[:, t] scores token t+1; completion tokens start at index prompt_input_len.
    B, L = input_ids.shape
    positions = torch.arange(1, L, device=input_ids.device).unsqueeze(0)
    is_completion = positions >= int(prompt_input_len)
    not_pad = attention_mask[:, 1:].to(input_ids.device) > 0
    return (is_completion & not_pad).float()


def approx_kl_from_logprobs(new_logprobs, ref_logprobs, mask, eps=1e-8, log_ratio_clip=20.0):
    """Positive KL proxy from sampled actions."""
    # k3 estimator of KL(p_new || p_ref) on sampled tokens:
    # with delta = log p_ref(a) - log p_new(a), a ~ p_new, E[exp(delta) - delta - 1] = KL.
    delta = (ref_logprobs - new_logprobs).clamp(-log_ratio_clip, log_ratio_clip)
    per_token = torch.exp(delta) - delta - 1.0
    return masked_mean(per_token, mask, eps=eps)
```

Logits are upcast to fp32 for the cross-entropy so that the log-probs, and therefore the
PPO ratio `exp(new - old)`, aren't quantized to bf16 precision. This costs memory. See §5.

### `src/llm_rl/rollout/rollout_buffer.py` — `iter_minibatches`

```python
    N = int(batch.input_ids.shape[0])
    if minibatch_size <= 0:
        raise ValueError(f"minibatch_size must be >= 1, got {minibatch_size}")
    if shuffle:
        gen_device = generator.device if generator is not None else batch.input_ids.device
        order = torch.randperm(N, generator=generator, device=gen_device).to(batch.input_ids.device)
    else:
        order = torch.arange(N, device=batch.input_ids.device)

    for start in range(0, N, minibatch_size):
        idx = order[start : start + minibatch_size]
        idx_list = idx.tolist()
        mb = RolloutBatch(
            input_ids=batch.input_ids[idx],
            attention_mask=batch.attention_mask[idx],
            completion_mask=batch.completion_mask[idx],
            old_logprobs=batch.old_logprobs[idx],
            ref_logprobs=batch.ref_logprobs[idx],
            rewards=batch.rewards[idx.to(batch.rewards.device)],
            advantages=batch.advantages[idx.to(batch.advantages.device)],
            task_names=[batch.task_names[i] for i in idx_list] if batch.task_names is not None else None,
            completion_texts=(
                [batch.completion_texts[i] for i in idx_list] if batch.completion_texts is not None else None
            ),
        )
        if device is not None:
            mb = mb.to(device)
        yield mb
```

### `src/llm_rl/rl/reinforce.py` — minibatch body (replaces the TODO)

```python
            new_logp = compute_per_token_logprobs(model, mb.input_ids, mb.attention_mask)
            seq_logp = masked_mean_per_row(new_logp, mask)
            pg_loss = -(adv * seq_logp).mean()
            kl = approx_kl_from_logprobs(new_logp, mb.ref_logprobs, mask)
            with torch.no_grad():
                entropy = -masked_mean(new_logp, mask)
```

### `src/llm_rl/rl/grpo.py` — minibatch body (replaces the TODO)

```python
                new_logp = compute_per_token_logprobs(model, mb.input_ids, mb.attention_mask)
                log_ratio = (new_logp - mb.old_logprobs).clamp(-20.0, 20.0)
                ratio = torch.exp(log_ratio)
                adv_tok = adv.unsqueeze(1)
                unclipped = ratio * adv_tok
                clipped = ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_tok
                per_token_obj = torch.minimum(unclipped, clipped) * mask
                seq_obj = masked_mean_per_row(per_token_obj, mask)
                pg_loss = -seq_obj.mean()
                kl = approx_kl_from_logprobs(new_logp, mb.ref_logprobs, mask)
                with torch.no_grad():
                    entropy = -masked_mean(new_logp, mask)
                    was_clipped = ((ratio < 1.0 - cfg.clip_eps) | (ratio > 1.0 + cfg.clip_eps)).float()
                    clipfrac = masked_mean(was_clipped, mask)
```

### `src/llm_rl/train.py` — advantages

```python
def compute_group_advantages(rewards: torch.Tensor, group_size: int, eps: float = 1e-6) -> torch.Tensor:
    # rewards is flat [N] in prompt-major order; groups are contiguous.
    rewards = rewards.float()
    if group_size <= 1:
        return torch.zeros_like(rewards)
    if rewards.numel() % group_size != 0:
        raise ValueError(f"rewards.numel()={rewards.numel()} is not divisible by group_size={group_size}")
    grouped = rewards.view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True, unbiased=False)
    centered = grouped - mean
    # Groups with (near-)identical rewards carry no learning signal: emit zero advantage.
    adv = torch.where(std > eps, centered / (std + eps), torch.zeros_like(centered))
    return adv.reshape(-1)


def maybe_normalize_advantages(advantages: torch.Tensor, enabled: bool, eps: float = 1e-6) -> torch.Tensor:
    if not enabled or advantages.numel() <= 1:
        return advantages
    return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + eps)
```

Edge-case choices: `group_size <= 1` → zero advantages (a group of one has no baseline);
a length not divisible by `group_size` → error (it would indicate a sampler bug);
near-zero std → zero advantage for that group.

## 3. Packaging (`pyproject.toml`)

```toml
[project]
name = "code-20260923-fine-tune"
version = "0.1.0"
description = "Single-GPU LoRA RL fine-tuning (GRPO / REINFORCE) of small LLMs, run locally"
readme = "README.md"
authors = [{ name = "Tianyang Li", email = "litianyang.research@gmail.com" }]
requires-python = ">=3.13"
dependencies = [
    "torch>=2.7",
    "transformers>=4.56",
    "peft>=0.13.0",
    "accelerate>=1.0",
    "datasets>=3.0",
    "tqdm>=4.66.0",
    "numpy>=2.1",
]

[project.optional-dependencies]
wandb = ["wandb>=0.19"]

[project.scripts]
llm-rl-train = "llm_rl.train:main"
llm-rl-eval = "llm_rl.eval:main"

[tool.uv.build-backend]
module-name = "llm_rl"

[build-system]
requires = ["uv_build>=0.12.18,<0.13.0"]
build-backend = "uv_build"
```

Why the pins changed from hw4: Python 3.13 (from the scaffold) needs `torch>=2.6` and
`numpy>=2.1` wheels. The original `numpy<2.0` and `torch<2.7` pins can't install on 3.13.
`transformers>=4.56` is required because `models/load.py` calls
`from_pretrained(..., dtype=...)`, the newer spelling of `torch_dtype`.

`.gitignore` additions: `runs/`, `wandb/`. `uv lock` produced `uv.lock` (committed).
Versions resolved on 2026-09-23: torch 2.14.0 (cu130), transformers 5.17.0, peft 0.21.0,
wandb 0.30.0, tokenizers 0.23.2, safetensors 0.8.0, pyarrow 25.0.1.

## 4. Environment setup (the uv issue)

Symptom: `uv sync` / `uv run` printed

```
warning: `VIRTUAL_ENV=/media/lty/hdd-20241206/code/20260923-fine-tune/.venv` does not match
the project environment path `.venv` and will be ignored
```

Cause: the shell has the shared parent venv `../.venv` activated, while uv defaults to a
project-local `./.venv`. The intended environment is the shared venv, which already has
`torch 2.14.0+cu130` working with this GPU.

Fix: install the project editable into the shared venv with the pip interface. This keeps
the existing torch (it already satisfies `torch>=2.7`) and leaves the venv's other
packages alone:

```bash
source /media/lty/hdd-20241206/code/20260923-fine-tune/.venv/bin/activate
cd /media/lty/hdd-20241206/code/20260923-fine-tune/code-20260923-fine-tune
uv pip install -e ".[wandb]"
python -c "import torch,transformers,peft;print(torch.__version__, torch.cuda.is_available(), transformers.__version__, peft.__version__)"
# -> 2.14.0+cu130 True 5.17.0 0.21.0
```

**Don't** use `uv sync --active` on the shared venv. It makes the venv exactly match
`uv.lock` and removes unrelated packages (Jupyter, black, …).

A project-local `./.venv` was also created earlier by `uv sync --extra wandb`. It works
too (`uv run llm-rl-train …` once the other venv is deactivated), but it's redundant with
the shared venv. It's gitignored and can be deleted with `rm -rf .venv`.

## 5. Bugs found while running on this machine, and fixes

### 5.1 transformers 5.x `apply_chat_template` return type

Error on the first run:

```
File "src/llm_rl/models/load.py", in tokenize_chat_prompts
    max_len = max(x.numel() for x in encs)
AttributeError: 'tokenizers.Encoding' object has no attribute 'numel'
```

In transformers 5.x, `apply_chat_template(tokenize=True, return_tensors="pt")[0]` no
longer returns a tensor row. Fix in `tokenize_chat_prompts`: render to text, then
tokenize (works on 4.x and 5.x):

```python
        # Render to text then tokenize: apply_chat_template(tokenize=True) changed its
        # return type in transformers 5.x.
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        ids = torch.tensor(tokenizer(text, add_special_tokens=False)["input_ids"], dtype=torch.long)
```

`add_special_tokens=False` because the chat template already inserts the special tokens.

### 5.2 generate() deprecation warning

`hf_sampler.py` passed `use_cache=True` both inside `GenerationConfig` and as a kwarg to
`generate()`, which transformers 5 flags as deprecated. Removed the duplicate kwarg (it
stays in `GenerationConfig`).

### 5.3 CUDA OOM scoring math_hard rollouts on 16 GB

Error in the rollout (before any training step):

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 16.85 GiB.
```

Cause: `HFSampler.rollout` computed old/ref log-probs for all
`batch_size × group_size = 64` sequences in one forward pass. At ~1k tokens × 152k vocab,
the logits alone are ~17 GB. That fits on an H100 but not here.

Fix: score in chunks. `HFSampler.rollout(..., logprob_batch_size=None)` gained a chunked
helper used for both the policy (old) and adapter-disabled (ref) passes:

```python
            # Score in chunks: full-vocab logits for all B*group_size sequences at once
            # do not fit on a 16GB GPU for long completions.
            chunk = int(logprob_batch_size) if logprob_batch_size else int(sequences.shape[0])

            def _chunked_logprobs() -> torch.Tensor:
                return torch.cat(
                    [
                        compute_per_token_logprobs(
                            policy_model,
                            sequences[i : i + chunk],
                            full_attention[i : i + chunk],
                            enable_grad=False,
                        )
                        for i in range(0, int(sequences.shape[0]), chunk)
                    ],
                    dim=0,
                )

            old_logp = _chunked_logprobs()
            ...
            with policy_model.disable_adapter():
                ref_logp = _chunked_logprobs()
```

This is wired through as `TrainConfig.logprob_batch_size: int = 4`, CLI
`--logprob_batch_size`, validated `>= 1` in `train.main`, and passed to
`sampler.rollout(...)`. For the training update itself, math_hard uses
`--minibatch_size 4 --grad_accum_steps 16`, the same 64 sequences per optimizer step as
the original `8 × 8`.

## 6. Verification

### 6.1 CPU unit checks (all passed)

```bash
CUDA_VISIBLE_DEVICES= python - <<'EOF'
import torch
from llm_rl.train import compute_group_advantages, maybe_normalize_advantages
from llm_rl.models.logprobs import build_completion_mask, approx_kl_from_logprobs
from llm_rl.rollout.rollout_buffer import RolloutBatch, iter_minibatches
r = torch.tensor([1.,0.,1.,0., 2.,2.,2.,2.])
a = compute_group_advantages(r, 4); print(a)
assert torch.allclose(a[:4], torch.tensor([1.,-1.,1.,-1.]), atol=1e-4) and (a[4:]==0).all()
print(maybe_normalize_advantages(a, True))
ids = torch.tensor([[0,0,5,6,7,8,0],[0,5,5,6,7,0,0]]); am = torch.tensor([[0,0,1,1,1,1,0],[0,1,1,1,1,0,0]])
print(build_completion_mask(ids, am, prompt_input_len=4, pad_token_id=0))
lp = torch.randn(2,6); print(approx_kl_from_logprobs(lp, lp, torch.ones(2,6)))
N=10; b = RolloutBatch(*(torch.arange(N).unsqueeze(1).repeat(1,3) for _ in range(5)), torch.arange(N).float(), torch.arange(N).float(), task_names=list(range(N)))
g=torch.Generator(); g.manual_seed(0)
seen=[]
for mb in iter_minibatches(b, 4, generator=g):
    assert (mb.input_ids[:,0].float()==mb.advantages).all() and mb.task_names==mb.input_ids[:,0].tolist(); seen += mb.task_names
assert sorted(seen)==list(range(N)); print("ok", seen)
EOF
```

Observed: advantages `[1,-1,1,-1,0,0,0,0]` (the zero-variance group gets 0); completion
mask `[[0,0,0,1,1,0],[0,0,0,1,0,0]]` (correct off-by-one alignment); KL of identical
logprobs = 0; minibatches cover all indices once with fields aligned.

### 6.2 GPU smoke test: format_copy + GRPO — PASSED

```bash
llm-rl-train --task format_copy --algo grpo --output_dir runs/smoke_fc_grpo --steps 3 \
  --batch_size 8 --group_size 6 --max_new_tokens 24 --ppo_epochs 2 --minibatch_size 8 \
  --grad_accum_steps 6 --clip_eps 0.2 --warmup_steps 10 --format_copy_eval_n 16 \
  --eval_interval 0 --save_interval 0
```

Result (Qwen2.5-Math-1.5B-Instruct): ~8.2 s/step, peak GPU allocated 8.41 GB, mean
reward per step 0.154 / 0.081 / 0.127 (3 steps is too few to learn), KL ≈ 0.001, and a
checkpoint written to `runs/smoke_fc_grpo/checkpoints/step_000003/{adapter,optimizer.pt,meta.json,adapter_manifest.json}`.
The base model's baseline format_copy exact-match was 0/16.

### 6.3 GPU smoke test: math_hard + GRPO (512-token completions) — PASSED

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True llm-rl-train --task math_hard --algo grpo \
  --output_dir runs/smoke_mh_grpo --steps 2 --batch_size 8 --group_size 8 \
  --min_new_tokens 8 --max_new_tokens 512 --max_prompt_tokens 512 --temperature 0.8 --top_p 0.95 \
  --lr 3e-5 --ppo_epochs 2 --minibatch_size 4 --grad_accum_steps 16 --clip_eps 0.2 \
  --math_hard_eval_n 32 --eval_interval 0 --save_interval 0
```

- First attempt (before §5.3, `--minibatch_size 8`, unchunked scoring): OOM in rollout scoring.
- Baseline eval (both attempts): 32 test problems in ~43 s (0.75 problems/s), boxed
  exact-match **0.406** (13/32) for the base model.
- With §5.3 and minibatch 4: **PASSED**, exit 0, checkpoint `step_000002` written.

  | step | mean reward | mean completion tokens | frac hit 512 limit | peak GPU alloc | wall-clock | optimizer steps |
  |---|---|---|---|---|---|---|
  | 0 | 0.222 | 462 | 0.67 | 9.80 GB | 201 s | 2 |
  | 1 | 0.261 | 467 | 0.61 | 8.21 GB | 151 s | 2 |

  Final eval after 2 steps: boxed exact-match 0.375 on the same 32 problems (noise at
  n=32). KL ≈ 0 after 2 steps with lr warmup, as expected. At ~150–200 s/step, the
  README's 501-step GRPO run takes ~1 day on this GPU.

## 7. Model suggestions for this hardware

See the table in `README.md` ("Which models can this machine fine-tune?"). Summary:
0.5B–1.7B models (Qwen2.5-0.5B/1.5B, Qwen2.5-Math-1.5B, Qwen3-0.6B/1.7B, SmolLM2-1.7B,
Llama-3.2-1B) fit comfortably. 3–4B models are tight (minibatch 1–2, shorter completions).
7B+ doesn't fit in bf16 and would need 4-bit QLoRA, which isn't implemented. Only the
default Qwen2.5-Math-1.5B-Instruct was actually run.

## 8. Reproduce from scratch (summary)

```bash
# 1. Get the source and this repo
git clone git@github.com:tianyang-li/20260923-fine-tune-01.git code-20260923-fine-tune
#    (or rebuild it: clone berkeleydeeprlcourse/homework_spring2026 @ 59e40fc and apply §1–§5)

# 2. Environment (Python 3.13 uv venv with CUDA torch)
uv venv --python 3.13 ../.venv && source ../.venv/bin/activate   # skip if the shared venv exists
cd code-20260923-fine-tune
uv pip install -e ".[wandb]"          # pulls torch 2.14 cu130 from PyPI if not already present

# 3. Sanity check
llm-rl-train --task format_copy --algo grpo --output_dir runs/smoke --steps 3 \
  --batch_size 8 --group_size 6 --max_new_tokens 24 --format_copy_eval_n 16 \
  --eval_interval 0 --save_interval 0

# 4. Full runs: see README.md "Running on this machine"
```
