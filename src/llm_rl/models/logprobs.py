from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def compute_per_token_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    enable_grad: bool = True,
) -> torch.Tensor:
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


def build_completion_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_input_len: int,
    pad_token_id: int,
) -> torch.Tensor:
    """Mask over per-token positions [B, L-1], selecting completion tokens only."""
    # mask[:, t] scores token t+1; completion tokens start at index prompt_input_len.
    B, L = input_ids.shape
    positions = torch.arange(1, L, device=input_ids.device).unsqueeze(0)
    is_completion = positions >= int(prompt_input_len)
    not_pad = attention_mask[:, 1:].to(input_ids.device) > 0
    return (is_completion & not_pad).float()


def masked_sum(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum(dim=1) / (mask.sum(dim=1) + eps)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum() / (mask.sum() + eps)


def masked_mean_per_row(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum(dim=1) / (mask.sum(dim=1) + eps)


def approx_kl_from_logprobs(
    new_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
    log_ratio_clip: float = 20.0,
) -> torch.Tensor:
    """Positive KL proxy from sampled actions."""
    # k3 estimator of KL(p_new || p_ref) on sampled tokens:
    # with delta = log p_ref(a) - log p_new(a), a ~ p_new, E[exp(delta) - delta - 1] = KL.
    delta = (ref_logprobs - new_logprobs).clamp(-log_ratio_clip, log_ratio_clip)
    per_token = torch.exp(delta) - delta - 1.0
    return masked_mean(per_token, mask, eps=eps)
