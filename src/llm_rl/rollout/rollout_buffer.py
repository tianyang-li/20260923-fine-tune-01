from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

import torch


@dataclass
class RolloutBatch:
    input_ids: torch.Tensor          # [N, L]
    attention_mask: torch.Tensor     # [N, L]
    completion_mask: torch.Tensor    # [N, L-1] float
    old_logprobs: torch.Tensor       # [N, L-1]
    ref_logprobs: torch.Tensor       # [N, L-1]
    rewards: torch.Tensor            # [N]
    advantages: torch.Tensor         # [N]

    # Optional debug
    task_names: Optional[list] = None
    completion_texts: Optional[list] = None

    def to(self, device: torch.device) -> "RolloutBatch":
        return RolloutBatch(
            input_ids=self.input_ids.to(device, non_blocking=True),
            attention_mask=self.attention_mask.to(device, non_blocking=True),
            completion_mask=self.completion_mask.to(device, non_blocking=True),
            old_logprobs=self.old_logprobs.to(device, non_blocking=True),
            ref_logprobs=self.ref_logprobs.to(device, non_blocking=True),
            rewards=self.rewards.to(device, non_blocking=True),
            advantages=self.advantages.to(device, non_blocking=True),
            task_names=self.task_names,
            completion_texts=self.completion_texts,
        )


def iter_minibatches(
    batch: RolloutBatch,
    minibatch_size: int,
    shuffle: bool = True,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> Iterator[RolloutBatch]:
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
