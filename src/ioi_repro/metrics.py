"""Metrics for IOI reproduction experiments."""

from __future__ import annotations

import torch
from torch import Tensor


def logit_diff(
    logits: Tensor,
    io_tokens: Tensor,
    s_tokens: Tensor,
    *,
    mean: bool = True,
) -> Tensor:
    """Return final-position IO-minus-subject logit difference.

    Higher values mean the model more strongly prefers the indirect object
    answer over the subject distractor.
    """

    final_logits = logits[:, -1, :]
    batch_indices = torch.arange(final_logits.shape[0], device=final_logits.device)
    io_tokens = io_tokens.to(final_logits.device)
    s_tokens = s_tokens.to(final_logits.device)
    diff = final_logits[batch_indices, io_tokens] - final_logits[batch_indices, s_tokens]
    return diff.mean() if mean else diff


def logit_diff_recovery(
    patched_diff: Tensor,
    clean_diff: Tensor,
    corrupted_diff: Tensor,
    *,
    eps: float = 1e-8,
) -> Tensor:
    """Normalize a patched score between corrupted and clean baselines."""

    return (patched_diff - corrupted_diff) / (clean_diff - corrupted_diff + eps)
