"""Activation patching primitives for IOI experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ioi_repro.metrics import logit_diff, logit_diff_recovery


@dataclass(frozen=True)
class ResidualPatchingResult:
    """Outputs from residual-stream activation patching."""

    recovery: Tensor
    clean_diff: Tensor
    corrupted_diff: Tensor
    component: str

    def as_dict(self) -> dict[str, Tensor | str]:
        return {
            "recovery": self.recovery.detach().cpu(),
            "clean_diff": self.clean_diff.detach().cpu(),
            "corrupted_diff": self.corrupted_diff.detach().cpu(),
            "component": self.component,
        }


@torch.no_grad()
def residual_stream_patching(
    model,
    clean_tokens: Tensor,
    corrupted_tokens: Tensor,
    io_tokens: Tensor,
    s_tokens: Tensor,
    *,
    component: str = "resid_pre",
) -> ResidualPatchingResult:
    """Patch clean residual activations into corrupted prompts by layer/position."""

    from transformer_lens import utils

    if clean_tokens.shape != corrupted_tokens.shape:
        raise ValueError(
            "clean_tokens and corrupted_tokens must have identical shapes; "
            f"got {tuple(clean_tokens.shape)} and {tuple(corrupted_tokens.shape)}"
        )

    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corrupted_logits = model(corrupted_tokens)
    clean_diff = logit_diff(clean_logits, io_tokens, s_tokens)
    corrupted_diff = logit_diff(corrupted_logits, io_tokens, s_tokens)

    n_layers = model.cfg.n_layers
    _, seq_len = clean_tokens.shape
    recovery = torch.empty((n_layers, seq_len), device=clean_tokens.device)

    for layer in range(n_layers):
        act_name = utils.get_act_name(component, layer)
        clean_activation = clean_cache[act_name]

        for position in range(seq_len):

            def patch_hook(corrupted_activation, hook, position=position):
                patched = corrupted_activation.clone()
                patched[:, position, :] = clean_activation[:, position, :]
                return patched

            patched_logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(act_name, patch_hook)],
            )
            patched_diff = logit_diff(patched_logits, io_tokens, s_tokens)
            recovery[layer, position] = logit_diff_recovery(
                patched_diff,
                clean_diff,
                corrupted_diff,
            )

    return ResidualPatchingResult(
        recovery=recovery,
        clean_diff=clean_diff,
        corrupted_diff=corrupted_diff,
        component=component,
    )
