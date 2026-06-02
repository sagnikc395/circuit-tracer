"""Model loading and tokenization helpers."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor

from ioi_repro.dataset import IOIExample


def infer_device() -> str:
    """Pick the best available local torch device."""

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_hooked_transformer(model_name: str = "gpt2-small", *, device: str = "auto"):
    """Load a TransformerLens HookedTransformer."""

    from transformer_lens import HookedTransformer

    resolved_device = infer_device() if device == "auto" else device
    return HookedTransformer.from_pretrained(model_name, device=resolved_device)


def _single_token_id(model, text: str) -> int:
    token_ids = model.to_tokens(text, prepend_bos=False).squeeze(0)
    if token_ids.numel() != 1:
        pieces = model.to_str_tokens(text, prepend_bos=False)
        raise ValueError(
            f"Expected {text!r} to be one token, got {len(pieces)} tokens: {pieces}"
        )
    return int(token_ids.item())


def answer_token_ids(
    model,
    examples: Iterable[IOIExample],
    *,
    prepend_space: bool = True,
) -> tuple[Tensor, Tensor]:
    """Return IO and subject answer-token IDs for each example."""

    io_tokens: list[int] = []
    s_tokens: list[int] = []
    prefix = " " if prepend_space else ""

    for example in examples:
        io_tokens.append(_single_token_id(model, f"{prefix}{example.io_name}"))
        s_tokens.append(_single_token_id(model, f"{prefix}{example.s_name}"))

    device = getattr(model, "cfg", None)
    model_device = getattr(device, "device", None) or infer_device()
    return (
        torch.tensor(io_tokens, dtype=torch.long, device=model_device),
        torch.tensor(s_tokens, dtype=torch.long, device=model_device),
    )


def assert_uniform_token_length(
    model,
    prompts: Iterable[str],
    *,
    prepend_bos: bool = True,
) -> int:
    """Return the common token length, or raise if prompts are not aligned."""

    lengths = [
        model.to_tokens(prompt, prepend_bos=prepend_bos).shape[-1]
        for prompt in prompts
    ]
    unique_lengths = sorted(set(lengths))
    if len(unique_lengths) != 1:
        raise ValueError(
            "Activation patching expects position-aligned prompts, but token "
            f"lengths vary: {unique_lengths}. Use a single template and "
            "one-token names, places, and objects."
        )
    return unique_lengths[0]
