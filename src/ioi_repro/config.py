"""Experiment configuration objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    """Defaults for the IOI residual-stream patching experiment."""

    model_name: str = "gpt2-small"
    n_examples: int = 32
    seed: int = 0
    device: str = "auto"
    component: str = "resid_pre"
    output_dir: Path = Path("artifacts/ioi")
