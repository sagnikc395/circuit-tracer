"""Run residual-stream activation patching on IOI prompts."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ioi_repro.config import ExperimentConfig
from ioi_repro.dataset import build_ioi_dataset, clean_prompts, corrupted_prompts
from ioi_repro.model import (
    answer_token_ids,
    assert_uniform_token_length,
    load_hooked_transformer,
)
from ioi_repro.patching import residual_stream_patching
from ioi_repro.plotting import residual_patching_heatmap


def parse_args() -> argparse.Namespace:
    defaults = ExperimentConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=defaults.model_name)
    parser.add_argument("--n-examples", type=int, default=defaults.n_examples)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--component", default=defaults.component)
    parser.add_argument("--template-id", default="abba_store")
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--show", action="store_true", help="Open the Plotly figure")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = build_ioi_dataset(
        args.n_examples,
        seed=args.seed,
        template_id=args.template_id,
    )
    model = load_hooked_transformer(args.model_name, device=args.device)

    clean = clean_prompts(examples)
    corrupted = corrupted_prompts(examples)
    assert_uniform_token_length(model, clean + corrupted, prepend_bos=True)

    clean_tokens = model.to_tokens(clean, prepend_bos=True)
    corrupted_tokens = model.to_tokens(corrupted, prepend_bos=True)
    io_tokens, s_tokens = answer_token_ids(model, examples)

    result = residual_stream_patching(
        model,
        clean_tokens,
        corrupted_tokens,
        io_tokens,
        s_tokens,
        component=args.component,
    )

    tensor_path = output_dir / f"{args.component}_patching.pt"
    html_path = output_dir / f"{args.component}_patching.html"
    torch.save(result.as_dict(), tensor_path)
    residual_patching_heatmap(
        result.recovery,
        component=args.component,
        output_path=html_path,
        show=args.show,
    )

    print(f"clean logit diff: {float(result.clean_diff):.4f}")
    print(f"corrupted logit diff: {float(result.corrupted_diff):.4f}")
    print(f"saved tensor artifact: {tensor_path}")
    print(f"saved heatmap: {html_path}")


if __name__ == "__main__":
    main()
