# circuit-tracer

Implementing the IOI Circuit Tracer paper from scratch.

## Layout

```text
./
├── LICENSE
├── prompts
│   └── ioi_templates.jsonl
├── pyproject.toml
├── README.md
├── src
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── experiments
│   │   ├── __init__.py
│   │   └── activation_patching.py
│   ├── logit_diff.py
│   ├── metrics.py
│   ├── model.py
│   ├── patching.py
│   ├── plotting.py
│   └── visualize.py
├── tests
│   ├── test_dataset.py
│   └── test_metrics.py
└── uv.lock
```

## Quickstart

```bash
uv sync
uv run python -m unittest discover -s tests
uv run ioi-resid-patch --n-examples 32 --template-id abba_store --output-dir artifacts/ioi
```

The experiment writes:

- `artifacts/ioi/resid_pre_patching.pt`
- `artifacts/ioi/resid_pre_patching.html`

The first run may download `gpt2-small` weights from Hugging Face. For a faster
smoke test, use fewer prompts:

```bash
uv run ioi-resid-patch --n-examples 4 --template-id abba_store --output-dir artifacts/smoke
```

## Reproduction Roadmap

This repo currently covers the first reproducibility layer:

1. Generate paired clean/corrupted IOI prompts with position-aligned templates.
2. Score IO-vs-subject logit difference.
3. Run residual-stream activation patching over layer and position.
4. Save a heatmap for inspection.

Next natural additions are attention-head patching, path patching, named circuit
node bookkeeping, and paper-aligned figure/table scripts.

### References

1. Wang et al. (2022), [Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small](https://arxiv.org/abs/2211.00593)
