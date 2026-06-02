# circuit-tracer

Boilerplate for reproducing the IOI experiments from Wang et al. (2022),
"Interpretability in the Wild: a Circuit for Indirect Object Identification in
GPT-2 Small."

The scaffold is intentionally narrow: it gives you a clean prompt/corrupted
prompt generator, logit-difference metrics, GPT-2 Small loading via
TransformerLens, a residual-stream activation patching experiment, Plotly
visualization, and small tests for the pieces that do not require downloading a
model.

## Layout

```text
src/ioi_repro/
  dataset.py        IOI prompt generation and template loading
  metrics.py        IO-vs-subject logit difference and recovery metrics
  model.py          TransformerLens model loading and answer token helpers
  patching.py       Residual stream activation patching loop
  plotting.py       Plotly heatmap helper
  experiments/
    activation_patching.py

prompts/
  ioi_templates.jsonl

tests/
  test_dataset.py
  test_metrics.py
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
