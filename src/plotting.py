"""Plot helpers for IOI reproduction artifacts."""

from __future__ import annotations

from pathlib import Path

from torch import Tensor


def residual_patching_heatmap(
    recovery: Tensor,
    *,
    component: str = "resid_pre",
    output_path: str | Path | None = None,
    show: bool = False,
):
    """Create a Plotly heatmap for layer-by-position patching recovery."""

    import plotly.express as px

    fig = px.imshow(
        recovery.detach().cpu().numpy(),
        labels={
            "x": "Position",
            "y": "Layer",
            "color": "Logit Diff Recovery",
        },
        title=f"Activation Patching: {component}",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0.0,
        aspect="auto",
    )
    fig.update_yaxes(dtick=1)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(path)
    if show:
        fig.show()
    return fig
