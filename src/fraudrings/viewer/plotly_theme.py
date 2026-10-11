"""The dark Plotly template, with no streamlit import so scripts (the site's chart export) can use it.

`viewer.theme` re-exports these colors and registers the template at app start.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

CARD = "#13171d"        # card / panel surface
BORDER = "#222831"
TEXT = "#e6e9ee"
TEXT_DIM = "#aeb5bf"
RING = "#d65f5f"        # ring / anomaly / "this set" accent
NEUTRAL = "#4c78a8"     # secondary

TEMPLATE_NAME = "fraudrings_dark"


def plotly_template() -> go.layout.Template:
    """The dark template: the design's surface, grid and font. Figures in charts.py set only explicit
    marks/heights, so they pick up these defaults for everything else."""
    tmpl = go.layout.Template()
    tmpl.layout = go.Layout(
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(family="'IBM Plex Sans', system-ui, sans-serif", color=TEXT_DIM, size=12),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
        legend=dict(font=dict(color=TEXT_DIM)),
        colorway=[NEUTRAL, RING, "#59a14f", "#e0a05c", "#a87ca5", "#9c755f"],
    )
    return tmpl


def register_plotly_template() -> None:
    """Register + activate the template as the Plotly default (called once at app start)."""
    pio.templates[TEMPLATE_NAME] = plotly_template()
    pio.templates.default = f"plotly_dark+{TEMPLATE_NAME}"
