# 0007 — One chart library, and charts never feed scoring

**Status:** accepted

## Context

The investigator UI needs a dozen views: anomaly profile, daily activity, action composition,
reported-position histogram, co-target heatmap, linkage subgraph, account-age strip, registration
burst, inter-event cadence, temporal raster and occupancy, plus the triage page's axis, contributor
and weekly charts. Some of them must be **selectable**: lasso a clump of accounts on the linkage
graph and re-run the investigation seeded on them.

Two risks: a patchwork of chart libraries with inconsistent theming and interaction, and charts that
quietly become features — a number computed for display that someone later wires into a score,
breaking the anti-tautology boundary ([0001](0001-ground-truth-is-physically-separate.md)).

## Decision

- **Plotly for every chart**, behind one dark template registered once (`viewer/theme.py`). Streamlit's
  native `on_select` turns any per-account Plotly surface into a lasso-to-reseed control. Each
  selectable point carries its account id in `customdata`.
- **Presentation-only, by construction.** Chart substrate queries live in `viewer/data.py` and
  `viewer/data_triage.py`. Figures live in `viewer/charts.py`, with no Streamlit import, so they are
  unit-tested. Nothing in `viewer/` is imported by `detector/` or `eval/`.
- **Event-derivable ordering only.** The temporal raster orders rows by first-event time, never by
  ground-truth membership — a truth-sorted raster would hand the eye the answer.

## Consequences

- One theme, one interaction model, one testing story (`tests/test_charts.py`, `tests/test_raster.py`).
- Re-seeding from a lasso uses the same `run_investigation` path as everything else. No new detector
  machinery.
- Plotly's toolbar is shown on hover only, since lasso selection needs its select tools.
