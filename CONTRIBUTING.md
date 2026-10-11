# Contributing

## Setup

You need Python 3.11+.

```sh
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev,viz,ml]"
```

| Extra | Adds | Needed for |
|---|---|---|
| `dev` | pytest, ruff, pyright | tests and lint |
| `viz` | Streamlit, Plotly, pandas, networkx | `fraudrings view` and the viewer tests |
| `ml` | scikit-learn, LightGBM | the learned rule: `grade`, `curve` and their tests |
| `docs` | websocket-client, Pillow, matplotlib | regenerating screenshots and figures |

## Test and lint

```sh
.venv/bin/pytest            # the whole suite, ~2 minutes; every world is built from a seed in a temp dir
.venv/bin/ruff check src tests scripts
.venv/bin/pyright           # standard mode
```

Tests never need the network, a private dataset, or a running service. They generate small seeded
worlds (the `smoke` preset) on the fly. Many of them pin *measured* claims — the shape of the
degradation curve, the learned rule clearing the bar, the ranker's recall — so a regression in the
method shows up as a failing test, not as a quietly worse number.

## Layout

| Path | Role |
|---|---|
| `src/fraudrings/schema.py` | The event, account and ground-truth schemas — the contract between generator and detector |
| `src/fraudrings/config.py` | Every knob for a run, plus the `smoke` and `default` presets |
| `src/fraudrings/generator/` | `simulator.py` builds worlds; `evasion.py` is the one table that maps evasion levels to behavior |
| `src/fraudrings/detector/ingest.py` | Per-account features, the identifier graph, rarity (IDF) weights |
| `src/fraudrings/detector/engine.py` | The seeded engine: rarity-gated retrieval, co-target widening, four-axis scoring |
| `src/fraudrings/detector/concentration.py` | The seedless ranker and its expansion gate |
| `src/fraudrings/eval/` | `evaluate` (per-ring metrics), `sweep`, `supervised` + `grade` (learned rule), `oracle` + `curve` (learning curve), `concentration` (ranker grade) |
| `src/fraudrings/viewer/` | `main` (entrypoint and shared sidebar), `app_triage` and `app` (the two pages), `data*` (queries), `charts`, `raster`, `theme`, `labels` |
| `src/fraudrings/pipeline.py` | The `fraudrings` command |
| `scripts/` | Screenshot and figure generation |
| `docs/` | Walkthrough, decision records, research, results, explainer lessons |

## The rules that keep results honest

Read [`docs/research/method.md`](docs/research/method.md) before changing the generator or detector.
In particular:

- **The detector never reads ground truth or the world's config.** `detector/` reads `events` and
  `accounts` only. Tests enforce it.
- **New stochastic generator features draw from their own RNG stream** and add events rather than
  interleaving draws, so they can't move existing results ([ADR 0008](docs/adr/0008-rng-isolation-for-additive-features.md)).
  `tests/test_lockstep.py` pins a content hash of the identity world. Re-pin it only alongside an
  intentional generator change, and say why in the commit.
- **Schema changes bump `SCHEMA_VERSION`** in `detector/ingest.py`. Cached viewer worlds rebuild
  themselves on the next load.
- **Charts are presentation-only.** Nothing in `viewer/` is imported by `detector/` or `eval/`.

Record a design decision that a future reader would otherwise have to reverse-engineer as a new ADR
in `docs/adr/`.

## Regenerating screenshots and figures

Screenshots come from seeded worlds, so they are reproducible:

```sh
.venv/bin/python -m pip install -e ".[viz,docs]"
.venv/bin/python scripts/capture_docs.py     # docs/images/walkthrough/*.png and hero.gif (~3 min)
```

The script builds the worlds, launches the viewer on a free port, drives headless Chrome or
Chromium, and crops each figure by finding its section heading in the page. Look at every image
before committing.

The README's results figures are drawn from the committed result files in `docs/results/`:

```sh
.venv/bin/python scripts/make_figures.py     # docs/images/results/*.png
```

The docstring of `scripts/make_figures.py` lists the commands that regenerate `docs/results/` itself
(about ten minutes on the default preset; the learning curve is the heavy part). If you change the
method, re-run them and update the numbers quoted in the README, the walkthrough and the ADRs.

## Pull requests

Keep the suite and lint green, add tests for behavior you change, and add yourself to
[CONTRIBUTORS.md](CONTRIBUTORS.md) in your first PR.
