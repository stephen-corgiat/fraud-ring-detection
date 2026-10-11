"""`scripts/export_charts.py` on the smoke preset: every chart is produced and is valid Plotly JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
pytest.importorskip("networkx")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import export_charts as X  # type: ignore[import-not-found]  # noqa: E402  # pyright: ignore[reportMissingImports]


def test_smoke_export_produces_every_chart(tmp_path: Path) -> None:
    out = tmp_path / "charts"
    X.export(out, preset_name="smoke", seed=7)
    assert sorted(p.stem for p in out.glob("*.json")) == sorted(X.NAMES)
    for name in X.NAMES:
        doc = json.loads((out / f"{name}.json").read_text())
        assert doc["data"] and "layout" in doc, name
        assert doc["plotlyjs"] and doc["source"], name
        assert (out / f"{name}.json").stat().st_size < 250_000, name


def test_investigate_charts_follow_the_creator_one_handoff() -> None:
    """The walkthrough's `?creator=1` handoff: creator 1 is the default world's rank-1 suspect."""
    doc = json.loads((ROOT / "docs" / "charts" / "investigate-linkage.json").read_text())
    assert "creator 1 handoff" in doc["source"] and doc["source"].startswith("default/naive seed=7")


def test_committed_charts_match_the_pinned_plotlyjs() -> None:
    """The site pins one plotly.js; every committed chart must have been exported for it."""
    import tomllib

    pinned = tomllib.loads((ROOT / "site" / "site.toml").read_text())["plotly_version"]
    files = sorted((ROOT / "docs" / "charts").glob("*.json"))
    assert {f.stem for f in files} == set(X.NAMES)
    assert {json.loads(f.read_text())["plotlyjs"] for f in files} == {pinned}
