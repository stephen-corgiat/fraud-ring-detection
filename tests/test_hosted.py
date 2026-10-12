"""Hosted mode (`FRAUDRINGS_HOSTED=1`): per-session labels, bounded cache, and the entrypoint."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import pytest

from fraudrings.viewer import labels as L


def test_in_memory_label_stores_do_not_share(tmp_path) -> None:
    one = L.LabelStore(str(tmp_path / "l.json"), backing={})
    two = L.LabelStore(str(tmp_path / "l.json"), backing={})
    one.set(1, "fraud")
    assert two.get(1) is None and not (tmp_path / "l.json").exists()
    assert one.counts()["fraud"] == 1


def test_hosted_labels_are_per_session(tmp_path, monkeypatch) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from fraudrings.viewer import main as M

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FRAUDRINGS_HOSTED", "1")

    def open_session() -> AppTest:
        at = AppTest.from_file(M.__file__, default_timeout=300)
        at.session_state["w_preset"] = "smoke"
        at.run()
        assert not at.exception, at.exception
        return at

    def current(at: AppTest) -> bool:
        return any("current:" in c.value for c in at.caption)

    a, b = open_session(), open_session()
    assert any("Everything here is simulated" in m.value for m in a.markdown)
    a.radio[0].set_value("ring")
    save = [x for x in a.button if x.label == "save label"]
    assert save, "triage should offer a save-label button"
    save[0].click().run()
    assert not a.exception, a.exception
    assert current(a)
    b.run()  # B reruns after A saved: a shared store would show A's label here
    assert not b.exception, b.exception
    assert not current(b)
    assert not list(tmp_path.rglob("suspect_labels.json")), "hosted labels never touch disk"


def test_streamlit_app_entrypoint(monkeypatch) -> None:
    pytest.importorskip("streamlit")
    from fraudrings.viewer import main as M

    called: list[int] = []
    monkeypatch.setattr(M, "main", lambda: called.append(1))
    monkeypatch.delenv("FRAUDRINGS_HOSTED", raising=False)
    monkeypatch.delenv("POLARS_MAX_THREADS", raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    root = Path(__file__).resolve().parents[1]
    for _ in range(2):  # a rerun must not pile up sys.path entries
        runpy.run_path(str(root / "streamlit_app.py"))
    assert len(called) == 2 and os.environ["FRAUDRINGS_HOSTED"] == "1"
    assert sys.path.count(str(root / "src")) == 1
    monkeypatch.delenv("FRAUDRINGS_HOSTED")
    monkeypatch.delenv("POLARS_MAX_THREADS")


def test_requirements_match_pyproject() -> None:
    """requirements.txt exists only for Community Cloud; keep it equal to core + viz deps."""
    import tomllib

    root = Path(__file__).resolve().parents[1]
    proj = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    want = set(proj["dependencies"]) | set(proj["optional-dependencies"]["viz"])
    have = {ln.strip() for ln in (root / "requirements.txt").read_text().splitlines()
            if ln.strip() and not ln.startswith("#")}
    assert have == want
