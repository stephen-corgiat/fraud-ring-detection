"""The Triage page's data layer and the two-page app, on a smoke world.

The data tests run without a Streamlit runtime. The app test renders both pages headlessly through
`streamlit.testing`, which catches wiring errors (missing session keys, bad widget args) that unit
tests on the query layer cannot.
"""

from __future__ import annotations

import pytest

from fraudrings.viewer import data as D
from fraudrings.viewer import data_triage as R


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return D.load_world("smoke", "naive", 7, base_dir=str(tmp_path_factory.mktemp("w")))


def test_shortlist_evidence_is_consistent_with_the_ranker(world) -> None:
    suspects = R.suspects(world.con, R.ConcentrationConfig(impact_floor=100))
    assert suspects
    top = suspects[0]
    contrib = R.top_contributors(world.con, top.target_id)
    assert contrib == sorted(contrib, key=lambda r: (-r["qv"], r["account_id"]))
    weekly = R.weekly_qv(world.con, top.target_id)
    assert sum(w["qv"] for w in weekly) == top.tot_qv  # the weekly bars add up to the ranker's volume
    if top.core:
        core = R.core_stats(world.con, top.core, top.target_id)
        assert {c["account_id"] for c in core} == set(top.core)
        assert all(c["share"] >= 0.80 for c in core)  # every core account is a dedicated booster


def test_target_truth_marks_every_ring_target(world) -> None:
    truth = R.target_truth(world)
    targets = {c for c, t in truth.items() if t.is_ring_target}
    rings = {r for t in truth.values() for r in t.ring_ids}
    assert rings == set(world.gt.filter(world.gt["is_fraud"])["ring_id"].unique().to_list())
    assert all(truth[c].fraud_qv_share > 0 for c in targets)


def test_triage_labels_are_graded_against_ring_targets(world, tmp_path) -> None:
    truth = R.target_truth(world)
    ring_target = next(c for c, t in truth.items() if t.is_ring_target)
    organic = next(c for c, t in truth.items() if not t.is_ring_target)
    store = R.SuspectLabelStore(str(tmp_path / "labels.json"))
    store.set(ring_target, "ring")
    store.set(organic, "ring")
    store.set(organic + 10_000, "unsure")
    sc = store.score(truth)
    assert (sc.tp, sc.fp, sc.n_decided) == (1, 1, 2)
    saved = R.SuspectLabelStore(str(tmp_path / "labels.json")).get(ring_target)
    assert saved is not None and saved["verdict"] == "ring"


def test_both_pages_render_headlessly(tmp_path, monkeypatch) -> None:
    pytest.importorskip("streamlit")
    monkeypatch.chdir(tmp_path)  # the app builds its world under ./data
    from streamlit.testing.v1 import AppTest

    from fraudrings.viewer import main as M

    at = AppTest.from_file(M.__file__, default_timeout=300)
    at.session_state["w_preset"] = "smoke"
    at.run()
    assert not at.exception, at.exception
    assert any("Triage queue" in m.value for m in at.markdown)

    # The seedless -> seeded handoff: the top suspect's core becomes the investigator's seeds.
    handoff = [b for b in at.button if "Investigate this core" in str(b.label)]
    assert handoff, "the top smoke suspect should have a dedicated core to hand off"
    handoff[0].click().run()
    assert not at.exception, at.exception
    assert at.session_state["seed_mode"] == "from triage"
    assert any("Related accounts" in m.value or "Profile" in m.value for m in at.markdown)


def test_deep_links_select_the_world(tmp_path, monkeypatch) -> None:
    """`?preset=smoke&level=moderate&seed=13` opens the app on exactly that world."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from fraudrings.viewer import main as M

    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(M.__file__, default_timeout=300)
    at.query_params["preset"] = "smoke"
    at.query_params["level"] = "moderate"
    at.query_params["seed"] = "13"
    at.run()
    assert not at.exception, at.exception
    world = at.session_state["world"]
    assert (world.preset_name, world.level, world.seed) == ("smoke", "moderate", 13)

