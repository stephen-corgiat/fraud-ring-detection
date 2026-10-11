"""End-to-end smoke test: the whole loop must run AND measure.

On the naive adversary the loop should recover rings easily — this proves the plumbing and
that we can measure. The evasion dial (`test_evasion.py`) is what makes it hard.
"""

from __future__ import annotations

import polars as pl

from fraudrings.config import preset
from fraudrings.detector.ingest import build_db
from fraudrings.eval.evaluate import average_precision, evaluate
from fraudrings.generator import generate
from fraudrings.schema import EVENT_COLUMNS, Event


def test_average_precision_basics() -> None:
    import numpy as np

    perfect = average_precision(np.array([3.0, 2.0, 1.0]), np.array([1.0, 1.0, 0.0]))
    assert perfect == 1.0
    none = average_precision(np.array([1.0, 2.0]), np.array([0.0, 0.0]))
    assert none == 0.0


def test_end_to_end_recovers_naive_rings(tmp_path) -> None:
    cfg = preset("smoke")
    cfg.data_dir = str(tmp_path)

    events_path, gt_path = generate(cfg.sim, cfg.data_dir)

    # schema contract: a sampled event row validates against the Pydantic model
    events = pl.read_parquet(events_path)
    assert tuple(events.columns) == EVENT_COLUMNS
    Event.model_validate(events.row(0, named=True))
    # time-ordered (incremental-path precondition)
    assert events["ts"].is_sorted()

    # detector reads ONLY events
    con = build_db(events_path, cfg.engine, db_path=str(tmp_path / "world.duckdb"))
    result = evaluate(con, gt_path, cfg.engine, cfg.eval)

    assert result.rings, "no rings evaluated"
    assert result.mean_stage1_recall > 0.80, result.mean_stage1_recall
    assert result.mean_ap > 0.60, result.mean_ap
