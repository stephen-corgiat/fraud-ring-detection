"""Per-session plumbing for the viewer, and the switch for the public hosted demo.

Locally the viewer is a single-user tool: labels are JSON files in the world dir. On a shared host
(`FRAUDRINGS_HOSTED=1`, set by the root `streamlit_app.py`) that would let every visitor see and
overwrite everyone else's labels, so hosted label stores live in `st.session_state` instead.
"""

from __future__ import annotations

import os
from dataclasses import replace

import streamlit as st

from fraudrings.viewer import data as D
from fraudrings.viewer import data_triage as R
from fraudrings.viewer.labels import LabelStore

HOSTED_ENV = "FRAUDRINGS_HOSTED"
# Hosted worlds kept in memory at once; each is a DuckDB connection plus its ground truth.
HOSTED_MAX_WORLDS = 3


def hosted() -> bool:
    return os.environ.get(HOSTED_ENV) == "1"


def _backing(kind: str, world_dir: str) -> dict[int, dict]:
    return st.session_state.setdefault(f"_labels:{kind}:{world_dir}", {})


def label_store(world_dir: str) -> LabelStore:
    """Account-level labels (Investigate)."""
    if hosted():
        return LabelStore(os.path.join(world_dir, "labels.json"), backing=_backing("account", world_dir))
    return LabelStore(os.path.join(world_dir, "labels.json"))


def suspect_label_store(world_dir: str) -> R.SuspectLabelStore:
    """Creator-level labels (Triage)."""
    if hosted():
        return R.SuspectLabelStore(
            os.path.join(world_dir, "suspect_labels.json"), backing=_backing("suspect", world_dir)
        )
    return R.store_for_world(world_dir)


def session_world(world: D.World) -> D.World:
    """The cached world with a connection of this session's own.

    The world is shared across sessions through `st.cache_resource`, but a DuckDB connection is not
    safe for concurrent use and Streamlit runs each session on its own thread. A cursor is an
    independent connection to the same database.
    """
    key = "_cursor"
    held = st.session_state.get(key)
    if held is None or held[0] != world.world_dir:
        held = (world.world_dir, world.con.cursor())
        st.session_state[key] = held
    return replace(world, con=held[1])
