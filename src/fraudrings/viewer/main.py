"""Streamlit entrypoint for `fraudrings view`: one app, two pages over one shared world.

  * **Triage** (`viewer.app_triage`) — seedless. The concentration ranker shortlists suspect
    creators from the event stream alone; you read the evidence and make a call.
  * **Investigate** (`viewer.app`) — seeded. Start from a few suspect accounts (a triage core, a
    known ring, or any account) and the engine recovers the rest of the ring.

The world picker and the ground-truth toggle live in the shared sidebar, so both pages always look
at the same world. A triage suspect's core hands off to Investigate as its seeds.
"""

from __future__ import annotations

import argparse
import sys

import streamlit as st

from fraudrings.viewer import app as investigate
from fraudrings.viewer import app_triage as triage
from fraudrings.viewer import data as D
from fraudrings.viewer import session as S
from fraudrings.viewer import theme as T

LEVELS = ["naive", "moderate", "sophisticated"]
PRESETS = ["default", "smoke"]
# Operator-concurrency dial (lock-step worlds). "off" = the normal world.
CONCURRENCY_OPTS: dict[str, int | None] = {"off": None, "4": 4, "12": 12, "30": 30, "60": 60}

AUTHOR = "Stephen Corgiat"
SITE_URL = "https://stephen-corgiat.github.io/fraud-ring-detection/"
REPO_URL = "https://github.com/stephen-corgiat/fraud-ring-detection"


def _cli_args() -> tuple[str, bool]:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--rebuild", action="store_true",
                   help="Force-regenerate every world loaded this session (e.g. after a generator "
                        "change). Stale worlds from an older schema auto-rebuild regardless.")
    args, _ = p.parse_known_args(sys.argv[1:])
    return args.data_dir, args.rebuild


HOSTED = S.hosted()  # read once per script run; `streamlit_app.py` sets the env before importing


# Hosted: bound the cache so visitors browsing many worlds can't exhaust the host's memory.
@st.cache_resource(show_spinner="Building world (generate → ingest)…",
                   max_entries=S.HOSTED_MAX_WORLDS if HOSTED else None)
def _world(preset: str, level: str, seed: int, cotarget: bool, base_dir: str,
           concurrency: int | None, rebuild: bool) -> D.World:
    return D.load_world(
        preset, level, seed, cotarget_on=cotarget, base_dir=base_dir,  # type: ignore[arg-type]
        concurrency=concurrency, rebuild=rebuild,
    )


def _q(name: str, options: list[str], default: str) -> int:
    """Index of a deep-linked query-param value among `options` (e.g. `?level=sophisticated`)."""
    value = st.query_params.get(name, default)
    return options.index(value) if value in options else options.index(default)


def _q_int(name: str, default: int) -> int:
    try:
        return int(st.query_params.get(name, default))
    except (TypeError, ValueError):
        return default


def _world_sidebar(base_dir: str, rebuild: bool) -> None:
    # Every world control can be deep-linked (`?preset=default&level=moderate&seed=7&conc=4&truth=0`),
    # so a view can be shared or captured exactly; the query only sets the initial value.
    with st.sidebar:
        T.eyebrow("World")
        preset = st.selectbox("preset", PRESETS, index=_q("preset", PRESETS, "default"), key="w_preset")
        level = st.selectbox("evasion level", LEVELS, index=_q("level", LEVELS, "naive"), key="w_level",
                             help="How hard the rings try to look organic. Each level is a fresh world.")
        world_seed = int(st.number_input("world seed (RNG)", value=_q_int("seed", 7),
                                         step=1, key="w_seed",
                                         help="Reproducibility dial for the synthetic world. "
                                              "Distinct from the investigation *seeds* (accounts)."))
        cotarget = st.checkbox("co-target widening", value=True, key="w_cotarget",
                               help="Widen retrieval on the shared rare target (needed at sophisticated).")
        concurrency = CONCURRENCY_OPTS[st.selectbox(
            "operator concurrency (lock-step)", list(CONCURRENCY_OPTS),
            index=_q("conc", list(CONCURRENCY_OPTS), "off"), key="w_conc",
            help="'off' = the normal world. Set C to build a LOCK-STEP world: each ring is a finite "
                 "operator with C lanes (≤C members active at once → a flat occupancy plateau), while a "
                 "benign bandwagon piles on uncapped (a tall peak). Enables the temporal raster.",
        )]
        show_truth = st.checkbox("show ground truth", value=st.query_params.get("truth", "1") != "0",
                                 key="w_truth",
                                 help="Reveal hidden labels. Turn OFF to label blind (the honest test).")

    world = _world(preset, level, world_seed, cotarget, base_dir, concurrency, rebuild)
    if st.session_state.get("_world_dir") != world.world_dir:  # a new world invalidates the handoff
        st.session_state["_world_dir"] = world.world_dir
        st.session_state.pop("handoff", None)
    st.session_state["world"] = S.session_world(world)
    st.session_state["show_truth"] = show_truth


def main() -> None:
    st.set_page_config(page_title="fraudrings · fraud-ring detection demo", layout="wide",
                       menu_items={"About": f"Fraud-ring detection demo by {AUTHOR}. {SITE_URL}"})
    T.inject()
    T.register_plotly_template()
    if HOSTED:  # visitors arrive cold (often from a search or a shared link): say what this is
        T.intro(SITE_URL, REPO_URL)
    base_dir, rebuild = _cli_args()

    pages = {
        "triage": st.Page(triage.page, title="Triage", icon="🗂️", url_path="triage", default=True),
        "investigate": st.Page(investigate.page, title="Investigate", icon="🔎", url_path="investigate"),
    }
    st.session_state["_pages"] = pages
    nav = st.navigation(list(pages.values()))
    _world_sidebar(base_dir, rebuild)
    nav.run()
    with st.sidebar:  # after the page, so it lands below the page's own sidebar controls
        T.credit(AUTHOR, SITE_URL, REPO_URL)


if __name__ == "__main__":
    main()
