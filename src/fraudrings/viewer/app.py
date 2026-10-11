"""The Investigate page — seeded profile / investigation viewer.

Part of `fraudrings view` (needs the [viz] extra; see `viewer.main`). Pick a world, seed an investigation on a known
ring (or any account), and inspect each related account: its anomaly profile vs the population,
its activity over time, the engine's score breakdown, the shared rare identifiers that linked it,
and — toggleable — the ground-truth verdict. Label accounts fraud/legit/unsure; the sidebar grades
your labels against the truth.

This module owns layout + interaction only. Presentation chrome (top bar, breadcrumb, section
headers, score-composition bars, diverging z-bars, scoreboard, badges) lives in `viewer.theme`;
every query lives in `viewer.data`; Plotly figures in `viewer.charts`; labels in `viewer.labels`.
Everything stays within Streamlit + Plotly: static panels are CSS + HTML, while every control
remains a Streamlit widget. Charts are presentation-only — nothing here feeds detector scoring.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from fraudrings.config import preset
from fraudrings.viewer import charts as C
from fraudrings.viewer import data as D
from fraudrings.viewer import data_triage as R
from fraudrings.viewer import labels as L
from fraudrings.viewer import session as S
from fraudrings.viewer import theme as T
from fraudrings.viewer.raster import RasterData

BUCKET_MINUTES = [1, 2, 5, 10, 15]
# Cap accounts shown on the per-account EDA charts (bars/heatmap rows/graph nodes) so they stay
# legible; seeds come first, then top-scored candidates.
CHART_SET_CAP = 40


def _query_int(name: str) -> int | None:
    try:
        return int(st.query_params[name]) if name in st.query_params else None
    except (TypeError, ValueError):
        return None


def _deep_link_handoff(world: D.World) -> None:
    """`?creator=N` hands creator N's dedicated core to this page as seeds, exactly as the Triage
    page's "Investigate this core" button does — so a handoff can be bookmarked or captured."""
    creator = _query_int("creator")
    if creator is None or st.session_state.get("_linked_creator") == (world.world_dir, creator):
        return
    st.session_state["_linked_creator"] = (world.world_dir, creator)
    suspect = next((s for s in R.suspects(world.con, preset(world.preset_name).concentration)
                    if s.target_id == creator), None)
    if suspect is not None and suspect.core:
        st.session_state["handoff"] = {"target_id": creator, "core": list(suspect.core)}
        st.session_state["seed_mode"] = "from triage"


def _label(store: L.LabelStore, account_id: int, verdict: L.Verdict) -> None:
    store.set(account_id, verdict)
    st.rerun()


def _weights(world: D.World) -> dict[str, float]:
    """Live 4-axis score weights pulled from the engine config (never hard-coded)."""
    e = world.engine
    return {"linkage": e.w_linkage, "behavioral": e.w_behavioral,
            "cotarget": e.w_cotarget, "actionmix": e.w_actionmix}


def page() -> None:
    """The Investigate page. The world and the truth toggle come from the shared sidebar
    (`viewer.main`)."""
    world: D.World = st.session_state["world"]
    show_truth: bool = st.session_state["show_truth"]
    level = world.level
    store = S.label_store(world.world_dir)

    # --- top bar (mirrors the sidebar ground-truth state) ------------------
    T.top_bar(world.n_accounts, world.n_fraud, show_truth)

    # --- sidebar: subject selection ---------------------------------------
    ring_infos = D.rings(world)
    with st.sidebar:
        st.divider()
        T.section_header(1, "Seeds", "— the input", state="on")
        st.caption(
            "Accounts you **already** suspect. The engine starts here and recovers **related** "
            "accounts. Seeds are inputs — not the result."
        )
        _deep_link_handoff(world)
        handoff = st.session_state.get("handoff")
        modes = (["from triage"] if handoff else []) + ["from a known ring", "by account id"]
        mode = st.radio(
            "choose seeds", modes, horizontal=True, key="seed_mode",
            help="Seed from a ground-truth fraud ring (a few known members → recover the rest), or "
                 "from any single account id you want to dig into.",
        )

        if mode == "from triage" and handoff:
            st.caption(f"The dedicated core of suspect **creator {handoff['target_id']}**, handed over "
                       "from the Triage page. The engine expands from these accounts.")
            seeds = st.multiselect("seed accounts", handoff["core"], default=handoff["core"][:5])
            # The reference ring for the truth tags (eval side only): the ring most of the seeds belong
            # to, so recovered members read "this-ring" rather than "other-ring".
            seed_rings = [g["ring_id"] for a in seeds
                          if (g := D.ground_truth_of(world, a)) and g["ring_id"] is not None]
            ring_id = max(set(seed_rings), key=seed_rings.count) if seed_rings else None
        elif mode == "from a known ring" and ring_infos:
            opts = {f"ring {r.ring_id} · size {r.size} · {r.evasion_level}": r.ring_id for r in ring_infos}
            linked = _query_int("ring")
            ring_index = next((i for i, r in enumerate(ring_infos) if r.ring_id == linked), 0)
            chosen = st.selectbox(
                "ring", list(opts), index=ring_index,
                help="A ground-truth fraud ring to investigate. Its first few members become the "
                     "seeds; the rest are 'held out' so you can see how many the engine recovers.",
            )
            ring_id = opts[chosen]
            members = D.ring_members(world, ring_id)
            default = D.default_seeds(world, ring_id)
            seeds = st.multiselect(
                "seed accounts", members, default=default,
                help="The known members handed to the engine. The other ring members are held back "
                     "as the recovery target. Changing this resets the profile to the first seed.",
            )
        else:
            ring_id = None
            aid = int(st.number_input(
                "account id", value=0, step=1,
                help="Type any account id to make it the investigation subject. The profile and the "
                     "related-accounts list both update to this account when you change it.",
            ))
            seeds = [aid] if D.account_exists(world, aid) else []
            if seeds:
                gt = D.ground_truth_of(world, aid)
                ring_id = gt["ring_id"] if gt else None
            elif aid:
                st.warning(f"account {aid} has no events")

    st.session_state["seeds"] = seeds
    # focus = the profiled account. Follows the sidebar *subject* (typed account, or a ring's
    # primary seed) and resets whenever that selection changes. Drilling into a candidate ("view
    # profile") changes focus WITHOUT changing the subject, so it persists across reruns until the
    # sidebar subject changes again.
    subject_key = f"{mode}|{ring_id}|{tuple(seeds)}"
    if st.session_state.get("_subject_key") != subject_key:
        st.session_state["_subject_key"] = subject_key
        st.session_state["focus"] = seeds[0] if seeds else None

    # --- sidebar: label scoreboard ----------------------------------------
    with st.sidebar:
        st.divider()
        T.eyebrow("Your labels")
        counts = store.counts()
        T.label_counts(counts["fraud"], counts["legit"], counts["unsure"])
        if show_truth:
            s = store.score(world.gt)
            if s.n_decided:
                T.kpi_cards(s.precision, s.recall, s.accuracy, s.tp, s.fp, s.tn, s.fn)
            else:
                st.caption("label some accounts to grade vs truth")
        else:
            st.caption("blind mode — turn on *show ground truth* to grade your labels")

    if not seeds:
        st.info("Pick a ring or a valid account id in the sidebar (① Seeds) to begin.")
        return

    # --- main: breadcrumb + sections --------------------------------------
    focus = st.session_state["focus"] or seeds[0]
    focus_is_seed = focus in seeds
    if mode == "from triage" and handoff:
        subject = f"Creator {handoff['target_id']} core · {level}"
    elif ring_id is not None and mode == "from a known ring":
        subject = f"Ring {ring_id} · {level}"
    else:
        subject = f"account {seeds[0]}"
    if focus_is_seed:
        seed_pos = (f"profiling {seeds.index(focus) + 1} of {len(seeds)} seeds — "
                    "click a chip or a candidate's view to switch")
    else:
        seed_pos = "a related account you opened — click a seed chip to go back"
    T.breadcrumb(subject, focus, focus_is_seed, seed_pos)

    _render_seed_bar(seeds, focus)
    _render_profile(world, store, focus, ring_id, show_truth)

    st.divider()
    _render_investigation(world, store, seeds, ring_id, show_truth)

    st.divider()
    _render_eda(world, seeds, ring_id, show_truth)

    if world.concurrency is not None:
        st.divider()
        _render_temporal(world, seeds, ring_id, show_truth)


def _render_seed_bar(seeds: list[int], focus: int) -> None:
    """The seed set as clickable chips, highlighting the one currently profiled — makes it visceral
    that seeds are a *set* of inputs and the profile shows one of them at a time."""
    T.section_header(1, "Seeds", f"— {len(seeds)} account(s) you're investigating (click one to profile it)",
                     state="off")
    cols = st.columns(min(len(seeds), 8) or 1)
    for i, s in enumerate(seeds):
        marker = "▶ " if s == focus else ""
        kind = "primary" if s == focus else "secondary"
        if cols[i % len(cols)].button(f"{marker}account {s}", key=f"seedbtn{s}", type=kind):
            st.session_state["focus"] = s
            st.rerun()


def _render_profile(world: D.World, store: L.LabelStore, account_id: int,
                    ring_id: int | None, show_truth: bool) -> None:
    ov = D.account_overview(world, account_id)
    if ov is None:
        st.warning(f"account {account_id} has no events")
        return

    head_l, head_r = st.columns([3, 2])
    with head_l:
        badge = ""
        if show_truth:
            badge = "  " + T.verdict_badge(D.truth_tag(world, account_id, ring_id))
        st.markdown(
            f'<div class="ath-sec"><span class="num on">2</span>'
            f'<span class="ttl">Profile · <span class="mono">account {account_id}</span></span>'
            f"{badge}</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"{int(ov['n_events'])} events · {int(ov['n_views'])} views · "
            f"{int(ov['n_skips'])} skips · {int(ov['n_sessions'])} sessions · "
            f"{int(ov['n_items'])} items"
        )
    with head_r:
        rec = store.get(account_id)
        verdict = f"current: `{rec['verdict']}`" if rec else "*unlabeled*"
        st.markdown(f"verdict on **account {account_id}** — {verdict}")
        b1, b2, b3, b4 = st.columns(4)
        if b1.button("🟥 fraud", key=f"f{account_id}", type="primary",
                     help=f"Mark account {account_id} as fraud."):
            _label(store, account_id, "fraud")
        if b2.button("🟩 legit", key=f"l{account_id}", help=f"Mark account {account_id} as a real viewer."):
            _label(store, account_id, "legit")
        if b3.button("❔ unsure", key=f"u{account_id}", help="Record uncertainty (excluded from scoring)."):
            _label(store, account_id, "unsure")
        if b4.button("✖ clear", key=f"c{account_id}", help=f"Remove the label on account {account_id}."):
            store.clear(account_id)
            st.rerun()

    left, right = st.columns([1.15, 1])
    with left:
        pop = D.population_stats(world)
        rows = [{"feature": a.feature, "z": round(a.z, 2)} for a in D.anomalies(ov, pop)]
        T.anomaly_bars(rows)
        st.caption("z = (value − mean) / std — std-deviations from a typical account. These are "
                   "exactly the features the detector scores on.")
    with right:
        T.card_title("Daily activity", "day 0 = world's first event")
        act = D.daily_activity(world, account_id)
        if act:
            st.plotly_chart(C.daily_activity_figure(act), width="stretch", key="profile-activity")
        T.card_title("Top creators", "qualified = past the payout threshold")
        ta = D.top_creators(world, account_id)
        if ta:
            T.mini_table(
                [("CREATOR", False), ("VIEWS", True), ("QUAL", True)],
                [[f"creator {r['creator_id']}", str(r["views"]), str(r["qualified"])] for r in ta],
                dim_cols={2},
            )


def _render_investigation(world: D.World, store: L.LabelStore, seeds: list[int],
                          ring_id: int | None, show_truth: bool) -> None:
    T.section_header(3, "Related accounts", f"— recovered from seeds {seeds}", state="off")
    st.caption(
        "The engine's **output**: accounts linked to your seeds via shared rare identifiers / "
        "target creators. Runs blind on events — never ground truth."
    )
    candidates = D.investigate(world, seeds)
    if not candidates:
        st.info("No related candidates retrieved for this seed set.")
        return

    weights = _weights(world)
    T.axis_legend(weights)

    top = candidates[: min(25, len(candidates))]
    rows = []
    for rank, c in enumerate(top, 1):
        lab = store.get(c.account_id)
        rows.append({
            "rank": rank, "account": c.account_id, "score": c.score,
            "contrib": {
                "linkage": weights["linkage"] * c.linkage,
                "behavioral": weights["behavioral"] * c.behavioral,
                "cotarget": weights["cotarget"] * c.cotarget,
                "actionmix": weights["actionmix"] * c.actionmix,
            },
            "truth": D.truth_tag(world, c.account_id, ring_id) if show_truth else None,
            "label": (lab["verdict"] if lab else ""),
        })
    T.score_table(rows, show_truth=show_truth)
    wtxt = (f"{weights['linkage']:.2f}·linkage + {weights['behavioral']:.2f}·behavioral + "
            f"{weights['cotarget']:.2f}·cotarget + {weights['actionmix']:.2f}·actionmix")
    st.caption(f"{len(candidates)} candidates retrieved · showing top {len(top)}. "
               f"Score = {wtxt} (current 4-axis weights).")

    # --- candidate inspector ----------------------------------------------
    ids = [c.account_id for c in top]
    insp_l, insp_r = st.columns(2)
    with insp_l:
        T.card_title("Inspect a candidate")
        pick = st.selectbox(
            "candidate", ids, format_func=lambda a: f"account {a}", label_visibility="collapsed",
            help="Pick a candidate to see the rare identifiers that linked it and to label it, or "
                 "open its full profile up top with 'view profile'.",
        )
        ic1, ic2, ic3 = st.columns(3)
        if ic1.button("🔍 view profile", help=f"Load account {pick}'s full profile in the panel above."):
            st.session_state["focus"] = pick
            st.rerun()
        if ic2.button("🟥 fraud", key=f"if{pick}", type="primary", help=f"Mark account {pick} as fraud."):
            _label(store, pick, "fraud")
        if ic3.button("🟩 legit", key=f"il{pick}", help=f"Mark account {pick} as a real viewer."):
            _label(store, pick, "legit")
    with insp_r:
        T.card_title("Shared rare identifiers", "the linkage evidence")
        shared = D.shared_with_group(world, pick, seeds)
        if not shared:
            st.caption("No shared identifiers — this candidate was retrieved by co-target, not infra "
                       "linkage (it must earn its rank on behavioral + co-target).")
        else:
            T.mini_table(
                [("TYPE", False), ("VALUE", False), ("IDF", True), ("SHARE", True), ("GATE", True)],
                [[s.id_type, s.id_value, f"{s.rarity:.2f}", str(s.distinct_accounts),
                  "✓" if s.in_gate else "·"] for s in shared],
                dim_cols={3},
            )
            st.caption("`in_rarity_gate` (GATE ✓) marks identifiers rare enough to be evidence — "
                       "only those carry linkage score.")


# --- axis explainers + EDA (the Plotly chart spine) -------------------------


def _selected_ids(event: object) -> list[int]:
    """Pull account ids out of a Plotly lasso/box-select event (each point carries its id in
    `customdata`). Tolerant of streamlit's dict/attr selection shapes."""
    sel = event.get("selection") if isinstance(event, dict) else getattr(event, "selection", None)
    pts = (sel or {}).get("points", []) if isinstance(sel, dict) else getattr(sel, "points", [])
    out: list[int] = []
    for p in pts or []:
        cd = p.get("customdata") if isinstance(p, dict) else getattr(p, "customdata", None)
        if cd is None:
            continue
        v = cd[0] if isinstance(cd, (list, tuple)) else cd
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            continue
    return sorted(set(out))


def _render_eda(world: D.World, seeds: list[int], ring_id: int | None, show_truth: bool) -> None:
    T.section_header(4, "Axis explainers & EDA", "— presentation-only · lasso to re-seed", state="off")
    cands = D.investigate(world, seeds)
    set_ids = list(dict.fromkeys([*seeds, *(c.account_id for c in cands)]))[:CHART_SET_CAP]
    st.caption(
        f"On-the-fly over the candidate set — {len(set_ids)} accounts "
        f"({len(seeds)} seed(s) + recovered, capped at {CHART_SET_CAP}). "
        "Faint grey is the population baseline; red is this set. Nothing here feeds scoring."
    )
    selected: set[int] = set()
    tab_axes, tab_eda = st.tabs(["explain scored axes", "EDA · new substrate"])

    with tab_axes:
        T.card_title("Action composition", "explains the action-mix axis · lasso accounts to re-seed")
        ev = st.plotly_chart(
            C.action_composition_figure(D.action_composition(world, set_ids)),
            width="stretch", on_select="rerun", key="eda-actioncomp",
        )
        selected |= set(_selected_ids(ev))

        c1, c2 = st.columns(2)
        with c1:
            T.card_title("Reported position %", "payout threshold dashed")
            st.plotly_chart(
                C.position_histogram_figure(
                    D.position_samples(world, set_ids), D.position_samples(world),
                    world.engine.qualified_view_threshold,
                ),
                width="stretch", key="eda-pos",
            )
        with c2:
            T.card_title("Co-target heatmap", "q-views per top creator")
            accs, arts, mat = D.cotarget_matrix(world, set_ids)
            if arts:
                st.plotly_chart(C.cotarget_heatmap_figure(accs, arts, mat),
                                width="stretch", key="eda-cotarget")
            else:
                st.caption("No qualified views in the set to co-target on.")

        T.card_title("Linkage subgraph", "shared rare-identifier graph · lasso nodes to re-seed")
        nodes, edges = D.linkage_graph(world, set_ids)
        ev2 = st.plotly_chart(
            C.linkage_graph_figure(nodes, edges),
            width="stretch", on_select="rerun", key="eda-linkage",
        )
        selected |= set(_selected_ids(ev2))

    with tab_eda:
        T.card_title("Account age", "ring clusters low + tight vs the broad legit cloud · lasso to re-seed")
        ev3 = st.plotly_chart(
            C.account_age_strip_figure(
                D.account_age_samples(world, set_ids), D.account_age_samples(world)
            ),
            width="stretch", on_select="rerun", key="eda-age",
        )
        selected |= set(_selected_ids(ev3))

        c3, c4 = st.columns(2)
        with c3:
            T.card_title("Registration burst", "in-window co-registration spike")
            st.plotly_chart(C.registration_burst_figure(D.registration_offsets(world, set_ids)),
                            width="stretch", key="eda-reg")
        with c4:
            T.card_title("Inter-event gaps", "cadence comb, log seconds")
            st.plotly_chart(
                C.gap_histogram_figure(D.gap_samples(world, set_ids), D.gap_samples(world)),
                width="stretch", key="eda-gap",
            )

    if selected:
        st.session_state["eda_selection"] = sorted(selected)
    _render_reseed(world, st.session_state.get("eda_selection", []), ring_id, show_truth)


def _render_reseed(world: D.World, selected: list[int], ring_id: int | None, show_truth: bool) -> None:
    st.divider()
    st.markdown('<div class="ath-sec"><span class="num off">↻</span>'
                '<span class="ttl">Select-to-reseed</span>'
                '<span class="desc">— run the engine on your selection (no new machinery)</span></div>',
                unsafe_allow_html=True)
    if not selected:
        st.caption("Select accounts on the action-composition, linkage, or account-age chart above "
                   "to seed a fresh investigation on them.")
        return
    valid = [a for a in selected if D.account_exists(world, a)]
    st.caption(f"Selected {len(valid)} account(s): {valid}")
    bc1, bc2, _ = st.columns([1, 1, 3])
    if bc1.button("↻ investigate selection", key="reseed-btn", type="primary", disabled=not valid):
        st.session_state["reseed_run"] = valid
    if bc2.button("✖ clear selection", key="reseed-clear"):
        st.session_state.pop("eda_selection", None)
        st.session_state.pop("reseed_run", None)
        st.rerun()

    run_seeds = st.session_state.get("reseed_run")
    if not run_seeds:
        return
    recands = D.investigate(world, run_seeds)
    if not recands:
        st.info(f"No related candidates retrieved for the re-seed {run_seeds}.")
        return
    weights = _weights(world)
    top = recands[: min(15, len(recands))]
    rows = []
    for rank, c in enumerate(top, 1):
        rows.append({
            "rank": rank, "account": c.account_id, "score": c.score,
            "contrib": {
                "linkage": weights["linkage"] * c.linkage,
                "behavioral": weights["behavioral"] * c.behavioral,
                "cotarget": weights["cotarget"] * c.cotarget,
                "actionmix": weights["actionmix"] * c.actionmix,
            },
            "truth": D.truth_tag(world, c.account_id, ring_id) if show_truth else None,
            "label": "",
        })
    st.caption(f"Re-seeded on {run_seeds} → {len(recands)} candidates (top {len(top)}):")
    T.score_table(rows, show_truth=show_truth)


# --- temporal raster + occupancy view --------------------------------------


def _raster_image(rd: RasterData, color: str) -> np.ndarray:
    """An RGB (H, W, 3) image on the dark canvas: a bucket with events takes the panel's color, an
    empty one is background. Rows are accounts in activity-onset order (event-derivable), so no
    ground-truth ordering leaks."""
    present = rd.matrix() > 0  # binary presence reads cleanest at a glance
    fg = np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.uint8)
    img = np.empty((*present.shape, 3), dtype=np.uint8)
    img[:] = (19, 23, 29)  # card background
    img[present] = fg
    return img


def _draw_raster_panel(rd: RasterData, *, title: str, cap: int | None, color: str = "#4c78a8") -> None:
    if rd.n_accounts == 0:
        st.caption(f"{title}: no events.")
        return
    st.markdown(f"**{title}** — {rd.n_accounts} accounts · peak simultaneity **{rd.peak_occupancy}**"
                + (f" (cap C={cap})" if cap is not None else ""))
    st.plotly_chart(C.occupancy_figure(rd, cap, color), width="stretch", key=f"occ-{title}")
    st.image(_raster_image(rd, color), width="stretch",
             caption="event raster · rows = accounts (activity-onset order) · X = time bucket")


def _render_temporal(world: D.World, seeds: list[int], ring_id: int | None, show_truth: bool) -> None:
    T.section_header(5, "Temporal raster", "— lock-step view: occupancy plateau vs peak", state="off")
    st.caption(
        "Occupancy = accounts active **at once** per bucket. A serialized ring takes turns on C "
        "lanes → a flat plateau hugging the red cap line; an uncapped bandwagon piles on → a tall "
        "peak. Raster rows are ordered by activity onset (event-derivable, never ground truth)."
    )
    bucket_min = st.select_slider("time bucket (minutes)", BUCKET_MINUTES, value=5)
    bucket_s = bucket_min * 60

    if show_truth and ring_id is not None:
        clusters = D.bandwagon_clusters(world)
        if clusters:
            # Oracle-clean view: serialized ring beside an uncapped bandwagon, on a SHARED time axis
            # so the plateau-vs-peak contrast is directly comparable.
            ring_mem = D.ring_members(world, ring_id)
            band_mem = D.bandwagon_members(world, clusters[0].cluster_id)
            t0, t1 = _shared_window(world, ring_mem, band_mem)
            st.caption("Oracle-clean view (ground truth on): the serialized ring vs a benign "
                       "bandwagon, same time axis. Compare the occupancy shapes.")
            lc, rc = st.columns(2)
            with lc:
                rd = D.account_raster(world, ring_mem, bucket_seconds=bucket_s, t0=t0, t1=t1)
                _draw_raster_panel(rd, title=f"ring {ring_id} (serialized)", cap=world.concurrency,
                                   color="#d65f5f")
            with rc:
                rd = D.account_raster(world, band_mem, bucket_seconds=bucket_s, t0=t0, t1=t1)
                _draw_raster_panel(rd, title=f"bandwagon {clusters[0].cluster_id} (uncapped)",
                                   cap=world.concurrency, color="#4c78a8")
            return

    # Realistic view (ground truth off, or no bandwagon tag): the retrieved candidate set the
    # detector actually surfaced — contamination and all.
    cands = D.investigate(world, seeds)
    retrieved = list(dict.fromkeys([*seeds, *(c.account_id for c in cands)]))
    rd = D.account_raster(world, retrieved, bucket_seconds=bucket_s)
    st.caption("Retrieved set (seeds + related accounts) — the realistic, possibly-polluted view. "
               "Turn on ground truth to see the oracle-clean ring-vs-bandwagon comparison.")
    _draw_raster_panel(rd, title="retrieved candidate set", cap=world.concurrency)


def _shared_window(world: D.World, a: list[int], b: list[int]) -> tuple[int, int]:
    """Common [t0, t1] across two account sets so two rasters share one X axis."""
    ra = D.account_raster(world, a)
    rb = D.account_raster(world, b)
    lo = min(ra.t0, rb.t0)
    hi = max(ra.t0 + ra.n_buckets * ra.bucket_seconds, rb.t0 + rb.n_buckets * rb.bucket_seconds)
    return lo, hi


