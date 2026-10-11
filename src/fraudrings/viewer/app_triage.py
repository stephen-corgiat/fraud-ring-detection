"""The Triage page — seedless suspect review.

The concentration ranker (`fraudrings.detector.concentration`) reads only the event stream and
shortlists suspect **creators** whose qualified views concentrate: on few accounts, on many
dedicated single-creator accounts, or inside one week. This page shows the evidence behind each
rank so a human can try to separate a coordinated ring from a benign superfan cohort. Calls are
graded against the hidden truth when the truth toggle is on.

A suspect's dedicated core can be handed to the Investigate page as seeds; that is the seedless →
seeded handoff. Layout and interaction only: queries live in `viewer.data_triage`, figures in
`viewer.charts`, chrome in `viewer.theme`. Nothing here feeds detector scoring.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fraudrings.config import ConcentrationConfig, preset
from fraudrings.detector.concentration import SuspectTarget, maybe_expand
from fraudrings.viewer import charts as C
from fraudrings.viewer import data as D
from fraudrings.viewer import data_triage as R
from fraudrings.viewer import session as S
from fraudrings.viewer import theme as T


@st.cache_data(show_spinner="ranking creators by concentration…")
def _suspects(world_dir: str, sort_by: str, cap: int, impact_floor: int, _con) -> list[SuspectTarget]:
    cfg = ConcentrationConfig(sort_by=sort_by, cap=cap, impact_floor=impact_floor)  # type: ignore[arg-type]
    return R.suspects(_con, cfg)


@st.cache_data(show_spinner=False)
def _truth(world_dir: str, _world: D.World) -> dict[int, R.TargetTruth]:
    return R.target_truth(_world)


@st.cache_data(show_spinner=False)
def _account_ring(world_dir: str, _world: D.World) -> dict[int, int | None]:
    return R.account_ring(_world)


@st.cache_data(show_spinner=False)
def _contrib(world_dir: str, target_id: int, _con):
    return R.top_contributors(_con, target_id, 20)


@st.cache_data(show_spinner=False)
def _weekly(world_dir: str, target_id: int, _con):
    return R.weekly_qv(_con, target_id)


@st.cache_data(show_spinner=False)
def _core(world_dir: str, target_id: int, core: tuple[int, ...], _con):
    return R.core_stats(_con, core, target_id)


@st.cache_data(show_spinner="checking whether the core shares rare infrastructure…")
def _expansion(world_dir: str, core: tuple[int, ...], _world: D.World):
    return maybe_expand(_world.con, core, _world.engine)


def _evidence_read(s: SuspectTarget) -> tuple[str, list[str]]:
    """A plain-language read of the evidence plus a three-way lean. The lean is a cue, never a
    verdict. A single dedicated account is the irreducible superfan≈booster confound: one devoted
    fan and one lone operator produce the same contribution shape, so it reads as **ambiguous**."""
    cues: list[str] = []
    if s.top1_share >= 0.60:
        cues.append(f"one contributor = **{s.top1_share:.0%}** of qualified views — single-source")
    elif s.top3_share >= 0.60:
        cues.append(f"top 3 contributors = **{s.top3_share:.0%}** — few-source")
    if s.booster_count >= 3:
        cues.append(f"**{s.booster_count}** dedicated boosters — coordination signal")
    if s.burst7 >= 0.50:
        cues.append(f"**{s.burst7:.0%}** of qualified views in one week — campaign")

    coordinated = s.booster_count >= 3 or s.burst7 >= 0.50
    single_user = s.top1_share >= 0.60 or s.top3_share >= 0.60
    if coordinated:
        lean = "leans **coordinated / ring-shaped**"
    elif single_user:
        lean = ("**ambiguous** — a few dedicated accounts: a genuine superfan and a lone operator "
                "look identical here (a label is needed)")
    else:
        lean = "leans **organic** — concentrated, but no dedicated core and no campaign window"
    if not cues:
        cues.append(f"low effective contributor count (n_eff {s.n_eff:.0f}) relative to its peers, "
                    "with no boosters and no burst — a small, loyal audience")
    return lean, cues


def _queue_df(suspects: list[SuspectTarget], store: R.SuspectLabelStore, top_n: int,
              truth: dict[int, R.TargetTruth] | None) -> pd.DataFrame:
    rows = []
    for i, s in enumerate(suspects[:top_n], 1):
        lab = store.get(s.target_id)
        row = {
            "rank": i, "creator": s.target_id, "suspicion": round(s.suspicion, 3),
            "tags": ", ".join(s.tags) or "—", "n_eff": round(s.n_eff, 1),
            "boosters": s.booster_count, "burst7": round(s.burst7, 2),
            "capped qv": s.capped_qv, "label": (lab["verdict"] if lab else "—"),
        }
        if truth is not None:
            t = truth.get(s.target_id)
            row["truth"] = (f"ring {', '.join(map(str, t.ring_ids))}" if t and t.is_ring_target else "organic")
        rows.append(row)
    return pd.DataFrame(rows)


def _sidebar(world: D.World, store: R.SuspectLabelStore, truth: dict[int, R.TargetTruth] | None
             ) -> tuple[str, int, int, int]:
    with st.sidebar:
        st.divider()
        T.eyebrow("Ranker")
        sort_by = st.radio("rank by", ["suspicion", "rank_score"], index=0, horizontal=True,
                           help="suspicion = pure concentration (max of the three entry-axis "
                                "percentiles); rank_score = suspicion × log10(capped qv), impact-first.")
        cap = int(st.number_input("payout cap (per account/day)", value=10, min_value=1, step=1,
                                  help="Views over the cap earn nothing, so a lone high-volume bot is "
                                       "deprioritized on impact."))
        impact_floor = int(st.number_input(
            "impact floor (min qualified views)", value=preset(world.preset_name).concentration.impact_floor,
            min_value=1, step=50, help="Creators below this volume are not payout-relevant."))
        top_n = int(st.number_input("show top N", value=30, min_value=5, max_value=500, step=5))

        st.divider()
        T.eyebrow("Your calls")
        counts = store.counts()
        st.caption(f"ring **{counts['ring']}** · legit **{counts['legit']}** · "
                   f"unsure **{counts['unsure']}** · escalated **{counts['escalate']}**")
        if truth is not None:
            sc = store.score(truth)
            if sc.n_decided:
                T.kpi_cards(sc.precision, sc.recall, sc.accuracy, sc.tp, sc.fp, sc.tn, sc.fn)
            else:
                st.caption("call some suspects ring or legit to grade them against the truth")
        else:
            st.caption("blind mode — turn on *show ground truth* to grade your calls")
    return sort_by, cap, impact_floor, top_n


def page() -> None:
    world: D.World = st.session_state["world"]
    show_truth: bool = st.session_state["show_truth"]
    wd = world.world_dir
    store = S.suspect_label_store(wd)
    truth = _truth(wd, world) if show_truth else None
    sort_by, cap, impact_floor, top_n = _sidebar(world, store, truth)

    T.top_bar(world.n_accounts, world.n_fraud, show_truth, page="triage")
    suspects = _suspects(wd, sort_by, cap, impact_floor, world.con)
    if not suspects:
        st.info("No creator clears the impact floor. Lower it in the sidebar.")
        return

    # --- headline numbers ---------------------------------------------------
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("suspect creators", f"{len(suspects):,}", help="every creator above the impact floor, ranked")
    m2.metric("top tag", (suspects[0].tags[0] if suspects[0].tags else "—"))
    if truth is not None:
        n_targets = sum(1 for t in truth.values() if t.is_ring_target)
        k = 2 * n_targets
        hit = sum(1 for s in suspects[:k] if truth.get(s.target_id, R.TargetTruth((), 0)).is_ring_target)
        m3.metric(f"ring targets in top {k}", f"{hit} / {n_targets}",
                  help="How many of the creators the rings were paid to boost the ranker put near the head.")
    else:
        m3.metric("ring targets in top k", "hidden")
    m4.metric("evasion level", world.level)

    # --- ① the queue ----------------------------------------------------------
    T.section_header(1, "Triage queue", "— suspects ranked from the event stream alone (leads, not verdicts)",
                     state="on")
    st.dataframe(_queue_df(suspects, store, top_n, truth), hide_index=True, width="stretch",
                 height=340)

    ranks = list(range(1, min(top_n, len(suspects)) + 1))
    try:  # deep-link a suspect via ?rank=N (shareable; also lets a screenshot target any suspect)
        q_rank = int(st.query_params.get("rank", 1))
    except (TypeError, ValueError):
        q_rank = 1
    pick = st.selectbox(
        "inspect rank", ranks, index=(q_rank - 1 if 1 <= q_rank <= len(ranks) else 0),
        format_func=lambda r: (f"#{r} · creator {suspects[r - 1].target_id} · "
                               f"{', '.join(suspects[r - 1].tags) or 'no tags'}"),
    )
    s = suspects[pick - 1]

    # --- ② the evidence -------------------------------------------------------
    t = truth.get(s.target_id) if truth is not None else None
    badge = ""
    if t is not None:
        badge = (f" · truth: ring target of ring {', '.join(map(str, t.ring_ids))}" if t.is_ring_target
                 else f" · truth: organic ({t.fraud_qv_share:.0%} ring traffic)")
    T.section_header(2, f"Suspect · creator {s.target_id}", f"— rank #{pick}{badge}", state="on")
    lean, cues = _evidence_read(s)

    left, right = st.columns([1, 1])
    with left:
        T.card_title("Why it's here", "the three entry axes (tallest = the axis that fired)")
        st.plotly_chart(C.suspect_axis_figure(s.pct_neff, s.pct_booster, s.pct_burst7),
                        width="stretch", key="axes")
        k1, k2, k3 = st.columns(3)
        k1.metric("n_eff", f"{s.n_eff:.1f}", help="effective number of contributors (low = concentrated)")
        k2.metric("boosters", s.booster_count,
                  help="dedicated accounts: ≥50 qualified views here AND ≥80% of all their qualified "
                       "viewing on this one creator (the benign twin is the superfan)")
        k3.metric("burst7", f"{s.burst7:.0%}", help="largest share of qualified views in any 7-day window")
        k4, k5 = st.columns(2)
        k4.metric("capped qv", f"{s.capped_qv:,}", help="payout-relevant impact: Σ min(daily qv, cap)")
        k5.metric("raw qv", f"{s.tot_qv:,}",
                  delta=f"{(1 - s.capped_qv / s.tot_qv):.0%} over cap" if s.tot_qv > s.capped_qv else None,
                  delta_color="off", help="raw volume; a big gap vs capped = a lone high-volume bot")
        st.markdown(f"**Evidence read — {lean}** *(a cue, not a verdict — you decide):*")
        for c in cues:
            st.markdown(f"- {c}")

    ring_of = _account_ring(wd, world) if show_truth else None
    with right:
        T.card_title("Contributor concentration",
                     "top contributors by qualified views" + (" (red = ring account)" if show_truth else ""))
        st.plotly_chart(C.suspect_contributor_figure(_contrib(wd, s.target_id, world.con), ring_of),
                        width="stretch", key="contrib")
        T.card_title("Temporal concentration", "qualified views per 7-day window (red = peak week)")
        st.plotly_chart(C.suspect_weekly_figure(_weekly(wd, s.target_id, world.con)),
                        width="stretch", key="weekly")

    # --- ③ the core -------------------------------------------------------------
    core = _core(wd, s.target_id, s.core, world.con)
    T.section_header(3, "The core", f"— {len(core)} dedicated booster account(s)", state="on" if core else "off")
    if core:
        st.caption(
            "A **booster** puts ≥50 qualified views on this creator *and* ≥80% of all its qualified "
            "viewing here. Its benign twin is the **superfan**, a real devotee, which is why these stay "
            "suspects. Infra columns are corroborating evidence only: a shared ASN or subnet across the "
            "core leans ring; one or two residential subnets per account leans genuine viewer."
        )
        rows = []
        for c in core:
            row = {
                "account": c["account_id"], "qv on creator": c["qv_on_target"], "total qv": c["total_qv"],
                "single-creator share": round(c["share"], 2), "# creators": c["n_targets"],
                "# subnets": c["n_subnet"], "# ASNs": c["n_asn"],
                "top ASN": (str(c["top_asn"]) if c["top_asn"] is not None else "—"),
            }
            if ring_of is not None:
                r = ring_of.get(c["account_id"])
                row["truth"] = f"ring {r}" if r is not None else "legit"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     height=min(300, 40 + 35 * len(core)))

        expanded = _expansion(wd, tuple(s.core), world)
        if expanded is None:
            st.markdown("**Expansion gate: closed.** The core does not interlink on rare infrastructure, "
                        "so automatic expansion would fan out into noise. The core itself is the output.")
        else:
            st.markdown(f"**Expansion gate: open.** The core shares rare infrastructure, so the seeded "
                        f"engine expands it: **{len(expanded)}** related accounts recovered.")
        if st.button("🔎 Investigate this core →", type="primary"):
            st.session_state["handoff"] = {"target_id": s.target_id, "core": list(s.core)}
            st.session_state["seed_mode"] = "from triage"
            st.switch_page(st.session_state["_pages"]["investigate"])
    else:
        st.caption("No dedicated boosters: this creator is on the list for concentration or timing alone.")

    # --- ④ your call --------------------------------------------------------------
    T.section_header(4, "Your call", "— label this suspect; the sidebar grades you against the truth",
                     state="on")
    existing = store.get(s.target_id)
    lc, rc = st.columns([2, 3])
    with lc:
        default = R.SUSPECT_VERDICTS.index(existing["verdict"]) if existing else 2  # 'unsure'
        verdict = st.radio("verdict", R.SUSPECT_VERDICTS, index=default, horizontal=True)
    with rc:
        note = st.text_input("note (optional)", value=(existing or {}).get("note", ""))
    b1, b2, _ = st.columns([1, 1, 4])
    if b1.button("save label", type="primary"):
        store.set(s.target_id, verdict, note=note, suspicion=s.suspicion, rank=pick)  # type: ignore[arg-type]
        st.rerun()
    if existing and b2.button("clear"):
        store.clear(s.target_id)
        st.rerun()
    if existing:
        st.caption(f"current: **{existing['verdict']}** · updated {existing['updated']}"
                   + (f" · “{existing['note']}”" if existing.get("note") else ""))
