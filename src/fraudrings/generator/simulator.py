"""Synthetic-world simulator: legit users, fraud rings, and benign look-alikes.

Produces a time-ordered event stream plus a separate ground-truth file. Three
populations:

  * legit       — diverse infra + behavior; a few coincidentally share a household
                  IP or a VPN/datacenter IP (noise that tests precision).
  * fraud rings — bots on a small shared datacenter IP/ASN pool, hammering a target
                  creator with regular timing and spoofed completion. How hard they
                  try to look organic is the evasion dial (`generator.evasion`).
  * benign fan-club — coordinated-looking (all view one release together) but on
                  diverse residential infra, and HONEST. A negative control: they
                  share behaviour/co-target but NOT rare infra, so Stage-1 must not
                  pull them into a ring.

Output is columnar Parquet written via Polars (fast); the schema matches
`fraudrings.schema` (validated on a sample in tests).
"""

from __future__ import annotations

import heapq
import os
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from fraudrings.config import SimConfig
from fraudrings.generator.evasion import EvasionProfile, profile_for

_HEX = np.array(list("0123456789abcdef"))
# Fallback only — every real call threads `cfg.complete_threshold`. Kept so `_emit_view` is usable
# stand-alone in tests without a cfg.
COMPLETE_THRESHOLD_DEFAULT = 0.90
# Enrichment draws (registration, lifecycle events, funnel search/save presence, the whole
# lurker segment) come from a SEPARATE rng seeded at cfg.seed + this offset. Isolating them keeps the
# core content stream (infra, timing, views, skips, positions — everything the OLD behavioral basket
# reads) independent of every enrichment knob, so turning enrichment features on or off cannot move
# the behavioral results. Only NEW funnel events are added to the stream.
ENRICH_SEED_OFFSET = 0x5747


def _rand_ip(rng: np.random.Generator) -> str:
    o = rng.integers(1, 255, size=3)
    return f"10.{o[0]}.{o[1]}.{o[2]}"


def _subnet(ip: str) -> str:
    a, b, c, _d = ip.split(".")
    return f"{a}.{b}.{c}"


def _rand_fp(rng: np.random.Generator) -> str:
    return "".join(_HEX[rng.integers(0, 16, size=12)].tolist())


def _draw_registration(
    rng: np.random.Generator,
    cfg: SimConfig,
    *,
    aged: bool,
    span: int,
    burst_anchor: int | None = None,
) -> tuple[int, int | None]:
    """Assign an account a `registration_ts` and, if it registered IN the observation window, the ts
    of its `register` event (else None — most legit registered before the window, so register is
    sparse). `aged` accounts registered pre-window (large, broad `account_age_days`); fresh accounts
    register in-window — `burst_anchor` set => a tight co-registration spike near a ring's campaign
    start (low age, low variance), else a uniform organic signup. Windowed-truncation fidelity: the
    durable axis is the in-window funnel; register is only a burst/recency tell, never required."""
    if aged:
        days = float(rng.uniform(cfg.pre_window_age_days_min, cfg.pre_window_age_days_max))
        return cfg.base_epoch - int(days * 86_400), None
    if burst_anchor is not None:
        reg = burst_anchor + int(rng.integers(0, 2 * 86_400))  # co-registered within ~2 days
    else:
        reg = cfg.base_epoch + int(rng.integers(0, max(1, span)))  # organic, uniform over the window
    reg = int(min(reg, cfg.base_epoch + span))
    return reg, reg


def _emit_lifecycle(
    cols: _Cols,
    rng: np.random.Generator,
    cfg: SimConfig,
    *,
    aid: int,
    ip: str,
    asn: int,
    fp: str,
    reg_event_ts: int | None,
    span: int,
) -> None:
    """Sparse `register` / `login` recency events on the account's PRIMARY infra. An in-window
    registrant emits a register (+ a likely first login); an aged account occasionally surfaces a
    lone login. These are supplementary burst/recency signals only — never required by a feature
    (account_age_days carries recency) — and use an existing ip/asn so they don't perturb n_ips/
    n_asns. They are excluded from the media-based old-basket features in ingest by construction."""
    if reg_event_ts is not None:
        _emit_event(cols, ts=reg_event_ts, account_id=aid, session_id=f"a{aid}-reg",
                    ip=ip, asn=asn, fp=fp, action="register")
        if rng.random() < 0.8:  # a fresh signup typically logs in right after registering
            _emit_event(cols, ts=reg_event_ts + 30, account_id=aid, session_id=f"a{aid}-reg",
                        ip=ip, asn=asn, fp=fp, action="login")
    elif cfg.login_prob > 0.0 and rng.random() < cfg.login_prob:
        lts = cfg.base_epoch + int(rng.integers(0, max(1, span)))
        _emit_event(cols, ts=lts, account_id=aid, session_id=f"a{aid}-login",
                    ip=ip, asn=asn, fp=fp, action="login")


@dataclass
class _Cols:
    """Growing columnar buffers for the event stream."""

    ts: list[int] = field(default_factory=list)
    account_id: list[int | None] = field(default_factory=list)
    session_id: list[str] = field(default_factory=list)
    ip: list[str] = field(default_factory=list)
    subnet: list[str] = field(default_factory=list)
    asn: list[int] = field(default_factory=list)
    fingerprint: list[str] = field(default_factory=list)
    action: list[str] = field(default_factory=list)
    item_id: list[int | None] = field(default_factory=list)
    creator_id: list[int | None] = field(default_factory=list)
    reported_position_pct: list[float | None] = field(default_factory=list)
    skip_flag: list[bool] = field(default_factory=list)
    challenge_result: list[str | None] = field(default_factory=list)

    def add(
        self,
        *,
        ts: int,
        account_id: int | None,
        session_id: str,
        ip: str,
        asn: int,
        fp: str,
        action: str,
        item_id: int | None,
        creator_id: int | None,
        position: float | None,
        skip: bool,
        challenge: str | None,
    ) -> None:
        self.ts.append(ts)
        self.account_id.append(account_id)
        self.session_id.append(session_id)
        self.ip.append(ip)
        self.subnet.append(_subnet(ip))
        self.asn.append(asn)
        self.fingerprint.append(fp)
        self.action.append(action)
        self.item_id.append(item_id)
        self.creator_id.append(creator_id)
        self.reported_position_pct.append(position)
        self.skip_flag.append(skip)
        self.challenge_result.append(challenge)

    def __len__(self) -> int:
        return len(self.ts)


def _items_of_creator(creator: int, items_per_creator: int) -> np.ndarray:
    base = creator * items_per_creator
    return np.arange(base, base + items_per_creator)


def _emit_event(
    cols: _Cols,
    *,
    ts: int,
    account_id: int | None,
    session_id: str,
    ip: str,
    asn: int,
    fp: str,
    action: str,
    item_id: int | None = None,
    creator_id: int | None = None,
) -> None:
    """Emit a non-view funnel event (register / login / search / save). No position/skip/challenge
    — those belong to a media event. `search` carries the searched creator_id; `save` carries the item."""
    cols.add(
        ts=ts, account_id=account_id, session_id=session_id, ip=ip, asn=asn, fp=fp,
        action=action, item_id=item_id, creator_id=creator_id, position=None, skip=False, challenge=None,
    )


def _emit_view(
    cols: _Cols,
    rng: np.random.Generator,
    *,
    ts: int,
    account_id: int | None,
    session_id: str,
    ip: str,
    asn: int,
    fp: str,
    creator: int,
    item: int,
    skip_prob: float,
    spoof_complete: bool,
    challenge: str | None,
    complete_threshold: float = COMPLETE_THRESHOLD_DEFAULT,
) -> None:
    skip = bool(rng.random() < skip_prob)
    if spoof_complete:
        position = float(rng.uniform(0.92, 1.0))  # bots claim near-full views
        skip = False
    elif skip:
        position = float(rng.uniform(0.0, 0.25))
    else:
        position = float(np.clip(rng.beta(5, 2), 0.0, 1.0))
    # `complete` is a relabel: a non-skipped view that reaches near-full is the media event's OUTCOME on the
    # SAME event — no new event, reported_position_pct still the granular spoofable fraction. The
    # detector folds view+complete back together for volume features, so the parity-tuned session/gap
    # structure is untouched; only the funnel SHAPE (completion_rate) the action-mix axis reads changes.
    if skip:
        action = "skip"
    elif position >= complete_threshold:
        action = "complete"
    else:
        action = "view"
    cols.add(
        ts=ts,
        account_id=account_id,
        session_id=session_id,
        ip=ip,
        asn=asn,
        fp=fp,
        action=action,
        item_id=item,
        creator_id=creator,
        position=position,
        skip=skip,
        challenge=challenge,
    )


def _emit_funnel_view(
    cols: _Cols,
    rng: np.random.Generator,
    *,
    frng: np.random.Generator,
    ts: int,
    account_id: int,
    session_id: str,
    ip: str,
    asn: int,
    fp: str,
    creator: int,
    item: int,
    skip_prob: float,
    spoof_complete: bool,
    challenge: str | None,
    complete_threshold: float,
    search_prob: float,
    save_prob: float,
) -> None:
    """A media event wrapped in its optional funnel context: an in-session `search` for the creator just
    before, the view itself, an optional `save` of the item just after. The search/save sit a few
    seconds either side of the view (one tight intra-session gap) and reference a creator/item the
    account already views, so they exercise the action-mix axis WITHOUT disturbing the old basket's
    n_creators / top_creator_share.

    The media event itself draws from `rng` (the core content stream); the funnel-PRESENCE decisions draw
    from `frng` (a dedicated enrichment stream). So the core media stream does not depend on the funnel
    knobs at all — the isolation is exact, not merely statistical (a test pins it). Only new
    search/save EVENTS are added."""
    if search_prob > 0.0 and frng.random() < search_prob:
        _emit_event(cols, ts=ts - 4, account_id=account_id, session_id=session_id, ip=ip, asn=asn,
                    fp=fp, action="search", creator_id=creator)
    _emit_view(
        cols, rng, ts=ts, account_id=account_id, session_id=session_id, ip=ip, asn=asn, fp=fp,
        creator=creator, item=item, skip_prob=skip_prob, spoof_complete=spoof_complete,
        challenge=challenge, complete_threshold=complete_threshold,
    )
    if save_prob > 0.0 and frng.random() < save_prob:
        _emit_event(cols, ts=ts + 4, account_id=account_id, session_id=session_id, ip=ip, asn=asn,
                    fp=fp, action="save", item_id=item, creator_id=creator)


# --- lock-step session scheduling --------------------------------------------
# The shared session primitive + the C-lane scheduler. Both the serialized ring and the
# uncapped bandwagon are built from these, so the concurrency cap is their ONLY difference.
LOCKSTEP_VIEW_GAP_MIN = 20   # seconds between consecutive views inside one session
LOCKSTEP_VIEW_GAP_MAX = 180
LOCKSTEP_LANE_GAP = 5        # idle seconds a lane waits before its next session
LOCKSTEP_FAN_TARGET_SHARE = 0.8  # bandwagon fans view the release creator most of the time (honest co-target)


def _session_offsets(rng: np.random.Generator, n_sessions: int, views_min: int, views_max: int) -> list[np.ndarray]:
    """A member's burst as a list of sessions; each session is the per-view time offsets
    (seconds from that session's start) of Uniform(views_min, views_max) views. The first view
    sits at offset 0, so a session's duration is its last offset."""
    sessions: list[np.ndarray] = []
    for _ in range(n_sessions):
        size = int(rng.integers(views_min, views_max + 1))
        gaps = rng.integers(LOCKSTEP_VIEW_GAP_MIN, LOCKSTEP_VIEW_GAP_MAX + 1, size=size)
        gaps[0] = 0
        sessions.append(np.cumsum(gaps).astype(np.int64))
    return sessions


def _expected_burst_width(n_members: int, cfg: SimConfig) -> float:
    """Width (seconds) of the burst window, DERIVED so the concurrency cap binds: enough total
    session-seconds to keep C lanes busy. slack>1 leaves the lanes saturated (ring max reaches ~C)
    without forcing large forward-shifts that would drift gap_mean from the bandwagon's."""
    e_size = (cfg.session_views_min + cfg.session_views_max) / 2.0
    e_gap = (LOCKSTEP_VIEW_GAP_MIN + LOCKSTEP_VIEW_GAP_MAX) / 2.0
    e_dur = max(1.0, (e_size - 1.0) * e_gap)
    total = n_members * cfg.lockstep_session_mean * e_dur
    return total / max(1, cfg.operator_concurrency or 1) * cfg.lockstep_window_slack


def _assign_starts(
    rng: np.random.Generator,
    durations: list[float],
    *,
    concurrency: int,
    w0: int,
    width: float,
    capped: bool,
) -> np.ndarray:
    """Assign each session a start time. Both cohorts draw the SAME desired-start distribution
    (Uniform over the window) — the cap is the only thing that moves them apart:

      * capped  -> earliest-free-lane packing onto `concurrency` lanes: a session starts at the
                   later of its desired time and its lane's free time, so <=C sessions are ever
                   active at once (the ring takes turns -> a flat plateau).
      * uncapped -> start == desired start (the bandwagon piles on -> a tall spike).
    """
    n = len(durations)
    desired = w0 + rng.uniform(0.0, width, size=n)
    starts = desired.copy()
    if not capped:
        return starts
    order = np.argsort(desired, kind="stable")
    lanes: list[tuple[float, int]] = [(float(w0), lane) for lane in range(max(1, concurrency))]
    heapq.heapify(lanes)
    for idx in order:
        free, lane = heapq.heappop(lanes)
        start = desired[idx] if desired[idx] > free else free
        starts[idx] = start
        heapq.heappush(lanes, (start + durations[idx] + LOCKSTEP_LANE_GAP, lane))
    return starts


def _generate_lockstep(
    cfg: SimConfig,
    rng: np.random.Generator,
    cols: _Cols,
    prof: EvasionProfile,
    *,
    erng: np.random.Generator,
    next_account_id: int,
    span: int,
    household_ips: list[str],
    residential_asns: np.ndarray,
    target_creators: np.ndarray,
    gt_account: list[int],
    gt_is_fraud: list[bool],
    gt_ring: list[int | None],
    gt_evasion: list[str],
    gt_bandwagon: list[int | None],
    acc_id: list[int],
    acc_reg: list[int],
) -> int:
    """The lock-step world. Each ring is one finite-concurrency operator: its members'
    sessions are packed onto `operator_concurrency` lanes inside a burst window, so <=C are active at
    once and the pool cycles (they take turns). The bandwagon reuses the SAME session primitive and
    the SAME window per target creator but UNCAPPED (fans pile on). Per-event CONTENT is generated
    exactly as the identity ring/fan path (via `prof`); only the TIMESTAMPS come from the scheduler.
    Returns the advanced `next_account_id`. Anti-tautology: nothing here is read by the detector.

    Funnel note: the lock-step branch keeps the media-only event stream (the `complete`
    relabel applies uniformly via `_emit_view`; NO search/save/register/login events) and registers
    BOTH ring and bandwagon as AGED, symmetric accounts — so the action-mix features introduce no
    new ring-vs-bandwagon separator and the parity guard (test_lockstep) is preserved."""
    w0 = cfg.base_epoch + span // 4
    windows: dict[int, float] = {}  # target creator -> burst-window width (start w0 is shared)

    def _clamp_width(width: float) -> float:
        return max(60.0, min(width, float(cfg.base_epoch + span - w0 - 1)))

    def _draw_sessions() -> list[np.ndarray]:
        n_sessions = max(1, int(rng.poisson(cfg.lockstep_session_mean)))
        return _session_offsets(rng, n_sessions, cfg.session_views_min, cfg.session_views_max)

    # ---- lock-step fraud rings (one C-lane operator per ring) -------------
    for ring_id in range(cfg.n_rings):
        base_size = int(rng.integers(cfg.ring_size_min, cfg.ring_size_max + 1))
        size = int(round(base_size * prof.account_multiplier))
        if prof.residential_asn:
            ring_asns = [int(a) for a in rng.choice(residential_asns, size=prof.asn_pool, replace=False)]
        else:
            ring_asns = [64500 + ring_id * 8 + j for j in range(prof.asn_pool)]
        n_resid = int(round(prof.ip_pool * prof.residential_ip_frac))
        ring_ips = [str(rng.choice(household_ips)) for _ in range(n_resid)]
        ring_ips += [_rand_ip(rng) for _ in range(prof.ip_pool - n_resid)]
        # Distinct IP strings only: the residential pool is sampled WITH replacement from a small
        # household pool, so it holds duplicates. Drawing per-member without de-duping first would let
        # a member realize fewer distinct IPs than the bandwagon's fresh IPs -> an n_ips parity leak.
        ring_ip_choices = list(dict.fromkeys(ring_ips))
        ring_fps = [_rand_fp(rng) for _ in range(prof.fp_pool)]
        target = int(target_creators[ring_id % len(target_creators)])
        width = _clamp_width(_expected_burst_width(size, cfg))
        windows.setdefault(target, width)

        n_ip = max(1, min(prof.ips_per_account, len(ring_ip_choices)))
        n_asn = max(1, min(prof.asns_per_account, len(ring_asns)))
        m_aid: list[int] = []
        m_ips: list[list[str]] = []
        m_asns: list[list[int]] = []
        m_fp: list[str] = []
        records: list[tuple[int, np.ndarray]] = []
        durations: list[float] = []
        for _m in range(size):
            aid = next_account_id
            next_account_id += 1
            mi = len(m_aid)
            m_aid.append(aid)
            m_ips.append([str(x) for x in rng.choice(ring_ip_choices, size=n_ip, replace=False)])
            m_asns.append([int(x) for x in rng.choice(ring_asns, size=n_asn, replace=False)])
            m_fp.append(_rand_fp(rng) if prof.per_session_fp else str(rng.choice(ring_fps)))
            for off in _draw_sessions():
                records.append((mi, off))
                durations.append(float(off[-1]))
            gt_account.append(aid)
            gt_is_fraud.append(True)
            gt_ring.append(ring_id)
            gt_evasion.append(cfg.evasion_level)
            gt_bandwagon.append(None)  # ring members are not bandwagon
            acc_id.append(aid)
            acc_reg.append(_draw_registration(erng, cfg, aged=True, span=span)[0])  # aged (parity)

        starts = _assign_starts(
            rng, durations, concurrency=cfg.operator_concurrency or 1, w0=w0, width=width, capped=True
        )
        sidx = [0] * len(m_aid)
        for (mi, off), start in zip(records, starts, strict=True):
            aid = m_aid[mi]
            sid = f"a{aid}-s{sidx[mi]}"
            sidx[mi] += 1
            fp = _rand_fp(rng) if prof.per_session_fp else m_fp[mi]
            start_i = int(start)
            for o in off:
                creator = target if rng.random() < prof.target_share else int(rng.integers(0, cfg.n_creators))
                _emit_view(
                    cols, rng, ts=start_i + int(o), account_id=aid, session_id=sid,
                    ip=str(rng.choice(m_ips[mi])), asn=int(rng.choice(m_asns[mi])), fp=fp,
                    creator=creator, item=int(rng.choice(_items_of_creator(creator, cfg.items_per_creator))),
                    skip_prob=prof.skip_prob, spoof_complete=bool(rng.random() < prof.spoof_fidelity),
                    challenge="fail" if rng.random() < prof.challenge_fail_prob else None,
                )

    # ---- bandwagon: same primitive + same window, UNCAPPED (piles on) ------
    # The benign control: co-target + time-correlated with a ring, but NOT serialized. Matched to the
    # ring on session structure and per-account IP COUNT, so only the cross-account overlap differs.
    n_clusters = max(1, min(cfg.n_benign_clusters, cfg.n_target_creators))
    fan_n_ip = max(1, prof.ips_per_account)  # match the ring's per-account IP count (n_ips parity)
    # gap_mean parity needs the SAME window width as a ring (gap_mean scales with width/events).
    # When a cohort's target has a co-located ring, reuse THAT ring's width; otherwise fall back to a
    # RING-scale width (mean ring size), NOT the cohort size, so density — and gap_mean — still match.
    mean_ring_size = max(1, int(round((cfg.ring_size_min + cfg.ring_size_max) / 2 * prof.account_multiplier)))
    fallback_width = _clamp_width(_expected_burst_width(mean_ring_size, cfg))
    for cluster_idx in range(n_clusters):
        target = int(target_creators[cluster_idx])
        width = windows.get(target, fallback_width)

        m_aid = []
        m_ips = []
        m_asn: list[int] = []
        m_fp = []
        records = []
        durations = []
        for _m in range(cfg.benign_cluster_size):
            aid = next_account_id
            next_account_id += 1
            mi = len(m_aid)
            m_aid.append(aid)
            m_ips.append([_rand_ip(rng) for _ in range(fan_n_ip)])  # DIVERSE residential infra (honest)
            m_asn.append(int(rng.choice(residential_asns)))
            m_fp.append(_rand_fp(rng))
            for off in _draw_sessions():
                records.append((mi, off))
                durations.append(float(off[-1]))
            gt_account.append(aid)
            gt_is_fraud.append(False)
            gt_ring.append(None)
            gt_evasion.append("none")
            gt_bandwagon.append(cluster_idx)  # eval-only tag: which uncapped cohort this fan belongs to
            acc_id.append(aid)
            acc_reg.append(_draw_registration(erng, cfg, aged=True, span=span)[0])  # aged (parity)

        starts = _assign_starts(
            rng, durations, concurrency=cfg.operator_concurrency or 1, w0=w0, width=width, capped=False
        )
        sidx = [0] * len(m_aid)
        for (mi, off), start in zip(records, starts, strict=True):
            aid = m_aid[mi]
            sid = f"a{aid}-s{sidx[mi]}"
            sidx[mi] += 1
            start_i = int(start)
            for o in off:
                creator = target if rng.random() < LOCKSTEP_FAN_TARGET_SHARE else int(rng.integers(0, cfg.n_creators))
                _emit_view(
                    cols, rng, ts=start_i + int(o), account_id=aid, session_id=sid,
                    ip=str(rng.choice(m_ips[mi])), asn=m_asn[mi], fp=m_fp[mi],
                    creator=creator, item=int(rng.choice(_items_of_creator(creator, cfg.items_per_creator))),
                    skip_prob=0.3, spoof_complete=False, challenge="pass" if rng.random() < 0.4 else None,
                )

    return next_account_id


def generate(cfg: SimConfig, out_dir: str) -> tuple[str, str]:
    """Generate the world. Returns (events_parquet_path, ground_truth_parquet_path).

    Also writes `accounts.parquet` (the detector-readable registration registry) into `out_dir`
    alongside `events.parquet`; `ingest.build_db` reads it from the same directory to derive
    `account_age_days`. It is NOT ground truth — the real platform knows registration dates."""
    rng = np.random.default_rng(cfg.seed)
    erng = np.random.default_rng(cfg.seed + ENRICH_SEED_OFFSET)  # enrichment-only stream (see above)
    os.makedirs(out_dir, exist_ok=True)

    mobile_asns = np.arange(1000, 1000 + cfg.mobile_asn_count)
    residential_asns = np.arange(2000, 2000 + cfg.residential_asn_count)
    household_ips = [_rand_ip(rng) for _ in range(cfg.shared_household_ips)]
    vpn_ips = [_rand_ip(rng) for _ in range(8)]  # shared datacenter-ish IPs (legit VPN noise)
    target_creators = np.arange(cfg.n_target_creators)

    cols = _Cols()
    gt_account: list[int] = []
    gt_is_fraud: list[bool] = []
    gt_ring: list[int | None] = []
    gt_evasion: list[str] = []
    gt_bandwagon: list[int | None] = []  # eval-only lock-step bandwagon tag (None outside that branch)
    acc_id: list[int] = []     # detector-readable accounts registry (parallels gt_account 1:1) ...
    acc_reg: list[int] = []    # ... registration_ts -> account_age_days in ingest

    span = cfg.span_days * 86_400
    next_account_id = 0

    def day_ts(rng: np.random.Generator, hour_center: float) -> int:
        day = int(rng.integers(0, cfg.span_days))
        hour = float(np.clip(rng.normal(hour_center, 3.0), 0, 23))
        return cfg.base_epoch + day * 86_400 + int(hour * 3600) + int(rng.integers(0, 3600))

    # ---- legit population -------------------------------------------------
    for _ in range(cfg.n_legit):
        aid = next_account_id
        next_account_id += 1
        cookieless = rng.random() < cfg.legit_cookieless_fraction
        on_mobile = rng.random() < 0.5
        asn = int(rng.choice(mobile_asns if on_mobile else residential_asns))
        if rng.random() < 0.10:  # household sharing
            home_ip = str(rng.choice(household_ips))
        else:
            home_ip = _rand_ip(rng)
        fp = _rand_fp(rng)
        hour_center = float(rng.uniform(8, 21))
        prefs = rng.choice(np.arange(cfg.n_creators), size=int(rng.integers(2, 6)), replace=False)
        skip_prob = float(rng.uniform(0.2, 0.5))
        n_ev = max(5, int(rng.poisson(cfg.legit_events_mean)))
        # Windowed truncation: most legit registered BEFORE the window (aged, no in-window register);
        # a trickle (legit_inwindow_signup_frac) are organic in-window signups. (Enrichment stream.)
        aged = erng.random() >= cfg.legit_inwindow_signup_frac
        reg_ts, reg_event_ts = _draw_registration(erng, cfg, aged=aged, span=span)

        for _e in range(n_ev):
            ts = day_ts(rng, hour_center)
            ip = str(rng.choice(vpn_ips)) if rng.random() < cfg.vpn_noise_fraction else home_ip
            session_id = _rand_fp(rng) if cookieless else f"a{aid}-d{(ts - cfg.base_epoch) // 86400}"
            # mostly preferred creators, occasionally a target creator (organic)
            creator = int(rng.choice(prefs)) if rng.random() < 0.85 else int(rng.choice(target_creators))
            item = int(rng.choice(_items_of_creator(creator, cfg.items_per_creator)))
            # Engaged legit exercise the full funnel (search before, save after) — the action-mix axis.
            _emit_funnel_view(
                cols, rng, frng=erng, ts=ts, account_id=aid, session_id=session_id, ip=ip, asn=asn,
                fp=fp, creator=creator, item=item, skip_prob=skip_prob, spoof_complete=False,
                challenge="pass" if rng.random() < 0.3 else None,
                complete_threshold=cfg.complete_threshold,
                search_prob=cfg.legit_search_prob, save_prob=cfg.legit_save_prob,
            )
        _emit_lifecycle(cols, erng, cfg, aid=aid, ip=home_ip, asn=asn, fp=fp,
                        reg_event_ts=reg_event_ts, span=span)
        gt_account.append(aid)
        gt_is_fraud.append(False)
        gt_ring.append(None)
        gt_evasion.append("none")
        gt_bandwagon.append(None)
        acc_id.append(aid)
        acc_reg.append(reg_ts)

    # ---- benign lurker segment (passive viewers) ------------------------
    # The anti-tautology confound: aged, honest, funnel-TRUNCATED users who consume
    # via autoplay/feeds — they rarely search or save and complete most views of a favoured creator.
    # Funnel-truncated WITHOUT being fraud, so "truncated funnel => fraud" cannot be a free tautology;
    # the action-mix axis must hold precision on them (they look funnel-thin like a naive ring but are
    # AGED and on diverse infra). Generated BEFORE the rings/clusters so the benign cohorts stay the
    # trailing account-id block (the convention the generator/lockstep tests key on).
    # Generated entirely on the ENRICHMENT stream so it adds a benign population WITHOUT perturbing
    # the core ring/legit content draws (the old behavioral basket stays byte-identical).
    for _ in range(cfg.n_lurkers):
        aid = next_account_id
        next_account_id += 1
        asn = int(erng.choice(residential_asns))  # diverse residential infra -> never infra-linked
        home_ip = _rand_ip(erng)
        fp = _rand_fp(erng)
        favorite = int(erng.choice(np.arange(cfg.n_creators)))  # one favoured creator, viewed to completion
        n_ev = max(5, int(erng.poisson(cfg.lurker_events_mean)))
        reg_ts, reg_event_ts = _draw_registration(erng, cfg, aged=True, span=span)  # long-time user
        for _e in range(n_ev):
            ts = day_ts(erng, hour_center=20.0)
            session_id = f"a{aid}-d{(ts - cfg.base_epoch) // 86400}"
            creator = favorite if erng.random() < 0.85 else int(erng.choice(np.arange(cfg.n_creators)))
            item = int(erng.choice(_items_of_creator(creator, cfg.items_per_creator)))
            # Passive: completes most views (low skip), but almost never searches or saves.
            _emit_funnel_view(
                cols, erng, frng=erng, ts=ts, account_id=aid, session_id=session_id, ip=home_ip,
                asn=asn, fp=fp, creator=creator, item=item, skip_prob=0.08, spoof_complete=False,
                challenge="pass" if erng.random() < 0.5 else None,
                complete_threshold=cfg.complete_threshold,
                search_prob=0.02, save_prob=0.02,
            )
        _emit_lifecycle(cols, erng, cfg, aid=aid, ip=home_ip, asn=asn, fp=fp,
                        reg_event_ts=reg_event_ts, span=span)
        gt_account.append(aid)
        gt_is_fraud.append(False)
        gt_ring.append(None)
        gt_evasion.append("none")
        gt_bandwagon.append(None)
        acc_id.append(aid)
        acc_reg.append(reg_ts)

    # ---- fraud rings + benign clusters ------------------------------------
    # One profile resolves the evasion level into concrete knobs; neither branch below ever
    # branches on the level name. Higher levels rotate infra, jitter timing, mint per-session
    # fingerprints, clear cookies, soften the spoof, and camouflage into legit behavior — see
    # fraudrings.generator.evasion. `operator_concurrency` is ORTHOGONAL to this dial.
    prof = profile_for(cfg.evasion_level, ip_pool_base=cfg.ring_ip_pool, fp_pool_base=cfg.ring_fp_pool)
    if cfg.operator_concurrency is None:
        # === identity branch: per-member independent-start rings + fan clubs. Output is
        # byte-for-byte independent of the lock-step knobs (a test pins it).
        for ring_id in range(cfg.n_rings):
            # Fixed-yield split: throttle per-account volume but grow the ring so total
            # shadowed views stay ~constant (the economic cost of evasion).
            base_size = int(rng.integers(cfg.ring_size_min, cfg.ring_size_max + 1))
            size = int(round(base_size * prof.account_multiplier))  # account_multiplier >= 1 -> >= base_size
            # ASNs: distinctive datacenter ASNs (rare, strong link) unless the ring routes
            # through residential proxies on the SAME ASNs legit use (kills ASN linkage).
            if prof.residential_asn:
                ring_asns = [int(a) for a in rng.choice(residential_asns, size=prof.asn_pool, replace=False)]
            else:
                ring_asns = [64500 + ring_id * 8 + j for j in range(prof.asn_pool)]
            # IPs: a mix of own datacenter IPs and residential IPs drawn from the pool legit
            # share (household_ips) — the residential fraction is what leaks legit into candidates.
            n_resid = int(round(prof.ip_pool * prof.residential_ip_frac))
            ring_ips = [str(rng.choice(household_ips)) for _ in range(n_resid)]
            ring_ips += [_rand_ip(rng) for _ in range(prof.ip_pool - n_resid)]
            ring_fps = [_rand_fp(rng) for _ in range(prof.fp_pool)]
            target = int(target_creators[ring_id % len(target_creators)])
            base_gap = float(rng.uniform(180, 900))  # ring's nominal cadence (seconds)
            jitter = max(1, int(base_gap * prof.gap_jitter))
            # Fresh ring accounts co-register in a tight in-window burst near the ring's campaign start
            # (the registration spike a viz/feature can read); a sophisticated ring instead AGES its
            # accounts (account_aged_frac -> 1.0) so account_age looks legit (evasion convergence).
            # All on the enrichment stream -> the ring's core content is unchanged.
            reg_burst_anchor = cfg.base_epoch + int(erng.integers(0, max(1, span // 8)))

            for _m in range(size):
                aid = next_account_id
                next_account_id += 1
                aged = erng.random() < prof.account_aged_frac
                reg_ts, reg_event_ts = _draw_registration(
                    erng, cfg, aged=aged, span=span, burst_anchor=reg_burst_anchor
                )
                # Rotate across accounts: each member is pinned to a few IPs/ASNs from the ring
                # pool, so its per-account footprint looks legit while the shared pool still links.
                n_ip = min(prof.ips_per_account, len(ring_ips))
                n_asn = min(prof.asns_per_account, len(ring_asns))
                member_ips = [str(x) for x in rng.choice(ring_ips, size=n_ip, replace=False)]
                member_asns = [int(x) for x in rng.choice(ring_asns, size=n_asn, replace=False)]
                hour_center = float(rng.uniform(8, 21))  # day-parting for scattered (legit-like) events
                sess_idx = 0
                # per-session fingerprints mint fresh (unshared) FPs; otherwise a shared spoofed FP.
                fp = _rand_fp(rng) if prof.per_session_fp else str(rng.choice(ring_fps))
                n_ev = max(20, int(rng.poisson(max(1.0, cfg.fraud_events_mean * prof.volume_scale))))
                t = cfg.base_epoch + int(rng.integers(0, span // 4))
                for _e in range(n_ev):
                    if rng.random() < prof.scatter_frac:
                        ts = day_ts(rng, hour_center)  # scattered like a real viewer (long idle gaps)
                    else:
                        t += int(base_gap) + int(rng.integers(-jitter, jitter + 1))  # cadenced burst
                        if t > cfg.base_epoch + span:
                            t = cfg.base_epoch + int(rng.integers(0, span))
                        ts = t
                    if rng.random() < prof.cookie_clear_prob:  # cookie cleared -> fresh session (+ fp)
                        sess_idx += 1
                        if prof.per_session_fp:
                            fp = _rand_fp(rng)
                    ip = str(rng.choice(member_ips))
                    asn = int(rng.choice(member_asns))
                    creator = target if rng.random() < prof.target_share else int(rng.integers(0, cfg.n_creators))
                    item = int(rng.choice(_items_of_creator(creator, cfg.items_per_creator)))
                    # Funnel convergence: naive views the target directly (search/save prob 0 ->
                    # short-circuit, no extra draws, truncated funnel); sophisticated fakes decoy
                    # searches/saves so the action-mix tell fades.
                    _emit_funnel_view(
                        cols, rng, frng=erng, ts=ts, account_id=aid, session_id=f"a{aid}-s{sess_idx}",
                        ip=ip, asn=asn, fp=fp, creator=creator, item=item, skip_prob=prof.skip_prob,
                        spoof_complete=bool(rng.random() < prof.spoof_fidelity),
                        challenge="fail" if rng.random() < prof.challenge_fail_prob else None,
                        complete_threshold=cfg.complete_threshold,
                        search_prob=prof.funnel_search_prob, save_prob=prof.funnel_save_prob,
                    )
                _emit_lifecycle(cols, erng, cfg, aid=aid, ip=member_ips[0], asn=member_asns[0], fp=fp,
                                reg_event_ts=reg_event_ts, span=span)
                gt_account.append(aid)
                gt_is_fraud.append(True)
                gt_ring.append(ring_id)
                gt_evasion.append(cfg.evasion_level)
                gt_bandwagon.append(None)
                acc_id.append(aid)
                acc_reg.append(reg_ts)

        # ---- benign coordinated clusters (fan clubs) ----------------------
        # One super-fan cohort per target creator (capped by n_benign_clusters): a coordinated-LOOKING
        # but honest release-day burst on a creator a ring ALSO targets — the hardest legit negative
        # (genuine fans of the very creator the ring pumps). Spreading the confound across SEVERAL
        # targets (not just creator 0) is what makes the leave-one-ring-out grade fair: the colliding
        # rings then appear in the training folds too, so the rule learns the ring-vs-superfan boundary
        # instead of meeting it for the first time on the held-out ring.
        n_clusters = max(1, min(cfg.n_benign_clusters, cfg.n_target_creators))
        for cluster_idx in range(n_clusters):
            release_creator = int(target_creators[cluster_idx])  # also a fraud target -> confound
            release_item = int(_items_of_creator(release_creator, cfg.items_per_creator)[0])
            release_day = int(rng.integers(0, cfg.span_days))
            for _ in range(cfg.benign_cluster_size):
                aid = next_account_id
                next_account_id += 1
                asn = int(rng.choice(residential_asns))  # DIVERSE infra -> no rare shared link
                home_ip = _rand_ip(rng)
                fp = _rand_fp(rng)
                n_ev = max(5, int(rng.poisson(cfg.legit_events_mean // 2)))
                # Honest long-time fans: aged accounts that exercise the full funnel for their creator.
                reg_ts, reg_event_ts = _draw_registration(erng, cfg, aged=True, span=span)
                for _e in range(n_ev):
                    # all view the same release around the same day (coordinated-LOOKING, honest)
                    ts = cfg.base_epoch + release_day * 86_400 + int(rng.normal(20 * 3600, 4 * 3600))
                    ts = int(np.clip(ts, cfg.base_epoch, cfg.base_epoch + span))
                    session_id = f"a{aid}-d{(ts - cfg.base_epoch) // 86400}"
                    _emit_funnel_view(
                        cols, rng, frng=erng, ts=ts, account_id=aid, session_id=session_id, ip=home_ip,
                        asn=asn, fp=fp, creator=release_creator, item=release_item, skip_prob=0.3,
                        spoof_complete=False, challenge="pass" if rng.random() < 0.4 else None,
                        complete_threshold=cfg.complete_threshold,
                        search_prob=cfg.legit_search_prob, save_prob=cfg.legit_save_prob,
                    )
                _emit_lifecycle(cols, erng, cfg, aid=aid, ip=home_ip, asn=asn, fp=fp,
                                reg_event_ts=reg_event_ts, span=span)
                gt_account.append(aid)
                gt_is_fraud.append(False)
                gt_ring.append(None)
                gt_evasion.append("none")
                gt_bandwagon.append(None)
                acc_id.append(aid)
                acc_reg.append(reg_ts)
    else:
        # === lock-step branch: a finite operator with C lanes SERIALIZES each ring
        # (members take turns -> <=C active at once), while the bandwagon uses the SAME session
        # primitive + SAME burst window but UNCAPPED (piles on). The concurrency cap is the sole
        # difference, so the temporal-overlap signal is earned. See _generate_lockstep.
        next_account_id = _generate_lockstep(
            cfg, rng, cols, prof,
            erng=erng,
            next_account_id=next_account_id,
            span=span,
            household_ips=household_ips,
            residential_asns=residential_asns,
            target_creators=target_creators,
            gt_account=gt_account,
            gt_is_fraud=gt_is_fraud,
            gt_ring=gt_ring,
            gt_evasion=gt_evasion,
            gt_bandwagon=gt_bandwagon,
            acc_id=acc_id,
            acc_reg=acc_reg,
        )

    # ---- write (time-ordered) ---------------------------------------------
    events = pl.DataFrame(
        {
            "ts": cols.ts,
            "account_id": cols.account_id,
            "session_id": cols.session_id,
            "ip": cols.ip,
            "subnet": cols.subnet,
            "asn": cols.asn,
            "fingerprint": cols.fingerprint,
            "action": cols.action,
            "item_id": cols.item_id,
            "creator_id": cols.creator_id,
            "reported_position_pct": cols.reported_position_pct,
            "skip_flag": cols.skip_flag,
            "challenge_result": cols.challenge_result,
        }
    ).sort("ts")  # time-ordered: enables micro-batch / incremental processing

    ground_truth = pl.DataFrame(
        {
            "account_id": gt_account,
            "is_fraud": gt_is_fraud,
            "ring_id": gt_ring,
            "evasion_level": gt_evasion,
            # Eval-only lock-step tag (see schema.GroundTruthRow). None outside the lock-step branch.
            "bandwagon_cluster_id": gt_bandwagon,
        },
        schema_overrides={"bandwagon_cluster_id": pl.Int64},
    )

    # Detector-readable account registry (NOT ground truth). Written next to events.parquet so
    # `ingest.build_db` can pick it up from the same directory and derive `account_age_days`.
    accounts = pl.DataFrame(
        {"account_id": acc_id, "registration_ts": acc_reg},
        schema={"account_id": pl.Int64, "registration_ts": pl.Int64},
    )

    events_path = os.path.join(out_dir, "events.parquet")
    gt_path = os.path.join(out_dir, "ground_truth.parquet")
    accounts_path = os.path.join(out_dir, "accounts.parquet")
    events.write_parquet(events_path)
    ground_truth.write_parquet(gt_path)
    accounts.write_parquet(accounts_path)
    return events_path, gt_path
