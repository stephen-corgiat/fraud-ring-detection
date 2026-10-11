"""Typed configuration.

Everything that controls the synthetic world, the detector, and the evaluation
lives here as Pydantic models, so a run is fully described by one object and a seed.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from fraudrings.schema import EvasionLevel

# 2024-01-01T00:00:00Z — fixed so runs are deterministic (no wall-clock).
BASE_EPOCH: int = 1_704_067_200


class SimConfig(BaseModel):
    """Synthetic-world parameters (the generator's only input besides the seed)."""

    seed: int = 7

    # Adversary sophistication for ALL rings in this world. The sweep
    # harness re-generates per level; one world = one point on the P/R-vs-sophistication
    # curve. "none" is invalid here (it tags non-fraud accounts only).
    evasion_level: EvasionLevel = "naive"

    # Population
    n_legit: int = 2_000
    n_rings: int = 8
    ring_size_min: int = 25
    ring_size_max: int = 55
    benign_cluster_size: int = 120  # fan-club: coordinated-looking but honest
    # Number of benign super-fan cohorts, each on a distinct target creator a ring also targets.
    # >1 so the ring-vs-superfan confound appears in the leave-one-ring-out TRAINING folds, not
    # only the held-out one — otherwise the lone colliding ring is always the test fold and the
    # learned rule meets the boundary for the first time on held-out data.
    n_benign_clusters: int = 4

    # A deliberate benign PASSIVE-VIEWER (lurker) segment — real users who
    # consume via autoplay/feeds, rarely search or save, and complete most views of a favoured
    # creator. They are funnel-TRUNCATED without being fraud, so the action-mix axis cannot read
    # "truncated funnel => fraud" as a free tautology — it must hold the line on these.
    # On diverse residential infra and AGED accounts, so Stage-1 infra linkage never retrieves them
    # (they pressure precision through the action-mix axis / co-target widening, not infra).
    n_lurkers: int = 300
    lurker_events_mean: int = 45

    # Catalog
    n_creators: int = 200
    items_per_creator: int = 10
    n_target_creators: int = 8  # fraud beneficiaries

    # Time
    span_days: int = 30
    base_epoch: int = BASE_EPOCH

    # Lock-step operator scheduling. Orthogonal to evasion_level.
    #   None  -> identity: the existing per-member-start ring loop + fan-club loop run UNCHANGED.
    #   int C -> a separate additive branch: a finite operator with C concurrent lanes serializes
    #            each ring (earliest-free-lane packing -> <=C members active at once, they take
    #            turns), while the benign bandwagon uses the SAME session primitive + SAME burst
    #            window but UNCAPPED (piles on). The concurrency cap is then the SOLE difference
    #            between ring and bandwagon -> the temporal-overlap signal is earned, not assumed.
    # operator_concurrency reschedules *when* events happen; it never changes population, infra, or
    # per-event content (those come from profile_for(evasion_level), exactly as the naive path).
    operator_concurrency: int | None = Field(default=None, ge=1)
    # Session primitive: an account wakes, views Uniform(session_views_min, session_views_max) items
    # with short intra-session delays, then idles. Used identically by ring and bandwagon.
    session_views_min: int = Field(default=3, ge=1)
    session_views_max: int = Field(default=6, ge=1)
    lockstep_session_mean: int = Field(default=12, ge=1)  # mean sessions per account in the burst (Poisson)
    # Burst-window width is DERIVED so the cap binds: width = total_session_seconds / C * slack.
    # slack > 1 leaves the lanes saturated (ring max reaches ~C) without large forward-shifts that
    # would drift the per-account gap_mean away from the bandwagon's (the anti-tautology parity).
    lockstep_window_slack: float = Field(default=1.2, gt=0.0)

    @model_validator(mode="after")
    def _check_session_views(self) -> SimConfig:
        if self.session_views_min > self.session_views_max:
            raise ValueError(
                f"session_views_min ({self.session_views_min}) > session_views_max ({self.session_views_max})"
            )
        return self

    # Activity volumes (events per account)
    legit_events_mean: int = 60
    fraud_events_mean: int = 240  # inflation: bots view a lot

    # --- the action funnel (search -> view -> complete -> save) -------------------------------
    # An engaged legit viewer exercises the whole funnel; the action-mix axis reads its shape.
    # `complete` is a relabel: a near-full, non-skipped view is relabelled `complete` on the SAME
    # event (no new event) — `reported_position_pct` stays the granular spoofable fraction, so the
    # parity-tuned session/gap structure is untouched. search/save are EXTRA events emitted in-session
    # next to the view they relate to (small gaps), on creators the account already views (so the
    # old basket's n_creators/top_creator_share are not disturbed). Rings' funnel shape converges with
    # sophistication (see evasion.py): naive truncates it, sophisticated fakes it.
    complete_threshold: float = 0.90  # reported_position_pct at/above which a non-skip view is `complete`
    legit_search_prob: float = 0.30   # P(an engaged legit view is preceded by a search for that creator)
    legit_save_prob: float = 0.15     # P(an engaged legit view is followed by a save of that item)

    # --- registration / account age (windowed truncation) --------------------------------------
    # Most legit accounts registered BEFORE the observation window -> no in-window `register` event
    # (register is sparse), and `account_age_days` is large and broadly distributed. A trickle of
    # organic signups + a burst of freshly-minted ring accounts register in-window. account_age is a
    # detector-READABLE attribute (the real platform has it), NOT ground truth.
    pre_window_age_days_min: int = 30    # aged accounts registered this many days before the window ...
    pre_window_age_days_max: int = 730   # ... up to this many (broad legit age distribution)
    legit_inwindow_signup_frac: float = 0.05  # fraction of legit who DO register in-window (organic trickle)
    login_prob: float = 0.15             # P(a session opens with an explicit `login` event) — sparse recency

    # Infrastructure realism
    mobile_asn_count: int = 6  # big shared ASNs (must be rarity-gated, not links)
    residential_asn_count: int = 60
    shared_household_ips: int = 150  # small pool -> a few legit coincidentally share
    legit_cookieless_fraction: float = 0.15  # clients that never persist a cookie
    vpn_noise_fraction: float = 0.04  # legit who occasionally use a datacenter IP
    ring_ip_pool: int = 5  # datacenter IPs per ring (shared, rare)
    ring_fp_pool: int = 3  # spoofed fingerprints per ring

    @property
    def n_items(self) -> int:
        return self.n_creators * self.items_per_creator


class EngineConfig(BaseModel):
    """Detector knobs."""

    qualified_view_threshold: float = 0.30  # position% to count for payouts

    # Stage-1 rarity gate: only expand identifiers shared by >=2 and <= cap accounts.
    expand_min_accounts: int = 2
    expand_max_accounts: int = 150  # > ring size, << mobile-ASN size

    link_id_types: tuple[str, ...] = ("ip", "subnet", "asn", "fp", "cookie")

    # Stage-1b behavioral candidate generation. When infra linkage
    # is too thin to have retrieved the whole ring (the sophisticated regime), widen retrieval on
    # the economic invariant the adversary can't escape: shared qualified views of the rare target
    # creator, time-localized to the ring's campaign window. ADAPTIVE — fires only when the seeds'
    # linkage cohesion is low, so the pristine naive/moderate regimes are untouched. Default OFF:
    # the widened net pulls in genuine fans of the target that only the learned Stage-2 rule can rank
    # out, so it is enabled together with that rule.
    cotarget_candidate_gen: bool = False
    cotarget_seed_cohesion_trigger: float = 0.8  # fire when < this fraction of seeds interlink
    # Widen retrieval for EVERY investigation, not only low-cohesion ones. The adaptive trigger
    # fires for so few moderate rings that the benign super-fan club ends up only in the held-out
    # fold under leave-one-ring-out — the learned rule then meets the ring-vs-superfan boundary for
    # the first time on test data and over-flags the fans. Always-widening puts the confound in the
    # TRAINING folds too (so the rule learns to rank fans out) and lifts recall the gate left behind.
    # The hand-weighted path keeps it OFF (it can't rank out the fans the wider net retrieves);
    # only the supervised rule turns it on (paired with the learned ranker).
    cotarget_always: bool = False
    cotarget_n_targets: int = 2  # how many target creators to expand on
    cotarget_min_qviews: int = 3  # min qualified views on a target (in window) to be a candidate
    cotarget_window_pad_frac: float = 0.1  # pad the seed-derived campaign window by this fraction

    # Stage-2 evidence-score weights (hand-set, validated by the eval harness). The action-mix term
    # is weighted so it carries diagnosis without diluting the trusted behavioral centroid.
    w_linkage: float = 0.45
    w_behavioral: float = 0.25
    w_cotarget: float = 0.15
    w_actionmix: float = 0.15

    # Engine hardening (config-gated, default OFF, regression-guarded). With both flags OFF, Stage 2
    # is byte-identical to the hand-weighted baseline, so every existing result is unchanged by
    # construction; turning them ON is measured for lift.
    #   robust_centroid — the seed centroid is the per-feature MEDIAN, not the mean, so one atypical or
    #     mislabelled seed cannot drag the whole ring's behavioral/action-mix reference off true. Low
    #     regression risk (identity on a clean, tight seed set; a clean win on a contaminated one).
    #   cotarget_concentration — the co-target evidence term measures a candidate's DEVOTION to the
    #     seed target (its qualified-view share ON that target — the booster/core signature the
    #     ranker scores by), replacing the baseline Jaccard of viewed-creator REPERTOIRES. Jaccard rewards
    #     any broad-taste viewer who overlaps the ring's catalog; concentration rewards only the
    #     accounts actually pouring views into the target. Higher risk (Jaccard worked on synthetic),
    #     so gated and measured. Reuses the same event-only, label-free target identification as the
    #     co-target candidate-gen (`_seed_targets`).
    robust_centroid: bool = False
    cotarget_concentration: bool = False

    # Behavioral feature columns used for the seed-relative distance.
    behavioral_features: tuple[str, ...] = (
        "skip_rate",
        "qualified_rate",
        "views_per_session",
        "gap_mean",
        "gap_std",
        "active_hours",
        "n_creators",
        "top_creator_share",
        "n_ips",
        "n_asns",
    )

    # Action-mix axis sub-features: the funnel SHAPE + account recency. A
    # separate top-level term (seed-centroid proximity, mirroring the behavioral term) so it gives
    # per-axis diagnosis instead of diluting the behavioral basket. Naive/moderate rings sit far
    # from a legit seed's funnel centroid (truncated funnel, fresh accounts); the signal fades as a
    # sophisticated ring fakes the funnel and ages its accounts. The lurker confound (truncated but
    # benign) is what keeps this honest — it must NOT separate cleanly on funnel-truncation alone.
    actionmix_features: tuple[str, ...] = (
        "completion_rate",
        "search_rate",
        "save_rate",
        "funnel_completeness",
        "account_age_days",
    )


class ConcentrationConfig(BaseModel):
    """Knobs for the seedless concentration ranker (`detector.concentration`), label-free.

    Mirrors `EngineConfig`: every knob that controls the suspect shortlist lives here, so a run is
    fully described by one object. The ranker needs no seeds, so it runs on any events table on its
    own. The defaults were tuned on the `default` preset (the `smoke` preset scales the impact floor
    down with the world); they are a starting point, not law. `n_eff` is a continuum, so the output
    is a *ranking*, never a hard cut. See
    `GLOSSARY.md` for the vocabulary.
    """

    # Qualified-view definition (mirrors EngineConfig.qualified_view_threshold). A view/complete at
    # or above this reported position, with a non-null target, counts toward payouts.
    qualified_view_threshold: float = 0.30

    # Per-(account, day) payout cap behind capped-qv = Σ min(daily qv, CAP). Raw volume overstates
    # a lone bot's harm (a 300/day bot nets ~10/day); capping points ranking at real payout harm.
    cap: int = 10

    # Relevance floor: a target enters the suspect pool only if its raw qualified-view volume clears
    # this. Below it the concentration signal is noise (a 3-view "ring" is not payout-relevant).
    impact_floor: int = 500

    # Booster (single-target-share) definition → the core + the booster-count entry axis. A booster
    # is an account dedicated to ONE target: single-target share ≥ `booster_min_share` at volume
    # ≥ `booster_min_qv`. Its benign twin is the superfan (why outputs stay suspects, not verdicts).
    booster_min_qv: int = 50
    booster_min_share: float = 0.80

    # Observed-topology tag thresholds (they define tags, never in/out gates). The tags name the
    # observed contributor shape, not proven coordination.
    tag_single_source_share: float = 0.60   # top1_share — one observed actor dominates
    tag_few_source_share: float = 0.60      # top3_share — a handful dominate
    tag_multi_source_boosters: int = 3      # booster_count — many dedicated contributors (coordination)
    tag_campaign_burst7: float = 0.50       # burst7 — temporal concentration

    # Expansion gate: hand a core to `run_investigation` ONLY when its members interlink on
    # rare infra (cohesion ≥ this). On sophisticated residential rings infra linkage is dead, so the
    # core IS the output and we do not auto-expand. Reuses `engine.seed_cohesion`.
    expand_cohesion_trigger: float = 0.80

    # Default shortlist ORDER. Both keys
    # are first-class fields on every `SuspectTarget`; this only picks which one the list is sorted by:
    #   "suspicion"  — pure contribution-concentration (max axis-percentile). The concentrated few-
    #                  dominant core surfaces at the head; a merely-popular high-volume target (the
    #                  superfan confound) can NOT float above it on volume alone,
    #                  because impact only breaks ties WITHIN an equal-suspicion band. The right lens
    #                  for a human triaging "how coordinated does this look".
    #   "rank_score" — impact-weighted (suspicion × log10 capped-qv): points the head at payout harm,
    #                  so high-volume targets (incl. legit) float up. Available for an impact-first view.
    # Tie-break within the primary key is capped-qv desc then target_id asc (deterministic).
    sort_by: Literal["suspicion", "rank_score"] = "suspicion"


class EvalConfig(BaseModel):
    """Evaluation protocol."""

    seeds_per_ring: int = 3  # K labeled members handed to the loop


class RunConfig(BaseModel):
    sim: SimConfig = Field(default_factory=SimConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    concentration: ConcentrationConfig = Field(default_factory=ConcentrationConfig)
    data_dir: str = "data"


def preset(name: str) -> RunConfig:
    """Named presets. 'smoke' is tiny (tests/CI); 'default' is the iteration size."""
    if name == "smoke":
        return RunConfig(
            sim=SimConfig(
                n_legit=250,
                n_rings=3,
                ring_size_min=12,
                ring_size_max=20,
                benign_cluster_size=25,
                n_lurkers=60,
                n_creators=40,
                span_days=14,
                legit_events_mean=40,
                fraud_events_mean=140,
                shared_household_ips=25,
            ),
            engine=EngineConfig(expand_max_accounts=40),
            concentration=ConcentrationConfig(impact_floor=100),
        )
    if name == "default":
        return RunConfig()
    raise ValueError(f"unknown preset: {name!r} (expected 'smoke' or 'default')")
