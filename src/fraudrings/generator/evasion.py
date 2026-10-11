"""The adversary as a parameterized dial.

`EvasionProfile` resolves an `evasion_level` into the concrete knobs the ring
generator reads. The generator itself is level-agnostic: it consumes a profile and
never branches on the level name. `profile_for` is the *only* place the three named
levels map to numbers — so the whole "how sophisticated is the adversary" axis is one
small, reviewable table.

The dials — IP/ASN rotation, timing jitter, fingerprint
randomization, cookie-clearing rate, position/skip spoofing fidelity, legit-behavior
camouflage — plus two learned from how careful real-world operators behave:

  * **Fixed-yield split** — a careful operator throttles per-account view volume and
    splits the same total fraudulent yield across MORE accounts, so each account looks
    as low-volume as a real viewer. `volume_scale` × `account_multiplier` ≈ 1, so the
    ring's total shadowed views stay ~constant while per-account volume drops. This turns
    sophistication into an explicit economic cost (accounts-needed vs. evasion).
  * **Rotate across accounts** — naive rotation makes one account hop dozens of IPs (a
    fresh anomaly). A careful operator instead pins each account to 1–2 residential IPs
    and spreads the ring's infra diversity ACROSS members, so every account's footprint
    looks legit while linkage survives only in the shared pool.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from fraudrings.schema import EvasionLevel


class EvasionProfile(BaseModel):
    """Concrete per-level adversary parameters. Resolved by `profile_for`."""

    level: EvasionLevel

    # --- Infrastructure: IP / ASN rotation -------------------------------------
    ip_pool: int  # ring-wide distinct IP pool (shared across members)
    residential_ip_frac: float = Field(ge=0.0, le=1.0)
    # fraction of the ring's IP pool drawn from the SAME residential pool legit use
    # (the rest are own datacenter IPs). >0 -> ring IPs collide with legit -> Stage-1
    # expansion over a shared IP pulls legit accounts in -> precision pressure.
    residential_asn: bool  # draw the ring's ASNs from legit's residential pool (kills ASN linkage)
    asn_pool: int  # ring-wide distinct ASN pool
    ips_per_account: int = Field(ge=1)  # IPs each member sticks to (rotation ACROSS accounts, not within)
    asns_per_account: int = Field(ge=1)  # ASNs each member sticks to

    # --- Fingerprint randomization ---------------------------------------------
    fp_pool: int  # size of the shared fingerprint pool (used only when not per-session)
    per_session_fp: bool  # mint a fresh fingerprint each session (no cross-member FP sharing)

    # --- Cookie-clearing -------------------------------------------------------
    cookie_clear_prob: float = Field(ge=0.0, le=1.0)  # P(any event starts a fresh session/cookie)

    # --- Timing ----------------------------------------------------------------
    gap_jitter: float = Field(ge=0.0)  # inter-event gap jitter as a fraction of the base cadence
    scatter_frac: float = Field(ge=0.0, le=1.0)  # fraction of events drawn with legit day-parting vs burst

    # --- Volume (the fixed-yield economic axis) --------------------------------
    volume_scale: float = Field(gt=0.0)  # per-account event-count multiplier
    account_multiplier: float = Field(gt=0.0)  # ring-size multiplier (~1/volume_scale -> ~constant yield)

    # --- Position/skip spoofing fidelity ---------------------------------------
    spoof_fidelity: float = Field(ge=0.0, le=1.0)  # P(a view is a near-perfect completion spoof)
    skip_prob: float = Field(ge=0.0, le=1.0)  # realistic skipping (naive bots ~never skip)

    # --- Legit-behavior camouflage ---------------------------------------------
    target_share: float = Field(ge=0.0, le=1.0)  # fraction of views on the target creator
    challenge_fail_prob: float = Field(ge=0.0, le=1.0)  # naive bots fail active challenges; sophisticated pass

    # --- Action-funnel convergence ---------------------------------------------
    # The funnel SHAPE the action-mix axis reads. A naive ring views the target directly with no
    # search/save and a perfect completion spoof (glaringly truncated funnel on fresh accounts); a
    # sophisticated ring fakes the whole funnel — decoy searches before the target, occasional decoy
    # saves, softened completes (spoof_fidelity already drops) — on AGED accounts so `account_age`
    # looks legit. So the action-diversity signal separates naive/moderate and FADES at sophisticated
    # (the anti-tautology convergence `tests/test_actionmix.py` checks).
    funnel_search_prob: float = Field(ge=0.0, le=1.0)  # P(a ring view is preceded by a (decoy) search
    funnel_save_prob: float = Field(ge=0.0, le=1.0)    # P(a ring view is followed by a (decoy) save)
    account_aged_frac: float = Field(ge=0.0, le=1.0)   # fraction of ring accounts registered PRE-window
    # (rest are freshly minted in-window -> low account_age + an in-window register burst)


def profile_for(level: EvasionLevel, *, ip_pool_base: int, fp_pool_base: int) -> EvasionProfile:
    """Map an evasion level to its concrete profile.

    `ip_pool_base` / `fp_pool_base` anchor the *naive* baseline to `SimConfig`
    (`ring_ip_pool` / `ring_fp_pool`) so the naive world is unchanged in character;
    higher levels scale up from there.
    """
    if level == "naive":
        # Everything shared and distinctive: a small datacenter IP/ASN pool, a handful of
        # spoofed fingerprints, metronomic burst timing, perfect completion spoof, ~all
        # target views, full per-account volume.
        return EvasionProfile(
            level="naive",
            ip_pool=ip_pool_base,
            residential_ip_frac=0.0,
            residential_asn=False,
            asn_pool=1,
            ips_per_account=ip_pool_base,
            asns_per_account=1,
            fp_pool=fp_pool_base,
            per_session_fp=False,
            cookie_clear_prob=0.0,
            gap_jitter=0.05,
            scatter_frac=0.0,
            volume_scale=1.0,
            account_multiplier=1.0,
            spoof_fidelity=1.0,
            skip_prob=0.02,
            target_share=0.90,
            challenge_fail_prob=0.6,
            # Glaringly truncated funnel on fresh, co-registered accounts.
            funnel_search_prob=0.0,
            funnel_save_prob=0.0,
            account_aged_frac=0.0,
        )
    if level == "moderate":
        # Infra still ring-specific enough to link on (datacenter ASNs survive), but FP/cookie/
        # timing degraded, HALF the IPs residential (legit starts leaking -> precision pressure),
        # per-account footprint tightened, and volume partially throttled (split across ~1.6x accounts).
        return EvasionProfile(
            level="moderate",
            ip_pool=ip_pool_base * 6,
            residential_ip_frac=0.5,
            residential_asn=False,
            asn_pool=3,
            ips_per_account=3,
            asns_per_account=1,
            fp_pool=fp_pool_base,
            per_session_fp=True,
            cookie_clear_prob=0.30,
            gap_jitter=0.5,
            scatter_frac=0.5,
            volume_scale=0.6,
            account_multiplier=1.6,
            spoof_fidelity=0.7,
            skip_prob=0.15,
            target_share=0.65,
            challenge_fail_prob=0.3,
            # Partial funnel; ~a third of accounts aged (the rest still a recency tell).
            funnel_search_prob=0.10,
            funnel_save_prob=0.05,
            account_aged_frac=0.3,
        )
    if level == "sophisticated":
        # Residential proxies on legit ASNs (ASN linkage dead), fully residential IP camouflage but
        # only 1-2 IPs per account (legit-like footprint), per-session fingerprints, human-like jitter,
        # mostly day-parted timing, realistic skips, heavy cover-view camo, and per-account volume
        # throttled to ~legit — the same yield split across ~3.5x more accounts.
        return EvasionProfile(
            level="sophisticated",
            ip_pool=ip_pool_base * 12,
            residential_ip_frac=1.0,
            residential_asn=True,
            asn_pool=6,
            ips_per_account=2,
            asns_per_account=1,
            fp_pool=fp_pool_base,
            per_session_fp=True,
            cookie_clear_prob=0.60,
            gap_jitter=1.0,
            scatter_frac=0.9,
            volume_scale=0.28,
            account_multiplier=3.5,
            spoof_fidelity=0.4,
            skip_prob=0.30,
            target_share=0.45,
            challenge_fail_prob=0.1,
            # Fakes the full funnel on fully-aged accounts -> the action-mix tell fades to legit-like.
            funnel_search_prob=0.30,
            funnel_save_prob=0.15,
            account_aged_frac=1.0,
        )
    raise ValueError(f"invalid ring evasion_level: {level!r} (expected naive|moderate|sophisticated)")
