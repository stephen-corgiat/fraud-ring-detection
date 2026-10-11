"""Canonical schemas.

`Event` is exactly what the detector is allowed to see; `GroundTruthRow` is the
hidden truth, written to a separate file the detector never opens.

The bulk generator writes columnar Parquet directly (millions of Pydantic objects
would be slow); these models define/validate the schema and document the contract.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

ActionType = Literal["register", "login", "view", "skip", "complete", "save", "follow", "search"]
# Adversary sophistication dial. "none" tags non-fraud accounts.
EvasionLevel = Literal["none", "naive", "moderate", "sophisticated"]
EVENT_COLUMNS: tuple[str, ...] = (
    "ts",
    "account_id",
    "session_id",
    "ip",
    "subnet",
    "asn",
    "fingerprint",
    "action",
    "item_id",
    "creator_id",
    "reported_position_pct",
    "skip_flag",
    "challenge_result",
)
GROUND_TRUTH_COLUMNS: tuple[str, ...] = (
    "account_id",
    "is_fraud",
    "ring_id",
    "evasion_level",
    "bandwagon_cluster_id",
)
# Account registry. A detector-READABLE side table — the real system knows
# when each account registered — so it is NOT ground truth. `ingest.py` derives `account_age_days`
# from it. Written alongside events.parquet; the detector may read it, eval never needs it.
ACCOUNT_COLUMNS: tuple[str, ...] = (
    "account_id",
    "registration_ts",
)


class Event(BaseModel):
    """One activity event as the collector would emit it (detector-visible)."""

    ts: int  # epoch seconds (time-ordered)
    account_id: int | None  # None => anonymous traffic
    session_id: str  # the spoofable session/device cookie
    ip: str
    subnet: str  # /24 of ip
    asn: int
    fingerprint: str
    action: ActionType
    item_id: int | None
    creator_id: int | None
    reported_position_pct: float | None  # CLIENT-reported, spoofable
    skip_flag: bool  # CLIENT-reported, spoofable
    challenge_result: str | None  # pass | fail | None (active-probing outcome)


class Account(BaseModel):
    """One row of the detector-readable account registry.

    `registration_ts` anchors `account_age_days`. Most legit accounts registered *before* the
    observation window (large, broadly-distributed age); freshly-minted ring accounts register
    *in* the window (low age, low variance) at naive/moderate — the burst/recency signal — while a
    sophisticated ring uses AGED accounts so age looks legit (evasion-dial convergence). Not GT:
    a real platform knows registration dates, so reading this is fair (anti-tautology preserved).
    """

    account_id: int
    registration_ts: int  # epoch seconds; may predate the event window (sparse in-window register)


class GroundTruthRow(BaseModel):
    """Hidden truth — read by eval only, never by the detector."""

    account_id: int
    is_fraud: bool
    ring_id: int | None
    evasion_level: EvasionLevel
    # Lock-step eval tag. The benign *bandwagon* cohort co-located
    # with a ring's burst window — the negative control the serialization signal must separate the
    # ring from. Populated ONLY by the lock-step generator branch (`operator_concurrency` set); None
    # for legit, fraud, and the identity-branch fan clubs. Eval-only: the detector never reads it
    # (anti-tautology). Without it the bandwagon is indistinguishable from generic legit in GT, so
    # the 2AFC / segmentation grading could not identify the bandwagon cluster.
    bandwagon_cluster_id: int | None = None
