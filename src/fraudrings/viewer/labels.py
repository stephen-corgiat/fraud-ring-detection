"""Human-label store + label-vs-truth scoring.

A per-world JSON file mapping account_id -> a human verdict. The viewer is the
human-in-the-loop tool, so it can grade those verdicts against the hidden ground
truth: how good is the human (or the human + the model's surfacing) at calling
fraud? `unsure` is recorded but excluded from precision/recall.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import polars as pl

Verdict = Literal["fraud", "legit", "unsure"]
VERDICTS: tuple[Verdict, ...] = ("fraud", "legit", "unsure")


@dataclass
class ScoreSummary:
    n_labeled: int
    n_decided: int  # fraud or legit (unsure excluded)
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.n_decided if self.n_decided else 0.0


class LabelStore:
    """Append/overwrite human labels in a JSON file alongside the world's data."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._labels: dict[int, dict] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            with open(self.path) as f:
                raw = json.load(f)
            self._labels = {int(k): v for k, v in raw.items()}

    def _flush(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({str(k): v for k, v in self._labels.items()}, f, indent=2, sort_keys=True)

    def set(self, account_id: int, verdict: Verdict, note: str = "") -> None:
        if verdict not in VERDICTS:
            raise ValueError(f"unknown verdict {verdict!r}; expected one of {VERDICTS}")
        self._labels[int(account_id)] = {
            "verdict": verdict,
            "note": note,
            "updated": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        self._flush()

    def clear(self, account_id: int) -> None:
        if int(account_id) in self._labels:
            del self._labels[int(account_id)]
            self._flush()

    def get(self, account_id: int) -> dict | None:
        return self._labels.get(int(account_id))

    def all(self) -> dict[int, dict]:
        return dict(self._labels)

    def counts(self) -> dict[str, int]:
        out = {v: 0 for v in VERDICTS}
        for rec in self._labels.values():
            out[rec["verdict"]] = out.get(rec["verdict"], 0) + 1
        return out

    def score(self, gt: pl.DataFrame) -> ScoreSummary:
        """Grade decided labels (fraud/legit) against ground truth's `is_fraud`."""
        truth = {int(r["account_id"]): bool(r["is_fraud"]) for r in gt.iter_rows(named=True)}
        tp = fp = tn = fn = 0
        decided = 0
        for aid, rec in self._labels.items():
            v = rec["verdict"]
            if v == "unsure" or aid not in truth:
                continue
            decided += 1
            is_fraud = truth[aid]
            called_fraud = v == "fraud"
            if called_fraud and is_fraud:
                tp += 1
            elif called_fraud and not is_fraud:
                fp += 1
            elif not called_fraud and not is_fraud:
                tn += 1
            else:
                fn += 1
        return ScoreSummary(
            n_labeled=len(self._labels), n_decided=decided, tp=tp, fp=fp, tn=tn, fn=fn
        )


def store_for_world(world_dir: str) -> LabelStore:
    return LabelStore(os.path.join(world_dir, "labels.json"))
