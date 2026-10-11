# 0002 — The adversary is a dial, and results are curves

**Status:** accepted

## Context

"Our detector has 0.97 precision" is meaningless without saying *against whom*. Fraud is a
strategic game: the adversary adapts to whatever the defender keys on (Hardt et al.'s strategic
classification; Perdomo et al.'s performative prediction). A single number on a single synthetic
world hides where the method breaks, which is the one thing a practitioner needs to know.

## Decision

- Every world has exactly one **evasion level** — `naive`, `moderate`, `sophisticated` — resolved
  by one reviewable table (`generator/evasion.py`) into concrete knobs: IP/ASN rotation, timing
  jitter, fingerprint randomization, cookie clearing, completion-spoof fidelity, target share,
  funnel faking, account aging. The ring generator never branches on the level name.
- Two knobs encode how careful operators actually behave:
  - **Fixed-yield split** — throttle per-account volume and spread the same total yield across
    more accounts (`volume_scale × account_multiplier ≈ 1`). Sophistication becomes an explicit
    economic cost: more accounts needed.
  - **Rotate across accounts, not within** — pin each account to one or two residential IPs and
    spread the diversity across members, so no single account's footprint looks odd.
- Results are reported as **curves over the dial**, from `fraudrings sweep`, with **two** curves
  rather than one:
  - (a) *per-ring ranking* — AP, Stage-1 recall, recall@k, precision@k;
  - (b) *operational* — of the accounts an investigation would action, how many are fraud at all
    (operational precision), how many are genuine users (legit FP), how many belong to another ring
    (cross-ring bleed).

## Consequences

The decomposition changes the story. At default scale over seeds 7, 13, 21:

| level | AP | Stage-1 recall | operational precision | legit FP | cross-ring |
|---|---|---|---|---|---|
| naive | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| moderate | 0.92 | 0.92 | 0.98 | 0.02 | 0.19 |
| sophisticated | 0.50 | 0.57 | 0.95 | 0.05 | 0.56 |

Ranking collapses with sophistication while the operational line mostly holds. The loss is the
engine confusing one ring for *another* (still fraud), not flagging genuine users. A single AP
number would have hidden that.

## References

- Hardt, Megiddo, Papadimitriou, Wootters, *Strategic Classification*, ITCS 2016.
- Perdomo, Zrnic, Mendler-Dünner, Hardt, *Performative Prediction*, ICML 2020.
- `src/fraudrings/generator/evasion.py`, `src/fraudrings/eval/sweep.py`, `tests/test_sweep.py`.
