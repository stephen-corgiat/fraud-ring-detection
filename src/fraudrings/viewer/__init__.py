"""The human-in-the-loop viewer (`fraudrings view`): Triage + Investigate.

A profile / related-accounts investigator and a seedless triage queue over the *same* tables the detector
queries (`account_features`, `assoc`, `identifier_stats`, `events`). It sits on the
eval / human side of the anti-tautology line, so — unlike `detector/` — it is allowed
to read `ground_truth` (the labeler's reveal + the label-vs-truth scoreboard).

Split so the query layer is testable without a UI runtime:
  * `data`   — world loading + all DuckDB queries (no streamlit import).
  * `data_triage` — the triage page's shortlist + evidence queries (no streamlit import).
  * `labels` — the per-world human-label store + label-vs-truth scoring.
  * `main`, `app`, `app_triage` — the Streamlit pages (import streamlit; need the [viz] extra).
"""
