# AGENTS.md — notes for AI coding agents working in this repo

Conventions and gotchas that aren't obvious from the code. Read [`GLOSSARY.md`](GLOSSARY.md) for the
vocabulary and [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup, layout and the rules that keep results
honest.

## Environment

- Use the project venv: `.venv/bin/python`, `.venv/bin/pytest`, `.venv/bin/fraudrings`.
- Extras: `[dev]` (tests), `[viz]` (viewer), `[ml]` (learned rule), `[docs]` (screenshots, figures).

## Things that bite

- **Never let `detector/` read ground truth.** It is the anti-tautology boundary; tests check it.
- **Generator changes move pinned results.** Draw new randomness from a separate RNG stream (see
  `simulator.ENRICH_SEED_OFFSET`). If `tests/test_lockstep.py`'s content hash changes, you changed
  the identity world — find out why before re-pinning.
- **Two words that look alike:** a *media event* is any `view`/`complete`/`skip`; a *view* is a
  non-skipped one. `n_media` and `n_views` are different features.
- **"Seed" means two things.** The world seed (an RNG integer) and investigation seeds (account ids).
- **Numbers in docs come from `docs/results/`.** If you change the method, regenerate those files
  (see `scripts/make_figures.py`) and update the README, walkthrough and ADRs.

## Screenshots: use headless Chrome, not a browser extension

`scripts/capture_docs.py` regenerates every doc image deterministically. For a one-off capture,
`scripts/capture_ui.py` drives headless Chrome or Chromium over the DevTools Protocol. It handles the
traps that make naive captures fail:

1. **Streamlit renders client-side over a websocket**, so it waits real seconds (`--wait`) before
   capturing.
2. **The scale factor is set only through CDP's `Emulation.setDeviceMetricsOverride`**, never also on
   the command line. Doing both double-scales and shifts the capture origin.
3. **The viewport is set to the full page height and captured without `captureBeyondViewport`.**
   Beyond-viewport capture mis-renders Streamlit's sidebar.

```bash
.venv/bin/python -m streamlit run src/fraudrings/viewer/main.py \
    --server.headless true --server.port 8533 &
.venv/bin/python scripts/capture_ui.py "http://localhost:8533/?level=naive&rank=1" /tmp/full.png --wait 15 --width 1680
```

Every world control is deep-linkable (`preset`, `level`, `seed`, `conc`, `truth`), as are `rank`
on Triage and `ring` / `creator` on Investigate. Stop a background viewer by its port's PID; `pkill
-f` on the command line can match your own shell.
