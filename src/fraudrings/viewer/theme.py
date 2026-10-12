"""Visual theme + HTML render helpers for the viewer.

The design is implemented *within* Streamlit (Streamlit + Plotly, custom CSS allowed but
limited — no bespoke JS framework). This module is the seam that carries
the design language:

* ``inject()`` pushes one global ``<style>`` block (IBM Plex fonts, dark card chrome, button /
  sidebar styling, hidden default Streamlit chrome).
* ``register_plotly_template()`` registers + activates a dark Plotly template so every figure in
  ``charts.py`` inherits the design's paper/grid/font colors without per-figure edits.
* the ``*_html`` builders return strings for the design's signature, non-interactive surfaces —
  the top bar, the subject▸focus breadcrumb, circled-number section headers, the diverging
  z-score anomaly bars, the stacked score-composition bars, the sidebar scoreboard, verdict
  badges, the axis legend (with *live* weights read from the engine config), and the
  styled mini-tables. Interactions stay Streamlit widgets placed next to these blocks.

Nothing here computes or mutates detector signal — it is presentation only.
"""

from __future__ import annotations

import html as _html

import streamlit as st

from fraudrings.viewer.plotly_theme import register_plotly_template  # noqa: F401  (re-exported: T.register_plotly_template)

# --- palette (the Plotly colors live in plotly_theme) -------------------------------------------
BG = "#0e1116"          # main canvas
BG_DEEP = "#0b0e12"     # top bar / sidebar
CARD_INSET = "#0e1116"  # inset track inside a card
BORDER_SOFT = "#1b2027"
TEXT_MUTE = "#8b929c"
TEXT_FAINT = "#6f7681"
TEXT_GHOST = "#5a6068"

BASELINE = "#bbbbbb"    # population baseline

# Score-axis colors (the ④ legend + ③ composition bars), matching the design.
AXIS_COLORS = {
    "linkage": "#5b8fd6",
    "behavioral": "#a87ca5",
    "cotarget": "#d65f5f",
    "actionmix": "#e0a05c",
}
# Anomaly z-score bands: |z|>=2 hot, 1<=|z|<2 warm, |z|<1 cool.
_Z_HOT = "#d65f5f"
_Z_WARM = "#dd9156"
_Z_COOL = "#4a5a72"
# Truth-tag dot + text colors.
_TRUTH = {
    "this-ring": ("#d65f5f", "#e79a9a", "this-ring"),
    "other-ring": ("#e0a05c", "#e0a05c", "other-ring"),
    "legit": ("#4f9d69", "#7fcf95", "legit"),
}

# --- global CSS -----------------------------------------------------------------------------

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root { --ath-mono: 'IBM Plex Mono', ui-monospace, monospace; }

html, body, [class*="css"], .stApp,
section[data-testid="stSidebar"] * , .stMarkdown, .stButton>button {
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
}
/* the font rule above would otherwise turn Material icons into their ligature names */
[data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded' !important; }
.stApp { background: #0e1116; }

/* hide default Streamlit chrome — the viewer supplies its own top bar. The header and toolbar
   stay in the layout (invisible) because they hold the button that re-opens a collapsed sidebar. */
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
[data-testid="stExpandSidebarButton"] { visibility: visible; }
/* Plotly toolbar: keep it (lasso-to-reseed needs its select tools) but only on hover */
.modebar-container { opacity: 0; transition: opacity .15s; }
.js-plotly-plot:hover .modebar-container { opacity: 1; }
.block-container { padding-top: 0.6rem; padding-bottom: 3rem; max-width: 1500px; }

/* monospace + tabular numerals helper */
.mono { font-family: var(--ath-mono); font-feature-settings: 'tnum' 1; }

/* sidebar surface */
section[data-testid="stSidebar"] { background: #0b0e12; border-right: 1px solid #222831; }
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

/* compact, dark form controls */
section[data-testid="stSidebar"] label, .stSelectbox label, .stMultiSelect label,
.stNumberInput label, .stRadio label, .stCheckbox label, .stSlider label {
  color: #8b929c !important; font-size: 12px !important; font-weight: 500;
}
div[data-baseweb="select"] > div, .stNumberInput input, .stTextInput input {
  background: #13171d !important; border-color: #232a33 !important; color: #d7dce3 !important;
}

/* buttons — dark, bordered, full-width in their column */
.stButton > button {
  background: #13171d; border: 1px solid #2a323d; color: #cfd5de;
  border-radius: 6px; font-size: 12.5px; padding: 5px 11px; font-weight: 500;
  transition: border-color .12s ease, background .12s ease; width: 100%;
}
.stButton > button:hover { border-color: #3a5a8c; color: #e6e9ee; background: #161b22; }
.stButton > button:focus { box-shadow: none; }
/* primary (fraud / focus) accent */
.stButton > button[kind="primary"] {
  background: #2a1d1f; border-color: #5a3438; color: #e79a9a;
}
.stButton > button[kind="primary"]:hover { background: #34232636; border-color: #6d4044; }

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #222831; }
.stTabs [data-baseweb="tab"] {
  background: #0e1116; border: 1px solid #232a33; border-radius: 6px 6px 0 0;
  color: #6f7681; padding: 6px 14px; font-size: 12px;
}
.stTabs [aria-selected="true"] { background: #161b22; border-color: #34465c; color: #cfe0f5; }

/* dataframes blend into the dark substrate */
[data-testid="stDataFrame"] { border: 1px solid #222831; border-radius: 6px; }

/* dividers a touch dimmer */
hr { border-color: #1b2027 !important; }

/* ---- custom components ---- */
.ath-topbar {
  display:flex; align-items:center; justify-content:space-between;
  height:46px; padding:0 16px; margin:-0.6rem -1rem 14px; background:#0b0e12;
  border-bottom:1px solid #222831;
}
.ath-brand { display:flex; align-items:center; gap:10px; }
.ath-brand .dot { width:9px; height:9px; border-radius:2px; background:#d65f5f; }
.ath-brand .name { font-weight:600; font-size:14px; color:#e6e9ee; }
.ath-brand .sub { font-size:13px; color:#5a6068; }
.ath-topbar .meta { display:flex; align-items:center; gap:14px; }
.ath-pill {
  display:inline-flex; align-items:center; gap:7px; padding:4px 10px; background:#161b22;
  border:1px solid #2a3543; border-radius:5px; font-size:12px; color:#b9c0ca;
}
.ath-pill { cursor:help; }
.ath-pill .d { width:7px; height:7px; border-radius:50%; display:inline-block; }
.ath-pill .d.on { background:#6f9fe0; } .ath-pill .d.off { background:#6f7681; }
.ath-brand .name, .ath-brand .sub, .ath-pill, .ath-topbar .meta .mono { white-space:nowrap; }
@media (max-width: 640px) {
  /* narrow screens: stack the brand over the stats instead of squeezing both onto one line */
  .ath-topbar { flex-wrap:wrap; height:auto; padding:8px 12px; gap:6px 12px; }
  .ath-topbar .meta { gap:10px; flex-wrap:wrap; }
}

.ath-intro { font-size:13px; line-height:1.55; color:#aeb5bf; max-width:1100px; margin:0 0 10px; }
.ath-intro b { color:#e6e9ee; font-weight:600; }
.ath-intro a, .ath-credit a { color:#8fb4e8; text-decoration:none; }
.ath-intro a:hover, .ath-credit a:hover { text-decoration:underline; }
.ath-credit { margin-top:28px; padding-top:12px; border-top:1px solid #1b2027;
  font-size:11.5px; line-height:1.6; color:#6f7681; }

.ath-crumb {
  display:flex; align-items:center; gap:10px; flex-wrap:wrap;
  padding:10px 14px; background:#0d1117; border:1px solid #222831; border-radius:6px; margin-bottom:16px;
}
.ath-crumb .lbl { font-family:var(--ath-mono); font-size:10px; letter-spacing:.12em; color:#5a6068; }
.ath-crumb .val { font-family:var(--ath-mono); font-size:12.5px; color:#c3c9d1; }
.ath-crumb .val.strong { color:#e6e9ee; font-weight:600; }
.ath-crumb .sep { color:#3a414b; }
.ath-tag { font-size:10.5px; padding:2px 8px; border-radius:10px; }
.ath-tag.seed { background:#1d2733; border:1px solid #34465c; color:#7fa8d9; }
.ath-tag.cand { background:#2a1d1f; border:1px solid #5a3438; color:#e79a9a; }
.ath-crumb .hint { margin-left:auto; font-size:11.5px; color:#5a6068; }

.ath-sec { display:flex; align-items:center; gap:10px; margin:4px 0 10px; }
.ath-sec .num {
  width:20px; height:20px; border-radius:50%; display:inline-flex; align-items:center;
  justify-content:center; font-size:12px; font-weight:600; flex:none;
}
.ath-sec .num.on  { background:#d65f5f; color:#fff; }
.ath-sec .num.off { background:#1b232e; border:1px solid #2f3a48; color:#8b929c; }
.ath-sec .num.dis { background:#13171d; border:1px dashed #2a323d; color:#4a515b; }
.ath-sec .ttl { font-weight:700; font-size:16px; color:#e6e9ee; }
.ath-sec .ttl.dis { color:#5a6068; }
.ath-sec .desc { font-size:12px; color:#5a6068; }

.ath-card {
  background:#13171d; border:1px solid #222831; border-radius:6px; padding:13px 14px; margin-bottom:2px;
}
.ath-card .hd {
  display:flex; justify-content:space-between; align-items:baseline; margin-bottom:11px;
}
.ath-card .hd .t { font-size:12.5px; font-weight:600; color:#e6e9ee; }
.ath-card .hd .s { font-size:10.5px; color:#5a6068; }

/* diverging z-score bars */
.zrow { display:grid; grid-template-columns:150px 1fr 44px; gap:9px; align-items:center; margin-bottom:7px; }
.zrow .f { font-family:var(--ath-mono); font-size:11px; color:#aeb5bf; }
.ztrack { position:relative; height:13px; background:#0e1116; border-radius:3px; }
.ztrack .mid { position:absolute; left:50%; top:0; bottom:0; width:1px; background:#2f3a48; }
.ztrack .fill { position:absolute; top:2px; bottom:2px; }
.zrow .v { font-family:var(--ath-mono); font-size:11px; text-align:right; }

/* score-composition table */
.ath-scoretbl { background:#13171d; border:1px solid #222831; border-radius:6px; overflow:hidden; }
.ath-scoretbl .row {
  display:grid; grid-template-columns:26px 64px 1fr 50px 92px; gap:10px; padding:9px 14px;
  align-items:center; border-bottom:1px solid #171c23;
}
.ath-scoretbl .row.hd {
  background:#0d1117; border-bottom:1px solid #222831; font-size:10px; letter-spacing:.05em;
  color:#5a6068; padding:8px 14px;
}
.ath-scoretbl .row:last-child { border-bottom:none; }
.ath-scoretbl .rk { font-family:var(--ath-mono); font-size:11px; color:#6f7681; }
.ath-scoretbl .ac { font-family:var(--ath-mono); font-size:12px; color:#e6e9ee; }
.ath-scoretbl .sc { font-family:var(--ath-mono); font-size:12px; text-align:right; color:#e6e9ee; font-weight:600; }
.compbar { display:flex; height:15px; background:#0e1116; border-radius:3px; overflow:hidden; }
.compbar > span { display:block; height:100%; }
.truthcell { font-size:10.5px; display:flex; align-items:center; gap:5px; }
.truthcell .d { width:7px; height:7px; border-radius:50%; flex:none; }

/* axis legend */
.ath-legend {
  display:flex; align-items:center; gap:16px; flex-wrap:wrap; margin-bottom:9px; padding:8px 12px;
  background:#0d1117; border:1px solid #1b2027; border-radius:5px;
}
.ath-legend .k { font-size:10.5px; color:#5a6068; letter-spacing:.05em; }
.ath-legend .item { display:flex; align-items:center; gap:6px; font-size:11px; color:#aeb5bf; }
.ath-legend .sw { width:11px; height:11px; border-radius:2px; }
.ath-legend .w { font-family:var(--ath-mono); color:#5a6068; }

/* sidebar scoreboard */
.ath-counts { display:flex; gap:6px; margin:4px 0 12px; }
.ath-counts .c { flex:1; text-align:center; }
.ath-counts .c .n { font-family:var(--ath-mono); font-size:18px; font-weight:700; }
.ath-counts .c .l { font-size:10px; color:#6f7681; }
.ath-kpis { display:flex; gap:6px; }
.ath-kpis .k {
  flex:1; text-align:center; padding:8px 0; background:#13171d; border:1px solid #232a33; border-radius:5px;
}
.ath-kpis .k .n { font-family:var(--ath-mono); font-size:15px; font-weight:600; color:#e6e9ee; }
.ath-kpis .k .l { font-size:9.5px; color:#6f7681; letter-spacing:.04em; }
.ath-conf { font-family:var(--ath-mono); font-size:10.5px; color:#5a6068; margin-top:8px; }

/* mini key/value tables (top creators, shared ids) */
.ath-mini { width:100%; border-collapse:collapse; font-size:11px; }
.ath-mini th {
  color:#5a6068; font-size:9.5px; font-weight:500; text-align:left; letter-spacing:.05em;
  padding:0 0 5px; border-bottom:1px solid #1b2027;
}
.ath-mini th.r, .ath-mini td.r { text-align:right; }
.ath-mini td { font-family:var(--ath-mono); padding:4px 0; color:#c3c9d1; }
.ath-mini td.dim { color:#8b929c; }

.ath-ct { display:flex; align-items:baseline; gap:8px; margin:2px 0 6px; }
.ath-ct .ct-t { font-size:12.5px; font-weight:600; color:#e6e9ee; }
.ath-ct .ct-sub { font-size:10.5px; color:#5a6068; }

.ath-badge { font-size:11.5px; padding:3px 9px; border-radius:11px; font-weight:600; }
.ath-eyebrow {
  font-family:var(--ath-mono); font-size:10px; letter-spacing:.16em; color:#5a6068;
  text-transform:uppercase; margin:6px 0 8px;
}
"""


def inject() -> None:
    """Push the global stylesheet. Call once, right after ``st.set_page_config``."""
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


def _md(html_str: str) -> None:
    st.markdown(html_str, unsafe_allow_html=True)


def _esc(v: object) -> str:
    return _html.escape(str(v))


# --- intro / credit -------------------------------------------------------------------------

def intro(site_url: str, repo_url: str) -> None:
    """One-paragraph "what is this" for the hosted demo, with links to the project site and source."""
    _md(
        '<div class="ath-intro"><b>fraudrings</b> finds <b>fraud rings</b>: groups of accounts run by one '
        "operator to look like many independent people, pumping up chosen creators to collect payouts "
        "no real audience earned. <b>Triage</b> ranks suspect creators from the raw event stream; "
        "<b>Investigate</b> starts from a few suspect accounts and recovers the rest of the ring. "
        "Everything here is simulated, and your labels stay in this tab. "
        f'<a href="{_esc(site_url)}" target="_blank">What is this?</a> · '
        f'<a href="{_esc(site_url)}tour.html" target="_blank">Two-minute tour</a> · '
        f'<a href="{_esc(repo_url)}" target="_blank">Source on GitHub</a></div>'
    )


def credit(author: str, site_url: str, repo_url: str) -> None:
    """Author line at the foot of the sidebar."""
    _md(
        f'<div class="ath-credit">Built by {_esc(author)}<br>'
        f'<a href="{_esc(site_url)}" target="_blank">Project site</a> · '
        f'<a href="{_esc(repo_url)}" target="_blank">GitHub</a></div>'
    )


# --- top bar / breadcrumb -------------------------------------------------------------------

def top_bar(n_accounts: int, n_fraud: int, show_truth: bool, page: str = "investigate") -> None:
    """Sticky-style brand bar: name + page, live world stats, and a ground-truth state badge. The
    badge is read-only (drawn as a dot, not a switch, so it doesn't invite a click): the real
    control is the sidebar's "show ground truth" checkbox."""
    knob = "on" if show_truth else "off"
    state = "ground truth shown" if show_truth else "blind mode"
    tip = "Toggle with “show ground truth” in the sidebar"
    _md(
        '<div class="ath-topbar">'
        '<div class="ath-brand"><span class="dot"></span>'
        '<span class="name">fraudrings</span>'
        f'<span class="sub">{_esc(page)}</span></div>'
        '<div class="meta">'
        f'<span class="mono" style="font-size:12px;color:#8b929c;">{n_accounts:,} accounts · {n_fraud} fraud</span>'
        f'<span class="ath-pill" title="{tip}"><span class="d {knob}"></span>{state}</span>'
        "</div></div>"
    )


def breadcrumb(subject: str, focus: int, focus_is_seed: bool, seed_pos: str) -> None:
    """SUBJECT ▸ FOCUS breadcrumb — keeps "what am I investigating" separate from "which account
    am I looking at". Shows what the investigation is and which single account is profiled."""
    tag = (
        '<span class="ath-tag seed">seed</span>' if focus_is_seed
        else '<span class="ath-tag cand">candidate</span>'
    )
    _md(
        '<div class="ath-crumb">'
        '<span class="lbl">SUBJECT</span>'
        f'<span class="val">{_esc(subject)}</span>'
        '<span class="sep">▸</span>'
        '<span class="lbl">FOCUS</span>'
        f'<span class="val strong">account {focus}</span>{tag}'
        f'<span class="hint">{_esc(seed_pos)}</span>'
        "</div>"
    )


def section_header(num: int, title: str, desc: str, state: str = "off") -> None:
    """A circled-number section marker (the ①→⑤ spine). ``state``: on (active) / off / dis."""
    _md(
        '<div class="ath-sec">'
        f'<span class="num {state}">{num}</span>'
        f'<span class="ttl {"dis" if state == "dis" else ""}">{_esc(title)}</span>'
        f'<span class="desc">{desc}</span>'
        "</div>"
    )


def eyebrow(text: str) -> None:
    _md(f'<div class="ath-eyebrow">{_esc(text)}</div>')


# --- profile: diverging z-score bars --------------------------------------------------------

def anomaly_bars(rows: list[dict], *, zmax: float = 3.0) -> None:
    """Diverging |z| bar panel (replaces the z-shaded dataframe). One row per behavioral feature,
    sorted by |z| desc; bar diverges from a center line, colored by band. ``rows`` items:
    {feature, z}. The exact features the detector scores on."""
    inner = []
    for r in rows:
        z = float(r["z"])
        a = abs(z)
        color = _Z_HOT if a >= 2 else (_Z_WARM if a >= 1 else _Z_COOL)
        tcol = "#e79a9a" if a >= 2 else ("#e0a05c" if a >= 1 else "#8b929c")
        half = min(a / zmax, 1.0) * 50.0
        if z >= 0:
            fill = (f'<div class="fill" style="left:50%;width:{half:.1f}%;background:{color};'
                    'border-radius:0 2px 2px 0;"></div>')
        else:
            fill = (f'<div class="fill" style="right:50%;width:{half:.1f}%;background:{color};'
                    'border-radius:2px 0 0 2px;"></div>')
        sign = "+" if z >= 0 else "−"
        inner.append(
            '<div class="zrow">'
            f'<span class="f">{_esc(r["feature"])}</span>'
            f'<div class="ztrack"><div class="mid"></div>{fill}</div>'
            f'<span class="v" style="color:{tcol};">{sign}{a:.2f}</span>'
            "</div>"
        )
    legend = (
        '<div style="display:flex;gap:14px;margin-top:9px;padding-top:9px;'
        'border-top:1px solid #1b2027;font-size:10px;color:#6f7681;">'
        f'<span style="display:flex;align-items:center;gap:5px;">'
        f'<span style="width:9px;height:9px;background:{_Z_HOT};border-radius:2px;"></span>|z|≥2</span>'
        f'<span style="display:flex;align-items:center;gap:5px;">'
        f'<span style="width:9px;height:9px;background:{_Z_WARM};border-radius:2px;"></span>1–2</span>'
        f'<span style="display:flex;align-items:center;gap:5px;">'
        f'<span style="width:9px;height:9px;background:{_Z_COOL};border-radius:2px;"></span>&lt;1</span>'
        "</div>"
    )
    _md(
        '<div class="ath-card"><div class="hd"><span class="t">Anomaly profile</span>'
        '<span class="s">z-score vs population · |z| sorted</span></div>'
        + "".join(inner) + legend + "</div>"
    )


# --- related accounts: score-composition table ----------------------------------------------

def axis_legend(weights: dict[str, float]) -> None:
    """The SCORE = legend with *live* weights read from the engine config."""
    items = []
    for ax in ("linkage", "behavioral", "cotarget", "actionmix"):
        items.append(
            f'<span class="item"><span class="sw" style="background:{AXIS_COLORS[ax]};"></span>'
            f'{ax} <span class="w">×{weights[ax]:.2f}</span></span>'
        )
    _md('<div class="ath-legend"><span class="k">SCORE =</span>' + "".join(items) + "</div>")


def _truth_cell(tag: str | None) -> str:
    if not tag or tag not in _TRUTH:
        return '<span class="truthcell" style="color:#5a6068;">—</span>'
    dot, txt, label = _TRUTH[tag]
    return (f'<span class="truthcell" style="color:{txt};">'
            f'<span class="d" style="background:{dot};"></span>{label}</span>')


def score_table(rows: list[dict], *, show_truth: bool) -> None:
    """Ranked related-accounts table rendered as stacked score-composition bars. Each ``rows`` item:
    {rank, account, score, contrib: {linkage, behavioral, cotarget, actionmix}, truth, label}.
    Segment widths are the *weighted contributions* (weight×axis), so the filled width is the
    score and the breakdown is legible at a glance."""
    head = (
        '<div class="row hd"><span>#</span><span>ACCOUNT</span>'
        '<span>SCORE COMPOSITION</span><span style="text-align:right;">SCORE</span>'
        f'<span>{"TRUTH" if show_truth else "YOUR LABEL"}</span></div>'
    )
    body = []
    for r in rows:
        segs = []
        for ax in ("linkage", "behavioral", "cotarget", "actionmix"):
            w = max(0.0, float(r["contrib"].get(ax, 0.0))) * 100.0
            if w > 0.05:
                segs.append(f'<span style="width:{w:.1f}%;background:{AXIS_COLORS[ax]};"></span>')
        if show_truth:
            last = _truth_cell(r.get("truth"))
        else:
            lab = r.get("label") or ""
            last = (f'<span class="truthcell" style="color:#9aa1ab;">{_esc(lab)}</span>'
                    if lab else '<span class="truthcell" style="color:#3a414b;">—</span>')
        body.append(
            '<div class="row">'
            f'<span class="rk">{r["rank"]}</span>'
            f'<span class="ac">{r["account"]}</span>'
            f'<div class="compbar">{"".join(segs)}</div>'
            f'<span class="sc">{r["score"]:.3f}</span>'
            f"{last}</div>"
        )
    _md('<div class="ath-scoretbl">' + head + "".join(body) + "</div>")


# --- mini tables ----------------------------------------------------------------------------

def mini_table(headers: list[tuple[str, bool]], rows: list[list[str]], *, dim_cols: set[int] | None = None) -> None:
    """A small dark key/value table (top creators, shared identifiers). ``headers`` is a list of
    (label, right_align); ``dim_cols`` are 0-based column indices rendered in a muted color."""
    dim_cols = dim_cols or set()
    th = "".join(f'<th class="{"r" if right else ""}">{_esc(h)}</th>' for h, right in headers)
    trs = []
    for row in rows:
        tds = []
        for i, (cell, (_h, right)) in enumerate(zip(row, headers, strict=False)):
            cls = " ".join(c for c in (("r" if right else ""), ("dim" if i in dim_cols else "")) if c)
            tds.append(f'<td class="{cls}">{_esc(cell)}</td>')
        trs.append("<tr>" + "".join(tds) + "</tr>")
    _md(f'<table class="ath-mini"><thead><tr>{th}</tr></thead><tbody>{"".join(trs)}</tbody></table>')


def card_title(title: str, sub: str = "") -> None:
    """A header line placed above a chart/table that follows as a separate Streamlit element."""
    s = f'<span class="ct-sub">{_esc(sub)}</span>' if sub else ""
    _md(f'<div class="ath-ct"><span class="ct-t">{_esc(title)}</span>{s}</div>')


# --- sidebar scoreboard ---------------------------------------------------------------------

def label_counts(fraud: int, legit: int, unsure: int) -> None:
    _md(
        '<div class="ath-counts">'
        f'<div class="c"><div class="n" style="color:#d65f5f;">{fraud}</div><div class="l">fraud</div></div>'
        f'<div class="c"><div class="n" style="color:#4f9d69;">{legit}</div><div class="l">legit</div></div>'
        f'<div class="c"><div class="n" style="color:#8b929c;">{unsure}</div><div class="l">unsure</div></div>'
        "</div>"
    )


def kpi_cards(precision: float, recall: float, accuracy: float,
             tp: int, fp: int, tn: int, fn: int) -> None:
    _md(
        '<div class="ath-kpis">'
        f'<div class="k"><div class="n">{precision:.2f}</div><div class="l">PRECISION</div></div>'
        f'<div class="k"><div class="n">{recall:.2f}</div><div class="l">RECALL</div></div>'
        f'<div class="k"><div class="n">{accuracy:.2f}</div><div class="l">ACC</div></div>'
        "</div>"
        f'<div class="ath-conf">TP {tp} · FP {fp} · TN {tn} · FN {fn}</div>'
    )


def verdict_badge(tag: str) -> str:
    """Inline colored verdict pill for the profile header (returns HTML)."""
    if tag not in _TRUTH:
        return ""
    dot, txt, label = _TRUTH[tag]
    bg = {"this-ring": "#2a1d1f", "other-ring": "#2a2218", "legit": "#16241b"}[tag]
    bd = {"this-ring": "#5a3438", "other-ring": "#5a4a2f", "legit": "#2f5a3c"}[tag]
    return (f'<span class="ath-badge" style="background:{bg};border:1px solid {bd};color:{txt};">'
            f'● {label}</span>')
