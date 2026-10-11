"""The static site build: every link resolves, every anchor exists, charts and images are handled."""

from __future__ import annotations

import json
import re
import shutil
import struct
import sys
import zlib
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

pytest.importorskip("markdown_it")
pytest.importorskip("jinja2")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_site as B  # type: ignore[import-not-found]  # noqa: E402  # pyright: ignore[reportMissingImports]


class _Scan(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.refs: list[tuple[str, str]] = []  # (tag, url)
        self.imgs: list[dict[str, str | None]] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if a.get("id"):
            self.ids.add(a["id"] or "")
        for key in ("href", "src"):
            if a.get(key):
                self.refs.append((tag, a[key] or ""))
        if tag == "img":
            self.imgs.append(a)


def _scan(path: Path) -> _Scan:
    s = _Scan()
    s.feed(path.read_text())
    return s


@pytest.fixture(scope="module")
def site(tmp_path_factory) -> Path:
    return B.build(ROOT, tmp_path_factory.mktemp("site"))


def test_slugs_match_github() -> None:
    assert B.github_slug("Act 1 — Triage — find somewhere to start") == "act-1--triage--find-somewhere-to-start"
    assert B.github_slug("6. Evaluation discipline") == "6-evaluation-discipline"
    assert B.github_slug("What you just did") == "what-you-just-did"


def test_every_internal_link_and_fragment_resolves(site: Path) -> None:
    pages = sorted(site.rglob("*.html"))
    assert len(pages) > 20
    scans = {p: _scan(p) for p in pages}
    repo = f"{B.Config.load(ROOT).repo_url}/"
    repo_links = 0
    for page, scan in scans.items():
        for _tag, url in scan.refs:
            if url.startswith(repo):  # a link into the repo: the path must exist at HEAD
                rel = unquote(urlsplit(url).path)[len(urlsplit(repo).path):]
                m = re.match(r"(blob|tree)/[^/]+/(.+)", rel)
                assert m and (ROOT / m.group(2)).exists(), f"{page}: {url}"
                repo_links += 1
                continue
            parts = urlsplit(url)
            if parts.scheme or url.startswith("//"):
                continue
            target = page if not parts.path else (page.parent / unquote(parts.path))
            if target.is_dir():
                target = target / "index.html"
            assert target.exists(), f"{page.relative_to(site)}: {url} does not resolve"
            assert not str(parts.path).endswith(".md"), f"{page.relative_to(site)}: {url} still points at markdown"
            if parts.fragment and target.suffix == ".html":
                ids = scans[target].ids if target in scans else _scan(target).ids
                assert parts.fragment in ids, f"{page.relative_to(site)}: #{parts.fragment} missing in {target.name}"
    assert repo_links >= 3, "expected links out to the repo (README results, GLOSSARY, ...)"


def test_home_links_the_live_app_and_nav_is_everywhere(site: Path) -> None:
    cfg = B.Config.load(ROOT)
    assert cfg.live_app_url in (site / "index.html").read_text()
    for rel in ("walkthrough.html", "explainer/index.html", "explainer/lessons/0001-narrow-then-rank.html",
                "research/method.html", "adr/index.html", "404.html"):
        text = (site / rel).read_text()
        assert 'id="site-nav"' in text and cfg.live_app_url in text, rel


def test_lessons_are_unchanged_apart_from_the_nav(site: Path) -> None:
    src = (ROOT / "docs/explainer/lessons/0003-four-lines-of-evidence.html").read_text()
    out = (site / "explainer/lessons/0003-four-lines-of-evidence.html").read_text()
    stripped = re.sub(r'  <link rel="stylesheet" href="[^"]*nav\.css">\n', "", out)
    stripped = re.sub(r'<header class="site-nav".*?</header>\n', "", stripped, flags=re.S)
    assert stripped == src


def test_images_are_shown_at_half_size_and_lazy(site: Path) -> None:
    imgs = [i for i in _scan(site / "walkthrough.html").imgs if (i.get("src") or "").endswith("02-triage-queue.png")]
    assert imgs
    w, h = B.image_size(ROOT / "docs/images/walkthrough/02-triage-queue.png") or (0, 0)
    assert (imgs[0]["width"], imgs[0]["height"]) == (str(w // 2), str(h // 2))
    assert imgs[0]["loading"] == "lazy"


def test_sitemap_and_meta(site: Path) -> None:
    cfg = B.Config.load(ROOT)
    assert f"<loc>{cfg.site_url}walkthrough.html</loc>" in (site / "sitemap.xml").read_text()
    head = (site / "walkthrough.html").read_text()
    assert 'rel="canonical"' in head and 'property="og:title"' in head and 'name="twitter:card"' in head


def _png(w: int, h: int) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    raw = b"".join(b"\x00" + b"\x00\x00\x00" * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def _mini_docs(tmp: Path, md: str, charts: dict[str, dict] | None = None) -> Path:
    """A copy of the real docs plus one extra page (`page.md`) and any extra chart files."""
    docs = tmp / "docs"
    shutil.copytree(ROOT / "docs", docs)
    (docs / "images" / "a.png").write_bytes(_png(4, 2))
    (docs / "page.md").write_text(md)
    for name, meta in (charts or {}).items():
        (docs / "charts" / f"{name}.json").write_text(json.dumps({"data": [], "layout": {}, **meta}))
    return docs


def test_chart_directive_becomes_a_figure(tmp_path: Path) -> None:
    cfg = B.Config.load(ROOT)
    docs = _mini_docs(tmp_path, "# T\n\nintro\n\n![A chart.](images/a.png)<!-- chart: demo -->\n",
                      {"demo": {"plotlyjs": cfg.plotly_version}})
    out = B.build(ROOT, tmp_path / "out", docs)
    html = (out / "page.html").read_text()
    assert 'data-chart="demo"' in html and 'data-src="charts/demo.json"' in html
    assert "<noscript><img" in html and cfg.plotly_url in html
    assert (out / "charts" / "demo.json").exists()


def test_unknown_chart_and_version_mismatch_are_errors(tmp_path: Path) -> None:
    cfg = B.Config.load(ROOT)
    docs = _mini_docs(tmp_path / "a", "# T\n\n![x](images/a.png)<!-- chart: nope -->\n")
    with pytest.raises(B.BuildError, match="unknown chart"):
        B.build(ROOT, tmp_path / "a" / "out", docs)
    docs = _mini_docs(tmp_path / "b", "# T\n", {"old": {"plotlyjs": "0.0.1"}})
    with pytest.raises(B.BuildError, match="plotly.js"):
        B.build(ROOT, tmp_path / "b" / "out", docs)
    assert cfg.plotly_version != "0.0.1"


def test_broken_link_is_an_error(tmp_path: Path) -> None:
    docs = _mini_docs(tmp_path, "# T\n\n[gone](missing.md)\n")
    with pytest.raises(B.BuildError, match="broken link"):
        B.build(ROOT, tmp_path / "out", docs)


def test_descriptions_keep_words_apart(site: Path) -> None:
    """Hard-wrapped markdown must not fuse the words on either side of a line break."""
    text = (site / "walkthrough.html").read_text()
    assert "number below comes" in text.split('name="description" content="', 1)[1].split('"', 1)[0]


def test_every_chart_directive_in_the_docs_is_consumed(site: Path) -> None:
    wanted = sum(p.read_text().count("<!-- chart:") for p in (ROOT / "docs").rglob("*.md"))
    got = sum(p.read_text().count('data-chart="') for p in site.rglob("*.html") if p.name != "index.html"
              or p.parent != site)  # the home page's own chart is placed by its template
    assert wanted == got and wanted >= 4


def test_nav_toggle_only_exists_with_js() -> None:
    css = (ROOT / "site/assets/nav.css").read_text()
    assert ".js .site-nav .nav-toggle { display: block; }" in css
    assert ".site-nav .nav-toggle { display: none;" in css


def test_headline_numbers_come_from_the_results_file(site: Path) -> None:
    row = next(r for r in json.loads((ROOT / "docs/results/rule_grades.json").read_text())["levels"]
               if r["level"] == "sophisticated")
    home = (site / "index.html").read_text()
    want = f"{row['baseline_recall_at_k']:.2f} → {row['recall']:.2f}"
    assert want in home and f"{row['legit_fp_rate']:.0%}" in home


def test_home_pitch_and_ring_paragraph_are_the_readmes(site: Path) -> None:
    readme = (ROOT / "README.md").read_text()
    home = (site / "index.html").read_text()
    for name in ("pitch", "ring"):
        block = re.search(rf"<!-- site:{name} -->\n(.*?)\n<!-- /site:{name} -->", readme, re.S)
        assert block
        probe = block.group(1).split(".")[0].replace("**", "")
        assert probe in home.replace("<strong>", "").replace("</strong>", ""), name


def test_every_live_app_link_is_the_configured_one_and_has_a_title(site: Path) -> None:
    cfg = B.Config.load(ROOT)
    host = urlsplit(cfg.live_app_url).netloc
    seen = 0
    for page in site.rglob("*.html"):
        for m in re.finditer(r'<a\b[^>]*href="([^"]*)"[^>]*>', page.read_text()):
            if host in m.group(1):
                seen += 1
                assert m.group(1).startswith(cfg.live_app_url), (page, m.group(1))
                assert ' title="' in m.group(0), f"{page.relative_to(site)}: live-app link without a title"
    assert seen > 10
    readme = (ROOT / "README.md").read_text()
    for m in re.finditer(r"\]\((https?://[^)\s]*" + re.escape(host) + r"[^)\s]*)( \"[^\"]+\")?\)", readme):
        assert m.group(2), f"README live-app link without a title: {m.group(1)}"


def test_nav_has_every_item_on_every_page(site: Path) -> None:
    labels = ["Home", "Tour", "Walkthrough", "Lessons", "Research", "Decisions", "Live app ↗", "GitHub ↗"]
    for page in site.rglob("*.html"):
        nav = re.search(r'<header class="site-nav".*?</header>', page.read_text(), re.S)
        assert nav, page
        assert re.findall(r">([^<]+)</a>", nav.group(0))[1:] == labels, page.relative_to(site)
