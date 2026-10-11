"""Build the project site: `docs/` (markdown, images, lessons, charts) -> `_site/`.

    .venv/bin/python scripts/build_site.py            # writes _site/
    python -m http.server --directory _site 8000      # preview; byte-identical to production

Needs only the `site` extra (markdown-it-py, jinja2), not `[viz]`. The markdown in `docs/` stays the
source of truth and keeps reading well on GitHub; this script only adds what GitHub can't show:
the shared layout, a nav bar, interactive charts and links that work on both.

What it does to the markdown:
  * heading ids match GitHub's slugs exactly, so existing `#fragment` links keep working;
  * `x.md` links become `x.html`, `README.md` becomes the directory index, and links that leave
    `docs/` become links into the repo on GitHub;
  * images get `width`/`height` at half their pixel size (captures are 2x), lazy loading and zoom;
  * `![alt](img.png)<!-- chart: name -->` becomes an interactive Plotly figure from
    `docs/charts/name.json`, with the PNG kept for GitHub and as the no-JS fallback.

Every broken link or unknown chart is a build error, not a warning.
"""
from __future__ import annotations

import argparse
import html
import json
import posixpath
import re
import shutil
import struct
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markdown_it import MarkdownIt
from markdown_it.token import Token

ROOT = Path(__file__).resolve().parents[1]
CHART_RE = re.compile(r"^<!--\s*chart:\s*([\w-]+)\s*-->$")
SCHEME_RE = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|//)", re.I)
MAX_CHART_BYTES = 250_000
MAX_PAGE_CHART_BYTES = 1_000_000
# Docs subtrees that are inputs to scripts, not part of the site.
SKIP_DIRS = {"results"}


class BuildError(Exception):
    pass


@dataclass
class Config:
    title: str
    tagline: str
    description: str
    site_url: str
    live_app_url: str
    repo_url: str
    repo_branch: str
    plotly_version: str

    @classmethod
    def load(cls, root: Path) -> Config:
        data = tomllib.loads((root / "site" / "site.toml").read_text())
        return cls(**data)

    @property
    def plotly_url(self) -> str:
        return f"https://cdn.plot.ly/plotly-{self.plotly_version}.min.js"


@dataclass
class Page:
    src: str  # docs-relative markdown path
    out: str  # site-relative output path
    title: str = ""
    description: str = ""
    html: str = ""
    charts: list[str] = field(default_factory=list)


# --- small pure helpers ------------------------------------------------------------------------


def github_slug(text: str) -> str:
    """GitHub's heading anchor: lowercase, drop anything but word chars, spaces and hyphens, then
    spaces become hyphens. (`Act 1 — Triage` -> `act-1--triage`.)"""
    return re.sub(r"[^\w\- ]", "", text.lower()).replace(" ", "-")


def out_path(src: str) -> str:
    """docs-relative `.md` path -> site-relative `.html` path (`README.md` -> `index.html`)."""
    d, name = posixpath.split(src)
    name = "index.html" if name == "README.md" else name[:-3] + ".html"
    return posixpath.join(d, name)


def rel_url(target_out: str, from_out: str) -> str:
    """URL of `target_out` as seen from the page at `from_out` (relative, so the site is portable)."""
    url = posixpath.relpath(target_out, posixpath.dirname(from_out) or ".")
    if url == "index.html":
        return "./"
    return url[: -len("index.html")] if url.endswith("/index.html") else url


def image_size(path: Path) -> tuple[int, int] | None:
    """Pixel size of a PNG or GIF, read from the header (no Pillow needed)."""
    with path.open("rb") as f:
        head = f.read(26)
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", head[16:24])
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return struct.unpack("<HH", head[6:10])
    return None


def inline_text(tok: Token) -> str:
    parts = []
    for c in tok.children or []:
        if c.type in ("text", "code_inline"):
            parts.append(c.content)
        elif c.type in ("softbreak", "hardbreak"):
            parts.append(" ")
    return "".join(parts)


# --- the builder -------------------------------------------------------------------------------


class Site:
    def __init__(self, root: Path, docs: Path, out: Path, cfg: Config) -> None:
        self.root, self.docs, self.out, self.cfg = root, docs, out, cfg
        self.md = MarkdownIt("commonmark", {"html": True}).enable(["table", "strikethrough"])
        self.env = Environment(
            loader=FileSystemLoader(root / "site" / "templates"),
            autoescape=select_autoescape(["html"]),
            trim_blocks=True, lstrip_blocks=True,
        )
        self.sources = sorted(p.relative_to(docs).as_posix() for p in docs.rglob("*.md"))
        self.pages: dict[str, Page] = {s: Page(s, out_path(s)) for s in self.sources}
        self.charts = self._load_charts()
        self.written: list[str] = []  # every html file, for the sitemap

    # -- charts ----------------------------------------------------------------------------

    def _load_charts(self) -> dict[str, int]:
        """name -> byte size. Checks that every chart was exported with the pinned plotly.js."""
        sizes: dict[str, int] = {}
        for f in sorted((self.docs / "charts").glob("*.json")):
            meta = json.loads(f.read_text())
            version = meta.get("plotlyjs")
            if version != self.cfg.plotly_version:
                raise BuildError(f"{f.name}: exported with plotly.js {version}, site pins {self.cfg.plotly_version}")
            if f.stat().st_size > MAX_CHART_BYTES:
                raise BuildError(f"{f.name}: {f.stat().st_size} bytes exceeds the {MAX_CHART_BYTES} budget")
            sizes[f.stem] = f.stat().st_size
        return sizes

    # -- links -----------------------------------------------------------------------------

    def rewrite(self, href: str, page: Page) -> tuple[str, bool]:
        """(new href, is_external)."""
        if SCHEME_RE.match(href):
            return href, True
        if href.startswith("#"):
            return href, False
        path, _, frag = href.partition("#")
        frag = f"#{frag}" if frag else ""
        target = posixpath.normpath(posixpath.join(posixpath.dirname(page.src), path))
        if target.startswith(".."):  # leaves docs/: link into the repo on GitHub
            repo_path = posixpath.normpath(posixpath.join("docs", target))
            if repo_path.startswith(".."):
                raise BuildError(f"{page.src}: link {href!r} leaves the repository")
            if not (self.root / repo_path).exists():
                raise BuildError(f"{page.src}: link {href!r} -> {repo_path} does not exist in the repo")
            kind = "tree" if (self.root / repo_path).is_dir() else "blob"
            return f"{self.cfg.repo_url}/{kind}/{self.cfg.repo_branch}/{repo_path}{frag}", True
        if target in self.pages:  # a markdown source
            return rel_url(self.pages[target].out, page.out) + frag, False
        if (self.docs / target).is_dir():
            index = posixpath.join(target, "README.md")
            if index not in self.pages:
                raise BuildError(f"{page.src}: link {href!r} is a directory with no README.md")
            return rel_url(self.pages[index].out, page.out) + frag, False
        if target.endswith(".html") and target[:-5] + ".md" in self.pages:  # link written against the output
            return rel_url(target, page.out) + frag, False
        if (self.docs / target).exists():  # image, lesson page, asset
            return rel_url(target, page.out) + frag, False
        raise BuildError(f"{page.src}: broken link {href!r}")

    # -- markdown --------------------------------------------------------------------------

    def render(self, page: Page) -> None:
        tokens = self.md.parse((self.docs / page.src).read_text())
        seen: dict[str, int] = {}
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok.type == "heading_open":
                text = inline_text(tokens[i + 1])
                if tok.tag == "h1" and not page.title:
                    page.title = text
                slug = github_slug(text)
                n = seen.get(slug, 0)
                seen[slug] = n + 1
                tok.attrSet("id", slug if n == 0 else f"{slug}-{n}")
            elif tok.type == "paragraph_open" and self._is_chart_paragraph(tokens[i + 1]):
                tokens[i : i + 3] = [self._chart_figure(tokens[i + 1], page)]
            elif tok.type == "inline":
                if not page.description and tokens[i - 1].type == "paragraph_open":
                    page.description = re.sub(r"\s+", " ", inline_text(tok)).strip()[:200]
                self._fix_inline(tok, page)
            i += 1
        page.html = self.md.renderer.render(tokens, self.md.options, {})

    @staticmethod
    def _is_chart_paragraph(inline: Token) -> bool:
        kids = inline.children or []
        return (inline.type == "inline" and len(kids) == 2 and kids[0].type == "image"
                and kids[1].type == "html_inline" and bool(CHART_RE.match(kids[1].content.strip())))

    def _chart_figure(self, inline: Token, page: Page) -> Token:
        img, comment = inline.children or []
        match = CHART_RE.match(comment.content.strip())
        assert match
        name = match.group(1)
        if name not in self.charts:
            raise BuildError(f"{page.src}: unknown chart {name!r} (no docs/charts/{name}.json)")
        page.charts.append(name)
        fallback, _ = self.rewrite(str(img.attrGet("src")), page)
        alt = html.escape(img.content, quote=True)
        src_json = rel_url(f"charts/{name}.json", page.out)
        tok = Token("html_block", "", 0)
        tok.content = (
            f'<figure class="chart" data-chart="{name}" data-src="{src_json}">'
            f'<noscript><img src="{html.escape(fallback)}" alt="{alt}"></noscript>'
            f"<figcaption>{html.escape(img.content)}</figcaption></figure>\n"
        )
        return tok

    def _fix_inline(self, inline: Token, page: Page) -> None:
        for tok in inline.children or []:
            if tok.type == "link_open":
                href, external = self.rewrite(str(tok.attrGet("href")), page)
                tok.attrSet("href", href)
                if external:
                    tok.attrSet("rel", "noopener")
            elif tok.type == "image":
                orig = str(tok.attrGet("src"))
                src, external = self.rewrite(orig, page)
                tok.attrSet("src", src)
                tok.attrSet("loading", "lazy")
                tok.attrSet("class", "zoom")
                if not external:
                    f = self.docs / posixpath.normpath(posixpath.join(posixpath.dirname(page.src), orig))
                    size = image_size(f) if f.is_file() else None
                    if size:  # captures are 2x: show them at half size
                        tok.attrSet("width", str(size[0] // 2))
                        tok.attrSet("height", str(size[1] // 2))

    # -- output ----------------------------------------------------------------------------

    def nav(self, out: str) -> list[dict]:
        cfg = self.cfg
        items = [
            ("Home", "index.html", "index.html"),
            ("Tour", "tour.html", "tour.html"),
            ("Walkthrough", "walkthrough.html", "walkthrough.html"),
            ("Lessons", "explainer/index.html", "explainer/"),
            ("Research", "research/index.html", "research/"),
            ("Decisions", "adr/index.html", "adr/"),
        ]
        nav = []
        for label, target, section in items:
            if not self._exists(target):
                continue  # a page that has not been written yet (the build prints what it skipped)
            active = out == section if section.endswith(".html") else out.startswith(section)
            nav.append({"label": label, "href": rel_url(target, out), "active": active})
        nav.append({"label": "Live app ↗", "href": cfg.live_app_url, "external": True,
                    "title": "Opens the live demo. If it has been idle it may take about 30 seconds to wake."})
        nav.append({"label": "GitHub ↗", "href": cfg.repo_url, "external": True})
        return nav

    def _exists(self, out: str) -> bool:
        if out == "index.html":
            return True
        return any(p.out == out for p in self.pages.values())

    def context(self, out: str, **extra) -> dict:
        root = posixpath.relpath(".", posixpath.dirname(out) or ".")
        root = "" if root == "." else root + "/"
        social = (self.docs / "images/site/social.png").exists()
        ctx = {
            "cfg": self.cfg, "root": root, "nav": self.nav(out),
            "canonical": self.cfg.site_url + out.removesuffix("index.html"),
            "og_image": self.cfg.site_url + "images/site/social.png" if social else "",
            "plotly_url": self.cfg.plotly_url, "charts": [],
            "title": self.cfg.title, "description": self.cfg.description,
        }
        ctx.update(extra)
        return ctx

    def write(self, out: str, content: str) -> None:
        dest = self.out / out
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)
        if out.endswith(".html"):
            self.written.append(out)

    def build(self) -> None:
        if self.out.exists():
            shutil.rmtree(self.out)
        self.out.mkdir(parents=True)

        # markdown pages
        page_tpl = self.env.get_template("page.html")
        for page in self.pages.values():
            self.render(page)
            budget = sum(self.charts[c] for c in page.charts)
            if budget > MAX_PAGE_CHART_BYTES:
                raise BuildError(f"{page.src}: {budget} bytes of chart JSON exceeds the {MAX_PAGE_CHART_BYTES} budget")
            title = page.title or self.cfg.title
            self.write(page.out, page_tpl.render(self.context(
                page.out, content=page.html, charts=page.charts,
                title=f"{title} · {self.cfg.title}" if page.out != "index.html" else title,
                description=page.description or self.cfg.description,
            )))

        # templated pages
        self.write("index.html", self.env.get_template("home.html").render(self.context("index.html")))
        self.write("404.html", self.env.get_template("404.html").render(
            self.context("404.html", title=f"Not found · {self.cfg.title}")))
        self._copy_static()
        self._write_sitemap()

    def _copy_static(self) -> None:
        shutil.copytree(self.root / "site" / "assets", self.out / "assets")
        nav_tpl = self.env.get_template("nav.html")
        for f in sorted(self.docs.rglob("*")):
            rel = f.relative_to(self.docs).as_posix()
            if f.is_dir() or f.suffix == ".md" or f.relative_to(self.docs).parts[0] in SKIP_DIRS:
                continue
            dest = self.out / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if rel.startswith("explainer/lessons/") and f.suffix == ".html":
                text = f.read_text()
                nav = nav_tpl.render(self.context(rel))
                css = f'  <link rel="stylesheet" href="{self.context(rel)["root"]}assets/nav.css">\n</head>'
                if "</head>" not in text or "<body>" not in text:
                    raise BuildError(f"{rel}: expected <head> and <body> to inject the nav")
                text = text.replace("</head>", css, 1).replace("<body>", "<body>\n" + nav, 1)
                dest.write_text(text)
                self.written.append(rel)
            else:
                shutil.copy2(f, dest)

    def _write_sitemap(self) -> None:
        urls = sorted(p for p in self.written if p != "404.html")
        body = "".join(f"  <url><loc>{html.escape(self.cfg.site_url + p.removesuffix('index.html'))}</loc></url>\n"
                       for p in urls)
        (self.out / "sitemap.xml").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n{body}</urlset>\n')


def build(root: Path = ROOT, out: Path | None = None, docs: Path | None = None) -> Path:
    out = out or root / "_site"
    Site(root, docs or root / "docs", out, Config.load(root)).build()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "_site")
    args = ap.parse_args()
    try:
        out = build(out=args.out)
    except BuildError as e:
        sys.exit(f"build error: {e}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
