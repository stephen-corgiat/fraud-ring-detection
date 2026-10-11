"""Regenerate every screenshot in the docs, deterministically.

    .venv/bin/python scripts/capture_docs.py            # writes docs/images/walkthrough/*.png + hero.gif

What it does, end to end:
  1. builds the seeded worlds the walkthrough uses (default preset, world seed 7) into a temp dir;
  2. pre-writes a few labels so the scoreboards have something to show;
  3. launches the viewer headless on a free port against that dir;
  4. drives headless Chrome over the DevTools Protocol to each deep link, waits for Streamlit's
     websocket render, captures the page at 2x, and crops each figure by *locating its heading text
     in the DOM* — so crops survive layout changes instead of relying on pixel offsets;
  5. assembles the README hero GIF from a subset of the frames.

Requires the [viz] and [docs] extras and Chrome/Chromium. Every world, label and crop is fixed, so
re-running produces the same images (modulo font rendering on a different machine).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from io import BytesIO

import websocket
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from capture_ui import find_chrome  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "docs", "images", "walkthrough")
WIDTH, HEIGHT, SCALE = 1680, 9000, 2
SIDEBAR_W = 300  # css px


@dataclass
class Crop:
    name: str
    top: str                     # text of the element the crop starts at (searched in the DOM)
    bottom: str | None = None    # text of the element the crop ends before; None => `height` px
    height: int = 600            # css px when `bottom` is None
    region: str = "main"         # "main" | "sidebar" | "full"
    pad_top: int = 12
    nth: int = 0                 # which match of `top` to use (0 = first)


@dataclass
class Shot:
    query: str                   # path + query string, e.g. "/?level=naive&rank=1"
    crops: list[Crop] = field(default_factory=list)
    wait: float = 14.0
    scroll_to: str | None = None # click nothing; just ensure lazy content renders


# --- the walkthrough's shot list --------------------------------------------------------------

NAIVE = "preset=default&level=naive&seed=7"
SOPH = "preset=default&level=sophisticated&seed=7"
SHOTS: list[Shot] = [
    Shot(f"/?{NAIVE}&rank=1", [
        Crop("01-sidebar-world", "WORLD", "RANKER", region="sidebar"),
        Crop("02-triage-queue", "suspect creators", "inspect rank", pad_top=20),
        Crop("03-suspect-ring", "Suspect · creator", "The core"),
        Crop("04-core", "The core", "Your call"),
        Crop("05-your-call", "Your call", None, height=230),
        Crop("06-triage-scoreboard", "YOUR CALLS", None, height=210, region="sidebar"),
    ], wait=22),
    Shot(f"/?{NAIVE}&rank=3", [
        Crop("07-suspect-organic", "Suspect · creator", "Your call"),
    ]),
    Shot("/?preset=default&level=moderate&seed=7&rank=1", [
        Crop("08-triage-queue-moderate", "suspect creators", "inspect rank", pad_top=20),
    ], wait=22),
    Shot(f"/investigate?{NAIVE}&creator=1", [
        Crop("10-handoff-seeds", "SUBJECT", "Profile"),
        Crop("11-profile", "Profile", "Related accounts"),
        Crop("12-related", "Related accounts", "Inspect a candidate"),
        Crop("13-inspector", "Inspect a candidate", "Axis explainers"),
        Crop("14-eda-axes", "Axis explainers", "Linkage subgraph"),
        Crop("15-linkage", "Linkage subgraph", None, height=520),
        Crop("16-sidebar-scoreboard", "YOUR LABELS", None, height=230, region="sidebar"),
    ], wait=30),
    Shot(f"/investigate?{SOPH}&ring=1", [
        Crop("20-related-sophisticated", "Related accounts", "Inspect a candidate"),
        Crop("21-eda-sophisticated", "Axis explainers", "Reported position"),
    ], wait=30),
    Shot(f"/investigate?{SOPH}&conc=4&ring=1", [
        Crop("30-temporal-raster", "Temporal raster", None, height=760),
    ], wait=40),
]

# Frames (in order) for the README hero GIF.
HERO_FRAMES = ["02-triage-queue", "03-suspect-ring", "04-core", "12-related", "15-linkage", "30-temporal-raster"]


# --- worlds + labels ---------------------------------------------------------------------------


def prepare_worlds(data_dir: str) -> None:
    """Build every world the shots use and seed the label stores the scoreboards read."""
    from fraudrings.config import preset
    from fraudrings.viewer import data as D
    from fraudrings.viewer import data_triage as R
    from fraudrings.viewer import labels as L

    for level, conc in (("naive", None), ("moderate", None), ("sophisticated", None), ("sophisticated", 4)):
        world = D.load_world("default", level, 7, cotarget_on=True, base_dir=data_dir, concurrency=conc)
        if level == "naive" and conc is None:
            suspects = R.suspects(world.con, preset("default").concentration)
            truth = R.target_truth(world)
            store = R.store_for_world(world.world_dir)
            ring = next(s for s in suspects if truth.get(s.target_id) and truth[s.target_id].is_ring_target)
            organic = next(s for s in suspects if not (truth.get(s.target_id) and truth[s.target_id].is_ring_target))
            store.set(ring.target_id, "ring", note="37 dedicated boosters + a one-week campaign",
                      suspicion=ring.suspicion, rank=suspects.index(ring) + 1)
            store.set(organic.target_id, "legit", note="concentrated, but no core and no burst",
                      suspicion=organic.suspicion, rank=suspects.index(organic) + 1)
            acct = L.store_for_world(world.world_dir)
            for a in ring.core[:6]:
                acct.set(a, "fraud")
        world.con.close()


# --- browser -----------------------------------------------------------------------------------


class Browser:
    def __init__(self, port: int = 9333) -> None:
        self.port = port
        self.profile = tempfile.mkdtemp(prefix="fraudrings-chrome-")
        self.proc = subprocess.Popen(
            [find_chrome(), "--headless=new", f"--remote-debugging-port={port}", "--remote-allow-origins=*",
             "--hide-scrollbars", f"--window-size={WIDTH},{HEIGHT}", "--no-first-run",
             "--no-default-browser-check", f"--user-data-dir={self.profile}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        ws_url = None
        for _ in range(80):
            try:
                pages = [t for t in json.loads(urllib.request.urlopen(f"http://localhost:{port}/json").read())
                         if t["type"] == "page"]
                if pages:
                    ws_url = pages[0]["webSocketDebuggerUrl"]
                    break
            except Exception:
                pass
            time.sleep(0.25)
        if not ws_url:
            raise SystemExit("could not reach Chrome over CDP")
        self.ws = websocket.create_connection(ws_url, max_size=None)
        self._id = 0
        self.rpc("Page.enable")
        self.rpc("Emulation.setDeviceMetricsOverride",
                 {"width": WIDTH, "height": HEIGHT, "deviceScaleFactor": SCALE, "mobile": False})

    def rpc(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        self.ws.send(json.dumps({"id": self._id, "method": method, "params": params or {}}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self._id:
                return msg.get("result", {})

    def eval(self, js: str):
        return self.rpc("Runtime.evaluate", {"expression": js, "returnByValue": True})["result"].get("value")

    def shot(self, url: str, wait: float) -> Image.Image:
        self.rpc("Page.navigate", {"url": url})
        time.sleep(wait)
        data = self.rpc("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})["data"]
        return Image.open(BytesIO(base64.b64decode(data))).convert("RGB")

    def locate(self, text: str, nth: int = 0) -> float | None:
        """Top (css px) of the innermost visible element whose text starts with `text`."""
        js = f"""(() => {{
            const want = {json.dumps(text)};
            const hits = [...document.querySelectorAll('body *')].filter(e => {{
                const t = (e.innerText || '').trim();
                if (!t.startsWith(want)) return false;
                const r = e.getBoundingClientRect();
                return r.height > 0 && [...e.children].every(c => !(c.innerText || '').trim().startsWith(want));
            }}).map(e => e.getBoundingClientRect().top + window.scrollY).sort((a, b) => a - b);
            return hits.length > {nth} ? hits[{nth}] : null;
        }})()"""
        return self.eval(js)

    def close(self) -> None:
        self.ws.close()
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
        shutil.rmtree(self.profile, ignore_errors=True)


def crop(img: Image.Image, browser: Browser, c: Crop) -> Image.Image:
    top = browser.locate(c.top, c.nth)
    if top is None:
        raise SystemExit(f"crop {c.name}: could not find {c.top!r}")
    if c.bottom is not None:
        bottom = browser.locate(c.bottom)
        if bottom is None or bottom <= top:
            raise SystemExit(f"crop {c.name}: could not find {c.bottom!r} below {c.top!r}")
        bottom -= 8
    else:
        bottom = top + c.height
    left, right = {"main": (SIDEBAR_W + 40, WIDTH - 40), "sidebar": (0, SIDEBAR_W), "full": (0, WIDTH)}[c.region]
    box = (left * SCALE, max(0, top - c.pad_top) * SCALE, right * SCALE, min(HEIGHT, bottom) * SCALE)
    return img.crop(box)


# --- app ---------------------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def launch_app(data_dir: str, port: int) -> subprocess.Popen:
    app = os.path.join(ROOT, "src", "fraudrings", "viewer", "main.py")
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", app, "--server.headless", "true",
         "--server.port", str(port), "--browser.gatherUsageStats", "false", "--", "--data-dir", data_dir],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(120):
        try:
            if urllib.request.urlopen(f"http://localhost:{port}/_stcore/health").read() == b"ok":
                return proc
        except Exception:
            pass
        time.sleep(0.5)
    proc.kill()
    raise SystemExit("viewer did not come up")


def make_gif(frames: list[str], out: str, width: int = 1200, ms: int = 2600) -> None:
    imgs = []
    for name in frames:
        im = Image.open(os.path.join(OUT_DIR, f"{name}.png")).convert("RGB")
        h = int(im.height * width / im.width)
        im = im.resize((width, h), Image.LANCZOS)
        canvas = Image.new("RGB", (width, 760), (14, 17, 22))
        canvas.paste(im.crop((0, 0, width, min(h, 760))), (0, 0))
        imgs.append(canvas.quantize(colors=200, method=Image.Quantize.MEDIANCUT))
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=ms, loop=0, optimize=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", help="capture only the named crops (default: all)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    data_dir = tempfile.mkdtemp(prefix="fraudrings-docs-")
    print(f"building worlds in {data_dir} ...")
    prepare_worlds(data_dir)
    port = free_port()
    app = launch_app(data_dir, port)
    browser = Browser()
    try:
        for s in SHOTS:
            wanted = [c for c in s.crops if not args.only or c.name in args.only]
            if not wanted:
                continue
            img = browser.shot(f"http://localhost:{port}{s.query}", s.wait)
            for c in wanted:
                out = os.path.join(OUT_DIR, f"{c.name}.png")
                crop(img, browser, c).save(out, optimize=True)
                print(f"wrote {os.path.relpath(out, ROOT)}")
    finally:
        browser.close()
        app.terminate()
        shutil.rmtree(data_dir, ignore_errors=True)
    if not args.only:
        make_gif(HERO_FRAMES, os.path.join(OUT_DIR, "hero.gif"))
        print("wrote docs/images/walkthrough/hero.gif")


if __name__ == "__main__":
    main()
