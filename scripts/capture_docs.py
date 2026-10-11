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

With `--video` it instead records the site's demo clip (docs/images/site/demo.mp4, demo.webm,
demo-poster.png): a real session driven over CDP with a visible cursor, captured with
`Page.startScreencast` and encoded by ffmpeg at a constant frame rate.

Requires the [viz] and [docs] extras and Chrome/Chromium (and ffmpeg for `--video`). Every world, label and
crop is fixed, so re-running produces the same images (modulo font rendering on a different machine).
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
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from io import BytesIO

import websocket
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
from capture_ui import find_chrome  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "docs", "images", "walkthrough")
WIDTH, HEIGHT, SCALE = 1680, 9000, 2
SIDEBAR_W = 300  # css px


@dataclass
class Callout:
    """A numbered marker for the Tour's annotated screenshots, drawn at the top-left of an element that
    is located in the DOM (by its text, or by a CSS selector for things with no text, such as the
    queue table) and never by a pixel offset."""
    n: int
    text: str | None = None
    selector: str | None = None
    nth: int = 0
    side: str = "left"  # which side of the element the disc sits on: "left" | "right"


@dataclass
class Crop:
    name: str
    top: str                     # text of the element the crop starts at (searched in the DOM)
    bottom: str | None = None    # text of the element the crop ends before; None => `height` px
    height: int = 600            # css px when `bottom` is None
    region: str = "main"         # "main" | "sidebar" | "full" | "column" (the column holding `top`)
    pad_top: int = 12
    nth: int = 0                 # which match of `top` to use (0 = first)
    callouts: list[Callout] = field(default_factory=list)


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
        # the Tour's annotated screenshots and chart fallbacks (docs/images/tour/)
        Crop("tour/world", "WORLD", "RANKER", region="sidebar", callouts=[
            Callout(1, "preset"), Callout(2, "evasion level"), Callout(3, "world seed (RNG)"),
            Callout(4, selector='[data-testid="stCheckbox"]', nth=1)]),
        Crop("tour/queue", "suspect creators", "Suspect · creator", pad_top=20, callouts=[
            Callout(1, "suspect creators"), Callout(2, "ring targets in top"),
            Callout(3, selector='[data-testid="stDataFrame"]'), Callout(4, "inspect rank")]),
        Crop("tour/axes", "Why it's here", "n_eff", region="column"),
        Crop("tour/contributors", "Contributor concentration", "Temporal concentration", region="column"),
        Crop("tour/weekly", "Temporal concentration", None, height=260, region="column"),
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
        Crop("tour/related", "Related accounts", "Inspect a candidate", callouts=[
            Callout(1, "The engine's output"), Callout(2, "SCORE ="), Callout(3, selector=".ath-scoretbl .compbar"),
            Callout(4, selector=".ath-scoretbl .truthcell", side="right")]),
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

    def rect(self, text: str | None = None, nth: int = 0, selector: str | None = None,
             column: bool = False) -> dict | None:
        """Page rect (css px: left, top, right, bottom) of the innermost visible element whose text
        starts with `text`, or of the `nth` element matching `selector`. With `column`, the rect of the
        Streamlit column that holds it."""
        js = f"""(() => {{
            const want = {json.dumps(text)}, sel = {json.dumps(selector)}, wantCol = {json.dumps(column)};
            let els;
            if (sel) els = [...document.querySelectorAll(sel)];
            else els = [...document.querySelectorAll('body *')].filter(e => {{
                const t = (e.innerText || '').trim();
                if (!t.startsWith(want)) return false;
                return e.getBoundingClientRect().height > 0
                    && [...e.children].every(c => !(c.innerText || '').trim().startsWith(want));
            }});
            els = els.map(e => [e, e.getBoundingClientRect()]).filter(([e, r]) => r.height > 0)
                     .sort((a, b) => a[1].top - b[1].top);
            if (els.length <= {nth}) return null;
            let [el, r] = els[{nth}];
            if (wantCol) {{
                const col = el.closest('[data-testid="stColumn"]');
                if (col) r = col.getBoundingClientRect();
            }}
            return {{left: r.left, top: r.top + window.scrollY, right: r.right, bottom: r.bottom + window.scrollY}};
        }})()"""
        return self.eval(js)

    def locate(self, text: str, nth: int = 0) -> float | None:
        """Top (css px) of the innermost visible element whose text starts with `text`."""
        r = self.rect(text, nth)
        return r["top"] if r else None

    def close(self) -> None:
        self.ws.close()
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
        shutil.rmtree(self.profile, ignore_errors=True)


MARKER_R = 13  # css px
GUTTER = 34    # css px of empty margin added left of an annotated screenshot


def _marker_font() -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=int(MARKER_R * 1.5 * SCALE))  # Pillow's bundled font: same everywhere


def draw_callouts(img: Image.Image, browser: Browser, callouts: list[Callout], origin: tuple[float, float],
                  name: str) -> Image.Image:
    """Draw each marker at its element's top-left. `origin` is the crop's top-left in css px. The disc
    has a white outline and a drop shadow so it reads as a layer on top of the app, not as one of the
    app's own numbered section markers."""
    img = img.convert("RGBA")
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    discs = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sd, dd = ImageDraw.Draw(shadow), ImageDraw.Draw(discs)
    font, r = _marker_font(), MARKER_R * SCALE
    for c in callouts:
        rect = browser.rect(c.text, c.nth, c.selector)
        if rect is None:
            raise SystemExit(f"crop {name}: could not find callout {c.n} ({c.text or c.selector!r})")
        # beside the element, level with its first line
        cx = (rect["left"] - 18 - origin[0]) * SCALE if c.side == "left" else (rect["right"] + 18 - origin[0]) * SCALE
        cy = (rect["top"] + 11 - origin[1]) * SCALE
        sd.ellipse((cx - r, cy - r + 3 * SCALE, cx + r, cy + r + 3 * SCALE), fill=(0, 0, 0, 200))
        dd.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(214, 95, 95, 255), outline=(255, 255, 255, 255),
                   width=3 * SCALE // 2 + 1)
        dd.text((cx, cy), str(c.n), font=font, fill=(255, 255, 255, 255), anchor="mm")
    shadow = shadow.filter(ImageFilter.GaussianBlur(3 * SCALE))
    return Image.alpha_composite(Image.alpha_composite(img, shadow), discs).convert("RGB")


def crop(img: Image.Image, browser: Browser, c: Crop) -> Image.Image:
    anchor = browser.rect(c.top, c.nth, column=c.region == "column")
    if anchor is None:
        raise SystemExit(f"crop {c.name}: could not find {c.top!r}")
    top = browser.locate(c.top, c.nth)
    assert top is not None
    if c.bottom is not None:
        bottom = browser.locate(c.bottom)
        if bottom is None or bottom <= top:
            raise SystemExit(f"crop {c.name}: could not find {c.bottom!r} below {c.top!r}")
        bottom -= 8
    else:
        bottom = top + c.height
    if c.region == "column":
        left, right = anchor["left"], anchor["right"]
    else:
        left, right = {"main": (SIDEBAR_W + 40, WIDTH - 40), "sidebar": (0, SIDEBAR_W), "full": (0, WIDTH)}[c.region]
    y0 = max(0, top - c.pad_top)
    out = img.crop((int(left * SCALE), int(y0 * SCALE), int(right * SCALE), int(min(HEIGHT, bottom) * SCALE)))
    if c.callouts:
        # a left gutter keeps the discs off the app's own labels and widgets
        gutter = GUTTER * SCALE
        padded = Image.new("RGB", (out.width + gutter, out.height), out.getpixel((3, out.height - 3)))
        padded.paste(out, (gutter, 0))
        out = draw_callouts(padded, browser, c.callouts, (left - GUTTER, y0), c.name)
    return out


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


# --- demo video ---------------------------------------------------------------------------------

VIDEO_W, VIDEO_H, VIDEO_FPS = 1280, 720, 25
VIDEO_DIR = os.path.join(ROOT, "docs", "images", "site")

# One page-side helper: find an element by visible text (as `Browser.rect` does), glide a visible
# cursor dot to it, scroll to it, click it.
PAGE_JS = """
window.__frFind = (want, nth = 0) => {
  const els = [...document.querySelectorAll('body *')].filter(e => {
    const t = (e.innerText || '').trim();
    return t.startsWith(want) && e.getBoundingClientRect().height > 0
      && [...e.children].every(c => !(c.innerText || '').trim().startsWith(want));
  }).sort((a, b) => a.getBoundingClientRect().top - b.getBoundingClientRect().top);
  return els[nth] || null;
};
window.__frCursor = () => {
  let c = document.getElementById('fr-cursor');
  if (!c) {
    c = document.createElement('div');
    c.id = 'fr-cursor';
    c.style.cssText = 'position:fixed;z-index:99999;width:22px;height:22px;margin:-11px 0 0 -11px;border-radius:50%;'
      + 'background:rgba(214,95,95,.9);border:2px solid #fff;box-shadow:0 2px 8px rgba(0,0,0,.6);pointer-events:none;'
      + 'left:640px;top:360px;opacity:0;transition:left .8s ease-in-out, top .8s ease-in-out, transform .15s';
    document.body.appendChild(c);
  }
  return c;
};
window.__frMove = (want, nth = 0) => {
  const el = window.__frFind(want, nth);
  if (!el) return false;
  const r = el.getBoundingClientRect(), c = window.__frCursor();
  c.style.opacity = 1;
  c.style.left = Math.min(Math.max(r.left + Math.min(r.width / 2, 120), 20), innerWidth - 20) + 'px';
  c.style.top = Math.min(Math.max(r.top + r.height / 2, 20), innerHeight - 20) + 'px';
  return true;
};
window.__frClick = (want, nth = 0) => {
  const el = window.__frFind(want, nth);
  if (!el) return false;
  const c = window.__frCursor();
  c.style.transform = 'scale(.7)';
  setTimeout(() => { c.style.transform = ''; }, 160);
  (el.closest('button') || el).click();
  return true;
};
window.__frScroll = (want, ms = 1500, offset = 90, nth = 0) => {
  const el = window.__frFind(want, nth);
  if (!el) return false;
  let box = el.parentElement;
  while (box && !(box.scrollHeight > box.clientHeight + 4 && /(auto|scroll)/.test(getComputedStyle(box).overflowY)))
    box = box.parentElement;
  box = box || document.scrollingElement;
  const from = box.scrollTop, to = from + el.getBoundingClientRect().top - offset, t0 = performance.now();
  const step = (now) => {
    const k = Math.min(1, (now - t0) / ms), e = k < .5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
    box.scrollTop = from + (to - from) * e;
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
  return true;
};
"""


class Screencast:
    """Headless Chrome over CDP with a reader thread that keeps every `Page.screencastFrame` (and acks
    it, or Chrome stalls after a few frames). `Browser.rpc` drops events, so it can't be used here."""

    def __init__(self, port: int) -> None:
        self.profile = tempfile.mkdtemp(prefix="fraudrings-video-")
        self.proc = subprocess.Popen(
            [find_chrome(), "--headless=new", f"--remote-debugging-port={port}", "--remote-allow-origins=*",
             "--hide-scrollbars", f"--window-size={VIDEO_W},{VIDEO_H}", "--no-first-run",
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
        self._send_lock, self._cv = threading.Lock(), threading.Condition()
        self._id, self._results = 0, {}
        self.segments: list[list[tuple[float, bytes]]] = []  # one list of (timestamp, jpeg) per recording
        self._recording = False
        threading.Thread(target=self._read, daemon=True).start()
        self.call("Page.enable")
        # exactly 1280x720 css px at scale 1: what the clip is, and what the app is laid out for
        self.call("Emulation.setDeviceMetricsOverride",
                  {"width": VIDEO_W, "height": VIDEO_H, "deviceScaleFactor": 1, "mobile": False})

    def _read(self) -> None:
        while True:
            try:
                msg = json.loads(self.ws.recv())
            except Exception:
                return
            if "id" in msg:
                with self._cv:
                    self._results[msg["id"]] = msg
                    self._cv.notify_all()
            elif msg.get("method") == "Page.screencastFrame":
                p = msg["params"]
                if self._recording and self.segments:
                    self.segments[-1].append((p["metadata"]["timestamp"], base64.b64decode(p["data"])))
                self._send("Page.screencastFrameAck", {"sessionId": p["sessionId"]}, wait=False)

    def _send(self, method: str, params: dict | None = None, wait: bool = True) -> dict:
        with self._send_lock:
            self._id += 1
            mid = self._id
            self.ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        if not wait:
            return {}
        with self._cv:
            self._cv.wait_for(lambda: mid in self._results, timeout=90)
            return self._results.pop(mid, {}).get("result", {})

    def call(self, method: str, params: dict | None = None) -> dict:
        return self._send(method, params)

    def js(self, expr: str):
        return self.call("Runtime.evaluate", {"expression": expr, "returnByValue": True}).get("result", {}).get("value")

    def start(self) -> None:
        self.segments.append([])
        self._recording = True
        self.call("Page.startScreencast", {"format": "jpeg", "quality": 85, "maxWidth": VIDEO_W,
                                           "maxHeight": VIDEO_H, "everyNthFrame": 1})

    def stop(self) -> None:
        self.call("Page.stopScreencast")
        self._recording = False

    def wait_text(self, text: str, timeout: float = 90) -> None:
        end = time.time() + timeout
        while time.time() < end:
            if self.js(f"!!window.__frFind && !!window.__frFind({json.dumps(text)})"):
                return
            time.sleep(0.4)
        raise SystemExit(f"video: timed out waiting for {text!r}")

    def goto(self, url: str, ready: str) -> None:
        self.call("Page.navigate", {"url": url})
        time.sleep(1.0)
        self.js(PAGE_JS)
        self.wait_text(ready)
        self.js("window.__frCursor()")
        time.sleep(1.0)

    def glide_click(self, text: str) -> None:
        if not self.js(f"window.__frMove({json.dumps(text)})"):
            raise SystemExit(f"video: no element {text!r} to click")
        time.sleep(1.0)  # the cursor glides (0.8 s) before the click
        self.js(f"window.__frClick({json.dumps(text)})")

    def scroll(self, text: str, ms: int = 1800, offset: int = 90) -> None:
        if not self.js(f"window.__frScroll({json.dumps(text)}, {ms}, {offset})"):
            raise SystemExit(f"video: no element {text!r} to scroll to")
        time.sleep(ms / 1000 + 0.3)

    def close(self) -> None:
        try:
            self.ws.close()
        except Exception:
            pass
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
        shutil.rmtree(self.profile, ignore_errors=True)


def resample(segments: list[list[tuple[float, bytes]]], out_dir: str, fps: int = VIDEO_FPS) -> int:
    """Write constant-frame-rate JPEGs. Chrome only sends a frame when the page changes, so each output
    frame repeats the latest frame at or before its time. Recording segments are joined end to end."""
    n = 0
    for seg in segments:
        if not seg:
            continue
        t0, t1 = seg[0][0], seg[-1][0] + 0.6  # hold the last frame a moment
        j = 0
        for i in range(int((t1 - t0) * fps)):
            t = t0 + i / fps
            while j + 1 < len(seg) and seg[j + 1][0] <= t:
                j += 1
            with open(os.path.join(out_dir, f"f{n:05d}.jpg"), "wb") as f:
                f.write(seg[j][1])
            n += 1
    return n


def encode(frames_dir: str, n_frames: int, poster_frame: int) -> None:
    os.makedirs(VIDEO_DIR, exist_ok=True)
    src = ["-framerate", str(VIDEO_FPS), "-i", os.path.join(frames_dir, "f%05d.jpg")]
    vf = "scale=trunc(iw/2)*2:trunc(ih/2)*2:in_range=full:out_range=limited,format=yuv420p"
    run = lambda args: subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *src, *args], check=True)  # noqa: E731
    run(["-vf", vf, "-c:v", "libx264", "-preset", "slow", "-crf", "30", "-movflags", "+faststart", "-an",
         os.path.join(VIDEO_DIR, "demo.mp4")])
    run(["-vf", vf, "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "37", "-row-mt", "1", "-an",
         os.path.join(VIDEO_DIR, "demo.webm")])
    Image.open(os.path.join(frames_dir, f"f{poster_frame:05d}.jpg")).convert("RGB").save(
        os.path.join(VIDEO_DIR, "demo-poster.png"), optimize=True)
    print(f"{n_frames} frames, {n_frames / VIDEO_FPS:.1f} s")


def record_video() -> None:
    data_dir = tempfile.mkdtemp(prefix="fraudrings-video-data-")
    print(f"building worlds in {data_dir} ...")
    prepare_worlds(data_dir)
    port = free_port()
    app = launch_app(data_dir, port)
    cast = Screencast(port + 1 if port < 65000 else 9444)
    base = f"http://localhost:{port}"
    frames = tempfile.mkdtemp(prefix="fraudrings-frames-")
    try:
        # 1. open Triage (naive) on the rank-1 suspect, 2. read the queue, 3. scroll through its evidence
        cast.goto(f"{base}/?{NAIVE}&rank=1", "Triage queue")
        cast.start()
        time.sleep(2.0)
        cast.scroll("Suspect · creator", 1600)
        time.sleep(1.2)
        cast.scroll("Contributor concentration", 1400, offset=110)
        time.sleep(1.8)
        cast.scroll("The core", 1600)
        time.sleep(1.8)
        # 4. click the handoff, 5. wait for related accounts
        cast.glide_click("🔎 Investigate this core")
        cast.wait_text("Related accounts")
        time.sleep(1.6)
        cast.scroll("Related accounts", 1500)
        time.sleep(2.2)
        # 6. scroll to the linkage graph
        cast.scroll("Linkage subgraph", 2600, offset=70)
        time.sleep(2.4)
        cast.stop()
        # 7. jump to the sophisticated, concurrency-4 temporal view (page load is cut out)
        cast.goto(f"{base}/investigate?{SOPH}&conc=4&ring=1", "Temporal raster")
        cast.start()
        time.sleep(1.2)
        cast.scroll("Temporal raster", 1600, offset=70)
        time.sleep(4.0)
        cast.stop()
    finally:
        cast.close()
        app.terminate()
        shutil.rmtree(data_dir, ignore_errors=True)
    n = resample(cast.segments, frames)
    first = cast.segments[0]
    t_poster = first[0][0] + 9.0  # after the evidence scroll: the "why it ranks" view
    poster = min(int((t_poster - first[0][0]) * VIDEO_FPS), n - 1)
    encode(frames, n, poster)
    shutil.rmtree(frames, ignore_errors=True)
    for f in ("demo.mp4", "demo.webm", "demo-poster.png"):
        path = os.path.join(VIDEO_DIR, f)
        print(f"wrote {os.path.relpath(path, ROOT)}  ({os.path.getsize(path) / 1e6:.2f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", help="capture only the named crops (default: all)")
    ap.add_argument("--video", action="store_true", help="record the site's demo clip instead of the stills")
    args = ap.parse_args()
    if args.video:
        record_video()
        return

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
                if c.name.startswith("tour/"):
                    out = os.path.join(ROOT, "docs", "images", f"{c.name}.png")
                os.makedirs(os.path.dirname(out), exist_ok=True)
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
