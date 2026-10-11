"""Capture full-page PNG screenshots of a Streamlit viewer via headless Chrome + CDP.

Why this exists (see AGENTS.md): the Chrome + Claude extension path cannot save screenshots
to a location the agent can read back, and a plain `chrome --screenshot` fires before
Streamlit's websocket-driven render lands. This drives Chrome over the DevTools Protocol:
navigate, wait real seconds for the render, then capture the full scroll height at 2x for
crisp doc figures. Trailing background is trimmed by scanning the MAIN content column only
(the Streamlit sidebar spans the full viewport height and would otherwise defeat a global
bbox trim).

Requires: a running Streamlit app, the [docs] extra (`websocket-client` + `Pillow`), and Chrome or
Chromium (found on PATH or at the usual macOS location; override with `--chrome`).

Usage:
    # 1. launch the app headless on a free port
    .venv/bin/python -m streamlit run src/fraudrings/viewer/main.py \
        --server.headless true --server.port 8533 &
    # 2. capture
    .venv/bin/python scripts/capture_ui.py "http://localhost:8533/?rank=1" out.png --wait 11 --width 1680

The `?rank=N` query param deep-links the Triage page to a given suspect, so a screenshot can target
any row of the queue.
"""
from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import time
import urllib.request
from io import BytesIO

import websocket  # websocket-client
from PIL import Image, ImageChops

PORT = 9222
_MAC_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def find_chrome() -> str:
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome"):
        path = shutil.which(name)
        if path:
            return path
    return _MAC_CHROME


def _rpc(ws, mid, method, params=None):
    ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
    while True:
        msg = json.loads(ws.recv())
        if msg.get("id") == mid:
            return msg.get("result", {})


def capture(url: str, out: str, wait: float = 8.0, width: int = 1600, height: int = 3200,
            chrome: str | None = None) -> None:
    # NOTE: do NOT also pass --force-device-scale-factor; combined with the Emulation
    # deviceScaleFactor below it double-scales and shifts the capture origin (clips the sidebar).
    proc = subprocess.Popen(
        [chrome or find_chrome(), "--headless=new", f"--remote-debugging-port={PORT}",
         "--remote-allow-origins=*", "--hide-scrollbars",
         f"--window-size={width},{height}", "--no-first-run", "--no-default-browser-check",
         "--user-data-dir=/tmp/chrome-fraudrings-shots"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        ws_url = None
        for _ in range(40):
            try:
                data = json.loads(urllib.request.urlopen(f"http://localhost:{PORT}/json").read())
                pages = [t for t in data if t["type"] == "page"]
                if pages:
                    ws_url = pages[0]["webSocketDebuggerUrl"]
                    break
            except Exception:
                pass
            time.sleep(0.25)
        if not ws_url:
            raise SystemExit("could not reach Chrome CDP — is Chrome/Chromium installed? (try --chrome PATH)")

        ws = websocket.create_connection(ws_url, max_size=None)
        mid = 0

        def rpc(method, params=None):
            nonlocal mid
            mid += 1
            return _rpc(ws, mid, method, params)

        rpc("Page.enable")
        rpc("Emulation.setDeviceMetricsOverride", {
            "width": width, "height": height, "deviceScaleFactor": 2, "mobile": False,
        })
        rpc("Page.navigate", {"url": url})
        time.sleep(wait)  # let the Streamlit websocket render land

        result = rpc("Page.captureScreenshot", {
            # the viewport is already full-height; beyond-viewport capture mis-renders the sidebar
            "format": "png", "captureBeyondViewport": False,
            "clip": {"x": 0, "y": 0, "width": width, "height": height, "scale": 1},
        })
        raw = base64.b64decode(result["data"])
        ws.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    # Trim the empty bottom band. The sidebar spans the full viewport height, so measure the
    # content bottom in the MAIN column (right of the sidebar) only.
    img = Image.open(BytesIO(raw)).convert("RGB")
    sidebar_w = int(img.width * 0.22)
    main = img.crop((sidebar_w, 0, img.width, img.height))
    bg = Image.new("RGB", main.size, main.getpixel((main.width - 3, main.height - 3)))
    bbox = ImageChops.difference(main, bg).getbbox()
    if bbox:
        img = img.crop((0, 0, img.width, min(img.height, bbox[3] + 40)))
    img.save(out)
    print(f"wrote {out}  ({img.width}x{img.height}px)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("url")
    p.add_argument("out")
    p.add_argument("--wait", type=float, default=8.0, help="seconds to wait for the render")
    p.add_argument("--width", type=int, default=1600, help="css viewport width (≥1680 avoids sidebar clip)")
    p.add_argument("--height", type=int, default=3200, help="css viewport height (tall = fit all content)")
    p.add_argument("--chrome", default=None, help="path to a Chrome/Chromium binary (default: auto-detect)")
    a = p.parse_args()
    capture(a.url, a.out, a.wait, a.width, a.height, a.chrome)
