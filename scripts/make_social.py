"""Compose the 1280x640 social-preview card: docs/images/site/social.png.

    .venv/bin/python scripts/make_social.py

The title, the README's pitch sentence and a cropped screenshot of the triage queue, on the app's dark
palette. Pillow's bundled font keeps the output the same on every machine. Upload the result under the
repo's Settings -> General -> Social preview (GitHub has no API for it); the site's meta tags use it too.
"""
from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
W, H = 1280, 640
BG, CARD, BORDER = (14, 17, 22), (19, 23, 29), (34, 40, 49)
TEXT, DIM, RING = (230, 233, 238), (174, 181, 191), (214, 95, 95)


def pitch_sentence() -> str:
    """The first sentence of the README pitch (the same block the home page renders)."""
    text = (ROOT / "README.md").read_text()
    block = re.search(r"<!-- site:pitch -->\n(.*?)\n<!-- /site:pitch -->", text, re.S)
    if not block:
        raise SystemExit("README.md: missing the site:pitch markers")
    return re.sub(r"\s+", " ", block.group(1)).split(". ")[0].rstrip(".") + "."


def wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines: list[str] = []
    for word in text.split():
        trial = f"{lines[-1]} {word}" if lines else word
        if lines and draw.textlength(trial, font=font) > width:
            lines.append(word)
        elif lines:
            lines[-1] = trial
        else:
            lines.append(word)
    return lines


def main() -> None:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((60, 58, 78, 76), radius=3, fill=RING)
    d.text((96, 54), "fraudrings", font=ImageFont.load_default(size=26), fill=DIM)
    d.text((60, 96), "fraud-ring-detection", font=ImageFont.load_default(size=68), fill=TEXT, stroke_width=1,
           stroke_fill=TEXT)
    body = ImageFont.load_default(size=27)
    y = 190
    for line in wrap(d, pitch_sentence(), body, W - 120):
        d.text((60, y), line, font=body, fill=DIM)
        y += 36

    shot = Image.open(ROOT / "docs/images/walkthrough/02-triage-queue.png").convert("RGB")
    width = W - 120
    shot = shot.resize((width, round(shot.height * width / shot.width)), Image.LANCZOS)
    top = max(y + 22, 270)
    card = Image.new("RGB", (width + 2, H - top + 2), BORDER)
    card.paste(shot.crop((0, 0, width, H - top)), (1, 1))
    mask = Image.new("L", card.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, card.width + 20, card.height + 20), radius=10, fill=255)
    img.paste(card, (59, top - 1), mask)
    out = ROOT / "docs/images/site/social.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, optimize=True)
    print(f"wrote {out.relative_to(ROOT)}  ({out.stat().st_size / 1000:.0f} KB)")


if __name__ == "__main__":
    main()
