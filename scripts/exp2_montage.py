"""Build report figures for Experiment 2:
  - docs/assets/exp2_teaser.png : a grid of rendered spheres (the new corpus)
  - docs/assets/exp2_pairs.png  : albedo (top) vs Mitsuba render (bottom) for a few materials
Uses only PIL/numpy so it runs without the ML stack."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

_CANDS = [Path(__file__).resolve().parent.parent,
          Path(r"D:\Seeing Machines - Final Project-20260708T194711Z-3-002\Seeing Machines - Final Project")]
BASE = next((p for p in _CANDS if (p / "data" / "cache" / "metadata.json").exists()), _CANDS[0])
CACHE = BASE / "data" / "cache"; CORPUS = BASE / "data" / "corpus"; REND = BASE / "data" / "corpus_rendered"
OUT = BASE / "docs" / "assets"; OUT.mkdir(parents=True, exist_ok=True)
META = json.load(open(CACHE / "metadata.json", encoding="utf-8"))

def load(p, px):
    return Image.open(p).convert("RGB").resize((px, px), Image.LANCZOS)

rendered = [r for r in META if (REND / r["file"]).exists()]
print(f"{len(rendered)} rendered materials available")

# ---- teaser: grid of renders, one per distinct category where possible ----
by_cat = {}
for r in rendered:
    by_cat.setdefault(r["category"], r)
picks = list(by_cat.values())[:12]
if len(picks) < 12:
    picks += [r for r in rendered if r not in picks][:12 - len(picks)]
px, cols = 200, 6
rows = (len(picks) + cols - 1) // cols
canvas = Image.new("RGB", (cols * px, rows * px), (240, 242, 245))
for i, r in enumerate(picks):
    canvas.paste(load(REND / r["file"], px), ((i % cols) * px, (i // cols) * px))
canvas.save(OUT / "exp2_teaser.png"); print("wrote exp2_teaser.png")

# ---- pairs: albedo vs render for 4 illustrative materials ----
want = ["acg_plastic_009", "tc_metal_029", "acg_wood_026", "st_white_ceramic"]
pairs = [r for r in rendered if r["name"] in want][:4] or rendered[:4]
px = 256; gap = 10; labelw = 70
W = labelw + len(pairs) * (px + gap); H = 2 * px + gap
canvas = Image.new("RGB", (W, H), (255, 255, 255))
for j, r in enumerate(pairs):
    x = labelw + j * (px + gap)
    canvas.paste(load(CORPUS / r["file"], px), (x, 0))
    canvas.paste(load(REND / r["file"], px), (x, px + gap))
canvas.save(OUT / "exp2_pairs.png"); print("wrote exp2_pairs.png")
