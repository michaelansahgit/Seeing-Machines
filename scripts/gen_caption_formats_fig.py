"""Figure(s): what text (if any) each retrieval route reads, for an example material.
Generates a failure case (plastic) and a success case (wood)."""
import json, textwrap
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageEnhance

BASE = Path("/Users/michael/Desktop/CompSci/Sem 2/Seeing Machines - Final Project")
DC = BASE / "Cache" / "SeeingMachines" / "data" / "cache"
CORPUS = BASE / "Cache" / "SeeingMachines" / "data" / "corpus"
CAPS = json.load(open(DC / "captions.json"))
SF = json.load(open(DC / "stub_format_captions.json"))
MD = {r["name"]: r for r in json.load(open(DC / "metadata.json"))}

def stub_text(nm):
    r = MD[nm]; tags = ", ".join(r["tags"][:4])
    return f"A {tags} {r['category'].lower()} surface texture.".replace("  ", " ")

def make_fig(nm, truth, truth_desc, verdicts, outname, brighten=1.0):
    c, s = CAPS[nm], SF[nm]
    rows = [
        ("1 · CLIP-only", "#3a5a8c",
         "no caption — the query is matched directly to the IMAGE embedding (the 512-d vector of the pixels).",
         verdicts["clip"], "P@5 0.536"),
        ("2 · Caption-only  (real Gemma)", "#c46a10",
         f'material_type: "{c["material_type"]}"   ·   caption: "{c["caption"]}"',
         verdicts["cap"], "P@5 0.368"),
        ("3 · Gemma “stub-format”", "#2e7d32",
         f'category: "{s["category"]}"   ·   description: "{s["description"]}"',
         verdicts["sfmt"], "P@5 0.424"),
        ("4 · Hybrid (α = 0.5)", "#7b3fb0",
         "α · (CLIP image score)  +  (1−α) · (Gemma caption score)  — fuses routes 1 and 2, no new text.",
         verdicts["hyb"], "P@5 0.496"),
        ("5 · Metadata stub  *", "#c62828",
         f'{stub_text(nm)}   (built from the dataset labels, NOT from Gemma)',
         verdicts["stub"], "P@5 0.888 *"),
    ]
    fig = plt.figure(figsize=(13.5, 8.6))
    fig.suptitle("What each retrieval route actually “reads” — one material, five representations",
                 fontsize=15, fontweight="bold", y=0.975)
    ax_img = fig.add_axes([0.035, 0.30, 0.22, 0.44])
    im = Image.open(CORPUS / f"{nm}.png").convert("RGB")
    if brighten != 1.0: im = ImageEnhance.Brightness(im).enhance(brighten)
    ax_img.imshow(im); ax_img.set_xticks([]); ax_img.set_yticks([])
    for sp in ax_img.spines.values(): sp.set_edgecolor("#222"); sp.set_linewidth(1.5)
    ax_img.set_title(nm, fontsize=9, color="#555")
    fig.text(0.145, 0.265, f"GROUND TRUTH:  {truth}", ha="center", fontsize=12, fontweight="bold")
    fig.text(0.145, 0.235, truth_desc, ha="center", fontsize=8.5, color="#555")

    ax = fig.add_axes([0.29, 0.05, 0.68, 0.86]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    n = len(rows); gap = 0.02; h = (1 - gap*(n-1)) / n
    for k, (title, color, rep, verdict, p5) in enumerate(rows):
        y = 1 - (k+1)*h - k*gap
        ax.add_patch(FancyBboxPatch((0, y), 1, h, boxstyle="round,pad=0.004,rounding_size=0.02",
                                    linewidth=1.4, edgecolor=color, facecolor=color+"14"))
        ax.text(0.02, y+h-0.045, title, fontsize=11.5, fontweight="bold", color=color, va="top")
        ax.text(0.985, y+h-0.045, p5, fontsize=10, color=color, va="top", ha="right", fontweight="bold")
        ax.text(0.02, y+h*0.52, textwrap.fill(rep, 92), fontsize=9.3, va="center", family="monospace", color="#222")
        ax.text(0.02, y+0.03, verdict, fontsize=9.2, va="bottom", color=color, style="italic")
    fig.text(0.29, 0.015, "*  Metadata stub is written from the dataset’s own category+tags — the labels being "
             "scored — so its score is leakage, shown only to expose the trap.", fontsize=8.3, color="#777")
    out = BASE / "docs" / "assets" / outname
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close(); print("wrote", out)

# --- failure case: plastic ---
make_fig("acg_plastic_009", "Plastic", "(a dark green procedural\nplastic texture)",
    {"clip": "✓  strongest route — reads the material, not words about it",
     "cap":  "✗  dropped the material → returned a COLOUR (“dark green”)",
     "sfmt": "✗  hallucinated the wrong material (“wood”)",
     "hyb":  "→  lands between CLIP and caption; best α = 1.0 (pure CLIP)",
     "stub": "✗  contains the true label “plastic” → evaluation LEAKAGE"},
    "caption_formats.png", brighten=2.4)

# --- success case: wood ---
make_fig("acg_wood_floor_015", "Wood", "(a light-brown herringbone\nwood floor)",
    {"clip": "✓  clean win — wood grain is a strong, unambiguous signature",
     "cap":  "✓  CORRECT — grain survives delighting, so Gemma reads “wood”",
     "sfmt": "✓  also correct — “wooden planks, parquet”",
     "hyb":  "→  both routes agree here, so fusion neither helps nor hurts",
     "stub": "≈  contains “wood” (leakage) — but here it happens to agree"},
    "caption_formats_wood.png")
