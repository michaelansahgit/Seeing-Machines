"""Figure: the caption SCHEMA (format + controlled vocabulary) each method produces."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BASE = Path("/Users/michael/Desktop/CompSci/Sem 2/Seeing Machines - Final Project")
OUT = BASE / "docs" / "assets" / "caption_schemas.png"

# (title, color, list_of_lines, subtitle)
panels = [
    ("2 · Structured schema  —  Caption-only (real Gemma)  ·  the L2 design decision", "#c46a10",
     '{\n'
     '  "material_type":    "wood | metal | fabric | stone | ground | plastic |\n'
     '                       ceramic | leather | concrete | plaster | marble | terracotta",\n'
     '  "colors":           ["1-3 dominant colours"],\n'
     '  "surface_finish":   "matte | glossy | rough | smooth | textured",        # ◆ controlled set\n'
     '  "pattern":          "woven | cracked | veined | grain | tiled | none",   # ◆ controlled set\n'
     '  "condition":        "new | worn | weathered | rusted | stained | clean", # ◆ controlled set\n'
     '  "notable_features": "one short phrase",\n'
     '  "caption":          "one natural sentence"                              # ► EMBEDDED for search\n'
     '}',
     "7 fields · 3 discretized vocabularies · only  caption  is embedded"),

    ("3 · Gemma “stub-format” schema  —  same image, catalogue-style fields", "#2e7d32",
     '{\n'
     '  "category":         "wood | metal | ... | terracotta",\n'
     '  "creation_method":  "procedural generation | photogrammetry | scanned | hand-painted", # ◆ controlled\n'
     '  "capture_method":   "substance designer | quixel mixer | adobe sampler | 3d rendered", # ◆ controlled\n'
     '  "tags":             ["3-7 descriptive tags"],\n'
     '  "description":      "one concise sentence"                              # ► EMBEDDED for search\n'
     '}',
     "5 fields · 2 discretized vocabularies · only  description  is embedded"),

    ("5 · Metadata stub schema  —  built from labels, NO Gemma  *", "#c62828",
     '{\n'
     '  "material_type":    "<category>",              # copied straight from the ground-truth label\n'
     '  "colors":           ["<colour tags>"],\n'
     '  "surface_finish":   "",   "pattern": "",   "condition": "",   # left BLANK — no VLM ever runs\n'
     '  "notable_features": "<all tags>",\n'
     '  "caption":          "A <tags> <category> surface texture.",   # ► EMBEDDED  ⚠ contains the label\n'
     '  "_stub": true\n'
     '}',
     "template only · no vision · the caption literally contains the answer key → leakage"),
]

fig = plt.figure(figsize=(14, 11.4))
fig.suptitle("The caption schema each method produces  —  format & controlled vocabulary",
             fontsize=16, fontweight="bold", y=0.985)

# top strip: the two schema-less routes
ax = fig.add_axes([0.03, 0.02, 0.94, 0.90]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.add_patch(FancyBboxPatch((0, 0.905), 1, 0.075, boxstyle="round,pad=0.004,rounding_size=0.015",
                            linewidth=1.4, edgecolor="#3a5a8c", facecolor="#3a5a8c14"))
ax.text(0.015, 0.943, "1 · CLIP-only", fontsize=12, fontweight="bold", color="#3a5a8c", va="center")
ax.text(0.16, 0.943, "no text schema — the image itself becomes a 512-d embedding vector.",
        fontsize=10.5, va="center", family="monospace", color="#222")
ax.text(0.015, 0.918, "4 · Hybrid", fontsize=11, fontweight="bold", color="#7b3fb0", va="center")
ax.text(0.16, 0.918, "no schema — fuses the CLIP vector and the caption embedding:  α·CLIP + (1−α)·caption.",
        fontsize=10, va="center", family="monospace", color="#222")

# three schema panels
top = 0.88; bottom = 0.02; gap = 0.025
h = (top - bottom - gap*len(panels)) / len(panels)
for k, (title, color, body, sub) in enumerate(panels):
    y = top - (k+1)*h - k*gap
    ax.add_patch(FancyBboxPatch((0, y), 1, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                linewidth=1.5, edgecolor=color, facecolor=color+"10"))
    ax.text(0.015, y+h-0.028, title, fontsize=12, fontweight="bold", color=color, va="top")
    ax.text(0.015, y+h-0.075, body, fontsize=9.0, va="top", family="monospace", color="#1a1a1a")
    ax.text(0.015, y+0.018, sub, fontsize=9.2, va="bottom", color=color, style="italic")

fig.text(0.03, 0.006, "◆ = discretized / controlled vocabulary (lowers caption variance)   "
         "► = the one field embedded for retrieval   "
         "* metadata stub is a leakage control, not a real route.",
         fontsize=8.6, color="#666")
plt.savefig(OUT, dpi=140, bbox_inches="tight"); print("wrote", OUT)
