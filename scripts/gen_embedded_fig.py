"""Figure: what each method ACTUALLY embeds for retrieval (only the used field), with a real example."""
import json, textwrap
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

BASE = Path("/Users/michael/Desktop/CompSci/Sem 2/Seeing Machines - Final Project")
DC = BASE / "Cache" / "SeeingMachines" / "data" / "cache"
CORPUS = BASE / "Cache" / "SeeingMachines" / "data" / "corpus"
OUT = BASE / "docs" / "assets" / "caption_schemas.png"

NM = "acg_wood_floor_015"
caps = json.load(open(DC / "captions.json"))[NM]
sf = json.load(open(DC / "stub_format_captions.json"))[NM]
md = {r["name"]: r for r in json.load(open(DC / "metadata.json"))}[NM]
stub_cap = f"A {', '.join(md['tags'][:4])} {md['category'].lower()} surface texture.".replace("  ", " ")

# (title, color, space_label, embedded_line, note)
rows = [
    ("1 · CLIP-only", "#3a5a8c", "→ 512-d CLIP image space",
     "‹ the image pixels ›   (no text is used at all)",
     "the query text is embedded into the SAME space and matched by cosine"),
    ("2 · Caption-only  (real Gemma)", "#c46a10", "→ 384-d MiniLM text space",
     f'caption = "{caps["caption"]}"',
     "only the caption field is embedded — material_type / colors / finish / … are NOT"),
    ("3 · Gemma “stub-format”", "#2e7d32", "→ 384-d MiniLM text space",
     f'description = "{sf["description"]}"',
     "only the description field is embedded — category / tags / methods are NOT"),
    ("4 · Hybrid (α = 0.5)", "#7b3fb0", "→ fused score",
     "α · (CLIP image vector · query)  +  (1−α) · (caption vector · query)",
     "reuses rows 1 and 2 — embeds nothing new"),
    ("5 · Metadata stub  *", "#c62828", "→ 384-d MiniLM text space",
     f'caption = "{stub_cap}"',
     "⚠ built from the labels, not from vision — the embedded text contains the answer"),
]

fig = plt.figure(figsize=(14, 8.6))
fig.suptitle("What each method actually EMBEDS for retrieval  (one field only)",
             fontsize=16, fontweight="bold", y=0.975)

# example thumbnail
ax_img = fig.add_axes([0.035, 0.60, 0.13, 0.26])
ax_img.imshow(Image.open(CORPUS / f"{NM}.png").convert("RGB"))
ax_img.set_xticks([]); ax_img.set_yticks([])
for s in ax_img.spines.values(): s.set_edgecolor("#222"); s.set_linewidth(1.3)
fig.text(0.10, 0.575, f"example:\n{NM}\n(Wood)", ha="center", fontsize=8.5, color="#555")

ax = fig.add_axes([0.20, 0.03, 0.78, 0.88]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
n = len(rows); gap = 0.022; h = (1 - gap*(n-1)) / n
for k, (title, color, space, line, note) in enumerate(rows):
    y = 1 - (k+1)*h - k*gap
    ax.add_patch(FancyBboxPatch((0, y), 1, h, boxstyle="round,pad=0.004,rounding_size=0.02",
                                linewidth=1.5, edgecolor=color, facecolor=color+"12"))
    ax.text(0.015, y+h-0.03, title, fontsize=12, fontweight="bold", color=color, va="top")
    ax.text(0.985, y+h-0.03, space, fontsize=10, color=color, va="top", ha="right", fontweight="bold")
    ax.text(0.03, y+h*0.50, textwrap.fill(line, 84), fontsize=10.2, va="center",
            family="monospace", color="#111")
    ax.text(0.03, y+0.028, note, fontsize=8.8, va="bottom", color=color, style="italic")

fig.text(0.20, 0.008, "The other structured fields are kept only for the mis-seeing audit — they are never "
         "embedded.   * metadata stub is a leakage control, not a real route.", fontsize=8.6, color="#666")
plt.savefig(OUT, dpi=140, bbox_inches="tight"); print("wrote", OUT)
