"""Regenerate documentation figures from the REAL 300-material Colab cache (real Gemma captions)."""
import json, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
import torch, open_clip
from sentence_transformers import SentenceTransformer
from collections import Counter

BASE = Path("/Users/michael/Desktop/CompSci/Sem 2/Seeing Machines - Final Project")
DCACHE = BASE / "Cache" / "SeeingMachines" / "data" / "cache"
CORPUS = BASE / "Cache" / "SeeingMachines" / "data" / "corpus"
OUT = BASE / "docs" / "assets"; OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

METADATA = json.load(open(DCACHE / "metadata.json"))
CAPTIONS = json.load(open(DCACHE / "captions.json"))
NAMES = [r["name"] for r in METADATA]; CATS = [r["category"] for r in METADATA]
IMG_EMB = np.load(DCACHE / "clip_image_embeds.npy")
CAP_EMB = np.load(DCACHE / "caption_text_embeds.npy")
N = len(METADATA)
uniq = sorted(set(CATS)); cmap = plt.get_cmap("tab20")
COLOR = {c: cmap(i % 20) for i, c in enumerate(uniq)}
def thumb(nm, px=240):
    im = Image.open(CORPUS / f"{nm}.png").convert("RGB"); im.thumbnail((px, px)); return im

clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
clip_tok = open_clip.get_tokenizer("ViT-B-32"); clip_model = clip_model.eval().to(DEVICE)
text_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
@torch.no_grad()
def clip_q(t):
    q = clip_model.encode_text(clip_tok([t]).to(DEVICE)); q = q/q.norm(dim=-1, keepdim=True); return q.cpu().numpy()[0]
def clip_search(q, k=5):
    s = IMG_EMB @ clip_q(q); o = np.argsort(-s)[:k]; return [(int(i), float(s[i])) for i in o]
def cap_search(q, k=5):
    s = CAP_EMB @ text_model.encode([q], normalize_embeddings=True)[0].astype(np.float32)
    o = np.argsort(-s)[:k]; return [(int(i), float(s[i])) for i in o]

# 1 teaser
sel, seen = [], set()
for r in METADATA:
    if r["category"] not in seen: seen.add(r["category"]); sel.append(r)
for r in METADATA:
    if len(sel) >= 32: break
    if r not in sel: sel.append(r)
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
for ax, r in zip(axes.flat, sel[:32]): ax.imshow(thumb(r["name"], 150)); ax.axis("off")
plt.subplots_adjust(wspace=.04, hspace=.04, left=.01, right=.99, top=.99, bottom=.01)
plt.savefig(OUT / "teaser.png", dpi=100, bbox_inches="tight"); plt.close(); print("teaser")

# 2 category dist
dist = sorted(Counter(CATS).items(), key=lambda kv: -kv[1])
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar([k for k,_ in dist], [v for _,v in dist], color=[COLOR[k] for k,_ in dist])
ax.set_ylabel("# materials"); ax.set_title(f"Corpus composition - {N} MatSynth materials, 13 categories")
plt.xticks(rotation=40, ha="right"); plt.tight_layout(); plt.savefig(OUT/"category_dist.png", dpi=120); plt.close(); print("dist")

# 3 PCA
xy = PCA(n_components=2, random_state=0).fit_transform(IMG_EMB)
fig, ax = plt.subplots(figsize=(9, 7))
for c in uniq:
    idx = [i for i in range(N) if CATS[i]==c]
    ax.scatter(xy[idx,0], xy[idx,1], s=42, color=COLOR[c], label=c, edgecolors="white", linewidths=.4)
ax.set_title("CLIP image-embedding space (PCA) - 300 materials by category")
ax.legend(ncol=2, fontsize=8); ax.grid(alpha=.25); plt.tight_layout()
plt.savefig(OUT/"embedding_space.png", dpi=120); plt.close(); print("pca")
sims = IMG_EMB @ IMG_EMB.T; intra, inter = [], []
for i in range(N):
    for j in range(i+1, N): (intra if CATS[i]==CATS[j] else inter).append(sims[i,j])
print(f"[STAT] intra {np.mean(intra):.3f} inter {np.mean(inter):.3f} gap {np.mean(intra)-np.mean(inter):+.3f}")

# 4 atlas grids
def grid(query, fname, k=5, route="clip"):
    hits = (clip_search if route=="clip" else cap_search)(query, k)
    fig, axes = plt.subplots(1, k, figsize=(2.5*k, 3.0))
    for ax,(i,s) in zip(axes, hits):
        ax.imshow(thumb(METADATA[i]["name"])); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{METADATA[i]['category']} - {s:.2f}", fontsize=9)
    fig.suptitle(f"[{route.upper()}]  “{query}”", fontsize=12); plt.tight_layout()
    plt.savefig(OUT/fname, dpi=110, bbox_inches="tight"); plt.close()
    print(fname, "::", ", ".join(f"{METADATA[i]['category']}({s:.2f})" for i,s in hits))
grid("woven fabric texture", "atlas_woven.png")
grid("rusty corroded metal", "atlas_rust.png")
grid("wood planks", "atlas_wood.png")
grid("polished marble with veins", "atlas_marble.png")
grid("something soft and cozy", "atlas_cozy.png")
grid("a photo of the number 7", "atlas_number7.png")
grid("a man-made building material", "compare_clip.png", k=4, route="clip")
grid("a man-made building material", "compare_caption.png", k=4, route="caption")

# 5 mis-seeing examples grid (real Gemma)
CANON={"ceramic","concrete","fabric","ground","leather","marble","metal","misc","plaster","plastic","stone","terracotta","wood"}
cases=[]
for r in METADATA:
    mt=(CAPTIONS[r["name"]].get("material_type","") or "").lower(); cat=r["category"].lower()
    if cat!="misc" and mt and mt!=cat and mt not in {"brick"}:  # skip brick~terracotta label coarseness
        cases.append(r)
pick = [r for r in cases if r["name"] in
        {"acg_plastic_009","2119_corten_steel","acg_concrete_026","0844_syntetic_foam"}] or cases[:4]
pick = pick[:4]
fig, axes = plt.subplots(1, len(pick), figsize=(3.2*len(pick), 3.8))
if len(pick)==1: axes=[axes]
for ax, r in zip(axes, pick):
    ax.imshow(thumb(r["name"])); ax.set_xticks([]); ax.set_yticks([])
    mt=CAPTIONS[r["name"]].get("material_type",""); cap=CAPTIONS[r["name"]].get("caption","")
    ax.set_title(f"TRUE: {r['category']}\nGemma: “{mt}”", fontsize=9, color="#c62828")
    ax.set_xlabel(cap[:46]+"...", fontsize=7.5, wrap=True)
fig.suptitle("Mis-seeing dossier - Gemma reads albedo maps by colour, not material", fontsize=12)
plt.tight_layout(); plt.savefig(OUT/"misseeing.png", dpi=115, bbox_inches="tight"); plt.close(); print("misseeing")
print("DONE")
