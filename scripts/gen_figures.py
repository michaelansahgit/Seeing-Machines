"""Generate all documentation figures from CACHED artifacts (no GPU, no re-download).
Outputs -> docs/assets/*.png. Also prints quantitative embedding-space stats for the write-up."""
import json, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image
from sklearn.decomposition import PCA

PROJ = Path("/Users/michael/Desktop/CompSci/Sem 2/Seeing Machines - Final Project")
CORPUS = PROJ / "data" / "corpus"
CACHE = PROJ / "data" / "cache"
ASSETS = PROJ / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

METADATA = json.load(open(CACHE / "metadata.json"))
CAPTIONS = json.load(open(CACHE / "captions.json"))
NAMES = [r["name"] for r in METADATA]
CATS  = [r["category"] for r in METADATA]
IMG_EMB = np.load(CACHE / "clip_image_embeds.npy")
CAP_EMB = np.load(CACHE / "caption_text_embeds.npy")
N = len(METADATA)
print(f"Loaded {N} materials, IMG_EMB {IMG_EMB.shape}, CAP_EMB {CAP_EMB.shape}")

# Stable color per category
uniq = sorted(set(CATS))
cmap = plt.get_cmap("tab20")
COLOR = {c: cmap(i % 20) for i, c in enumerate(uniq)}

def thumb(name, px=256):
    im = Image.open(CORPUS / f"{name}.png").convert("RGB")
    im.thumbnail((px, px)); return im

# ----------------------------------------------------------------------
# CLIP text encoder (loaded once) for query result grids
# ----------------------------------------------------------------------
import torch, open_clip
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
clip_tok = open_clip.get_tokenizer("ViT-B-32")
clip_model = clip_model.eval().to(DEVICE)
from sentence_transformers import SentenceTransformer
text_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

@torch.no_grad()
def clip_q(text):
    q = clip_model.encode_text(clip_tok([text]).to(DEVICE))
    q = q / q.norm(dim=-1, keepdim=True)
    return q.cpu().numpy()[0]

def clip_search(q, k=5):
    s = IMG_EMB @ clip_q(q); o = np.argsort(-s)[:k]
    return [(int(i), float(s[i])) for i in o]

def cap_search(q, k=5):
    qt = text_model.encode([q], normalize_embeddings=True)[0].astype(np.float32)
    s = CAP_EMB @ qt; o = np.argsort(-s)[:k]
    return [(int(i), float(s[i])) for i in o]

# ======================================================================
# FIG 1 — teaser montage
# ======================================================================
sel = []
seen = set()
for r in METADATA:                       # one per category first, then fill
    if r["category"] not in seen:
        seen.add(r["category"]); sel.append(r)
for r in METADATA:
    if len(sel) >= 24: break
    if r not in sel: sel.append(r)
sel = sel[:24]
fig, axes = plt.subplots(3, 8, figsize=(16, 6))
for ax, r in zip(axes.flat, sel):
    ax.imshow(thumb(r["name"], 160)); ax.axis("off")
plt.subplots_adjust(wspace=0.04, hspace=0.04, left=0.01, right=0.99, top=0.99, bottom=0.01)
plt.savefig(ASSETS / "teaser.png", dpi=110, bbox_inches="tight"); plt.close()
print("wrote teaser.png")

# ======================================================================
# FIG 2 — category distribution
# ======================================================================
from collections import Counter
dist = Counter(CATS)
items = sorted(dist.items(), key=lambda kv: -kv[1])
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar([k for k, _ in items], [v for _, v in items],
       color=[COLOR[k] for k, _ in items])
ax.set_ylabel("# materials"); ax.set_title("Corpus composition — 89 MatSynth materials by category")
plt.xticks(rotation=40, ha="right"); plt.tight_layout()
plt.savefig(ASSETS / "category_dist.png", dpi=120); plt.close()
print("wrote category_dist.png")

# ======================================================================
# FIG 3 — PCA of CLIP image embeddings, colored by category
# ======================================================================
pca = PCA(n_components=2, random_state=0)
xy = pca.fit_transform(IMG_EMB)
fig, ax = plt.subplots(figsize=(9, 7))
for c in uniq:
    idx = [i for i in range(N) if CATS[i] == c]
    ax.scatter(xy[idx, 0], xy[idx, 1], s=70, color=COLOR[c], label=c,
               edgecolors="white", linewidths=0.6)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}% var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}% var)")
ax.set_title("CLIP image-embedding space (PCA) — does the model cluster materials by category?")
ax.legend(ncol=2, fontsize=8, loc="best"); ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig(ASSETS / "embedding_space.png", dpi=120); plt.close()
print("wrote embedding_space.png")

# quantitative embedding-space stat: intra vs inter category cosine
sims = IMG_EMB @ IMG_EMB.T
intra, inter = [], []
for i in range(N):
    for j in range(i + 1, N):
        (intra if CATS[i] == CATS[j] else inter).append(sims[i, j])
print(f"\n[STAT] mean cosine — same-category pairs: {np.mean(intra):.3f} | "
      f"different-category: {np.mean(inter):.3f} | gap: {np.mean(intra)-np.mean(inter):+.3f}")

# ======================================================================
# FIG 4 — retrieval atlas grids (CLIP route), incl. failures
# ======================================================================
def relevant_for(query, idx):
    """Heuristic relevance for highlighting (category/tag match) — for the gallery only."""
    r = METADATA[idx]; q = query.lower(); tags = [t.lower() for t in r["tags"]]
    cat = r["category"].lower()
    keys = {"woven": ("fabric", "woven"), "fabric": ("fabric",), "rusty": ("metal",),
            "metal": ("metal",), "wood": ("wood",), "marble": ("marble",),
            "plastic": ("plastic",), "ground": ("ground",), "brick": ("terracotta",)}
    for key, oks in keys.items():
        if key in q:
            return cat in oks or any(o in tags for o in oks)
    return None  # unknown / adversarial -> no highlight

def atlas_grid(query, fname, k=5, route="clip"):
    hits = (clip_search if route == "clip" else cap_search)(query, k=k)
    fig, axes = plt.subplots(1, k, figsize=(2.6*k, 3.1))
    for ax, (i, s) in zip(axes, hits):
        r = METADATA[i]; ax.imshow(thumb(r["name"], 240))
        rel = relevant_for(query, i)
        color = {True: "#2e7d32", False: "#c62828", None: "#555"}[rel]
        for sp in ax.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(3)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{r['category']} · {s:.2f}", fontsize=9, color=color)
    fig.suptitle(f"[{route.upper()}]  “{query}”", fontsize=12)
    plt.tight_layout(); plt.savefig(ASSETS / fname, dpi=115, bbox_inches="tight"); plt.close()
    cats = ", ".join(f"{METADATA[i]['category']}({s:.2f})" for i, s in hits)
    print(f"wrote {fname}  ::  {cats}")

atlas_grid("woven fabric texture",       "atlas_woven.png")
atlas_grid("rusty corroded metal",       "atlas_rust.png")
atlas_grid("wood planks",                "atlas_wood.png")
atlas_grid("polished marble with veins", "atlas_marble.png")
atlas_grid("something soft and cozy",    "atlas_cozy.png")
atlas_grid("a photo of the number 7",    "atlas_number7.png")

# ======================================================================
# FIG 5 — CLIP vs caption on a reasoning query (caption should win)
# ======================================================================
q = "a man-made building material"
for route in ("clip", "caption"):
    atlas_grid(q, f"compare_{route}.png", k=4, route=route)

# ======================================================================
# FIG 6 — precision@k + alpha sweep (recompute from cache)
# ======================================================================
def by_cat(c): return lambda r: r["category"].lower() == c
def by_tag(t): return lambda r: t.lower() in [x.lower() for x in r["tags"]]
def all_(*p): return lambda r: all(f(r) for f in p)
def any_(*p): return lambda r: any(f(r) for f in p)
GOLD = [(c, by_cat(c.lower())) for c in uniq]
for q_, pred in [("rusty metal", all_(by_cat("metal"), any_(by_tag("rusty"), by_tag("rust")))),
                 ("red surface", by_tag("red")), ("green surface", by_tag("green")),
                 ("blue surface", by_tag("blue")), ("brown surface", by_tag("brown")),
                 ("smooth material", by_tag("smooth")), ("rough material", by_tag("rough")),
                 ("dark material", by_tag("dark")), ("cracked surface", any_(by_tag("cracked"), by_tag("crack"))),
                 ("tiled pattern", any_(by_tag("tiles"), by_tag("tile"), by_tag("tiled"))),
                 ("wooden planks", all_(by_cat("wood"), any_(by_tag("planks"), by_tag("plank"))))]:
    if any(pred(r) for r in METADATA): GOLD.append((q_, pred))
GSETS = [(q_, {r["idx"] for r in METADATA if pred(r)}) for q_, pred in GOLD]

def minmax(x):
    lo, hi = x.min(), x.max(); return np.zeros_like(x) if hi-lo < 1e-9 else (x-lo)/(hi-lo)
def hybrid(q, alpha, k):
    cs = IMG_EMB @ clip_q(q)
    ts = CAP_EMB @ text_model.encode([q], normalize_embeddings=True)[0].astype(np.float32)
    f = alpha*minmax(cs) + (1-alpha)*minmax(ts)
    o = np.argsort(-f)[:k]; return [(int(i), float(f[i])) for i in o]
def p_at_k(fn, k):
    ps = [sum(1 for i, _ in fn(q, k) if i in rel)/k for q, rel in GSETS if rel]
    return float(np.mean(ps))

KS = [1, 3, 5, 10]
routes = {"CLIP-only": lambda q, k: clip_search(q, k),
          "Caption-only": lambda q, k: cap_search(q, k),
          "Hybrid α=0.5": lambda q, k: hybrid(q, 0.5, k)}
res = {n: [p_at_k(fn, k) for k in KS] for n, fn in routes.items()}
alphas = np.linspace(0, 1, 11)
p5 = [p_at_k(lambda q, k, a=a: hybrid(q, a, k), 5) for a in alphas]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4))
for n, row in res.items(): a1.plot(KS, row, marker="o", label=n)
a1.set_xlabel("k"); a1.set_ylabel("precision@k"); a1.set_title("Precision@k by route")
a1.legend(); a1.grid(alpha=0.3)
a2.plot(alphas, p5, marker="s", color="purple")
a2.axhline(res["CLIP-only"][2], ls="--", color="tab:blue", label="CLIP-only P@5")
a2.axhline(res["Caption-only"][2], ls="--", color="tab:orange", label="Caption-only P@5")
a2.set_xlabel("α (CLIP weight)"); a2.set_ylabel("precision@5")
a2.set_title("Fusion sweep (stub captions)"); a2.legend(); a2.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(ASSETS / "precision.png", dpi=120); plt.close()
print("wrote precision.png")
print(f"\n[STAT] precision@k: {json.dumps(res)}")
print(f"[STAT] best alpha P@5: {alphas[int(np.argmax(p5))]:.1f} -> {max(p5):.3f}")
print(f"[STAT] gold queries: {len(GSETS)}")
print("\nDONE.")
