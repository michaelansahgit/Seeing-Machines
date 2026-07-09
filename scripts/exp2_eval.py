"""Experiment 2 evaluation — rendered corpus vs albedo corpus, run locally.

Mirrors run_l3_colab.py but adds the rendered corpus. The CLIP route needs only images, so the
albedo-vs-render CLIP comparison is fully reproducible on this laptop. The caption/hybrid routes
need the Gemma pass over the renders (captions_rendered.json); until that Colab run exists this
script reports CLIP-only for real and marks the caption route as pending.

Run:  python scripts/exp2_eval.py
"""
import json, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, open_clip

_CANDS = [Path(__file__).resolve().parent.parent,
          Path(r"D:\Seeing Machines - Final Project-20260708T194711Z-3-002\Seeing Machines - Final Project")]
BASE = next((p for p in _CANDS if (p / "data" / "cache" / "metadata.json").exists()), _CANDS[0])
CACHE = BASE / "data" / "cache"
CORPUS = BASE / "data" / "corpus"
REND = BASE / "data" / "corpus_rendered"
OUT = BASE / "docs" / "assets"; OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

METADATA = json.load(open(CACHE / "metadata.json", encoding="utf-8"))
NAMES = [r["name"] for r in METADATA]
CATEGORIES = [r["category"] for r in METADATA]
N = len(METADATA)

from PIL import Image
clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
clip_tok = open_clip.get_tokenizer("ViT-B-32"); clip_model = clip_model.eval().to(DEVICE)

@torch.no_grad()
def clip_q(t):
    q = clip_model.encode_text(clip_tok([t]).to(DEVICE)); q = q / q.norm(dim=-1, keepdim=True)
    return q.cpu().numpy()[0]

@torch.no_grad()
def embed_dir(directory):
    vecs = []
    for r in METADATA:
        img = Image.open(Path(directory) / r["file"]).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(DEVICE)
        f = clip_model.encode_image(x); f = f / f.norm(dim=-1, keepdim=True)
        vecs.append(f.cpu().numpy()[0])
    return np.vstack(vecs).astype(np.float32)

# albedo embeds: reuse Experiment-1 cache if present, else recompute
alb_path = CACHE / "clip_image_embeds.npy"
if alb_path.exists() and json.load(open(CACHE / "clip_image_keys.json")) == NAMES:
    IMG_ALB = np.load(alb_path); print(f"albedo CLIP embeds (cached): {IMG_ALB.shape}")
else:
    print("embedding albedo corpus..."); IMG_ALB = embed_dir(CORPUS)

# rendered embeds: cache
ren_key = CACHE / "clip_image_keys_rendered.json"; ren_path = CACHE / "clip_image_embeds_rendered.npy"
if ren_path.exists() and ren_key.exists() and json.load(open(ren_key)) == NAMES:
    IMG_REN = np.load(ren_path); print(f"rendered CLIP embeds (cached): {IMG_REN.shape}")
else:
    missing = [r["file"] for r in METADATA if not (REND / r["file"]).exists()]
    if missing:
        raise SystemExit(f"rendered corpus incomplete ({len(missing)} missing) — run scripts/render_pbr.py first")
    print("embedding rendered corpus..."); IMG_REN = embed_dir(REND)
    np.save(ren_path, IMG_REN); json.dump(NAMES, open(ren_key, "w"))

# ---- gold set (identical to run_l3_colab.py) ----
def by_cat(c): return lambda r: r["category"].lower() == c
def by_tag(t): return lambda r: t.lower() in [x.lower() for x in r["tags"]]
def all_(*p): return lambda r: all(f(r) for f in p)
def any_(*p): return lambda r: any(f(r) for f in p)
GOLD = [(c, by_cat(c.lower())) for c in sorted(set(CATEGORIES))]
for q_, pred in [("rusty metal", all_(by_cat("metal"), any_(by_tag("rusty"), by_tag("rust")))),
                 ("woven fabric", all_(by_cat("fabric"), by_tag("woven"))),
                 ("red surface", by_tag("red")), ("green surface", by_tag("green")),
                 ("blue surface", by_tag("blue")), ("brown surface", by_tag("brown")),
                 ("smooth material", by_tag("smooth")), ("rough material", by_tag("rough")),
                 ("dark material", by_tag("dark")),
                 ("cracked surface", any_(by_tag("cracked"), by_tag("crack"))),
                 ("tiled pattern", any_(by_tag("tiles"), by_tag("tile"), by_tag("tiled"))),
                 ("wooden planks", all_(by_cat("wood"), any_(by_tag("planks"), by_tag("plank"))))]:
    if any(pred(r) for r in METADATA): GOLD.append((q_, pred))
GSETS = [(q_, {r["idx"] for r in METADATA if pred(r)}) for q_, pred in GOLD]
print(f"gold queries: {len(GSETS)}")

def clip_search(emb, q, k):
    s = emb @ clip_q(q); return [int(i) for i in np.argsort(-s)[:k]]
def p_at_k(emb, k):
    return float(np.mean([sum(1 for i in clip_search(emb, q, k) if i in rel)/k for q, rel in GSETS if rel]))

KS = [1, 3, 5, 10]
alb = [p_at_k(IMG_ALB, k) for k in KS]
ren = [p_at_k(IMG_REN, k) for k in KS]
print("\n=== CLIP-only precision@k ===")
print("k        " + "".join(f"P@{k:<4d}" for k in KS))
print("albedo   " + "".join(f"{v:.3f} " for v in alb))
print("render   " + "".join(f"{v:.3f} " for v in ren))

# caption route (only if a real Gemma-over-renders cache exists)
cap_note = "pending Colab Gemma pass over renders"
capr = CACHE / "captions_rendered.json"
if capr.exists():
    cj = json.load(open(capr))
    if not any(v.get("_stub") for v in cj.values()):
        cap_note = "real captions present — see notebook R.6/R.7 for caption+hybrid numbers"

# figure
fig, ax = plt.subplots(figsize=(7.5, 4.4))
x = np.arange(len(KS)); w = 0.38
ax.bar(x - w/2, alb, w, label="albedo (Exp 1)", color="#c9a24b")
ax.bar(x + w/2, ren, w, label="Mitsuba render (Exp 2)", color="#3a5a8c")
for xi, (a, r) in enumerate(zip(alb, ren)):
    ax.text(xi - w/2, a + .01, f"{a:.2f}", ha="center", fontsize=8)
    ax.text(xi + w/2, r + .01, f"{r:.2f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([f"P@{k}" for k in KS])
ax.set_ylabel("precision@k"); ax.set_ylim(0, max(alb+ren)*1.2)
ax.set_title("CLIP-only retrieval: albedo vs Mitsuba render (89 materials, same gold set)")
ax.legend(); ax.grid(alpha=.3, axis="y")
plt.tight_layout(); plt.savefig(OUT / "exp2_precision.png", dpi=130); plt.close()
print("\nwrote docs/assets/exp2_precision.png")

json.dump({"clip_albedo": dict(zip([f"P@{k}" for k in KS], alb)),
           "clip_rendered": dict(zip([f"P@{k}" for k in KS], ren)),
           "n_gold": len(GSETS), "N": N, "caption_route": cap_note},
          open(BASE / "scripts" / "exp2_results.json", "w"), indent=2)
print("wrote scripts/exp2_results.json ·", cap_note)
