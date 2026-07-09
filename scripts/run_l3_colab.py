"""Run Level-3 (hybrid retrieval + precision@k) locally against the downloaded Colab cache.
Corpus = 300 MatSynth materials, real Gemma 3 4B captions. Produces numbers + figures for the deductions."""
import json, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, open_clip
from sentence_transformers import SentenceTransformer

BASE = Path("/Users/michael/Desktop/CompSci/Sem 2/Seeing Machines - Final Project")
CACHE = BASE / "Cache" / "SeeingMachines" / "data" / "cache"
OUT = BASE / "docs" / "assets"; OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

METADATA = json.load(open(CACHE / "metadata.json"))
NAMES = [r["name"] for r in METADATA]
CATEGORIES = [r["category"] for r in METADATA]
CAPTIONS = json.load(open(CACHE / "captions.json"))
STUBFMT = json.load(open(CACHE / "stub_format_captions.json"))
N = len(METADATA)

IMG_EMB       = np.load(CACHE / "clip_image_embeds.npy")            # (N,512)
GEMMA_CAP_EMB = np.load(CACHE / "caption_text_embeds.npy")         # real structured captions
STUB_CAP_EMB  = np.load(CACHE / "stub_caption_text_embeds.npy")    # metadata stub
SFMT_EMB      = np.load(CACHE / "stub_format_caption_text_embeds.npy")  # Gemma stub-format
print(f"N={N}  IMG{IMG_EMB.shape} GEMMA{GEMMA_CAP_EMB.shape} STUB{STUB_CAP_EMB.shape} SFMT{SFMT_EMB.shape}")

# --- query encoders ---
clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
clip_tok = open_clip.get_tokenizer("ViT-B-32"); clip_model = clip_model.eval().to(DEVICE)
text_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

@torch.no_grad()
def clip_q(t):
    q = clip_model.encode_text(clip_tok([t]).to(DEVICE)); q = q/q.norm(dim=-1, keepdim=True)
    return q.cpu().numpy()[0]
def txt_q(t): return text_model.encode([t], normalize_embeddings=True)[0].astype(np.float32)

def make_search(emb, is_clip):
    def fn(query, top_k=5):
        s = emb @ (clip_q(query) if is_clip else txt_q(query))
        o = np.argsort(-s)[:top_k]; return [(int(i), float(s[i])) for i in o]
    return fn
clip_search = make_search(IMG_EMB, True)
caption_search = make_search(GEMMA_CAP_EMB, False)
stub_search = make_search(STUB_CAP_EMB, False)
sfmt_search = make_search(SFMT_EMB, False)

def minmax(x):
    lo, hi = x.min(), x.max(); return np.zeros_like(x) if hi-lo < 1e-9 else (x-lo)/(hi-lo)
def hybrid_search(query, alpha=0.5, top_k=5):
    cs = IMG_EMB @ clip_q(query); ts = GEMMA_CAP_EMB @ txt_q(query)
    f = alpha*minmax(cs) + (1-alpha)*minmax(ts); o = np.argsort(-f)[:top_k]
    return [(int(i), float(f[i])) for i in o]

# ---- gold-standard query set ----
def by_cat(c): return lambda r: r["category"].lower() == c
def by_tag(t): return lambda r: t.lower() in [x.lower() for x in r["tags"]]
def all_(*p): return lambda r: all(f(r) for f in p)
def any_(*p): return lambda r: any(f(r) for f in p)
uniq = sorted(set(CATEGORIES))
GOLD = [(c, by_cat(c.lower())) for c in uniq]
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
print(f"\nGold queries: {len(GSETS)}")
for q_, rel in GSETS: print(f"  {q_:18s} {len(rel):3d} relevant")

def p_at_k(fn, k):
    return float(np.mean([sum(1 for i,_ in fn(q, top_k=k) if i in rel)/k for q, rel in GSETS if rel]))

ROUTES = {
    "CLIP-only":        lambda q, top_k: clip_search(q, top_k),
    "Caption-only":     lambda q, top_k: caption_search(q, top_k),
    "GemmaStubFmt":     lambda q, top_k: sfmt_search(q, top_k),
    "MetadataStub":     lambda q, top_k: stub_search(q, top_k),
    "Hybrid a=0.5":     lambda q, top_k: hybrid_search(q, 0.5, top_k),
}
KS = [1, 3, 5, 10]
print("\n=== precision@k ===")
print(f"{'route':16s}" + "".join(f"  P@{k:<3d}" for k in KS))
res = {}
for name, fn in ROUTES.items():
    row = [p_at_k(fn, k) for k in KS]; res[name] = row
    print(f"{name:16s}" + "".join(f"  {v:.3f}" for v in row))

# alpha sweep
alphas = np.linspace(0, 1, 11)
p5 = [p_at_k(lambda q, top_k, a=a: hybrid_search(q, a, top_k), 5) for a in alphas]
best_a = float(alphas[int(np.argmax(p5))])
print(f"\nBest alpha @P5: {best_a:.1f} -> {max(p5):.3f}")

# ---- mis-seeing quantification (real Gemma) ----
CANON = {"ceramic","concrete","fabric","ground","leather","marble","metal","misc",
         "plaster","plastic","stone","terracotta","wood"}
ALIAS = {"textile":"fabric","cloth":"fabric","rock":"stone","soil":"ground","dirt":"ground",
         "tile":"ceramic","brick":"terracotta","steel":"metal","iron":"metal"}
def canon(w):
    w=(w or "").strip().lower(); return w if w in CANON else ALIAS.get(w, w)
mis_struct = mis_sfmt = 0
mismatches = []
for r in METADATA:
    truth = r["category"].lower()
    g = canon(CAPTIONS[r["name"]].get("material_type",""))
    s = canon(STUBFMT[r["name"]].get("category",""))
    if truth != "misc" and g and g != truth and g not in CANON:  # material_type wasn't even a material word
        pass
    if truth != "misc" and g in CANON and g != truth:
        mis_struct += 1; mismatches.append((r["name"], truth, g))
    if truth != "misc" and s in CANON and s != truth:
        mis_sfmt += 1
# also count captions whose material_type is not a material at all (e.g. "dark green")
non_material = sum(1 for r in METADATA if canon(CAPTIONS[r["name"]].get("material_type","")) not in CANON)
print(f"\n=== Mis-seeing (real Gemma) ===")
print(f"structured material_type != category (both material words): {mis_struct}/{N}")
print(f"structured material_type NOT a material word at all: {non_material}/{N}")
print(f"stub-format category != true category: {mis_sfmt}/{N}")
print("examples:", mismatches[:8])

# ---- figures ----
fig,(a1,a2)=plt.subplots(1,2,figsize=(13,4.4))
core = ["CLIP-only","Caption-only","GemmaStubFmt","Hybrid a=0.5"]
for name in core: a1.plot(KS, res[name], marker="o", label=name)
a1.set_xlabel("k"); a1.set_ylabel("precision@k"); a1.set_title("Precision@k by route (300 materials, real Gemma)")
a1.legend(); a1.grid(alpha=.3)
a2.plot(alphas, p5, marker="s", color="purple")
a2.axhline(res["CLIP-only"][2], ls="--", color="tab:blue", label="CLIP-only P@5")
a2.axhline(res["Caption-only"][2], ls="--", color="tab:orange", label="Caption-only P@5")
a2.set_xlabel("alpha (CLIP weight)"); a2.set_ylabel("P@5"); a2.set_title("Hybrid fusion sweep")
a2.legend(); a2.grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT/"colab_l3_precision.png", dpi=120); plt.close()
print("\nwrote docs/assets/colab_l3_precision.png")

# dump numbers for the writeup
json.dump({"precision": res, "best_alpha": best_a, "best_p5": max(p5),
           "n_gold": len(GSETS), "mis_struct": mis_struct, "non_material": non_material,
           "mis_sfmt": mis_sfmt, "N": N},
          open(BASE/"scripts"/"l3_results.json","w"), indent=2)
print("wrote scripts/l3_results.json")
