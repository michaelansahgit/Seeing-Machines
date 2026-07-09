"""Side-by-side ViT-B/32 vs ViT-B/16 CLIP retrieval over the same 89-material corpus.
Embeds with B/16 (cached), then compares rankings, embedding-space separation, and precision@k.
Also writes side-by-side figures to docs/assets/."""
import json, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch, open_clip

PROJ = Path("/Users/michael/Desktop/CompSci/Sem 2/Seeing Machines - Final Project")
CORPUS, CACHE = PROJ/"data"/"corpus", PROJ/"data"/"cache"
ASSETS = PROJ/"docs"/"assets"; ASSETS.mkdir(parents=True, exist_ok=True)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

METADATA = json.load(open(CACHE/"metadata.json"))
NAMES = [r["name"] for r in METADATA]; CATS = [r["category"] for r in METADATA]
N = len(METADATA)

MODELS = {
    "B/32": ("ViT-B-32", "laion2b_s34b_b79k", CACHE/"clip_image_embeds.npy"),
    "B/16": ("ViT-B-16", "laion2b_s34b_b88k", CACHE/"clip16_image_embeds.npy"),
}

def load_and_embed(arch, pretrain, cache_path):
    model, _, pre = open_clip.create_model_and_transforms(arch, pretrained=pretrain)
    tok = open_clip.get_tokenizer(arch); model = model.eval().to(DEVICE)
    keys_path = cache_path.with_suffix(".keys.json")
    if cache_path.exists() and keys_path.exists() and json.load(open(keys_path)) == NAMES:
        emb = np.load(cache_path); print(f"  {arch}: loaded cache {emb.shape}")
    else:
        print(f"  {arch}: embedding {N} images...")
        vecs = []
        with torch.no_grad():
            for r in METADATA:
                x = pre(Image.open(CORPUS/r["file"]).convert("RGB")).unsqueeze(0).to(DEVICE)
                f = model.encode_image(x); f = f/f.norm(dim=-1, keepdim=True)
                vecs.append(f.cpu().numpy()[0])
        emb = np.vstack(vecs).astype(np.float32)
        np.save(cache_path, emb); json.dump(NAMES, open(keys_path, "w"))
        print(f"  {arch}: cached {emb.shape}")
    return model, tok, emb

BUNDLE = {}
for tag, (arch, pretrain, cp) in MODELS.items():
    print(f"Loading {tag} ({arch})")
    BUNDLE[tag] = load_and_embed(arch, pretrain, cp)

def q_search(tag, query, k=5):
    model, tok, emb = BUNDLE[tag]
    with torch.no_grad():
        q = model.encode_text(tok([query]).to(DEVICE)); q = q/q.norm(dim=-1, keepdim=True)
    s = emb @ q.cpu().numpy()[0]; o = np.argsort(-s)[:k]
    return [(int(i), float(s[i])) for i in o]

# ---- embedding-space separation ----
print("\n=== Category separation (mean cosine intra vs inter) ===")
sep = {}
for tag in MODELS:
    emb = BUNDLE[tag][2]; sims = emb @ emb.T
    intra, inter = [], []
    for i in range(N):
        for j in range(i+1, N):
            (intra if CATS[i]==CATS[j] else inter).append(sims[i,j])
    gap = np.mean(intra)-np.mean(inter); sep[tag]=(np.mean(intra),np.mean(inter),gap)
    print(f"  {tag}: intra={np.mean(intra):.3f} inter={np.mean(inter):.3f} gap={gap:+.3f}")

# ---- precision@k (CLIP-only) ----
def by_cat(c): return lambda r: r["category"].lower()==c
def by_tag(t): return lambda r: t.lower() in [x.lower() for x in r["tags"]]
def all_(*p): return lambda r: all(f(r) for f in p)
def any_(*p): return lambda r: any(f(r) for f in p)
uniq = sorted(set(CATS))
GOLD = [(c, by_cat(c.lower())) for c in uniq]
for q_, pred in [("rusty metal", all_(by_cat("metal"), any_(by_tag("rusty"),by_tag("rust")))),
                 ("red surface",by_tag("red")),("green surface",by_tag("green")),
                 ("blue surface",by_tag("blue")),("brown surface",by_tag("brown")),
                 ("smooth material",by_tag("smooth")),("rough material",by_tag("rough")),
                 ("dark material",by_tag("dark")),("cracked surface",any_(by_tag("cracked"),by_tag("crack"))),
                 ("tiled pattern",any_(by_tag("tiles"),by_tag("tile"),by_tag("tiled"))),
                 ("wooden planks",all_(by_cat("wood"),any_(by_tag("planks"),by_tag("plank"))))]:
    if any(pred(r) for r in METADATA): GOLD.append((q_, pred))
GSETS=[(q_,{r["idx"] for r in METADATA if pred(r)}) for q_,pred in GOLD]
def p_at_k(tag,k): return float(np.mean([sum(1 for i,_ in q_search(tag,q,k) if i in rel)/k
                                          for q,rel in GSETS if rel]))
print("\n=== Precision@k (CLIP-only) ===")
KS=[1,3,5,10]; pk={}
print(f"{'model':6s}"+"".join(f"  P@{k:<3d}" for k in KS))
for tag in MODELS:
    row=[p_at_k(tag,k) for k in KS]; pk[tag]=row
    print(f"{tag:6s}"+"".join(f"  {v:.3f}" for v in row))

# ---- qualitative queries ----
QUERIES=["polished marble with veins","woven fabric texture","rusty corroded metal",
         "wood planks","smooth glossy plastic","cracked dry ground"]
print("\n=== Top-5 rankings side by side ===")
for q in QUERIES:
    print(f"\nQUERY: {q}")
    for tag in MODELS:
        hits=q_search(tag,q,5)
        print(f"  {tag}: "+", ".join(f"{METADATA[i]['category']}({s:.2f})" for i,s in hits))

# ---- side-by-side figures for the two most illustrative queries ----
def thumb(nm,px=230):
    im=Image.open(CORPUS/f"{nm}.png").convert("RGB"); im.thumbnail((px,px)); return im
def side_fig(query, fname, k=5):
    fig,axes=plt.subplots(2,k,figsize=(2.5*k,5.3))
    for row,tag in enumerate(MODELS):
        hits=q_search(tag,query,k)
        for col,(i,s) in enumerate(hits):
            ax=axes[row,col]; r=METADATA[i]; ax.imshow(thumb(r["name"]))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{r['category']} · {s:.2f}",fontsize=9)
        axes[row,0].set_ylabel(f"ViT-{tag}",fontsize=12,fontweight="bold")
    fig.suptitle(f"“{query}” — ViT-B/32 (top) vs ViT-B/16 (bottom)",fontsize=12)
    plt.tight_layout(); plt.savefig(ASSETS/fname,dpi=115,bbox_inches="tight"); plt.close()
    print(f"wrote {fname}")

side_fig("polished marble with veins","vit_compare_marble.png")
side_fig("woven fabric texture","vit_compare_woven.png")

# ---- summary bar: P@5 + separation gap ----
fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4))
tags=list(MODELS)
a1.bar(tags,[pk[t][2] for t in tags],color=["#7fa0d6","#3a5a8c"])
a1.set_title("Precision@5 (CLIP-only)"); a1.set_ylim(0,max(pk[t][2] for t in tags)*1.3)
for i,t in enumerate(tags): a1.text(i,pk[t][2]+0.005,f"{pk[t][2]:.3f}",ha="center")
a2.bar(tags,[sep[t][2] for t in tags],color=["#7fa0d6","#3a5a8c"])
a2.set_title("Category separation gap (intra−inter cosine)")
for i,t in enumerate(tags): a2.text(i,sep[t][2]+0.001,f"{sep[t][2]:+.3f}",ha="center")
plt.tight_layout(); plt.savefig(ASSETS/"vit_compare_summary.png",dpi=120); plt.close()
print("wrote vit_compare_summary.png")
print("\nDONE")
