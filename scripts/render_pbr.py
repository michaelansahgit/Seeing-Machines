"""Experiment 2 · "Render, then caption" — LOCAL GPU STEP (beyond Colab scope).

Fetches the full PBR map stack (basecolor / normal / roughness / metallic / height) for the
same MatSynth `test`-split materials used in Experiment 1, then renders each onto a lit sphere
with Mitsuba 3 (cuda_ad_rgb) under a fixed synthetic studio environment. Output is one shaded
RGB PNG per material in data/corpus_rendered/, a drop-in replacement for the delit albedo corpus.

This is the only step in the project that does not run on the free Colab tier: Mitsuba's CUDA
variant needs a local NVIDIA GPU (developed on an RTX A1000 6 GB). Everything downstream
(CLIP embedding, Gemma captioning, hybrid fusion, precision@k) is unchanged from Experiment 1.

Run:  python scripts/render_pbr.py                # all materials in metadata.json
      LIMIT=3 python scripts/render_pbr.py        # quick smoke test (first 3)
"""
import os, sys, json, time
from pathlib import Path
import numpy as np
from PIL import Image

# --- paths: portable across the Mac dev machine and this Windows laptop --------
_CANDIDATES = [
    Path(__file__).resolve().parent.parent,                       # scripts/ -> project root
    Path(r"D:\Seeing Machines - Final Project-20260708T194711Z-3-002\Seeing Machines - Final Project"),
    Path("/Users/michael/Desktop/CompSci/Sem 2/Seeing Machines - Final Project"),
]
PROJECT_DIR = next((p for p in _CANDIDATES if (p / "data" / "cache" / "metadata.json").exists()), _CANDIDATES[0])
CACHE_DIR = PROJECT_DIR / "data" / "cache"
MAPS_DIR  = PROJECT_DIR / "data" / "pbr_maps"
REND_DIR  = PROJECT_DIR / "data" / "corpus_rendered"
ENV_PATH  = MAPS_DIR / "_studio_env.exr"
MAPS_DIR.mkdir(parents=True, exist_ok=True)
REND_DIR.mkdir(parents=True, exist_ok=True)

# --- render parameters (held constant across ALL materials — the control) ------
MAP_PX    = 1024          # stored map resolution (down from MatSynth's 4096)
REND_PX   = 512           # render output resolution (matches the albedo corpus at 512px)
SPP       = 128           # samples per pixel
UV_SCALE  = 3.0           # texture tiling on the sphere (reveals pattern/frequency)
MAP_KEYS  = ["basecolor", "normal", "roughness", "metallic", "height"]

METADATA = json.load(open(CACHE_DIR / "metadata.json", encoding="utf-8"))
LIMIT = int(os.environ.get("LIMIT", str(len(METADATA))))
TARGETS = [r["name"] for r in METADATA][:LIMIT]


def fetch_maps(target_names):
    """Stream MatSynth, projecting to the 5 maps we need, saving each downscaled to MAP_PX.
    Idempotent: a material whose folder already has all 5 maps is skipped."""
    from datasets import load_dataset
    want = set(target_names)
    have = {n for n in want if all((MAPS_DIR / n / f"{k}.png").exists() for k in MAP_KEYS)}
    todo = want - have
    print(f"[fetch] {len(have)} cached · {len(todo)} to download", flush=True)
    if not todo:
        return
    ds = load_dataset("gvecchio/MatSynth", split="test", streaming=True).select_columns(["name"] + MAP_KEYS)
    done = 0
    for ex in ds:
        if not todo:
            break
        name = ex["name"]
        if name not in todo:
            continue
        d = MAPS_DIR / name; d.mkdir(parents=True, exist_ok=True)
        for k in MAP_KEYS:
            img = ex[k]
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            img.thumbnail((MAP_PX, MAP_PX), Image.LANCZOS)
            img.save(d / f"{k}.png")
        todo.discard(name); done += 1
        print(f"[fetch] {name}  ({len(have)+done}/{len(want)})", flush=True)


def build_envmap(path, h=512, w=1024):
    """Synthetic lat-long studio HDRI: cool sky gradient + one warm key light + a cooler fill.
    Self-contained (no external HDRI download) and identical for every material, so any
    difference between renders comes from the material maps, not the lighting."""
    if path.exists():
        return
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    v = yy / (h - 1); u = xx / (w - 1)
    img = (0.55 - 0.4 * v)[..., None] * np.array([0.9, 0.95, 1.0], np.float32)
    def disk(cu, cv, r, I, warm):
        d2 = ((u - cu) * 2.0) ** 2 + (v - cv) ** 2
        return (I * np.exp(-d2 / (2 * r * r)))[..., None] * warm
    img = img + disk(0.32, 0.30, 0.10, 9.0, np.array([1.0, 0.97, 0.90], np.float32))   # key
    img = img + disk(0.72, 0.45, 0.16, 2.2, np.array([0.85, 0.90, 1.0], np.float32))   # fill
    img = np.maximum(img, 0.0).astype(np.float32)
    import mitsuba as mi
    mi.Bitmap(img, pixel_format=mi.Bitmap.PixelFormat.RGB).write(str(path))
    print(f"[env] wrote {path}", flush=True)


def render_material(mi, name):
    d = MAPS_DIR / name
    def tex(fname, raw):
        return {"type": "bitmap", "filename": str(d / fname), "raw": raw,
                "to_uv": mi.ScalarTransform4f().scale([UV_SCALE, UV_SCALE, 1.0])}
    principled = {"type": "principled",
                  "base_color": tex("basecolor.png", raw=False),
                  "roughness":  tex("roughness.png", raw=True),
                  "metallic":   tex("metallic.png",  raw=True)}
    bsdf = {"type": "normalmap", "normalmap": tex("normal.png", raw=True), "bsdf": principled}
    scene = mi.load_dict({
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 8},
        "sensor": {"type": "perspective", "fov": 35,
                   "to_world": mi.ScalarTransform4f().look_at(origin=[0, 0, 4.2], target=[0, 0, 0], up=[0, 1, 0]),
                   "film": {"type": "hdrfilm", "width": REND_PX, "height": REND_PX,
                            "rfilter": {"type": "gaussian"}, "pixel_format": "rgb"},
                   "sampler": {"type": "independent", "sample_count": SPP}},
        "env": {"type": "envmap", "filename": str(ENV_PATH)},
        "sphere": {"type": "sphere", "radius": 1.0, "bsdf": bsdf},
    })
    img = mi.render(scene, spp=SPP)
    out = REND_DIR / f"{name}.png"
    mi.Bitmap(img).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, srgb_gamma=True).write(str(out))
    return out


def main():
    print(f"[render_pbr] project: {PROJECT_DIR}")
    print(f"[render_pbr] {len(TARGETS)} materials · maps@{MAP_PX}px · render@{REND_PX}px/{SPP}spp", flush=True)
    fetch_maps(TARGETS)
    import mitsuba as mi
    mi.set_variant("cuda_ad_rgb")
    build_envmap(ENV_PATH)
    t0 = time.time(); done = 0
    for i, name in enumerate(TARGETS):
        out = REND_DIR / f"{name}.png"
        if out.exists():
            continue
        render_material(mi, name); done += 1
        if done % 10 == 0 or i == len(TARGETS) - 1:
            print(f"[render] {i+1}/{len(TARGETS)}  ({(time.time()-t0):.0f}s)", flush=True)
    print(f"[render_pbr] done · {done} rendered · {len(TARGETS)} total in {REND_DIR}", flush=True)


if __name__ == "__main__":
    main()
