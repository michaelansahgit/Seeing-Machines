# Experiment 2 — Render, then caption

A follow-up to the main [Seeing Machines](README.md) project. Experiment 1 found an honest negative result on
**delit albedo maps**: a small VLM reads *colour, not material* (~43 % mis-seeing), so the caption route loses
to CLIP and hybrid fusion buys nothing. Its Limitations section named the fix the literature endorses (MatPedia,
Make-it-Real): **render the material first, then caption the render.** Experiment 2 builds exactly that.

It changes **one variable only** — the input image — and re-runs the identical pipeline (same CLIP model, same
Gemma schema and prompt, the same gold-set construction, same precision@k). Every artifact is duplicated with a
`_rendered` suffix, so Experiment 1 stays intact for a clean side-by-side.

> 🖥️ **The one step beyond Colab.** Each material's full PBR stack — **base-color + normal + roughness +
> metallic**, re-fetched from MatSynth — is rendered onto a lit sphere with **Mitsuba 3** under a fixed in-code
> studio environment (one warm key light, a cool fill, held constant across all 89 materials so only the
> material changes). Mitsuba's `cuda_ad_rgb` path needs a local NVIDIA GPU, so this is the **only** part of the
> project that does not run on the free Colab tier — it was produced on my laptop's **RTX A1000 (6 GB)**. The
> renders are cached to `data/corpus_rendered/`, so everything downstream stays Colab-reproducible with no GPU.

| Albedo (Experiment 1 input) | Mitsuba render (Experiment 2 input) |
|---|---|
| flat, delit texture | shaded sphere: base-color → `principled` BSDF, roughness/metallic maps set reflectance, normal map adds mesostructure |

Figures: `docs/assets/exp2_teaser.png` (rendered-sphere grid), `docs/assets/exp2_pairs.png` (albedo vs render),
`docs/assets/exp2_precision.png` (CLIP precision bar chart).

## Result 1 — the CLIP route (measured locally, real)

Both sides recomputed on the **same 89-material corpus** with the **same 24-query gold set**, so the albedo
baseline here (P@5 0.425) is that subset — *not* the 300-material headline quoted in Experiment 1.

| CLIP-only route | P@1 | P@3 | P@5 | P@10 |
|---|---|---|---|---|
| Albedo (Exp 1 input) | 0.500 | 0.486 | **0.425** | 0.313 |
| Mitsuba render (Exp 2 input) | **0.583** | 0.486 | 0.358 | 0.300 |

**A rank-dependent trade, not a clean win.** Rendering improves the single best hit (P@1 0.500 → 0.583) but
hurts deeper in the ranking (P@5 0.425 → 0.358). Mechanistically: the lit sphere gives CLIP a cleaner,
higher-confidence *top* match, but the fixed grey background and spherical-UV distortion inject a shared,
material-agnostic signal that pulls unrelated spheres together further down the list. For a joint-embedding
model matching pixels to text, presentation is a double-edged cue.

## Result 2 — the caption route (the actual hypothesis, pending Colab)

The literature's claim is about the *VLM*: a lit render gives it the finish and specular cues a flat map cannot.
The test is to re-run the Gemma 3 4B pass over the renders and re-measure the 43 % material-error rate and the
caption-route P@5. That is GPU-VLM work and runs on Colab exactly as in Experiment 1.

**Falsifiable prediction, stated in advance:** the mis-seeing rate should fall well below 43 %, and the
caption-route P@5 should rise above its albedo value of 0.368 — and if it clears the CLIP baseline, hybrid
fusion should for the first time prefer an α strictly inside (0, 1).

> **Honesty note.** As submitted, the rendered *CLIP* numbers are real and reproducible on this machine; the
> rendered *caption/hybrid* numbers await the one Colab Gemma pass. The render is a single-view sphere preview,
> not the material's real-world appearance: any recovery would be evidence that *lit context*, not photographic
> realism, is what the VLM needed.

## Reproduce

**1. Render locally (needs an NVIDIA GPU + Mitsuba):**
```bash
pip install mitsuba datasets numpy pillow
python scripts/render_pbr.py           # fetch PBR maps + render 89 spheres -> data/corpus_rendered/
```

**2. CLIP comparison locally (CPU is fine):**
```bash
pip install torch open_clip_torch matplotlib
python scripts/exp2_eval.py            # -> scripts/exp2_results.json + docs/assets/exp2_precision.png
python scripts/exp2_montage.py         # -> docs/assets/exp2_teaser.png, exp2_pairs.png
```

**3. Caption route on Colab (T4) — the remaining step:** open `seeing_machines.ipynb`, set
`CONFIG['RUN_VLM'] = True`, and run **cell R.5** to caption the renders → `data/cache/captions_rendered.json`.
Cells **R.6–R.9** then produce the real caption/hybrid numbers and the final albedo-vs-render comparison.

Notebook: the Experiment-2 cells are **Part R** in `seeing_machines.ipynb`. The full write-up with figures is
the "Experiment 2 — Render, then caption" section of `docs/index.html`.
