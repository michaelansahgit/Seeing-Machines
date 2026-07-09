# Live demo script (~5 minutes)

The brief's final session asks for **one working query path, one documented failure, one thing you'd build
next**. Run-of-show below. Have the Colab notebook open (cached artifacts loaded) with the Critic UI cell
ready (`LAUNCH_UI=True`), and the docs page in a second tab.

---

### 0 · Frame it (30 s)
> "My archive is **300 PBR material textures** from MatSynth — surfaces, not objects. I built three retrieval
> routes: CLIP image search, a caption route where **Gemma 3 4B** describes each image, and a hybrid — and I
> *measured* all three with precision@k. The headline isn't a win; it's an honest negative result."

Show the **teaser** and the **pipeline diagram** (docs page, top).

### 1 · One working query path (75 s)
In the Critic UI, search **"wood planks"** with the **α slider at 1.0 (pure CLIP)**.
> "All five results are wood at ~0.30. A distinctive material noun with a strong visual signature — CLIP's
> happy path. This is the L1 Finder working end to end: query in, images out."

Drag α toward 0 (caption route).
> "As I lean on Gemma's captions instead of pixels, the ranking degrades — which is the whole story."

### 2 · One documented failure (90 s)
Two failures, escalating:

**CLIP failure —** search **"woven fabric texture"**.
> "Top results are *ceramic tiles*, not fabric. A regular tile grid and a cloth weave are the same frequency
> pattern to CLIP. Backed by the PCA: same-category materials are only **+0.048 cosine** closer than random —
> CLIP barely separates materials by category."

**The real finding — the mis-seeing dossier.**
> "But the bigger failure is the VLM. Gemma names the **wrong material 43 % of the time** on these albedo maps.
> A green plastic becomes 'a dark green surface'; corten steel becomes 'weathered concrete'. It reads
> **colour, not material** — because an albedo map is delit and out-of-distribution."

Show the precision table:
> "So with *real* captions the caption route scores **P@5 0.37 vs CLIP's 0.54**, and the hybrid sweep peaks at
> **α = 1.0** — fusion buys nothing, because one route is just weaker. That's a legitimate negative result."

### 3 · The honesty beat (30 s)
> "The metadata-stub route scores 0.89 — but that's **leakage**: it's built from the labels I'm scoring
> against. I keep it in the table only to show what a too-good number looks like."

### 4 · One thing I'd build next (30 s)
> "**Render, then caption.** The papers that caption PBR materials successfully — MatPedia, Make-it-Real — feed
> the model a *lit render*, not raw albedo. That single change should cut the 43 % mis-seeing rate and might
> finally make the caption route competitive. That's my next step."

---

**Backup if the UI won't launch:** run the `compare_routes_with_gemma_stub_format(...)` and the L3
`precision_at_k` cells — the printed results carry the whole story without Gradio.
