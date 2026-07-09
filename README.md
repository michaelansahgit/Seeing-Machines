# Seeing Machines — A Multimodal Archive Companion

Final project for **CompSci for Designers 2** (MA Design for Digital Futures, TH Nürnberg).
A multimodal retrieval + conversation system over a curated archive of **MatSynth** PBR material textures,
built in three layers: **L1 Finder → L2 Companion → L3 Critic**.

| | |
|---|---|
| **Level** | L3 "The Critic" (built sequentially over L1 and L2) |
| **Corpus** | 89 base-color material textures from the MatSynth `test` split, 13 categories |
| **Models** | CLIP `ViT-B-32` (LAION-2B) · `all-MiniLM-L6-v2` text · Gemma 3 4B (Colab) |

## Layout
```
seeing_machines.ipynb     # the whole project, runs top-to-bottom on CPU/Mac
environment.yml           # local conda env (no CUDA deps)
colab/                    # how to run the GPU-only Gemma steps + requirements
data/
  corpus/                 # 512px basecolor PNGs (the archive)
  cache/                  # metadata.json, *.npy embeddings, captions.json  (reproducible artifacts)
```

## Run locally
```bash
conda env create -f environment.yml      # one time
conda activate seeing-machines
jupyter lab seeing_machines.ipynb        # select the "Python (seeing-machines)" kernel
```
Everything is cached, so a fresh run is fast and never needs a GPU. The Gemma captioning step runs on
Colab (see `colab/README.md`); until then the notebook uses a clearly-labelled metadata stub so it still
runs end-to-end.

## What each level delivers
- **L1 Finder** — CLIP joint-embedding search, a 10-query retrieval atlas, a Gradio search box.
- **L2 Companion** — a structured VLM description schema + prompt, caption-text retrieval, a CLIP-vs-caption
  comparison, and a mis-seeing dossier.
- **L3 Critic** — α-weighted hybrid (CLIP+caption) retrieval, multimodal answering, and a precision@k
  evaluation across all three routes on a label-derived gold query set.
