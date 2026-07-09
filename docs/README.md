# Documentation site — publishing & PDF export

This folder is a self-contained static site (`index.html` + `assets/`). It is the project's
**GitHub Pages** deliverable (Section 6 of the brief).

## Preview locally
```bash
cd "docs"
python3 -m http.server 8000
# open http://localhost:8000
```
(Or just double-click `index.html` — but the local server renders the SVG diagram more reliably.)

## Publish to GitHub Pages (free = public repo)
1. Create a **public** GitHub repo (GitHub Pages on free accounts requires public). MatSynth is CC0, so the
   texture images are safe to publish.
2. Push the project. Two common layouts:
   - **Project site from `/docs`:** push everything, then in **Settings → Pages** set
     *Source = Deploy from a branch*, *Branch = main*, *Folder = /docs*. Your site is at
     `https://<user>.github.io/<repo>/`.
   - **Or** copy the contents of `docs/` to the repo root (or a `gh-pages` branch) if you prefer the URL
     without `/docs`.
3. Wait ~1 minute for the first build, then open the URL.

## PDF export (also required)
The brief accepts browser print-to-PDF:
1. Open the **published** page (or the local preview) in Chrome.
2. `Cmd+P` → **Destination: Save as PDF** → **Background graphics: ON** → Save as
   `SeeingMachines_documentation.pdf`.
3. Submit the PDF together with the live link.

## Before you submit — fill the provisional bits
The page is honest about what's still stubbed. After the Colab Gemma captioning pass:
- Re-run `docs`-figure generation so the caption/hybrid figures reflect **real** captions.
- Replace the "stub" notes with the real numbers and add 2–3 concrete mis-seeing cases to the atlas/dossier.
