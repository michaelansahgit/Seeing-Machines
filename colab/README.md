# Running the VLM steps on Google Colab

The local Mac (8 GB, no CUDA) runs everything *except* the Gemma 3 4B vision-language model. Do those two
GPU steps on Colab's free T4, cache the result, and bring it back. The main notebook is the same file in
both places — only `CONFIG['RUN_VLM']` changes.

## One-time setup
1. Accept the Gemma license on https://huggingface.co/google/gemma-3-4b-it (required — the model is gated).
2. Upload the project's `data/` folder to Google Drive at `MyDrive/SeeingMachines/data/`
   (you mainly need `data/corpus/` and `data/cache/metadata.json`).

## In Colab
1. Open `seeing_machines.ipynb` in Colab. **Runtime → Change runtime type → T4 GPU.**
2. First code cell: install GPU deps →
   ```python
   !pip install -q -r /content/drive/MyDrive/SeeingMachines/colab/requirements_colab.txt
   ```
3. Authenticate to Hugging Face: `from huggingface_hub import login; login()` (paste a token).
4. In the setup cell (Part 0) set `CONFIG['RUN_VLM'] = True`. `ON_COLAB` auto-detects and mounts Drive.
5. Run **L2.3** → writes `data/cache/captions.json` to Drive (~a few min for 89 images).
6. (Optional, L3) Run **L3.3** `answer_multimodal(...)` at `n_images = 1,2,3,4` and log the degradation.

## Back on the Mac
Copy the refreshed `captions.json` into local `data/cache/`, then re-run the notebook top-to-bottom.
L2.4 will detect the real captions (no longer a stub) and every downstream comparison / precision@k
number becomes the honest result.

> **Quota tip (from the brief):** develop on a CPU runtime and attach the GPU only for the captioning
> run; idling with a GPU attached can lock you out. If you exhaust Colab quota, Kaggle's free tier gives
> ~30 GPU-hours/week.
