# Putting TomatoLeafAI online

The app runs anywhere Docker runs. The trained models need **about 4–8 GB of RAM** together on a CPU (the two
EfficientNetB7-based models are the largest), so very small free tiers (512 MB–1 GB) are not enough.

| Option | Cost | RAM / CPU | Notes |
|---|---|---|---|
| **Hugging Face Spaces (Docker)** — recommended | free "CPU basic" tier | 16 GB / 2 vCPU | HTTPS link, no server to manage; sleeps after a period without visitors (first visit then waits for the start-up); history is lost on restart unless you add persistent storage |
| Any VM (Hetzner, DigitalOcean, AWS Lightsail, a university server) | paid / free if you have one | choose ≥ 8 GB | full control, persistent disk |
| Google Cloud Run / Render / Railway | paid tiers | choose ≥ 8 GB | `Dockerfile` works unchanged |

Plans, limits and prices change — check the provider's current pricing page before choosing.

Online the app runs with `PUBLIC_MODE=1` (the Docker image's default):

* every visitor gets an anonymous random id in their browser and only ever sees **their own** history and reports;
* diagnoses are rate-limited per visitor (`RATE_LIMIT_PER_MIN`, default 12 photos / minute) and queued (`MAX_QUEUE`);
* **model upload and rescan are switched off** — a `.keras` file can contain code, so strangers must never upload one.
  Set an `ADMIN_TOKEN` secret if *you* want to manage models from the browser (Models tab → **Admin**);
* error details and server paths are not shown to visitors.

---

## A. Hugging Face Spaces in one command (recommended)

Run this on the machine that has the trained project folder (the one with `models/` and `results/`).

1. Create a free account at <https://huggingface.co> and a **write** token at <https://huggingface.co/settings/tokens>.
2. In the app folder:

   ```bash
   pip install huggingface_hub
   export HF_TOKEN=hf_xxx            # Windows PowerShell:  $env:HF_TOKEN="hf_xxx"
   python tools/push_to_hf.py --project /path/to/TomatoLeafAI_Project_v5 \
          --space <hf-user>/tomatoleafai \
          --models-repo <hf-user>/tomatoleafai-models \
          --private-models                     # optional: keep the model files private
          # --admin-token "a-long-random-phrase"   optional: model management in the browser
   ```

   It uploads (1) every trained classifier + `ood_gate.keras`, `feature_ood_v21.json`, `class_indices.json` and the
   notebook's `results/` (json, csv, eval/, plots/ — no datasets, no caches) to a model repo, (2) the app code to a
   Docker Space, and (3) sets the Space variables `PUBLIC_MODE=1`, `HF_MODEL_REPO=<models repo>` (and the secrets
   `HF_TOKEN` / `ADMIN_TOKEN` when needed).
3. Open `https://huggingface.co/spaces/<hf-user>/tomatoleafai`. The first build installs TensorFlow (about 5–10
   minutes), then the container downloads the models and loads them (the page shows the progress).
   The direct link for sharing is `https://<hf-user>-tomatoleafai.hf.space`.

**Updating** after retraining or a new notebook run: run the same command again (only changed files are uploaded),
then **Settings → Restart this Space** so it downloads the new models. App-code-only update: add `--skip-models`.

## B. Hugging Face Spaces by hand (web browser only)

1. **New model repo** (e.g. `tomatoleafai-models`) → *Files → Add file → Upload files*: create the folders exactly as
   `models/…` and `results/…` (drag the two folders in; leave out datasets, `*.npz`, `_pred_cache*`).
2. **New Space** → SDK **Docker** → blank template. Upload every file of this app folder **except** `models/`,
   `results/`, `instance/` and `tests/`. Put this header at the very top of the Space's `README.md`:

   ```yaml
   ---
   title: TomatoLeafAI
   emoji: 🍅
   colorFrom: green
   colorTo: green
   sdk: docker
   app_port: 7860
   ---
   ```
3. Space **Settings → Variables and secrets**: variable `HF_MODEL_REPO` = `<hf-user>/tomatoleafai-models`; if that repo
   is private, secret `HF_TOKEN` = a *read* token. Optional secret `ADMIN_TOKEN`.

Alternative without a model repo: upload `models/` and `results/` into the Space itself (Git LFS handles the large
files) — then nothing is downloaded at start.

**Keep history across restarts (optional, paid):** enable *Persistent storage* in the Space settings and add the
variable `HISTORY_DIR=/data/history`.

## C. Docker on any server

```bash
# on the server, in the app folder, with models/ and results/ copied next to app.py
docker build -t tomatoleafai .
docker run -d --restart unless-stopped -p 80:7860 \
  -e ADMIN_TOKEN="a-long-random-phrase" \
  -v "$PWD/models:/home/user/app/models" -v "$PWD/results:/home/user/app/results" \
  -v tomatoleafai-history:/home/user/app/instance \
  --name tomatoleafai tomatoleafai
```

Put HTTPS in front with the host's load balancer or a reverse proxy (Caddy / nginx) — the app already trusts the
proxy's `X-Forwarded-*` headers in public mode. Without Docker:
`pip install -r requirements-docker.txt && PUBLIC_MODE=1 gunicorn -c gunicorn.conf.py wsgi:app`.

To download the models at start instead of mounting them, set `HF_MODEL_REPO` (+ `HF_TOKEN`) or `MODEL_URLS`:

```bash
-e MODEL_URLS='[{"url":"https://example.org/Hybrid_CNN_ViT_final.keras","path":"models/Hybrid_CNN_ViT_final.keras"}]'
```

## Online settings

| Variable | Default (Docker) | Meaning |
|---|---|---|
| `PUBLIC_MODE` | `1` | per-visitor history, rate limits, model management off (set `0` for a private single-user server) |
| `ADMIN_TOKEN` | — | enables model upload / rescan for whoever enters it (Models tab → Admin). Use a long random phrase. |
| `HF_MODEL_REPO` / `HF_TOKEN` | — | model repo to download `models/` + `results/` from at start (`HF_TOKEN` only for a private repo) |
| `MODEL_URLS` | — | JSON list of direct download links instead of a model repo |
| `BATCH_MAX` | `10` | photos per batch |
| `RATE_LIMIT_PER_MIN` | `12` | photos per visitor per minute |
| `MAX_QUEUE` | `6` | diagnoses waiting at once; more get "server busy, try again" |
| `LOAD_MODELS` | all | e.g. `Hybrid_CNN_ViT,VGG16,RegNetY008,CNN_Only` to save RAM |
| `THREADS` | `8` | web threads (inference itself runs one diagnosis at a time) |
| `HISTORY_MAX_ITEMS` / `HISTORY_MAX_PER_VISITOR` | `1000` / `200` | stored diagnoses in total / per visitor (oldest removed first) |
| `FRAME_ANCESTORS` | `'self' https://huggingface.co https://*.hf.space` | sites allowed to show the app in a frame |

## Troubleshooting

| Problem | Fix |
|---|---|
| Build fails installing TensorFlow | The image uses Python 3.11 and `tensorflow-cpu==2.17.1`; do not change the Python version in the Dockerfile. |
| Container restarts / "out of memory" | Use a bigger tier, or load fewer models with `LOAD_MODELS`. |
| "Models are still loading" for a long time | Normal for 1–2 minutes after a (re)start on a CPU tier; the page shows the progress. Check the container log for download errors (`[fetch] …`). |
| "No trained models found" online | `HF_MODEL_REPO` is wrong, the repo is private without `HF_TOKEN`, or the repo has no `models/` folder. |
| Visitors see "Too many diagnoses" / "server busy" | Raise `RATE_LIMIT_PER_MIN` / `MAX_QUEUE`, or use a faster tier. |
| Results tab empty online | The model repo has no `results/` folder — re-run `push_to_hf.py` with `--project` pointing at the folder that contains `results/`. |
