# TomatoLeafAI v24 — web app (local + online)

Photograph a tomato leaf → background removal → "is this a tomato leaf?" check → disease diagnosis with **any trained
model you put in `models/`** (or all of them side by side) → Grad-CAM + Leaf-Focus Score → treatment advice.

**Nothing is hard-coded.** The app finds every trained classifier in `models/` — the v21 notebook's models
(LAG-HViT, VGG16, EfficientNetB7, …) or any other Keras model — reads its input size, scaling, outputs and
Grad-CAM layer from the model itself, and matches it to its evaluation files in `results/`. Add a model by copying
it into `models/` or with **Models → Add a model**; it appears within seconds, no code change, no restart.

Developed by **Dammar Khadayat** — MIT in Artificial Intelligence, Gandaki University.

The app runs **exactly the pipeline the v21 notebook evaluated**. The code in `utils/v21/` is copied unchanged from
the notebook cells (2.1b background removal V3, 2.2b tomato-leaf gate, 5.1b model layers, 9.5 feature OOD,
10.1b Grad-CAM / LFS), and `utils/treatment.py` is Cell 11.1. Every number on the **Models** tab is read from the
notebook's own result files; nothing is recomputed or invented.

**New in v24**

* **Results tab** — the thesis evidence from the v24 notebook, as interactive charts with table views: RQ1–RQ3 and
  H1–H3 verdicts, lab vs field accuracy with 95% CIs, the *"is the hybrid the best model?"* scorecard (every model
  ranked on every criterion, paired tests, per-class F1), robustness to damaged photos, Grad-CAM faithfulness and
  sanity check, Leaf-Focus Score per disease, the tomato-leaf gate (per-source AUROC, stress test), data validation
  (duplicates, leakage fix, stratification) and background-removal quality.
* **Batch diagnosis** — many photos at once with one model; live progress, then a table, a **CSV** and a **PDF report**.
* **PDF report** for every diagnosis (photo, background removal, Grad-CAM, top-5, advice) and **CSV export** of History.
* **Online mode** (`PUBLIC_MODE=1`) — ready for Hugging Face Spaces / Docker: every visitor sees only their own
  history, rate limits, no model upload for strangers (optional admin token). See **[DEPLOY.md](DEPLOY.md)**.

---

## 1. Folder layout

```
TomatoLeafAI_Flask_v21/
├─ app.py                 Flask routes (diagnose, compare, batch, history, exports, Results, models)
├─ wsgi.py, gunicorn.conf.py, start.sh, Dockerfile   online hosting (DEPLOY.md)
├─ requirements.txt       same TensorFlow/Keras/NumPy versions as training
├─ run_app.bat / run_app.sh
├─ models/                ← put the trained files here (step 2)
├─ results/               ← put the notebook's results folder here (step 2)
├─ model_info.json        OPTIONAL display names / types / descriptions per model
├─ custom_layers/         OPTIONAL .py code for custom Keras layers your models use
├─ utils/
│  ├─ registry.py         finds the models in models/ (no fixed list)
│  ├─ adapters.py         runs any model: v21 trunk/head, any Keras/.h5 model, SavedModel
│  ├─ inference.py        the v21 pipeline (ModelBundle) + auto-rescan
│  ├─ metrics.py          reads results/ (Models tab)
│  ├─ history.py          saved diagnoses (instance/history/, per visitor online)
│  ├─ reports.py          CSV / PDF exports
│  ├─ thesis.py           reads the v24 results files for the Results tab
│  ├─ treatment.py        Cell 11.1
│  └─ v21/                notebook cells 2.1b, 2.2b, 5.1b, 9.5, 10.1b (unchanged)
├─ templates/index.html   static/app.js  results.js  net.js  style.css
├─ tools/collect_artifacts.py   copies models/ + results/ from the training folder
├─ tools/push_to_hf.py   publishes models + app to Hugging Face (DEPLOY.md)
├─ tools/fetch_models.py downloads models + results when the container starts
└─ tests/smoke_test.py
```

## 2. Copy the trained files (once)

The notebook writes everything under its project folder (`LOCAL_ROOT` in Cell 1.2), in `models/` and `results/`.
The v21 and v22 notebooks write the same file names, so either works. With v22's `gate_v21_config.json`
(`"ood_gate_input": "background_removed"`) the app feeds the OOD-gate CNN the background-removed crop it
was trained on, and the classifier always gets exactly the notebook's input (background-removed crop, or a clean
letterboxed photo when background removal is unreliable).

**Easiest:** on the machine that has the project folder:

```bash
python tools/collect_artifacts.py /path/to/TomatoLeafAI_Project_v5/TomatoLeafAI_Project_v5
```

**Or copy by hand:**

| Into the app's `models/` | Written by |
|---|---|
| every `<Model>_final.keras` you trained (any number, any names) | Cell 7.3 / 7.3R |
| `ood_gate.keras` | Cell 4.3 |
| `feature_ood_v21.json` | Cell 9.5 |
| `class_indices.json` | Cell 1.4 |

| Into the app's `results/` | Written by | Used for |
|---|---|---|
| `calibration_summary.json` | Cell 8.2 (+ 9.2R) | calibrated confidence (temperature per model) |
| `gate_v21_config.json` | Cell 9.5 | calibrated tomato-leaf gate thresholds |
| `eval/<Model>/…_metrics.json` + PNGs | Cell 9.2 / 9.2R | Models tab: accuracy, F1, per-class, charts |
| `final_comparison_table.csv`, `plots/` | Cell 9.3 | comparison table, charts |
| `statistical_validation.json` | Cell 9.4 | McNemar test |
| `xai_v21_summary.csv`, `plots/xai_v21/` | Cell 10.2 | Leaf-Focus / Lesion-Focus table, Grad-CAM examples |
| `hypothesis_verdicts_v21.json`, `latency_model_size_v21.csv` | Cell 10.3 | H1–H3 verdicts, speed table |
| `research_questions_v23.json`, `rq1_lab_field_table_v23.csv`, `hybrid_proof_scorecard_v24.json`, `per_class_f1_hybrid_vs_best_backbone_v24.csv`, `robustness_field_v24.csv`, `gradcam_faithfulness_v24.csv`, `lfs_by_class_v24.csv`, `ood_per_source_auroc_v24.json`, `ood_stress_test_v24.csv`, `data_validation_v24.json`, `segmentation_qa_v24.json`, `seed_runs_v24.json` | Cells 1.6c, 2.5b, 7.4, 9.5b, 9.6, 10.2b, 10.3, 10.3c | **Results** tab |

Only the model files are required. Without `gate_v21_config.json` the leaf check uses its heuristic stages only;
without `calibration_summary.json` confidences are uncalibrated (T = 1). The page lists anything missing under
"startup notices".

## 2b. Adding ANY trained model

| What you have | What to do |
|---|---|
| `MyNet.keras`, `MyNet_final.keras`, `MyNet_best.keras` | copy into `models/` (name shown = file name without `_final`/`_best`) |
| `MyNet.h5` (older tf.keras) | copy into `models/` |
| SavedModel folder (`saved_model.pb` inside) | copy the folder into `models/` (prediction + saliency map) |
| from the browser | **Models → Add a model** (.keras / .h5, optional display name, type, input scaling, class names) |

The app reads from each model: input size (e.g. 128×128, 224×224, 380×380), grayscale/RGB, softmax or logit
output, dict/list outputs, and the layer for Grad-CAM (last convolutional map; ViT token grid; otherwise a gradient
saliency map). v21 trunk/head models keep the notebook's exact 14×14 lesion-map Grad-CAM and feature-OOD path.

Optional, per model (only when the defaults are wrong):

* **Input scaling** — auto-detected (a `Rescaling` layer inside the model → raw 0–255 pixels, otherwise pixels/255).
  Override in `model_info.json`: `"preprocess": "0-1" | "0-255" | "-1-1" | "caffe" | "torch"`.
* **Class names** — `models/class_indices.json` is used when the model has the same number of outputs. A model trained
  on a different class list needs `models/<Name>_classes.json` (a list, or the notebook/Keras `class_indices` format).
  Any class naming works (`Early_blight`, `Tomato___Early_blight`, …) — it is matched to the treatment guide.
* **Label / type / description / order / default** — `model_info.json` (app folder or `models/`), or `models/<Name>.json`.
* **Custom layers** — put their code in `custom_layers/*.py` (registered with `@keras.saving.register_keras_serializable()`).
* **Evaluation numbers** — `results/eval/<Name>/<Name>_<Split>_metrics.json` (notebook Cell 9.2 format). Every split
  found is shown; names are matched loosely (`vgg16` = `VGG16`).
* **Calibration** — a `"<Name>": {"temperature": T}` entry in `results/calibration_summary.json`; without it T = 1.

Skipped automatically: `ood_gate*.keras`, `*_pvonly.keras`, stage checkpoints (`*_s1`, `*_s2`), `*.weights.h5`.
Removing a file from `models/` unloads that model.

## 3. Install and run

Use **Python 3.9 – 3.12** (TensorFlow 2.17 has no wheels for 3.13).

**Windows** — double-click `run_app.bat`, or in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

**Linux / macOS**

```bash
./run_app.sh            # or: python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && python app.py
```

Open **http://127.0.0.1:5000**. The page opens immediately; the models load in the background (about 20–60 s on a
CPU, shown on the page). Check the setup without a browser: `python tests/smoke_test.py path/to/leaf.jpg`.

### Settings (environment variables, all optional)

| Variable | Default | Meaning |
|---|---|---|
| `MODELS_DIR` / `RESULTS_DIR` | `./models` / `./results` | where the trained files are (you can point straight at the training folder) |
| `CONF_THRESHOLD` | `0.60` | below this calibrated confidence the app says "Not sure" |
| `DEFAULT_MODEL` | automatic | model selected by default (else `"default": true` in model_info.json, else the proposed model, else the most accurate) |
| `LOAD_MODELS` | all found | e.g. `Hybrid_CNN_ViT,VGG16` to load fewer models (less RAM, faster start) |
| `EXCLUDE_MODELS` | — | names in `models/` to ignore |
| `TTA` | `auto` | test-time augmentation (original + h-flip + v-flip, as notebook v23 evaluated and calibrated); `auto` = on when the notebook used it |
| `AUTO_RESCAN` | `5` | seconds between checks of `models/` and `results/` for changes (`0` = off; use the Rescan button) |
| `MODEL_UPLOAD_MAX_MB` | `2048` | size limit for **Add a model** |
| `PORT` / `HOST` | `5000` / `127.0.0.1` | use `HOST=0.0.0.0` to open it from a phone on the same Wi-Fi |
| `CPU_FLOAT32` | `1` | rebuild the mixed-precision models in float32 for CPU speed (same weights) |
| `HISTORY_DIR` | `instance/history` | where saved diagnoses are kept |
| `BATCH_MAX` | `30` (online `10`) | photos per batch |
| `PUBLIC_MODE` | `0` (Docker image: `1`) | online hosting: per-visitor history, rate limits, model management off |
| `ADMIN_TOKEN` | — | online only: whoever enters it (Models tab → Admin) may upload models / rescan |
| `RATE_LIMIT_PER_MIN` | `0` (online `12`) | photos per visitor per minute (`0` = no limit) |
| `MAX_QUEUE` | `0` (online `6`) | diagnoses waiting at once before new ones get "server busy" |
| `HF_MODEL_REPO`, `HF_TOKEN`, `MODEL_URLS` | — | online: where `tools/fetch_models.py` downloads the models / results at start (DEPLOY.md) |

## 4. What the app does for one photo

1. **Resize** in the browser to ≤ 1280 px (fast upload).
2. **Background removal V3** (Cell 2.1b): crop to the leaf; soil, stones, mulch, grass and shadows removed; lesions kept.
3. **Tomato-leaf gate V21** (Cell 2.2b): photo quality → leaf shape / toothed edge / veins / colour / texture →
   deep tomato identity (OOD-gate CNN, Relative-Mahalanobis and energy p-values on the hybrid's features, max
   softmax), with the thresholds calibrated in Cell 9.5. Rejections come with a plain-language reason.
4. **Classifier** on the background-removed crop — the same input Cell 9.2 evaluated, resized to each model's own
   input size — then **temperature scaling** with the model's own fitted T (Cell 8.2). Under 60 % → "Not sure".
5. **Grad-CAM / Grad-CAM++** on the native 14×14 lesion map (v21 models) or the model's last feature map, **Leaf-Focus Score** (attention on the leaf),
   **Lesion-Focus Score** (attention on lesion-coloured tissue), pointing game, disease-region boxes, and the
   hybrid's own lesion-attention map (Cell 10.1b formulas).
6. **Affected-area estimate** (share of leaf with lesion colour — an estimate, not a severity grade) and
   **treatment / management advice** (Cell 11.1) with its disclaimer.

**Compare all** runs every loaded model on the same photo (one leaf check), shows each model's prediction,
confidence, Leaf-Focus Score and Grad-CAM, and the consensus.

## 5. API

| Method | Path | |
|---|---|---|
| POST | `/api/jobs/predict`, `/api/jobs/compare` | start a diagnosis (`leaf_image`, `model_name`) → `job_id` |
| POST | `/api/jobs/batch` | several photos (`leaf_images` repeated, `model_name`) → `job_id`, `batch_id` |
| GET | `/api/history/export.csv[?batch_id=]` | History (or one batch) as CSV |
| GET | `/api/history/<id>/report.pdf`, `/api/batch/<batch_id>/report.pdf` | PDF reports |
| GET | `/api/thesis` | everything the Results tab shows (read from `results/`) |
| GET | `/api/jobs/<job_id>` | progress (`stage`, `pct`, partial results) and the final `result` |
| POST | `/api/predict` | synchronous JSON (`leaf_image`, `model_name`, `mode=single|compare`) |
| GET | `/api/history`, `/api/history/<id>` · DELETE same | saved diagnoses |
| GET | `/api/models`, `/api/status`, `/api/performance`, `/api/diseases`, `/health` | |
| POST | `/api/models/rescan` | look again in `models/` and `results/` |
| POST | `/api/models/upload` | add a model (`model_file`, optional `label`, `kind`, `preprocess`, `classes_file`) |
| GET | `/eval-charts/<file>.png` | notebook charts from `results/` |

```bash
curl -F leaf_image=@leaf.jpg -F mode=compare http://127.0.0.1:5000/api/predict
```

## 6. Put it online

Short version (details, other hosts and costs in **[DEPLOY.md](DEPLOY.md)**):

```bash
pip install huggingface_hub
python tools/push_to_hf.py --project /path/to/TomatoLeafAI_Project_v5 \
       --space <hf-user>/tomatoleafai --models-repo <hf-user>/tomatoleafai-models
```

This uploads the trained models + results to a Hugging Face model repo and the app to a Docker Space (free CPU
tier, 16 GB RAM). The site is live at `https://<hf-user>-tomatoleafai.hf.space` after the first build.

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `pip` cannot find `tensorflow==2.17.1` | Your Python is 3.13+. Install Python 3.12 (`winget install -e --id Python.Python.3.12`) and recreate the venv. |
| A model shows "failed to load" | The Models tab shows the reason. Usual fixes: Keras 3.15.1 (`pip install -r requirements.txt`); custom layers → `custom_layers/`; different class count → `models/<Name>_classes.json`. |
| A model predicts nonsense | Its input scaling differs from the guess — set `"preprocess"` for it in `model_info.json`. |
| Port 5000 already in use | Another copy is running; stop it, or `set PORT=5001` (Windows) / `PORT=5001 python app.py`. |
| Every photo "Not a tomato leaf" | Photograph one leaf filling the frame, in focus, in daylight. If it still happens, check `results/gate_v21_config.json` is from the latest Cell 9.5 run. |
| Out of memory on a small PC | `LOAD_MODELS=Hybrid_CNN_ViT,VGG16` (EfficientNetB7 is the largest). |
| Results tab says "No thesis results yet" | Copy the notebook's `results/` folder (Cells 1.6c, 2.5b, 9.5b, 9.6, 10.2b, 10.3, 10.3c write its files); the list at the bottom of the tab shows which files were found. |
| PDF download fails | `pip install reportlab` (it is in requirements.txt). |

The advice is decision support, not a substitute for a local agronomist or the pesticide label.


py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py


linux:
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py


pip install tensorflow==2.17.1 keras==3.15.1 numpy==1.26.4 "opencv-python-headless>=4.9,<5" "scipy>=1.11" "scikit-learn>=1.3" "pillow>=10.0" "flask>=3.0,<4" "reportlab>=4.0"

hugging face:

python tools\push_to_hf.py --project "C:\Users\ASUS\Downloads\TomatoLeafAI" --space YOUR_HF_NAME/tomatoleafai --models-repo Dammar/tomatoleafai-models

python tools\push_to_hf.py --project "C:\Users\ASUS\Downloads\TomatoLeafAI" --space Dammar/tomatoleafai --models-repo Dammar/tomatoleafai-models
