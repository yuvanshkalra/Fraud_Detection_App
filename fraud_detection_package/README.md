# Fraud Call Detection — Inference Package

A trained Random Forest model that classifies call transcripts as **fraud** or
**normal**. The model is already trained and saved — you only need to install
the environment and run `predict_fraud.py`. No training step is required.

Model performance on the held-out test set (1,280 calls):

| Metric | Value |
|---|---|
| Accuracy | 0.9805 |
| Precision (fraud) | 0.9433 |
| Recall (fraud) | 0.9289 |
| F1 (fraud) | 0.9361 |
| ROC-AUC | 0.9942 |

Confusion matrix: `[[1072, 11], [14, 183]]` (rows = true, columns = predicted).

---

## 1. Package contents

```
fraud_detection_package/
├── README.md                  <- this file
├── requirements.txt           <- direct dependencies (pinned)
├── requirements-lock.txt      <- full environment freeze (exact versions)
├── fraud_rf_model.joblib      <- trained model + both TF-IDF vectorizers
├── predict_fraud.py           <- inference entry point (no training)
├── examples/sample_calls.txt  <- sample transcripts for a quick test
├── model_metrics.json         <- evaluation metrics
├── classification_report.txt  <- precision/recall/F1 report
└── training/                  <- optional: everything needed to retrain
    ├── build_dataset.py
    ├── train_random_forest_gpu.py
    ├── combined_dataset.csv
    ├── English_NonScam.txt
    ├── English_Scam.txt
    └── fraud_call.file
```

The saved artifact contains:

| Key | Object |
|---|---|
| `word_vectorizer` | sklearn `TfidfVectorizer` (word 1–2 grams, 30k features) |
| `char_vectorizer` | sklearn `TfidfVectorizer` (char 3–5 grams, 30k features) |
| `model` | cuML `RandomForestClassifier` (500 trees, balanced class weights) |
| `cuml_version` | version string used at training time |

> **Important:** the model is a **cuML** (GPU) estimator. Loading it requires
> `cuml` in the environment — a scikit-learn-only environment cannot unpickle
> it.

---

## 2. Requirements

### Hardware / OS

- Linux x86_64 (tested on Ubuntu with an NVIDIA RTX 2050, 4 GB VRAM)
- NVIDIA GPU supported by CUDA 13 (compute capability 7.0+)
- NVIDIA driver supporting CUDA 13.x (tested with driver **580.178.04**,
  CUDA 13.0). You do **not** need to install the full CUDA toolkit — the pip
  wheels bundle the CUDA 13 runtime libraries.
- ~4 GB free VRAM for training; inference needs far less (~1 GB).
- Python **3.13** (cuML wheels are `cp311-abi3`, so 3.11+ works, but this
  environment was built and tested on 3.13).

### Libraries used

| Library | Version | Used for |
|---|---|---|
| `cuml-cu13` | 26.10.0 | GPU Random Forest (`cuml.ensemble.RandomForestClassifier`) |
| `cupy-cuda13x` | 14.2.0 | GPU arrays, required by cuML |
| `scikit-learn` | 1.9.1 | `TfidfVectorizer`, train/test split, CV, metrics |
| `pandas` | 3.0.6 | Reading/combining CSV data |
| `numpy` | 2.4.6 | Array operations |
| `scipy` | 1.18.1 | Sparse matrices (`hstack`) |
| `joblib` | 1.6.0 | Saving/loading the model artifact |

cuML also pulls in its own runtime stack (`cudf`, `libcuml`, `libcudf`,
`treelite`, `nvforest`, `nvidia-*` CUDA libraries). The exact versions are in
`requirements-lock.txt`.

### Why the NVIDIA package index?

RAPIDS cuML packages are published on NVIDIA's index, not (fully) on PyPI:

```
--extra-index-url https://pypi.nvidia.com
```

**Pin the version** (`cuml-cu13==26.10.0`). An unpinned `cuml-cu13` can resolve
to a 1 KB placeholder (`0.0.0a0`) that installs "successfully" but provides no
model. With `uv`, also pass `--index-strategy unsafe-best-match` so uv may take
different packages from PyPI and the NVIDIA index.

---

## 3. Set up the environment (one time)

Run these commands from inside `fraud_detection_package/`.

### Option A — with `uv` (what was used to build this package)

```bash
uv venv --python 3.13 .venv

uv pip install --python .venv/bin/python -r requirements.txt \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match
```

### Option B — with standard `pip`

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install --upgrade pip

.venv/bin/pip install -r requirements.txt \
    --extra-index-url https://pypi.nvidia.com
```

### Option C — exact reproduction of the original environment

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match
```

Verify the GPU stack:

```bash
.venv/bin/python -c "import cuml, cupy as cp; print(cuml.__version__, cp.cuda.runtime.getDeviceProperties(0)['name'])"
# expected: 26.10.00 b'NVIDIA GeForce RTX 2050'
```

> The `.venv` directory is not part of this package. Create it once on the
> machine where you will run inference.

---

## 4. Run the model (no training)

Classify one transcript:

```bash
.venv/bin/python predict_fraud.py "Congratulations! You won a free vacation. Send your credit card details to claim it."
```

Classify several transcripts:

```bash
.venv/bin/python predict_fraud.py \
  "Your debit card is about to expire, share the OTP to renew it." \
  "This is your pharmacy calling about a prescription refill."
```

Classify a text file (one transcript per line):

```bash
.venv/bin/python predict_fraud.py --file examples/sample_calls.txt
```

Run with no arguments to use the built-in examples:

```bash
.venv/bin/python predict_fraud.py
```

Example output:

```
Model: 26.10.00 cuML | 2 transcript(s)

FRAUD  (fraud probability 0.842)  Congratulations! You have won a free vacation. Send your credit card details to claim the
normal (fraud probability 0.314)  Hi, this is your pharmacy calling about a prescription refill. Would you like us to auto-rene
```

Predictions use a 0.5 probability threshold. To trade recall for precision,
adjust the threshold in `predict_fraud.py`.

### Using the model from your own Python code

```python
import joblib, numpy as np
from scipy.sparse import hstack

art = joblib.load("fraud_rf_model.joblib")
texts = ["Your card is about to expire, share your PIN."]

X = hstack([
    art["word_vectorizer"].transform(texts),
    art["char_vectorizer"].transform(texts),
]).astype(np.float32).toarray()

proba = np.asarray(art["model"].predict_proba(X))[:, 1]
print(proba)          # fraud probability per transcript
```

---

## 5. Retrain (optional)

The package already contains the trained model. You only need this section if
you change the data or model settings.

```bash
cd training

# 1. Rebuild combined_dataset.csv from the three source files
../.venv/bin/python build_dataset.py

# 2. Train on the GPU (logs progress to train_gpu.log)
../.venv/bin/python -u train_random_forest_gpu.py > train_gpu.log 2>&1
```

Training writes its outputs into `training/`:

- `training/fraud_rf_model.joblib`
- `training/model_metrics_gpu.json`
- `training/classification_report_gpu.txt`

To use the retrained model for inference, replace the package-level artifact:

```bash
cp training/fraud_rf_model.joblib .
```

Training on the RTX 2050 takes roughly 25 minutes (5 CV folds + final
500-tree fit). Expect much faster times on a larger GPU.

### Dataset

`training/combined_dataset.csv` has 6,398 rows (983 fraud, 5,415 normal):

| Source | Label | Rows |
|---|---|---|
| `English_NonScam.txt` | normal | 400 |
| `English_Scam.txt` (numbering prefix stripped) | fraud | 394 |
| `fraud_call.file` (TSV) | fraud / normal | 5,604 |

329 exact duplicate (text, label) pairs were removed. `build_dataset.py`
performs the parsing, normalization, deduplication, and writes both
`combined_dataset.csv` and `combined_dataset.txt` (tab-separated).

---

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'cuml'` | Install with `requirements.txt` and the NVIDIA index; check you are using the `.venv` Python. |
| `cuml-cu13==0.0.0a0` installed but import fails | Unpinned cuML resolved to the placeholder. Pin `cuml-cu13==26.10.0`. |
| `CUDA driver version is insufficient` | Update the NVIDIA driver to one supporting CUDA 13.x (580+). |
| `CUDA out of memory` during training | Close other GPU processes, or lower `max_features` / `n_estimators` in `training/train_random_forest_gpu.py`. |
| Treelite "newer version" warning | Harmless; the GPU forest is still trained/predicted correctly. |
| Inference is slow on first call | cuML initializes CUDA context on first use; subsequent calls are fast. |

---

## 7. Notes and limitations

- The model was trained on English call transcripts. Performance on other
  languages, domains, or very short SMS-style text may differ.
- A large share of the fraud examples come from templated scam texts, so the
  model may rely on wording patterns. Treat predictions as a screening signal,
  not a final decision.
- The TF-IDF vocabulary is capped at 30k word + 30k char features because
  cuML requires dense input and the reference GPU has 4 GB VRAM.
- Probability outputs are Random Forest vote frequencies, not calibrated
  probabilities.
