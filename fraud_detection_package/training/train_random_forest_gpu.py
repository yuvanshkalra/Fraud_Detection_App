"""Train a GPU Random Forest fraud-call classifier with RAPIDS cuML.

Model: TF-IDF (word 1-2 grams + char 3-5 grams) -> cuML RandomForestClassifier.
Outputs: fraud_rf_model.joblib, model_metrics_gpu.json, classification_report_gpu.txt.
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "combined_dataset.csv"
MODEL_PATH = BASE_DIR / "fraud_rf_model.joblib"


def build_vectorizers() -> tuple[TfidfVectorizer, TfidfVectorizer]:
    # Vocabulary is capped because cuML 26.10 requires dense input and the
    # GPU has 4 GB of VRAM.
    word = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=3,
        sublinear_tf=True, strip_accents="unicode",
        max_features=30_000, dtype=np.float32,
    )
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=3,
        sublinear_tf=True, strip_accents="unicode",
        max_features=30_000, dtype=np.float32,
    )
    return word, char


def fit_model(X_text, y, word, char, n_estimators=500):
    """Fit vectorizers + cuML Random Forest; returns artifacts."""
    from cuml.ensemble import RandomForestClassifier as cuRF

    X_word = word.fit_transform(X_text)
    X_char = char.fit_transform(X_text)
    X = hstack([X_word, X_char], format="csr").astype(np.float32).toarray()

    model = cuRF(
        n_estimators=n_estimators,
        max_features="sqrt",
        n_bins=64,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X, y)
    return model, X


def main() -> None:
    try:
        import cuml
        from cuml.ensemble import RandomForestClassifier as cuRF  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"cuML is not installed in this interpreter: {exc}")

    try:
        import cupy as cp
        gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception:
        gpu_name = "unknown GPU"
    print(f"cuML {cuml.__version__} | GPU: {gpu_name}")

    df = pd.read_csv(DATA_PATH)
    X, y = df["text"].astype(str).to_numpy(), df["label"].to_numpy(dtype=np.int32)
    print(f"Loaded {len(df)} rows ({int(y.sum())} fraud, {int((y == 0).sum())} normal)",
          flush=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train/test split: {len(X_train)}/{len(X_test)}", flush=True)

    # 5-fold CV on the training split (vectorizers refit per fold, no leakage).
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1, cv_acc = [], []
    for fold, (tr, va) in enumerate(skf.split(X_train, y_train), 1):
        print(f"[CV] fold {fold}/5: vectorizing + GPU fit...", flush=True)
        t_fold = time.perf_counter()
        w, c = build_vectorizers()
        m, _ = fit_model(X_train[tr], y_train[tr], w, c, n_estimators=300)
        pred = np.asarray(m.predict(
            hstack([w.transform(X_train[va]), c.transform(X_train[va])],
                   format="csr").astype(np.float32).toarray()))
        cv_f1.append(f1_score(y_train[va], pred))
        cv_acc.append(accuracy_score(y_train[va], pred))
        print(f"[CV] fold {fold}/5 done in {time.perf_counter() - t_fold:.1f}s: "
              f"accuracy={cv_acc[-1]:.4f}  f1={cv_f1[-1]:.4f}", flush=True)
    print(f"5-fold CV: accuracy={np.mean(cv_acc):.4f} +/- {np.std(cv_acc):.4f}  "
          f"f1={np.mean(cv_f1):.4f} +/- {np.std(cv_f1):.4f}")

    # Final model on the full training split.
    print("[final] vectorizing full training set...", flush=True)
    word, char = build_vectorizers()
    t0 = time.perf_counter()
    model, _ = fit_model(X_train, y_train, word, char)
    fit_seconds = time.perf_counter() - t0
    print(f"[final] GPU fit done in {fit_seconds:.2f}s", flush=True)

    X_test_mat = hstack([word.transform(X_test), char.transform(X_test)],
                        format="csr").astype(np.float32).toarray()
    y_pred = np.asarray(model.predict(X_test_mat)).astype(int)
    y_prob = np.asarray(model.predict_proba(X_test_mat))[:, 1].astype(float)

    metrics = {
        "framework": f"cuml {cuml.__version__}",
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "label_counts": {k: int(v) for k, v in df["label_name"].value_counts().items()},
        "gpu_fit_seconds": round(fit_seconds, 2),
        "cv_accuracy_mean": round(float(np.mean(cv_acc)), 4),
        "cv_f1_mean": round(float(np.mean(cv_f1)), 4),
        "cv_f1_std": round(float(np.std(cv_f1)), 4),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    report = classification_report(y_test, y_pred,
                                   target_names=["normal", "fraud"], digits=4)

    print("\nHold-out test metrics:")
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"  {key:<10} {metrics[key]:.4f}")
    print(f"\nConfusion matrix (rows=true, cols=pred): "
          f"{metrics['confusion_matrix']}")
    print(f"\n{report}")

    joblib.dump({"word_vectorizer": word, "char_vectorizer": char,
                 "model": model, "cuml_version": cuml.__version__},
                MODEL_PATH)
    (BASE_DIR / "model_metrics_gpu.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    (BASE_DIR / "classification_report_gpu.txt").write_text(
        report, encoding="utf-8")

    print(f"Saved model: {MODEL_PATH.name}")
    print("Saved: model_metrics_gpu.json, classification_report_gpu.txt")


if __name__ == "__main__":
    main()
