"""Classify call transcripts with the already-trained fraud detection model.

No training is performed here: the TF-IDF vectorizers and the GPU Random
Forest are loaded from fraud_rf_model.joblib.

Usage:
    python predict_fraud.py "text one" "text two"
    python predict_fraud.py --file calls.txt     # one transcript per line
    python predict_fraud.py                      # built-in examples
"""

import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT = BASE_DIR / "fraud_rf_model.joblib"

EXAMPLES = [
    "Congratulations! You have won a free vacation. Send your credit card details to claim the prize.",
    "Hi, this is your pharmacy calling about a prescription refill. Would you like us to auto-renew it?",
]


def load_texts(argv: list[str]) -> list[str]:
    if len(argv) >= 2 and argv[0] == "--file":
        lines = Path(argv[1]).read_text(encoding="utf-8").splitlines()
        texts = [line.strip() for line in lines if line.strip()]
        if not texts:
            raise SystemExit(f"No non-empty lines found in {argv[1]}")
        return texts
    return argv or EXAMPLES


def main() -> None:
    texts = load_texts(sys.argv[1:])

    art = joblib.load(ARTIFACT)
    X = hstack([
        art["word_vectorizer"].transform(texts),
        art["char_vectorizer"].transform(texts),
    ]).astype(np.float32).toarray()

    proba = np.asarray(art["model"].predict_proba(X))[:, 1]
    print(f"Model: {art.get('cuml_version', 'unknown')} cuML | {len(texts)} transcript(s)\n")
    for text, p in zip(texts, proba):
        label = "FRAUD" if p >= 0.5 else "normal"
        print(f"{label:<6} (fraud probability {p:.3f})  {text[:90]}")


if __name__ == "__main__":
    main()
