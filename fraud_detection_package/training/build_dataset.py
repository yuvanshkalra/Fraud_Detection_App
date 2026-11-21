"""Build a single combined fraud-call dataset from the three source files.

Sources
-------
English_NonScam.txt : one normal (non-scam) call transcript per line.
English_Scam.txt    : one scam transcript per line, prefixed with "N.<TAB>".
fraud_call.file     : tab-separated "<label>\t<text>", label in {fraud, normal}.

Outputs
-------
combined_dataset.csv : text,label,label_name,source
combined_dataset.txt : tab-separated text,label,label_name,source
"""

import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
NUMBER_PREFIX = re.compile(r"^\s*\d+\s*[.)]\s*")


def normalize(text: str) -> str:
    """Collapse all whitespace runs (incl. CR/LF/tabs) into single spaces."""
    return " ".join(text.split())


def read_plain(path: Path, label: int, source: str) -> list[dict]:
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        text = normalize(raw)
        if text:
            rows.append({"text": text, "label": label,
                         "label_name": "fraud" if label else "normal",
                         "source": source})
    return rows


def read_numbered(path: Path, label: int, source: str) -> list[dict]:
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        text = normalize(NUMBER_PREFIX.sub("", raw))
        if text:
            rows.append({"text": text, "label": label,
                         "label_name": "fraud" if label else "normal",
                         "source": source})
    return rows


def read_labeled(path: Path) -> list[dict]:
    label_map = {"fraud": 1, "normal": 0}
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        parts = raw.split("\t", 1)
        if len(parts) != 2:
            continue
        name, text = parts[0].strip().lower(), normalize(parts[1])
        if name not in label_map or not text:
            continue
        rows.append({"text": text, "label": label_map[name],
                     "label_name": name, "source": path.name})
    return rows


def main() -> None:
    rows = []
    rows += read_plain(BASE_DIR / "English_NonScam.txt", 0, "English_NonScam")
    rows += read_numbered(BASE_DIR / "English_Scam.txt", 1, "English_Scam")
    rows += read_labeled(BASE_DIR / "fraud_call.file")

    df = pd.DataFrame(rows)

    # Exact duplicate removal (same normalized text and label).
    before = len(df)
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    removed = before - len(df)

    # Report any text that appears with both labels (label noise).
    conflicts = (
        df.groupby("text")["label"].nunique().loc[lambda s: s > 1].index
    )
    if len(conflicts):
        print(f"Note: {len(conflicts)} text(s) appear with conflicting labels "
              f"(kept as-is).")

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.insert(0, "id", range(1, len(df) + 1))

    csv_path = BASE_DIR / "combined_dataset.csv"
    txt_path = BASE_DIR / "combined_dataset.txt"
    df.to_csv(csv_path, index=False)
    df.to_csv(txt_path, sep="\t", index=False)

    print(f"Rows written: {len(df)} (removed {removed} exact duplicates)")
    print(f"Label counts:\n{df['label_name'].value_counts().to_string()}")
    print(f"Per-source counts:\n"
          f"{df.groupby(['source', 'label_name']).size().to_string()}")
    print(f"Saved: {csv_path.name}, {txt_path.name}")


if __name__ == "__main__":
    main()
