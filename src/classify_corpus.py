"""
classify_corpus.py — Apply the saved B.4a polemic classifier to the full corpus.

Loads the fine-tuned model from data/models/best_polemic_classifier/ and
writes data/full_corpus_predictions.parquet with predicted label + class probabilities.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parent.parent

LABEL_ORDER = [
    "non-polemic",
    "implicit polemic",
    "explicit polemic",
    "meta-polemic (descriptive)",
]


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding=True,
            max_length=max_length, return_tensors="pt",
        )

    def __len__(self):
        return self.encodings["input_ids"].size(0)

    def __getitem__(self, i):
        return {
            "input_ids":      self.encodings["input_ids"][i],
            "attention_mask": self.encodings["attention_mask"][i],
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="data/models/best_polemic_classifier")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--chunk-size", type=int, default=512,
                   help="Rows tokenized at a time (memory control)")
    p.add_argument("--output", default="data/full_corpus_predictions.parquet")
    args = p.parse_args()

    model_path = ROOT / args.model_path
    if not (model_path / "config.json").exists():
        sys.exit(f"ERROR: no saved model at {model_path}")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path)).to(device)
    model.eval()

    corpus = pd.read_parquet(ROOT / "corpus.parquet", columns=["doc_id", "text"])
    corpus["text"] = corpus["text"].astype(str)
    print(f"Corpus: {len(corpus)} texts")

    all_preds = []
    all_probs = []
    n = len(corpus)
    for start in range(0, n, args.chunk_size):
        end = min(start + args.chunk_size, n)
        chunk = corpus.iloc[start:end]
        ds = TextDataset(chunk["text"].tolist(), tokenizer)
        loader = DataLoader(ds, batch_size=args.batch_size)
        with torch.no_grad():
            for batch in loader:
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                ).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_preds.extend(probs.argmax(axis=-1).tolist())
                all_probs.append(probs)
        if (start // args.chunk_size) % 4 == 0 or end == n:
            print(f"  {end}/{n} ({end/n:.1%}) done")

    probs_arr = np.concatenate(all_probs, axis=0)
    out = corpus.copy()
    out["predicted_label"] = [LABEL_ORDER[i] for i in all_preds]
    out["confidence"] = probs_arr.max(axis=1)
    for i, name in enumerate(LABEL_ORDER):
        key = name.split()[0].replace("-", "_")  # non / implicit / explicit / meta
        out[f"prob_{key}"] = probs_arr[:, i]

    output_path = ROOT / args.output
    out.drop(columns=["text"]).to_parquet(output_path, index=False)
    print(f"\nSaved {len(out)} predictions to {output_path}")
    print("\nLabel distribution:")
    counts = out["predicted_label"].value_counts()
    for label in LABEL_ORDER:
        c = int(counts.get(label, 0))
        print(f"  {label:35s} {c:6d} ({c/len(out):.1%})")


if __name__ == "__main__":
    main()
