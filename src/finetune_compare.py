"""
finetune_compare.py — B.4a model selection

Compares dicta-il/dictabert vs avichr/heBERT for 4-tier polemic classification.
Uses calibration_v2.parquet labels + corpus.parquet texts.

Usage:
    python src/finetune_compare.py                  # compare both models
    python src/finetune_compare.py --model dictabert # run one model only
    python src/finetune_compare.py --save            # save best model after comparison
    python src/finetune_compare.py --epochs 5        # override epoch count
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "dictabert": "dicta-il/dictabert",
    "hebert":    "avichr/heBERT",
}

LABEL_ORDER = [
    "non-polemic",
    "implicit polemic",
    "explicit polemic",
    "meta-polemic (descriptive)",
]
LABEL2ID = {l: i for i, l in enumerate(LABEL_ORDER)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    cal_path = ROOT / "data" / "calibration_v2.parquet"
    if not cal_path.exists():
        sys.exit("ERROR: calibration_v2.parquet not found. Run the calibration first.")

    cal = pd.read_parquet(cal_path)
    cal = cal[cal["polemic_label"].notna() & cal["polemic_label"].isin(LABEL_ORDER)].copy()
    print(f"Calibration labels loaded: {len(cal)}")

    corpus = pd.read_parquet(
        ROOT / "corpus.parquet",
        columns=["doc_id", "text"],
    )
    df = cal[["doc_id", "polemic_label"]].merge(corpus, on="doc_id", how="inner")
    missing = len(cal) - len(df)
    if missing:
        print(f"  Warning: {missing} calibration rows not found in corpus, skipped.")

    df["label_id"] = df["polemic_label"].map(LABEL2ID)
    print("Label distribution:")
    for label in LABEL_ORDER:
        n = (df["polemic_label"] == label).sum()
        print(f"  {label:35s} {n:4d} ({n/len(df):.1%})")
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    train, test = train_test_split(
        df, test_size=test_size, stratify=df["label_id"], random_state=seed
    )
    print(f"\nSplit: {len(train)} train / {len(test)} test (stratified 80/20)")
    return train.reset_index(drop=True), test.reset_index(drop=True)


# ── Dataset ───────────────────────────────────────────────────────────────────

class PoemicDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ── Training ──────────────────────────────────────────────────────────────────

def train_and_eval(
    model_key: str,
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    epochs:   int = 3,
    batch_size: int = 8,
    lr: float = 2e-5,
    save_path: Path = None,
) -> dict:
    model_id = MODELS[model_key]
    print(f"\n{'='*60}")
    print(f"Model: {model_id}")
    print(f"{'='*60}")

    print("  Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=len(LABEL_ORDER),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,  # replaces any existing classification head
    )

    device = torch.device("cpu")
    model = model.to(device)

    # Class weights to handle label imbalance
    class_weights = compute_class_weight(
        "balanced",
        classes=np.array(range(len(LABEL_ORDER))),
        y=train_df["label_id"].values,
    )
    weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    print("  Tokenizing...")
    train_dataset = PoemicDataset(train_df["text"], train_df["label_id"], tokenizer)
    test_dataset  = PoemicDataset(test_df["text"],  test_df["label_id"],  tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights_tensor)

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            loss = loss_fn(outputs.logits, batch["labels"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch}/{epochs} — train loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    report = classification_report(
        all_labels, all_preds,
        target_names=LABEL_ORDER,
        digits=3,
    )
    print(f"\n  Test results ({model_key}):")
    print(report)
    print(f"  Macro F1: {macro_f1:.3f}")

    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"  Saved to {save_path}")

    return {
        "model_key":  model_key,
        "model_id":   model_id,
        "macro_f1":   macro_f1,
        "report":     report,
        "per_class":  f1_score(all_labels, all_preds, average=None).tolist(),
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="B.4a: compare Hebrew models for polemic classification")
    parser.add_argument("--model", choices=list(MODELS.keys()),
                        help="Run one model only (default: both)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size (default: 8; reduce to 4 if OOM on CPU)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data for test set (default: 0.2)")
    parser.add_argument("--save", action="store_true",
                        help="Save best model to data/models/best_polemic_classifier/")
    args = parser.parse_args()

    df = load_data()
    train_df, test_df = split_data(df, test_size=args.test_size)

    models_to_run = [args.model] if args.model else list(MODELS.keys())
    results = []

    for model_key in models_to_run:
        save_path = None
        result = train_and_eval(
            model_key, train_df, test_df,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_path=save_path,
        )
        results.append(result)

    # Comparison summary
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"{'Model':<12} {'Macro F1':>10}  Per-class F1 (non / impl / expl / meta)")
        print("-" * 60)
        for r in results:
            pc = [f"{v:.3f}" for v in r["per_class"]]
            print(f"{r['model_key']:<12} {r['macro_f1']:>10.3f}  {' / '.join(pc)}")

        best = max(results, key=lambda r: r["macro_f1"])
        print(f"\nRecommended model: {best['model_key']} ({best['model_id']}) — Macro F1 {best['macro_f1']:.3f}")

        if args.save:
            print(f"\nSaving best model ({best['model_key']})...")
            save_path = ROOT / "data" / "models" / "best_polemic_classifier"
            train_and_eval(
                best["model_key"], train_df, test_df,
                epochs=args.epochs,
                batch_size=args.batch_size,
                save_path=save_path,
            )

    # Save results JSON for the plan
    results_path = ROOT / "data" / "b4a_model_comparison.json"
    with open(results_path, "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "report"} for r in results], f, indent=2)
    print(f"\nResults saved to {results_path}")
    print("\nNext step: update polemiconPlan.md B.4a section with chosen model and F1.")


if __name__ == "__main__":
    main()
