"""Validate the length-artifact hypothesis on the 22 BenYehuda RA-gold cases.

Chunks each text and re-classifies with the saved B.4a model, aggregates
per-document with max(prob_polemic), and compares to RA gold.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "data" / "models" / "best_polemic_classifier"
MAX_TOKENS = 512
STRIDE = 0  # 0 = no overlap; bump to 128 if validation suggests boundary misses


def chunk_text(text: str, tokenizer, max_tokens: int = MAX_TOKENS, stride: int = STRIDE):
    """Paragraph-aware chunking with sliding-window fallback for long paragraphs."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    buf_ids = []
    for para in paragraphs:
        para_ids = tokenizer.encode(para, add_special_tokens=False)
        # Long paragraph — slide it
        if len(para_ids) > max_tokens:
            if buf_ids:
                chunks.append(buf_ids); buf_ids = []
            for start in range(0, len(para_ids), max_tokens - stride if stride else max_tokens):
                chunks.append(para_ids[start:start + max_tokens])
            continue
        # Accumulate paragraphs until we'd exceed budget
        if len(buf_ids) + len(para_ids) > max_tokens:
            chunks.append(buf_ids)
            buf_ids = para_ids
        else:
            buf_ids = buf_ids + para_ids
    if buf_ids:
        chunks.append(buf_ids)
    return chunks


def main():
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH)).to(device).eval()

    # 22 BenYehuda RA-gold cases (excluding uncertain)
    ra = pd.read_parquet(ROOT / "data" / "ra_gold_labels.parquet")
    ra = ra[(ra["source"] == "bypc_excel") & (ra["ra_label_4tier"] != "uncertain")].copy()
    ra["ra_binary"] = ra["ra_label_4tier"] != "non-polemic"
    print(f"BenYehuda RA-gold non-uncertain cases: {len(ra)}")

    corpus = pd.read_parquet(ROOT / "corpus.parquet", columns=["doc_id", "text"])
    df = ra.merge(corpus, on="doc_id", how="inner")
    print(f"Matched to corpus.parquet: {len(df)}")

    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id

    rows = []
    for _, r in df.iterrows():
        text = str(r["text"])
        chunk_ids_list = chunk_text(text, tokenizer)
        max_polemic = 0.0
        n_chunks = 0
        chunk_probs = []
        for chunk_ids in chunk_ids_list:
            input_ids = [cls_token_id] + chunk_ids[:MAX_TOKENS - 2] + [sep_token_id]
            t = torch.tensor([input_ids]).to(device)
            mask = torch.ones_like(t)
            with torch.no_grad():
                logits = model(input_ids=t, attention_mask=mask).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            prob_polemic = 1.0 - probs[0]  # 1 - prob(non-polemic)
            chunk_probs.append(prob_polemic)
            max_polemic = max(max_polemic, prob_polemic)
            n_chunks += 1
        rows.append({
            "doc_id":       r["doc_id"],
            "ra_label":     r["ra_label_4tier"],
            "ra_binary":    bool(r["ra_binary"]),
            "n_chunks":     n_chunks,
            "n_tokens":     sum(len(c) for c in chunk_ids_list),
            "max_polemic":  max_polemic,
            "mean_polemic": float(np.mean(chunk_probs)) if chunk_probs else float("nan"),
            "is_polemic_chunked": bool(max_polemic > 0.5),
        })
        print(f"  {r['doc_id']:18s}  chunks={n_chunks:3d}  tokens={sum(len(c) for c in chunk_ids_list):6d}  "
              f"max_pol={max_polemic:.3f}  mean_pol={float(np.mean(chunk_probs)):.3f}  ra={r['ra_label_4tier']}")

    out = pd.DataFrame(rows)
    out.to_parquet(ROOT / "data" / "bypc_chunk_validation.parquet", index=False)

    # Compare to baseline single-pass predictions for these same 22 docs
    preds = pd.read_parquet(ROOT / "data" / "full_corpus_predictions.parquet")
    preds["is_polemic_baseline"] = preds["prob_non_polemic"] < 0.5
    cmp = out.merge(preds[["doc_id", "is_polemic_baseline", "prob_non_polemic"]], on="doc_id")

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    k_base    = cohen_kappa_score(cmp["ra_binary"], cmp["is_polemic_baseline"])
    k_chunked = cohen_kappa_score(cmp["ra_binary"], cmp["is_polemic_chunked"])
    agree_b   = (cmp["ra_binary"] == cmp["is_polemic_baseline"]).mean()
    agree_c   = (cmp["ra_binary"] == cmp["is_polemic_chunked"]).mean()
    pol_recall_b = ((cmp["ra_binary"]) & (cmp["is_polemic_baseline"])).sum() / max(1, cmp["ra_binary"].sum())
    pol_recall_c = ((cmp["ra_binary"]) & (cmp["is_polemic_chunked"])).sum() / max(1, cmp["ra_binary"].sum())

    print(f"  baseline (single-pass 512-token truncation):")
    print(f"    agree={agree_b:.1%}  kappa={k_base:+.3f}  polemic-recall={pol_recall_b:.1%}")
    print(f"  chunked (max prob across chunks):")
    print(f"    agree={agree_c:.1%}  kappa={k_chunked:+.3f}  polemic-recall={pol_recall_c:.1%}")
    print()
    print("Confusion matrix — chunked (rows=RA, cols=model):")
    cm = confusion_matrix(cmp["ra_binary"], cmp["is_polemic_chunked"])
    print(f"                 model:non  model:pol")
    print(f"    RA:non-pol   {cm[0,0]:8d}   {cm[0,1]:8d}")
    print(f"    RA:polemic   {cm[1,0]:8d}   {cm[1,1]:8d}")

    print()
    print("Verdict:")
    if k_chunked >= 0.3 and k_chunked - k_base > 0.2:
        print("  ✓ Length-artifact hypothesis CONFIRMED. Invest in full BenYehuda chunking.")
    elif k_chunked > k_base + 0.1:
        print("  ~ Moderate improvement. Worth chunking but may not be sufficient alone.")
    else:
        print("  ✗ No meaningful improvement. Length is not the dominant problem.")


if __name__ == "__main__":
    main()
