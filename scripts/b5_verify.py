"""B.5 verification — three plan-stated sanity checks on the fine-tuned classifier."""
import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent


def main():
    preds = pd.read_parquet(ROOT / "data" / "full_corpus_predictions.parquet")
    corpus = pd.read_parquet(
        ROOT / "corpus.parquet",
        columns=["doc_id", "source", "intertextual_reference", "אזכור מכ״ע"],
    )
    df = preds.merge(corpus, on="doc_id", how="left")
    df["is_polemic"] = df["prob_non_polemic"] < 0.5

    print("=" * 60)
    print("B.5 Check 1 — Press polemic prevalence (target 10–30%)")
    print("=" * 60)
    for src, sub in df.groupby("source"):
        rate = sub["is_polemic"].mean()
        flag = "✓" if (src != "press" or 0.10 <= rate <= 0.30) else "✗"
        print(f"  {flag} {src:22s} n={len(sub):6,d}  polemic={int(sub['is_polemic'].sum()):5,d} ({rate:.1%})")

    print()
    print("=" * 60)
    print("B.5 Check 2 — Cross-reference items flagged polemic")
    print("=" * 60)
    manual = df[df["אזכור מכ״ע"].notna()]
    mech   = df[df["intertextual_reference"].notna()]
    base   = df["is_polemic"].mean()
    print(f"  Manual cross-refs (אזכור מכ״ע):    n={len(manual):4d}  polemic={int(manual['is_polemic'].sum()):4d} ({manual['is_polemic'].mean():.1%})")
    print(f"  Mechanical intertextual_reference:  n={len(mech):4d}  polemic={int(mech['is_polemic'].sum()):4d} ({mech['is_polemic'].mean():.1%})")
    print(f"  Corpus baseline:                              {base:.1%}")
    print(f"  Manual lift:     {manual['is_polemic'].mean() / base:.2f}×")
    print(f"  Mechanical lift: {mech['is_polemic'].mean() / base:.2f}×")

    print()
    print("=" * 60)
    print("B.5 Check 3 — RA gold vs model (binary, kappa target > 0.6)")
    print("=" * 60)
    ra = pd.read_parquet(ROOT / "data" / "ra_gold_labels.parquet")
    # Map RA 4-tier to binary; drop uncertain.
    ra = ra[ra["ra_label_4tier"] != "uncertain"].copy()
    ra["ra_binary"] = ra["ra_label_4tier"] != "non-polemic"
    m = ra.merge(df[["doc_id", "is_polemic", "predicted_label"]], on="doc_id", how="inner")
    print(f"  RA gold available (non-uncertain): {len(ra)}")
    print(f"  Matched to predictions:            {len(m)}")
    if len(m) > 0:
        kappa = cohen_kappa_score(m["ra_binary"], m["is_polemic"])
        agree = (m["ra_binary"] == m["is_polemic"]).mean()
        cm = confusion_matrix(m["ra_binary"], m["is_polemic"])
        print(f"  Agreement: {agree:.1%}")
        print(f"  Cohen's kappa: {kappa:.3f}  {'✓' if kappa > 0.6 else '✗ (below 0.6 target)'}")
        print(f"  Confusion matrix (rows=RA, cols=model):")
        print(f"                 model:non  model:pol")
        print(f"    RA:non-pol   {cm[0,0]:8d}   {cm[0,1]:8d}")
        print(f"    RA:polemic   {cm[1,0]:8d}   {cm[1,1]:8d}")

        # 4-class confusion (informational, not pass/fail)
        print()
        print("  4-class breakdown (RA gold vs model predicted_label):")
        ct = pd.crosstab(m["ra_label_4tier"], m["predicted_label"], margins=True)
        print(ct.to_string())


if __name__ == "__main__":
    main()
