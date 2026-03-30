"""
backfill_metadata.py - Enrich corpus.parquet with metadata from source files

Backfills:
- Press: newspaper-code, intertextual reference, headline
- E-geret: authorString, Recipient
- Polemic candidates: author_string (already partially there)

The corpus doc_ids map to source file row indices.
"""
import os
import csv
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

PRESS_PATH = "MGD-LBN-MLZ-HZF-HZTfull2021-08-14-(1)-tsv.csv"
EGERET_PATH = "e-geret-batch-export.tsv"
BYPC_PATH = "Ben-Yehuda-Project-polemic-candidates.csv"


def main():
    csv.field_size_limit(2**30)

    print("Loading corpus...")
    corpus = pd.read_parquet("corpus.parquet")
    print(f"  {len(corpus)} texts")

    # Extract numeric index from doc_id (skip compact_memory — no source file to backfill from)
    corpus["_src_idx"] = pd.to_numeric(
        corpus["doc_id"].str.replace(r"^[a-z]+_", "", regex=True),
        errors="coerce"
    )

    # --- Press ---
    print("\nBackfilling press metadata...")
    press_meta = pd.read_csv(PRESS_PATH, usecols=["newspaper-code", "intertextual reference", "headline"])
    press_meta["_src_idx"] = press_meta.index

    press_mask = corpus["source"] == "press"
    press_corpus = corpus.loc[press_mask].merge(
        press_meta, on="_src_idx", how="left", suffixes=("", "_new")
    )

    corpus.loc[press_mask, "newspaper"] = press_corpus["newspaper-code"].values
    corpus.loc[press_mask, "intertextual_reference"] = press_corpus["intertextual reference"].values
    if "headline" not in corpus.columns:
        corpus["headline"] = None
    corpus.loc[press_mask, "headline"] = press_corpus["headline"].values

    print(f"  newspaper: {corpus.loc[press_mask, 'newspaper'].notna().sum()} filled")
    print(f"  intertextual_reference: {corpus.loc[press_mask, 'intertextual_reference'].notna().sum()} filled")
    print(f"  headline: {corpus.loc[press_mask, 'headline'].notna().sum()} filled")

    # --- E-geret ---
    print("\nBackfilling e-geret metadata...")
    egeret_meta = pd.read_csv(EGERET_PATH, sep="\t", encoding="utf-8-sig",
                               usecols=["authorString", "Recipient"])
    egeret_meta["_src_idx"] = egeret_meta.index

    egeret_mask = corpus["source"] == "egeret"
    egeret_corpus = corpus.loc[egeret_mask].merge(
        egeret_meta, on="_src_idx", how="left", suffixes=("", "_new")
    )

    corpus.loc[egeret_mask, "author"] = egeret_corpus["authorString"].values
    if "recipient" not in corpus.columns:
        corpus["recipient"] = None
    corpus.loc[egeret_mask, "recipient"] = egeret_corpus["Recipient"].values

    print(f"  author: {corpus.loc[egeret_mask, 'author'].notna().sum()} filled")
    print(f"  recipient: {corpus.loc[egeret_mask, 'recipient'].notna().sum()} filled")

    # --- Polemic candidates ---
    print("\nBackfilling polemic candidate metadata...")
    bypc_meta = pd.read_csv(BYPC_PATH, usecols=["author_string"])
    bypc_meta["_src_idx"] = bypc_meta.index

    bypc_mask = corpus["source"] == "polemic_candidates"
    bypc_corpus = corpus.loc[bypc_mask].merge(
        bypc_meta, on="_src_idx", how="left", suffixes=("", "_new")
    )

    corpus.loc[bypc_mask, "author"] = bypc_corpus["author_string"].values

    print(f"  author: {corpus.loc[bypc_mask, 'author'].notna().sum()} filled")

    # --- Add intertextual_reference column for non-press ---
    if "intertextual_reference" not in corpus.columns:
        corpus["intertextual_reference"] = None

    # --- Cleanup ---
    corpus.drop(columns=["_src_idx"], inplace=True)

    # Summary
    print("\n" + "=" * 50)
    print("METADATA BACKFILL SUMMARY")
    print("=" * 50)
    print(f"Total texts: {len(corpus)}")
    for col in ["author", "newspaper", "recipient", "headline", "intertextual_reference"]:
        if col in corpus.columns:
            n = corpus[col].notna().sum()
            print(f"  {col}: {n} ({n/len(corpus):.1%})")

    print(f"\nAuthor coverage by source:")
    for src in corpus["source"].unique():
        sub = corpus[corpus["source"] == src]
        print(f"  {src}: {sub['author'].notna().sum()}/{len(sub)} ({sub['author'].notna().mean():.1%})")

    print(f"\nNewspaper distribution:")
    print(corpus["newspaper"].value_counts().to_string())

    print(f"\nTop authors:")
    print(corpus["author"].value_counts().head(15).to_string())

    # Save
    corpus.to_parquet("corpus.parquet", index=False)
    print(f"\nSaved enriched corpus.parquet ({len(corpus)} texts, {len(corpus.columns)} columns)")
    print(f"Columns: {list(corpus.columns)}")


if __name__ == "__main__":
    main()
