"""
ingest_ra_gold.py
-----------------
Parse RA annotation files into data/ra_gold_labels.parquet.

Sources:
  1. ~/Downloads/annotations cheap diverge (1).csv  — 16 pilot review cases (4-tier labels)
  2. ~/Downloads/annotations_disagree.csv            — 7 pilot review cases  (4-tier labels)
  3. Candidates- test.xlsx  → 'Letters'              — egeret gold labels (Hebrew)
  4. Candidates- test.xlsx  → 'BenYehudaProject'     — bypc gold labels (Hebrew)

Output columns:
  doc_id, ra_label_4tier, ra_is_polemic, ra_describes_polemic,
  ra_reference_in_text, ra_notes, ra_comment, source
"""

import os
import re
import pandas as pd
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data"
DOWNLOADS  = Path.home() / "Downloads"

CHEAP_DIVERGE_CSV = DOWNLOADS / "annotations cheap diverge (1).csv"
DISAGREE_CSV      = DOWNLOADS / "annotations_disagree.csv"
EXCEL_FILE        = ROOT / "Candidates- test.xlsx"
EGERET_TSV        = ROOT / "e-geret-batch-export.tsv"
BYPC_CSV          = ROOT / "Ben-Yehuda-Project-polemic-candidates.csv"
OUTPUT_PARQUET    = DATA_DIR / "ra_gold_labels.parquet"


# ── Hebrew label → 4-tier mapping ────────────────────────────────────────────
# is_polemic: כן / לא / לדיון / NaN
# describes_polemic: כן / לא / לדיון / NaN

def derive_4tier(is_polemic, describes_polemic):
    """Map Hebrew RA labels to the 4-tier English scheme."""
    ip = str(is_polemic).strip() if pd.notna(is_polemic) else ""
    dp = str(describes_polemic).strip() if pd.notna(describes_polemic) else ""

    if ip == "כן":
        return "explicit polemic"
    if ip == "לדיון":
        return "implicit polemic"
    if ip == "לא":
        if dp == "כן":
            return "meta-polemic (descriptive)"
        if dp == "לדיון":
            return "uncertain"
        return "non-polemic"
    return "unlabeled"


# ── 1. Load annotation CSVs (already 4-tier) ─────────────────────────────────

def load_annotation_csvs():
    frames = []
    for path, src_name in [
        (CHEAP_DIVERGE_CSV, "cheap_diverge_csv"),
        (DISAGREE_CSV,      "disagree_csv"),
    ]:
        df = pd.read_csv(path, encoding="latin-1")
        df = df.rename(columns={"comment": "ra_comment"})
        df["ra_label_4tier"]        = df["label"]
        df["ra_is_polemic"]         = None
        df["ra_describes_polemic"]  = None
        df["ra_reference_in_text"]  = None
        df["ra_notes"]              = None
        df["source"]                = src_name
        frames.append(df[["doc_id", "ra_label_4tier", "ra_is_polemic",
                           "ra_describes_polemic", "ra_reference_in_text",
                           "ra_notes", "ra_comment", "source"]])
    return pd.concat(frames, ignore_index=True)


# ── 2. Build egeret URL → row-index lookup ────────────────────────────────────

def build_egeret_lookup():
    """Returns dict: (url, letterIndex_int) → doc_id"""
    df = pd.read_csv(EGERET_TSV, sep="\t", encoding="utf-8-sig",
                     usecols=["url", "letterIndex"])
    lookup = {}
    for idx, row in df.iterrows():
        url = str(row["url"]).strip() if pd.notna(row["url"]) else ""
        li  = row["letterIndex"]
        if url and pd.notna(li):
            try:
                lookup[(url, int(li))] = f"egeret_{idx}"
            except (ValueError, TypeError):
                pass
    return lookup


# ── 3. Build bypc URL → row-index lookup ─────────────────────────────────────

def build_bypc_lookup():
    """Returns dict: url → doc_id"""
    df = pd.read_csv(BYPC_CSV, usecols=["link"])
    return {str(row["link"]).strip(): f"bypc_{idx}"
            for idx, row in df.iterrows() if pd.notna(row["link"])}


# ── 4. Load Letters sheet ────────────────────────────────────────────────────

LETTERS_COLS = {
    "link":                                          "link",
    "_ - letterIndex":                               "letter_index",
    "האם פולמוסי?":                                  "ra_is_polemic",
    "האם מתאר פולמוס?":                              "ra_describes_polemic",
    "ההפנייה בטקסט למושא הביקורת (מפורטת ככל שניתן)":"ra_reference_in_text",
    "הערות חופשיות/דיון/הרחבה":                      "ra_notes",
}

def load_letters(egeret_lookup):
    df = pd.read_excel(EXCEL_FILE, sheet_name="Letters")
    df = df.rename(columns={c: v for c, v in LETTERS_COLS.items() if c in df.columns})

    # Drop rows with no RA label
    df = df[df["ra_is_polemic"].notna()].copy()

    # Resolve doc_id
    def resolve(row):
        url = str(row.get("link", "")).strip()
        li  = row.get("letter_index")
        if url and pd.notna(li):
            return egeret_lookup.get((url, int(li)))
        return None

    df["doc_id"] = df.apply(resolve, axis=1)
    df = df[df["doc_id"].notna()].copy()

    df["ra_label_4tier"] = df.apply(
        lambda r: derive_4tier(r["ra_is_polemic"], r["ra_describes_polemic"]), axis=1)
    df["ra_comment"] = None
    df["source"]     = "letters_excel"

    return df[["doc_id", "ra_label_4tier", "ra_is_polemic", "ra_describes_polemic",
               "ra_reference_in_text", "ra_notes", "ra_comment", "source"]]


# ── 5. Load BenYehudaProject sheet ───────────────────────────────────────────

BYPC_COLS = {
    "link":                                          "link",
    "האם פולמוסי?":                                  "ra_is_polemic",
    "האם מתאר פולמוס?":                              "ra_describes_polemic",
    "ההפנייה בטקסט למושא הביקורת (מפורטת ככל שניתן)":"ra_reference_in_text",
    "הערות חופשיות/דיון/הרחבה":                      "ra_notes",
}

def load_bypc(bypc_lookup):
    df = pd.read_excel(EXCEL_FILE, sheet_name="BenYehudaProject")
    df = df.rename(columns={c: v for c, v in BYPC_COLS.items() if c in df.columns})

    # Drop rows with no RA label
    df = df[df["ra_is_polemic"].notna()].copy()

    df["doc_id"] = df["link"].map(
        lambda x: bypc_lookup.get(str(x).strip()) if pd.notna(x) else None)
    df = df[df["doc_id"].notna()].copy()

    df["ra_label_4tier"] = df.apply(
        lambda r: derive_4tier(r["ra_is_polemic"], r["ra_describes_polemic"]), axis=1)
    df["ra_comment"] = None
    df["source"]     = "bypc_excel"

    return df[["doc_id", "ra_label_4tier", "ra_is_polemic", "ra_describes_polemic",
               "ra_reference_in_text", "ra_notes", "ra_comment", "source"]]


# ── 6. Combine & save ─────────────────────────────────────────────────────────

def main():
    print("Loading annotation CSVs …")
    csv_df = load_annotation_csvs()
    print(f"  {len(csv_df)} rows from annotation CSVs")

    print("Building e-geret lookup …")
    egeret_lookup = build_egeret_lookup()
    print("Building bypc lookup …")
    bypc_lookup   = build_bypc_lookup()

    print("Loading Letters sheet …")
    letters_df = load_letters(egeret_lookup)
    print(f"  {len(letters_df)} rows from Letters sheet")

    print("Loading BenYehudaProject sheet …")
    bypc_df = load_bypc(bypc_lookup)
    print(f"  {len(bypc_df)} rows from BenYehudaProject sheet")

    combined = pd.concat([csv_df, letters_df, bypc_df], ignore_index=True)

    # Dedup: annotation CSVs (reviewed pilot) take priority over Excel sheets
    priority = {"cheap_diverge_csv": 0, "disagree_csv": 1,
                "letters_excel": 2, "bypc_excel": 3}
    combined["_priority"] = combined["source"].map(priority)
    combined = (combined
                .sort_values("_priority")
                .drop_duplicates(subset="doc_id", keep="first")
                .drop(columns="_priority")
                .reset_index(drop=True))

    DATA_DIR.mkdir(exist_ok=True)
    combined.to_parquet(OUTPUT_PARQUET, index=False)

    print(f"\nSaved {len(combined)} gold labels → {OUTPUT_PARQUET}")
    print("\nLabel distribution:")
    print(combined["ra_label_4tier"].value_counts().to_string())
    print("\nSource distribution:")
    print(combined["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
