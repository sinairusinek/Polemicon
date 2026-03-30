"""
Phase 2b: Full quality assessment for the 8 text-available periodicals.
Uses already-downloaded PDFs where available, downloads others.
Computes quality scores for each volume and summarizes per periodical.
"""

import json
import re
import sys
import time
from io import BytesIO
from pathlib import Path

import pdfplumber
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cleaning import compute_quality_score, is_long_enough

RAW_DIR = Path("data/compact_memory/raw")
OUTPUT_DIR = Path("data/compact_memory")

# Periodicals with text (from probe)
TEXT_AVAILABLE = {
    "4785731", "9582285", "3769475", "8003959",
    "10719318", "4789469", "3773345", "9582265",
}

# Image-only (for the report)
IMAGE_ONLY = {"4861829", "4782723"}


def get_or_download_pdf(cm_id: str, volume_id: str) -> bytes:
    """Get PDF content, using cache if available."""
    cached = RAW_DIR / cm_id / f"{volume_id}.pdf"
    if cached.exists():
        return cached.read_bytes()
    url = f"https://sammlungen.ub.uni-frankfurt.de/download/pdf/{volume_id}"
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(resp.content)
    return resp.content


def extract_and_assess(pdf_content: bytes) -> dict:
    """Extract text from PDF and compute quality metrics."""
    with pdfplumber.open(BytesIO(pdf_content)) as pdf:
        total_pages = len(pdf.pages)
        all_text = []
        heb_pages = 0
        nonempty_pages = 0

        for page in pdf.pages:
            text = page.extract_text() or ""
            all_text.append(text)
            if len(text) > 50:
                nonempty_pages += 1
            q = compute_quality_score(text)
            if q["hebrew_ratio"] > 0.3:
                heb_pages += 1

    full_text = "\n".join(all_text)
    quality = compute_quality_score(full_text)
    heb_words = len(re.findall(r"[\u05D0-\u05EA]{2,}", full_text))

    return {
        "total_pages": total_pages,
        "nonempty_pages": nonempty_pages,
        "hebrew_majority_pages": heb_pages,
        "total_chars": len(full_text),
        "total_words": len(full_text.split()),
        "hebrew_words": heb_words,
        "hebrew_ratio": quality["hebrew_ratio"],
        "avg_word_len": quality["avg_word_len"],
        "meets_min_length": is_long_enough(full_text),
    }


def main():
    with open("data/compact_memory/volume_map.json") as f:
        volume_map = json.load(f)

    all_results = []

    for cm_id, info in volume_map.items():
        title = info["title"]
        hebrew = info.get("hebrew", "")

        if cm_id in IMAGE_ONLY:
            print(f"\n{title}: IMAGE-ONLY (skipping)")
            continue

        if cm_id not in TEXT_AVAILABLE:
            continue

        volumes = info["volumes"]
        print(f"\n{'='*60}")
        print(f"{title} ({hebrew}) — {len(volumes)} volumes")
        print(f"{'='*60}")

        for vol in volumes:
            vid = vol["id"]
            label = f"{vol['caption']} ({vol['date']})"
            print(f"  {label} (ID={vid})...", end=" ", flush=True)

            try:
                content = get_or_download_pdf(cm_id, vid)
                metrics = extract_and_assess(content)
                metrics["cm_id"] = cm_id
                metrics["title"] = title
                metrics["hebrew"] = hebrew
                metrics["volume_id"] = vid
                metrics["volume_label"] = label
                metrics["year"] = vol["date"]
                all_results.append(metrics)

                print(f"pages={metrics['total_pages']}, "
                      f"heb_ratio={metrics['hebrew_ratio']:.3f}, "
                      f"heb_words={metrics['hebrew_words']}")

            except Exception as e:
                print(f"ERROR: {e}")
                all_results.append({
                    "cm_id": cm_id, "title": title, "hebrew": hebrew,
                    "volume_id": vid, "volume_label": label, "year": vol["date"],
                    "error": str(e),
                })

            time.sleep(1)

    # Save
    df = pd.DataFrame(all_results)
    out_path = OUTPUT_DIR / "quality_assessment.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\n\nSaved {len(df)} volume assessments to {out_path}")

    # Summary per periodical
    print("\n=== SUMMARY BY PERIODICAL ===")
    valid = df[df["hebrew_ratio"].notna()].copy()
    summary = valid.groupby("title").agg(
        volumes=("volume_id", "count"),
        avg_heb_ratio=("hebrew_ratio", "mean"),
        total_heb_words=("hebrew_words", "sum"),
        avg_pages=("total_pages", "mean"),
        all_meet_min=("meets_min_length", "all"),
    ).sort_values("avg_heb_ratio", ascending=False)
    print(summary.to_string())

    # Flag periodicals with low Hebrew ratio
    print("\n=== POTENTIAL ISSUES ===")
    for _, row in summary.iterrows():
        if row["avg_heb_ratio"] < 0.3:
            print(f"  LOW HEBREW: {row.name} — avg ratio {row['avg_heb_ratio']:.3f}")


if __name__ == "__main__":
    main()
