"""
Phase 4: Extract, clean, and filter segmented CM articles.

Takes the raw segmented articles from segment.py, applies the same
cleaning pipeline used for the rest of the Polemicon corpus, filters
by quality and length, and outputs cm_articles.parquet.

All rule-based, no AI tokens.
"""

import json
import sys
from pathlib import Path

import pandas as pd

# Add project root so we can import src.cleaning
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cleaning import normalize_hebrew, compute_quality_score, is_long_enough
from src.compact_memory.segment import segment_all

OUTPUT_PATH = Path("data/compact_memory/extracted/cm_articles.parquet")


def clean_cm_text(text: str) -> str:
    """Clean CM article text: normalize Hebrew, remove OCR noise."""
    text = normalize_hebrew(text)
    # Remove common OCR artifacts: sequences of punctuation/digits/symbols
    # that aren't meaningful Hebrew text (but keep Hebrew + basic punctuation)
    import re
    # Remove lines that are mostly non-Hebrew (< 30% Hebrew chars)
    cleaned_lines = []
    for line in text.split("\n"):
        if not line.strip():
            cleaned_lines.append("")
            continue
        heb = len(re.findall(r"[\u05D0-\u05EA]", line))
        total = len(line.strip())
        if total > 0 and (heb / total >= 0.3 or total < 10):
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def extract_all():
    """Run the full extraction pipeline: segment → clean → filter → save."""
    print("Phase 4: Extracting and cleaning CM articles")
    print("=" * 60)

    # Step 1: Segment all volumes
    print("\nStep 1: Segmenting volumes...")
    raw_articles = segment_all()

    if not raw_articles:
        print("No articles found. Exiting.")
        return

    print(f"\nStep 2: Cleaning {len(raw_articles)} articles...")

    records = []
    for i, article in enumerate(raw_articles):
        text = article.get("text", "")
        if not text:
            continue

        # Clean text
        cleaned = clean_cm_text(text)

        # Quality check
        quality = compute_quality_score(cleaned)
        if quality["hebrew_ratio"] < 0.3:
            continue
        if not is_long_enough(cleaned, min_words=200):
            continue

        # Build record matching corpus schema
        records.append({
            "doc_id": f"cm_{article['periodical'].lower().replace('-', '_').replace(' ', '_')}_{article['volume_id']}_{article['printed_start_page']}",
            "source": "compact_memory",
            "newspaper": article["periodical"],
            "text": cleaned,
            "date": article.get("year", ""),
            "year": int(article["year"]) if article.get("year", "").isdigit() else None,
            "author": None,  # ToC entries mix author/title; unreliable to split
            "title": article["entry"],
            "genre": None,
            "quality_score": quality,
            "in_overlap": (1850 <= int(article["year"]) <= 1900)
                          if article.get("year", "").isdigit() else False,
            "volume_id": article["volume_id"],
            "cm_id": article["cm_id"],
            "printed_start_page": article["printed_start_page"],
            "printed_end_page": article["printed_end_page"],
            "num_pages": article["num_pages"],
            "hebrew_ratio": quality["hebrew_ratio"],
            "avg_word_len": quality["avg_word_len"],
            "char_count": len(cleaned),
            "segmentation_method": "toc",
        })

    df = pd.DataFrame(records)

    print(f"\nStep 3: Results after filtering")
    print(f"  Total articles passing filters: {len(df)}")
    print(f"  Dropped (low hebrew_ratio or too short): {len(raw_articles) - len(df)}")
    print(f"\n  By periodical:")
    for name, group in df.groupby("newspaper"):
        print(f"    {name}: {len(group)} articles, "
              f"median {group['char_count'].median():.0f} chars, "
              f"mean hebrew_ratio {group['hebrew_ratio'].mean():.2f}")
    print(f"\n  Year range: {df['year'].min()}-{df['year'].max()}")
    print(f"  All in overlap window: {df['in_overlap'].sum()}/{len(df)}")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")

    return df


if __name__ == "__main__":
    extract_all()
