"""
Phase 2: Download sample volumes from diverse CM periodicals,
extract text via pdfplumber, and assess quality vs JPRESS baseline.

Samples chosen to cover:
- Early (1850s) vs late (1890s) periodicals
- Different publishers/locations
- Different types (almanac, literary collection, cultural journal)
"""

import json
import sys
import time
from pathlib import Path

import pdfplumber
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cleaning import compute_quality_score, is_long_enough, normalize_hebrew

DOWNLOAD_DIR = Path("data/compact_memory/raw")
OUTPUT_DIR = Path("data/compact_memory")

# Sample volumes: one early, one mid, one late
SAMPLES = [
    # Kokhve Yitsḥaḳ vol 5 (1850) — early, Wien almanac
    {"cm_id": "4785731", "volume_id": "4786146", "title": "Kokhve Yitsḥaḳ",
     "volume_label": "vol 5 (1850)", "year": 1850},
    # Mim-mizraḥ vol 1 (1893) — late, Berlin/Wien cultural journal
    {"cm_id": "4861829", "volume_id": "6448950", "title": "Mim-mizraḥ ū-mim-maʿarāv",
     "volume_label": "vol 1 (1893)", "year": 1893},
    # Bikkūrē hā-'ittīm vol 1 (1820) — pre-1850 Haskalah, Wien
    {"cm_id": "4782723", "volume_id": "4782725", "title": "Bikkūrē hā-'ittīm",
     "volume_label": "vol 1 (1820)", "year": 1820},
]


def download_pdf(volume_id: str, dest: Path) -> bool:
    """Download a volume PDF from CM. Returns True on success."""
    if dest.exists():
        print(f"  Already downloaded: {dest}")
        return True
    url = f"https://sammlungen.ub.uni-frankfurt.de/download/pdf/{volume_id}"
    print(f"  Downloading {url} ...")
    resp = requests.get(url, timeout=120)
    if resp.status_code != 200:
        print(f"  FAILED: status {resp.status_code}")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    print(f"  Saved {len(resp.content) / 1e6:.1f} MB to {dest}")
    return True


def extract_text(pdf_path: Path) -> list[dict]:
    """Extract text from each page of a PDF. Returns list of {page, text}."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append({"page": i + 1, "text": text})
    return pages


def assess_volume(sample: dict, pages: list[dict]) -> dict:
    """Compute quality metrics for a full volume."""
    full_text = "\n".join(p["text"] for p in pages)
    quality = compute_quality_score(full_text)

    # Per-page stats
    page_lengths = [len(p["text"]) for p in pages]
    nonempty_pages = sum(1 for l in page_lengths if l > 50)
    heb_pages = 0
    for p in pages:
        q = compute_quality_score(p["text"])
        if q["hebrew_ratio"] > 0.3:
            heb_pages += 1

    word_count = len(full_text.split())
    heb_word_count = len(__import__("re").findall(r"[\u05D0-\u05EA]{2,}", full_text))

    return {
        "title": sample["title"],
        "volume_label": sample["volume_label"],
        "year": sample["year"],
        "total_pages": len(pages),
        "nonempty_pages": nonempty_pages,
        "hebrew_majority_pages": heb_pages,
        "total_chars": len(full_text),
        "total_words": word_count,
        "hebrew_words": heb_word_count,
        "hebrew_ratio": quality["hebrew_ratio"],
        "avg_word_len": quality["avg_word_len"],
        "meets_min_length": is_long_enough(full_text),
    }


def print_samples(pages: list[dict], sample: dict, n=3):
    """Print sample Hebrew text from a volume."""
    print(f"\n  Sample Hebrew text from {sample['title']} {sample['volume_label']}:")
    shown = 0
    for p in pages:
        q = compute_quality_score(p["text"])
        if q["hebrew_ratio"] > 0.4 and len(p["text"]) > 200:
            print(f"  --- Page {p['page']} (heb_ratio={q['hebrew_ratio']:.2f}) ---")
            print(f"  {p['text'][:300]}")
            shown += 1
            if shown >= n:
                break
    if shown == 0:
        print("  (no pages with >40% Hebrew and >200 chars found)")


def main():
    results = []

    for sample in SAMPLES:
        print(f"\n{'='*60}")
        print(f"{sample['title']} — {sample['volume_label']}")
        print(f"{'='*60}")

        pdf_path = DOWNLOAD_DIR / sample["cm_id"] / f"{sample['volume_id']}.pdf"
        if not download_pdf(sample["volume_id"], pdf_path):
            continue

        print("  Extracting text...")
        pages = extract_text(pdf_path)
        print(f"  Extracted {len(pages)} pages")

        assessment = assess_volume(sample, pages)
        results.append(assessment)

        print(f"\n  Quality metrics:")
        print(f"    Total pages:          {assessment['total_pages']}")
        print(f"    Non-empty pages:      {assessment['nonempty_pages']}")
        print(f"    Hebrew-majority pages: {assessment['hebrew_majority_pages']}")
        print(f"    Total words:          {assessment['total_words']}")
        print(f"    Hebrew words:         {assessment['hebrew_words']}")
        print(f"    Hebrew ratio:         {assessment['hebrew_ratio']:.3f}")
        print(f"    Avg word length:      {assessment['avg_word_len']:.2f}")
        print(f"    Meets 200-word min:   {assessment['meets_min_length']}")

        print_samples(pages, sample)

        time.sleep(2)

    # Save results
    df = pd.DataFrame(results)
    out_path = OUTPUT_DIR / "quality_assessment.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\n\nQuality assessment saved to {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
