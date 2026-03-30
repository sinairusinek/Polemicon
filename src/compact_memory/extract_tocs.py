"""Extract and display ToC pages from Pardes, Ha-Eshkol, Aḥiasaf volumes."""

import json
import re
from pathlib import Path

import pdfplumber

RAW_DIR = Path("data/compact_memory/raw")

# Periodicals with ToCs
TOC_PERIODICALS = {
    "8003959": "Pardes",
    "3773345": "Ha-Eshkol",
    "10719318": "Aḥiasaf",
}

with open("data/compact_memory/volume_map.json") as f:
    volume_map = json.load(f)


def find_toc_pages(pdf_path: Path, max_pages=30):
    """Find pages that contain ToC markers, searching early pages of the volume."""
    toc_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i in range(min(max_pages, len(pdf.pages))):
            text = pdf.pages[i].extract_text() or ""
            # Look for ToC markers
            if re.search(r"ןכות|ןכת", text) or re.search(r"דומע", text[:100]):
                toc_pages.append((i, text))
    return toc_pages


for cm_id, periodical_name in TOC_PERIODICALS.items():
    info = volume_map.get(cm_id)
    if not info:
        continue

    print(f"\n{'#'*70}")
    print(f"# {periodical_name} ({info['hebrew']})")
    print(f"{'#'*70}")

    for vol in info["volumes"]:
        vid = vol["id"]
        pdf_path = RAW_DIR / cm_id / f"{vid}.pdf"
        if not pdf_path.exists():
            print(f"\n  {vol['caption']} ({vol['date']}): PDF not found")
            continue

        toc_pages = find_toc_pages(pdf_path)
        print(f"\n  === {vol['caption']} ({vol['date']}) — {len(toc_pages)} ToC pages ===")

        for page_idx, text in toc_pages:
            print(f"\n  --- PDF page {page_idx + 1} ---")
            # Print each line with line numbers for analysis
            for j, line in enumerate(text.split("\n")):
                print(f"    {j:3d}| {line}")
