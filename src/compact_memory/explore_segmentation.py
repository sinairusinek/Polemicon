"""
Explore segmentation signals in CM volume text.
Look at: line length distribution, blank lines, short lines (potential headlines),
language switches, page breaks, and structural patterns.
"""

import re
import sys
from io import BytesIO
from pathlib import Path

import pdfplumber

RAW_DIR = Path("data/compact_memory/raw")

# Pick diverse samples: one almanac, one literary collection, one academic journal
SAMPLES = [
    ("4785731", "4786740", "Kokhve Yitsḥaḳ vol 9 (1854)"),
    ("8003959", "8003960", "Pardes vol 1 (1892)"),
    ("3773345", "5103602", "Ha-Eshkol vol 1 (1898)"),
    ("10719318", "10749702", "Aḥiasaf vol 1 (1893)"),
]


def analyze_page(page_num, text):
    """Analyze structural signals on a single page."""
    lines = text.split("\n")
    heb_chars = len(re.findall(r"[\u05D0-\u05EA]", text))
    total_chars = len(text)
    heb_ratio = heb_chars / total_chars if total_chars else 0

    short_lines = [l for l in lines if 0 < len(l.strip()) < 40]
    blank_lines = sum(1 for l in lines if l.strip() == "")

    # Potential headlines: short lines with Hebrew, not just numbers/punctuation
    headlines = []
    for l in lines:
        stripped = l.strip()
        if 3 < len(stripped) < 60:
            heb_in_line = len(re.findall(r"[\u05D0-\u05EA]", stripped))
            if heb_in_line > 2 and heb_in_line / max(len(stripped), 1) > 0.5:
                # Check if it looks like a title (followed by longer text or standalone)
                headlines.append(stripped)

    return {
        "page": page_num,
        "lines": len(lines),
        "chars": total_chars,
        "heb_ratio": heb_ratio,
        "blank_lines": blank_lines,
        "short_lines": len(short_lines),
        "potential_headlines": headlines[:5],
    }


def find_section_breaks(pages_text):
    """Look for patterns that indicate article/section boundaries."""
    breaks = []
    for i, text in enumerate(pages_text):
        lines = text.split("\n")
        for j, line in enumerate(lines):
            stripped = line.strip()

            # Ornamental separators
            if re.match(r"^[-=*~_]{3,}$", stripped):
                breaks.append({"page": i+1, "line": j+1, "type": "separator", "text": stripped})

            # Centered short Hebrew line (potential title)
            if 5 < len(stripped) < 50:
                heb = len(re.findall(r"[\u05D0-\u05EA]", stripped))
                if heb > 3 and heb / len(stripped) > 0.6:
                    # Check if preceded by blank line or page start
                    if j == 0 or (j > 0 and lines[j-1].strip() == ""):
                        breaks.append({"page": i+1, "line": j+1, "type": "title_candidate", "text": stripped})

            # Roman/Hebrew numerals as section markers
            if re.match(r"^[IVXivx]+\.?\s*$", stripped) or re.match(r"^[א-י][\.']\s*$", stripped):
                breaks.append({"page": i+1, "line": j+1, "type": "numeral", "text": stripped})

    return breaks


for cm_id, vol_id, label in SAMPLES:
    pdf_path = RAW_DIR / cm_id / f"{vol_id}.pdf"
    if not pdf_path.exists():
        print(f"\n{label}: PDF not found at {pdf_path}")
        continue

    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"{'='*70}")

    with pdfplumber.open(pdf_path) as pdf:
        pages_text = []
        for page in pdf.pages:
            pages_text.append(page.extract_text() or "")

        print(f"Total pages: {len(pages_text)}")

        # Show first few Hebrew-content pages in detail
        shown = 0
        for i, text in enumerate(pages_text):
            if len(text) < 100:
                continue
            heb = len(re.findall(r"[\u05D0-\u05EA]", text))
            if heb / max(len(text), 1) < 0.3:
                continue

            info = analyze_page(i+1, text)
            if shown < 3:
                print(f"\n--- Page {i+1} ---")
                print(f"  Lines: {info['lines']}, Chars: {info['chars']}, "
                      f"Heb ratio: {info['heb_ratio']:.2f}, "
                      f"Blank lines: {info['blank_lines']}, "
                      f"Short lines: {info['short_lines']}")
                if info['potential_headlines']:
                    print(f"  Potential headlines:")
                    for h in info['potential_headlines'][:3]:
                        print(f"    → {h}")

                # Show actual text structure (first 600 chars)
                print(f"  Text preview:")
                for line in text.split("\n")[:15]:
                    marker = "  "
                    if 3 < len(line.strip()) < 50:
                        marker = "→ "
                    print(f"    {marker}{line}")
            shown += 1

        # Find section breaks across all pages
        breaks = find_section_breaks(pages_text)
        print(f"\n  Section break candidates: {len(breaks)}")
        by_type = {}
        for b in breaks:
            by_type.setdefault(b["type"], []).append(b)
        for t, items in by_type.items():
            print(f"    {t}: {len(items)} found")
            for item in items[:3]:
                print(f"      p.{item['page']}: {item['text'][:60]}")
