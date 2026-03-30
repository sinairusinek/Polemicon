"""
Phase 3: ToC-based article segmentation for CM periodicals.

Parses tables of contents from Pardes (3 vols) and Ha-Eshkol (3 vols).
Aḥiasaf is excluded: two-column ToC layout merges columns into garbled lines,
and vol 2 has no ToC at all.

All rule-based, no AI tokens.
"""

import json
import re
from pathlib import Path
from typing import Optional, List, Dict

import pdfplumber

RAW_DIR = Path("data/compact_memory/raw")


# ── Pardes ToC parser ──────────────────────────────────────────────────

def parse_pardes_toc_line(line: str) -> Optional[dict]:
    """Parse a single Pardes ToC line: 'page_num . . . author . title'

    Examples:
        '5 . . . םעה דחא .ךרועה לא בתכמ'
        '86. םולבנעיליל .ל .מ,ינשה תיבה ןמזב לארשי ימכח תולועפ'
        '221 • • . . וואנבוד ן ועמש !הרוקחנו השפחנ'
    """
    # Match: number at start, then dots/spaces, then rest (author.title or title)
    m = re.match(r"^\s*(\d+)\s*[.\s•*,]+\s*(.+)$", line)
    if not m:
        # Try with $ as OCR'd page number
        m = re.match(r"^\s*\$(\d+)\s*[.\s•*,]+\s*(.+)$", line)
    if not m:
        return None

    page_num = int(m.group(1))
    rest = m.group(2).strip()

    # Try to split author and title at the last period before Hebrew text
    # The format is usually: author . title  (RTL, so visually title comes first)
    # In the OCR output, it appears as: title . author (reversed)
    # We'll keep the full text as the entry and not try to split author/title
    # since the OCR makes this unreliable

    return {"page": page_num, "entry": rest}


def parse_pardes_toc_no_page_line(line: str) -> Optional[dict]:
    """Parse a Pardes ToC line WITHOUT a page number (vols 2 & 3 first ToC page).

    These lines have author and title but no leading page number.
    Examples (from vol 2 PDF page 7):
        '. םעה דחא .ךרועה לא ינש בתכמ <'
        '.ק ילאיב .נ .ח •)ייש( הדגאה לא'
    """
    stripped = line.strip()
    if len(stripped) < 10:
        return None
    # Skip headers and garbled lines
    if re.search(r"(ןכות|דומע|םירפס|כ\*ע)", stripped):
        return None
    # Must have meaningful Hebrew content (at least 5 Hebrew chars)
    heb_chars = len(re.findall(r"[\u05D0-\u05EA]", stripped))
    if heb_chars < 5:
        return None
    # Lines starting with a digit are page-number entries, not this format
    if re.match(r"^\s*\d", stripped):
        return None
    # Clean: remove leading dots/spaces/markers and trailing markers
    cleaned = re.sub(r'^[.\s•*,]+', '', stripped)
    cleaned = re.sub(r'\s*[<>^]+\s*$', '', cleaned)
    # Remove trailing standalone numbers with parens like '3)'
    cleaned = re.sub(r'\s*\d+\)\s*$', '', cleaned)
    cleaned = cleaned.strip()
    if len(cleaned) < 5:
        return None
    return {"entry": cleaned}


def _extract_search_terms(text: str) -> List[str]:
    """Extract Hebrew search terms from a ToC entry.

    Handles OCR-split words by concatenating adjacent short Hebrew fragments
    (e.g., 'ק ילאיב' → also searches for 'קילאיב').
    """
    # Standard words (3+ Hebrew chars)
    words = re.findall(r"[\u05D0-\u05EA]{3,}", text)

    # Also build compound terms from adjacent Hebrew fragments
    # Split on non-Hebrew chars, keeping Hebrew-only tokens
    tokens = re.findall(r"[\u05D0-\u05EA]+", text)
    for i in range(len(tokens) - 1):
        compound = tokens[i] + tokens[i + 1]
        if len(compound) >= 4 and compound not in words:
            words.append(compound)

    return words


def locate_entries_by_text(pdf_path: Path, entries: List[dict],
                           offset: int,
                           skip_pages: Optional[set] = None) -> List[dict]:
    """Find printed page numbers for ToC entries by matching their text in the PDF.

    Searches PDF pages sequentially for Hebrew word matches from each entry.
    Skips ToC pages (which contain all entry words) to avoid false matches.
    Returns only entries where a match was found.
    """
    if not entries:
        return []
    if skip_pages is None:
        skip_pages = set()

    located = []
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        # Pre-extract all page texts
        page_texts = []
        for i in range(total):
            page_texts.append(pdf.pages[i].extract_text() or "")

        last_found_idx = 0
        for entry in entries:
            words = _extract_search_terms(entry["entry"])
            if not words:
                continue

            best_idx = None
            best_score = 0

            # Search forward from last found position, skipping ToC pages
            for idx in range(last_found_idx, total):
                if idx in skip_pages:
                    continue
                score = sum(1 for w in words if w in page_texts[idx])
                if score > best_score:
                    best_score = score
                    best_idx = idx
                # Once we've passed the best match by 10 pages, stop
                if best_idx is not None and idx > best_idx + 10:
                    break

            # Accept match if at least 1 term matches (sequential ordering
            # constrains false positives)
            if best_idx is not None and best_score >= 1:
                printed_page = best_idx + 1 - offset
                located.append({"page": printed_page, "entry": entry["entry"]})
                last_found_idx = best_idx

    return located


def extract_pardes_toc(pdf_path: Path) -> list[dict]:
    """Extract ToC entries from a Pardes volume.

    Handles two formats:
    - Vol 1: single ToC page with page numbers
    - Vols 2 & 3: first ToC page has entries without page numbers,
      second page (דומע) has a few more entries with page numbers
    """
    page_entries = []      # entries with page numbers
    no_page_entries = []   # entries without page numbers
    toc_page_indices = set()  # PDF page indices that are ToC pages (to skip in text matching)

    with pdfplumber.open(pdf_path) as pdf:
        for i in range(min(15, len(pdf.pages))):
            text = pdf.pages[i].extract_text() or ""
            lines = text.split("\n")

            has_toc = any(re.search(r"ןכות|ןכת", line) for line in lines[:3])
            has_page_header = any(re.search(r"דומע", line) for line in lines[:3])

            if not has_toc and not has_page_header:
                continue

            toc_page_indices.add(i)

            for line in lines:
                # Skip header lines
                if re.search(r"^:?(םיניינעה|םינינעה)\s*ןכות", line.strip()):
                    continue
                if re.search(r"^(דומע|:םישדח םירפס)", line.strip()):
                    continue
                # Skip book review entries (numbered with parentheses)
                if re.match(r"^\s*\)", line.strip()):
                    continue

                if has_page_header:
                    # Page-number format
                    parsed = parse_pardes_toc_line(line)
                    if parsed:
                        page_entries.append(parsed)
                elif has_toc and not has_page_header:
                    # No-page-number format (try page-number first, fallback)
                    parsed = parse_pardes_toc_line(line)
                    if parsed:
                        page_entries.append(parsed)
                    else:
                        parsed = parse_pardes_toc_no_page_line(line)
                        if parsed:
                            no_page_entries.append(parsed)

    # If we have no-page entries, locate them via text matching
    if no_page_entries:
        # Find offset using page-number entries if available
        if page_entries:
            offset = find_page_offset(pdf_path, page_entries)
        else:
            offset = 4  # reasonable default for Pardes
        located = locate_entries_by_text(pdf_path, no_page_entries, offset,
                                        skip_pages=toc_page_indices)
        print(f"  Text-matched {len(located)}/{len(no_page_entries)} entries without page numbers")
        page_entries.extend(located)

    # Sort by page number and deduplicate
    page_entries.sort(key=lambda e: e["page"])
    seen = set()
    deduped = []
    for e in page_entries:
        if e["page"] not in seen:
            seen.add(e["page"])
            deduped.append(e)

    return deduped


# ── Ha-Eshkol ToC parser ───────────────────────────────────────────────

def parse_eshkol_toc_line(line: str) -> Optional[dict]:
    """Parse a Ha-Eshkol ToC line: 'start — end  author  title'

    Examples:
        '1 — 5 ל״ומה תאמ ןבללו ררבל'
        '6 — 16 ־רנזירלקףסוי /; םולשב המחלמ'
        '96 ןודרוג ביל לאומש II )ייש( ןורצבל יבוש'
    """
    # Pattern: start_page — end_page rest
    m = re.match(r"^\s*(\d+)\s*[-—־]+\s*([\dCc]+)\s+(.+)$", line)
    if m:
        start = int(m.group(1))
        # Handle OCR'd 0 for page numbers (e.g., "1C0" = 100, "2C9" = 209)
        end_str = m.group(2).replace("C", "0").replace("c", "0")
        try:
            end = int(end_str)
        except ValueError:
            end = start
        rest = m.group(3).strip()
        return {"page": start, "end_page": end, "entry": rest}

    # Pattern: single page number (for poems, short pieces)
    m = re.match(r"^\s*(\d+)\s+([^\d].{5,})$", line)
    if m:
        page = int(m.group(1))
        rest = m.group(2).strip()
        # Filter out lines that are actually article text, not ToC entries
        # ToC entries have author names with abbreviations or specific markers
        if re.search(r"(תאמ|ר״ד|II|//|״|«|" + r"\)ריש\)|\)רופס\))", rest):
            return {"page": page, "end_page": page, "entry": rest}
        # Short entries with Hebrew author indicators
        if len(rest) < 80:
            return {"page": page, "end_page": page, "entry": rest}

    return None


def extract_eshkol_toc(pdf_path: Path) -> list[dict]:
    """Extract ToC entries from a Ha-Eshkol volume."""
    entries = []
    with pdfplumber.open(pdf_path) as pdf:
        for i in range(min(10, len(pdf.pages))):
            text = pdf.pages[i].extract_text() or ""
            lines = text.split("\n")

            # Check if this is a ToC page
            is_toc = any(
                re.search(r"ןכת|ןכות|ןכ1ו", line) for line in lines[:3]
            )
            if not is_toc:
                continue

            for line in lines:
                # Skip headers
                if re.search(r"^[:\s]*(םינינעה|םיענעזד|םינינאה)\s*(ןכת|ןכות|ןכ1ו)", line.strip()):
                    continue
                if re.match(r"^\s*דצ\.?\s*$", line.strip()):
                    continue

                parsed = parse_eshkol_toc_line(line)
                if parsed:
                    entries.append(parsed)

    entries.sort(key=lambda e: e["page"])
    seen = set()
    deduped = []
    for e in entries:
        if e["page"] not in seen:
            seen.add(e["page"])
            deduped.append(e)

    return deduped


# ── Page number mapping ────────────────────────────────────────────────

def find_page_offset(pdf_path: Path, toc_entries: list[dict], search_range=30) -> int:
    """Find the offset between printed page numbers and PDF page indices.

    Strategy: take the first ToC entry's page number, search nearby PDF pages
    for text that matches the entry's content.
    """
    if not toc_entries:
        return 0

    first_entry = toc_entries[0]
    target_page = first_entry["page"]
    # Extract a few distinctive Hebrew words from the entry
    entry_words = re.findall(r"[\u05D0-\u05EA]{3,}", first_entry["entry"])

    with pdfplumber.open(pdf_path) as pdf:
        best_offset = None
        best_score = 0

        for offset in range(-2, search_range):
            pdf_idx = target_page + offset - 1  # page numbers are 1-based
            if 0 <= pdf_idx < len(pdf.pages):
                page_text = pdf.pages[pdf_idx].extract_text() or ""
                # Score: how many entry words appear on this page
                score = sum(1 for w in entry_words if w in page_text)
                if score > best_score:
                    best_score = score
                    best_offset = offset

        if best_offset is not None and best_score > 0:
            return best_offset

    # Fallback: estimate from front matter
    # Most CM volumes have a few pages of front matter before page 1
    return 4  # common default


# ── Article extraction ─────────────────────────────────────────────────

def _find_content_end(pdf, start_pdf: int, total_pages: int) -> int:
    """Find the last page with significant Hebrew content, scanning backward.

    Used for the last article to avoid including indices/ads/back matter.
    """
    # Scan backward from the end
    for idx in range(total_pages - 1, start_pdf, -1):
        text = pdf.pages[idx].extract_text() or ""
        heb = len(re.findall(r"[\u05D0-\u05EA]", text))
        total = len(text)
        if total > 100 and heb / total > 0.4:
            return idx + 1  # exclusive upper bound
    return total_pages


def extract_articles(pdf_path: Path, toc_entries: list[dict], offset: int) -> list[dict]:
    """Extract article text by splitting at ToC page boundaries."""
    articles = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

        for i, entry in enumerate(toc_entries):
            start_pdf = entry["page"] + offset - 1  # 0-based PDF index
            if i + 1 < len(toc_entries):
                end_pdf = toc_entries[i + 1]["page"] + offset - 1
            else:
                # Last article: use end_page if available (Ha-Eshkol),
                # otherwise find where Hebrew content ends
                if entry.get("end_page") and entry["end_page"] > entry["page"]:
                    end_pdf = entry["end_page"] + offset  # inclusive end
                else:
                    # Scan backward from end to find last page with Hebrew content
                    end_pdf = _find_content_end(pdf, start_pdf, total_pages)

            # Bounds check
            start_pdf = max(0, min(start_pdf, total_pages - 1))
            end_pdf = max(start_pdf + 1, min(end_pdf, total_pages))

            # Extract text from these pages
            page_texts = []
            for j in range(start_pdf, end_pdf):
                text = pdf.pages[j].extract_text() or ""
                page_texts.append(text)

            full_text = "\n".join(page_texts)

            articles.append({
                "entry": entry["entry"],
                "printed_start_page": entry["page"],
                "printed_end_page": entry.get("end_page", entry["page"]),
                "pdf_start_page": start_pdf + 1,  # back to 1-based for display
                "pdf_end_page": end_pdf,
                "num_pages": end_pdf - start_pdf,
                "text": full_text,
                "char_count": len(full_text),
            })

    return articles


# ── Main segmentation pipeline ─────────────────────────────────────────

def segment_volume(cm_id: str, volume_id: str, periodical: str) -> list[dict]:
    """Segment a single volume into articles using its ToC."""
    pdf_path = RAW_DIR / cm_id / f"{volume_id}.pdf"
    if not pdf_path.exists():
        print(f"  PDF not found: {pdf_path}")
        return []

    # Parse ToC
    if periodical == "pardes":
        toc = extract_pardes_toc(pdf_path)
    elif periodical == "eshkol":
        toc = extract_eshkol_toc(pdf_path)
    else:
        print(f"  Unknown periodical type: {periodical}")
        return []

    if not toc:
        print(f"  No ToC entries found")
        return []

    print(f"  ToC: {len(toc)} entries (pages {toc[0]['page']}-{toc[-1]['page']})")

    # Find page offset
    offset = find_page_offset(pdf_path, toc)
    print(f"  Page offset: {offset} (printed page 1 = PDF page {1 + offset})")

    # Extract articles
    articles = extract_articles(pdf_path, toc, offset)

    return articles


def segment_all():
    """Segment all volumes with parseable ToCs."""
    with open("data/compact_memory/volume_map.json") as f:
        volume_map = json.load(f)

    all_articles = []

    # Pardes
    pardes = volume_map.get("8003959", {})
    print(f"\n{'='*60}")
    print(f"Pardes ({pardes.get('hebrew', '')})")
    print(f"{'='*60}")
    for vol in pardes.get("volumes", []):
        print(f"\n  --- {vol['caption']} ({vol['date']}) ---")
        articles = segment_volume("8003959", vol["id"], "pardes")
        for a in articles:
            a["cm_id"] = "8003959"
            a["volume_id"] = vol["id"]
            a["periodical"] = "Pardes"
            a["year"] = vol["date"]
        all_articles.extend(articles)
        print(f"  Extracted {len(articles)} articles")
        for a in articles[:3]:
            print(f"    p.{a['printed_start_page']}: {a['entry'][:60]} ({a['char_count']} chars)")

    # Ha-Eshkol
    eshkol = volume_map.get("3773345", {})
    print(f"\n{'='*60}")
    print(f"Ha-Eshkol ({eshkol.get('hebrew', '')})")
    print(f"{'='*60}")
    for vol in eshkol.get("volumes", []):
        print(f"\n  --- {vol['caption']} ({vol['date']}) ---")
        articles = segment_volume("3773345", vol["id"], "eshkol")
        for a in articles:
            a["cm_id"] = "3773345"
            a["volume_id"] = vol["id"]
            a["periodical"] = "Ha-Eshkol"
            a["year"] = vol["date"]
        all_articles.extend(articles)
        print(f"  Extracted {len(articles)} articles")
        for a in articles[:3]:
            print(f"    p.{a['printed_start_page']}: {a['entry'][:60]} ({a['char_count']} chars)")

    print(f"\n\nTotal articles extracted: {len(all_articles)}")
    print(f"  Pardes: {sum(1 for a in all_articles if a['periodical'] == 'Pardes')}")
    print(f"  Ha-Eshkol: {sum(1 for a in all_articles if a['periodical'] == 'Ha-Eshkol')}")

    return all_articles


if __name__ == "__main__":
    articles = segment_all()

    # Save preview (without full text) for review
    preview = [{k: v for k, v in a.items() if k != "text"} for a in articles]
    with open("data/compact_memory/segmentation_preview.json", "w") as f:
        json.dump(preview, f, ensure_ascii=False, indent=2)
    print(f"\nPreview saved to data/compact_memory/segmentation_preview.json")
