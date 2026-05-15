"""
verify_citations.py — DOI/Crossref pass on the literature-review citations.

For each citation in data/thread_literature_review.parquet, attempt to:
  1. Extract a DOI from the URL (if any).
  2. If DOI present, fetch Crossref metadata and compare author family-name +
     publication year against the parquet entry.
  3. Mark each citation as 'verified' | 'flagged' | 'unverifiable' and record
     the discrepancy (if any).

Citations without a resolvable DOI are 'unverifiable' (the URL may still be
valid — manual curation needed; this script does not chase non-DOI URLs).

Outputs:
  data/thread_literature_review_verified.parquet  per-citation rows with status
  logs/citation_verification.md                   summary report

Run:
  python scripts/verify_citations.py
"""
from __future__ import annotations
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LOGS = ROOT / "logs"

CROSSREF_URL = "https://api.crossref.org/works/{doi}"
USER_AGENT = "Polemicon-citation-verifier/1.0 (mailto:sinai.rusinek@gmail.com)"
DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"<>]+", re.IGNORECASE)
SLEEP = 0.2


def extract_doi(url: str) -> Optional[str]:
    if not url:
        return None
    m = DOI_RE.search(url)
    if not m:
        return None
    doi = m.group(0)
    doi = doi.rstrip(".,);]")
    return urllib.parse.unquote(doi)


def fetch_crossref(doi: str) -> Optional[dict]:
    try:
        req = urllib.request.Request(CROSSREF_URL.format(doi=urllib.parse.quote(doi)),
                                     headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode("utf-8", errors="replace"))
            return body.get("message")
    except Exception as e:
        return {"_error": str(e)[:160]}


def family_from_crossref(msg: dict) -> Optional[str]:
    authors = msg.get("author") or []
    if not authors:
        return None
    fam = authors[0].get("family") or ""
    return fam.strip().lower() or None


def year_from_crossref(msg: dict) -> Optional[int]:
    for k in ("issued", "published-print", "published-online", "created"):
        parts = (msg.get(k) or {}).get("date-parts") or []
        if parts and parts[0]:
            try:
                return int(parts[0][0])
            except Exception:
                pass
    return None


def author_family(author_str: str) -> Optional[str]:
    if not author_str:
        return None
    s = author_str.strip()
    # Pattern "Lastname, Firstname" — take before comma
    if "," in s:
        return s.split(",", 1)[0].strip().lower()
    # Otherwise take last whitespace token
    toks = s.split()
    return toks[-1].lower() if toks else None


def main():
    src = DATA / "thread_literature_review.parquet"
    if not src.exists():
        raise SystemExit(f"{src} not found")
    lit = pd.read_parquet(src)

    rows = []
    for _, r in lit.iterrows():
        tid = int(r["thread_id"])
        raw = r.get("key_sources") or "[]"
        try:
            sources = json.loads(raw) if isinstance(raw, str) else list(raw)
        except Exception:
            sources = []
        for idx, src_obj in enumerate(sources):
            author = src_obj.get("author") or ""
            title = src_obj.get("title") or ""
            year = src_obj.get("year") or 0
            url = src_obj.get("url") or ""
            doi = extract_doi(url)
            status = "unverifiable"
            issues = []
            cr_family = cr_year = None
            if doi:
                msg = fetch_crossref(doi)
                time.sleep(SLEEP)
                if msg is None or msg.get("_error"):
                    status = "unverifiable"
                    issues.append(f"crossref_fetch_failed: {(msg or {}).get('_error','no-response')}")
                else:
                    cr_family = family_from_crossref(msg)
                    cr_year = year_from_crossref(msg)
                    parquet_family = author_family(author)
                    flags = []
                    if cr_family and parquet_family and cr_family != parquet_family:
                        # Handle cases where parquet has organization name w/o personal author
                        if not (parquet_family in cr_family or cr_family in parquet_family):
                            flags.append(f"author_mismatch: parquet='{author}' crossref_family='{cr_family}'")
                    if cr_year and year and abs(cr_year - int(year)) > 0:
                        # Allow trivial off-by-one if year is 0
                        flags.append(f"year_mismatch: parquet={year} crossref={cr_year}")
                    status = "flagged" if flags else "verified"
                    issues.extend(flags)
            rows.append({
                "thread_id": tid,
                "src_index": idx,
                "author": author,
                "title": title[:200],
                "year": year,
                "url": url,
                "doi": doi or "",
                "crossref_family": cr_family or "",
                "crossref_year": cr_year or 0,
                "status": status,
                "issues": "; ".join(issues),
            })

    out = pd.DataFrame(rows)
    out_path = DATA / "thread_literature_review_verified.parquet"
    out.to_parquet(out_path, index=False)
    print(f"wrote {out_path} ({len(out)} citation rows)")

    counts = out["status"].value_counts().to_dict()
    print("Status counts:", counts)

    LOGS.mkdir(exist_ok=True)
    lines = ["# Citation verification report", ""]
    lines.append(f"Total citations: {len(out)}")
    for s in ("verified", "flagged", "unverifiable"):
        lines.append(f"- **{s}**: {counts.get(s, 0)}")
    lines.append("")
    flagged = out[out["status"] == "flagged"]
    if len(flagged):
        lines.append(f"## Flagged ({len(flagged)})")
        lines.append("")
        lines.append("| thread | author (parquet) | crossref family | year (parquet/crossref) | issue |")
        lines.append("|---|---|---|---|---|")
        for _, r in flagged.iterrows():
            lines.append(f"| {r['thread_id']} | {r['author']} | {r['crossref_family']} | "
                         f"{r['year']}/{r['crossref_year']} | {r['issues']} |")
    (LOGS / "citation_verification.md").write_text("\n".join(lines))
    print(f"wrote {LOGS / 'citation_verification.md'}")


if __name__ == "__main__":
    main()
