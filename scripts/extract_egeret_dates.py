"""
Backfill composition dates for egeret rows in polemic_pool.

Sources, in priority order, recorded per row in `source_of_date`:
  - tsv_dateiso_day      DateISO is full YYYY-MM-DD
  - tsv_dateiso_month    YYYY-MM
  - tsv_dateiso_year     YYYY
  - tsv_dateiso_partial  XXXX-MM-DD / XXXX-MM / 18XX / --MM-DD / range / list
  - tsv_date_hebrew      Tier B: parsed Hebrew-calendar date from raw `Date`
  - volume_metadata      origPublicationDate as upper bound (year_max only)
  - author_lifespan      no other signal, bounded by author birth+15..death
  - none                 nothing usable

`partial_policy` records how Tier A4 strings were collapsed:
  - list_min            list "1906-02-16, 1906-04-20, ..." -> earliest
  - range_span          "1881-1882" / "1922-02-09; 1923-03-08" -> year_min/max
  - month_day_only      "--02-22" / "XXXX-MM-DD" -> month/day kept, year null
  - month_only          "XXXX-MM" -> month kept, year null
  - century_known       "18XX" / "19XX-08" -> year_min/max bound a century
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
from pyluach import dates as pdates

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "data" / "polemic_pool.parquet"
TSV = ROOT / "e-geret-batch-export.tsv"
OUT = ROOT / "data" / "egeret_dates.parquet"

# Author lifespans for the 29 authors in the all-null bucket.
# Birth, death (Gregorian years). For collaborations ("X / Y") use union.
LIFESPANS: dict[str, tuple[int, int]] = {
    "נחום סוקולוב": (1859, 1936),
    "ראובן בריינין": (1862, 1939),
    "חיים ארלוזורוב": (1899, 1933),
    "אהרן דוד גורדון": (1856, 1922),
    "קלמן שולמן": (1819, 1899),
    "בנימין זאב הרצל / מיכל ברקוביץ": (1860, 1955),  # union: Herzl 1860-1904, Berkowicz 1865-1955
    "אחד העם": (1856, 1927),
    "משה ליב לילינבלום": (1843, 1910),
    'איתמר בן־אב"י': (1882, 1943),
    "צבי שץ": (1890, 1921),
    "מנחם מנדל איילבום": (1811, 1893),
    "מיכה יוסף ברדיצ'בסקי": (1865, 1921),
    "מרדכי בן הלל הכהן": (1856, 1936),
    "מיכה יוסף לבנזון": (1828, 1852),
    "יצחק פרנהוף": (1868, 1919),
    "יוסף זליגר": (1872, 1919),
    "דוד פרישמן": (1859, 1922),
    "יהודה ליב גורדון": (1830, 1892),
    "שמואל יוסף פין": (1818, 1890),
    "אברהם שטרן": (1907, 1942),
    "יהודה ליב קצנלסון": (1846, 1917),
    "שאול טשרניחובסקי": (1875, 1943),
    "אהרן דוד מרקסון": (1885, 1939),
    "שמריהו לוין": (1867, 1935),
    "חיים נחמן ביאליק": (1873, 1934),
    "שמואל הנגיד": (993, 1056),
    "חיים דוד נוסבוים": (1880, 1928),
    "אליעזר בן־יהודה": (1858, 1922),
    "ברל כצנלסון": (1887, 1944),
    "ישעיהו ברשדסקי": (1871, 1908),
}

# ---------- Hebrew-calendar parsing (Tier B, conservative) ----------

HEB_MONTHS = {
    "תשרי": 7, "מרחשון": 8, "חשון": 8, "חשוון": 8, "מרחשוון": 8,
    "כסלו": 9, "כסליו": 9, "טבת": 10, "שבט": 11,
    "אדר": 12, "אדר א": 12, "אדר ב": 13, 'אד"ר': 12, 'אד"ש': 13,
    "ניסן": 1, "אייר": 2, "סיון": 3, "סיוון": 3, "תמוז": 4,
    "אב": 5, "מנחם אב": 5, "אלול": 6,
}

GERESH = "[׳'׳]"
GERSHAYIM = "[״\"״]"

# Strip gershayim/geresh for numeric-letter parsing
def _heb_num(s: str) -> int | None:
    s = re.sub(GERESH, "", s)
    s = re.sub(GERSHAYIM, "", s)
    s = s.replace(" ", "")
    vals = {"א":1,"ב":2,"ג":3,"ד":4,"ה":5,"ו":6,"ז":7,"ח":8,"ט":9,
            "י":10,"כ":20,"ך":20,"ל":30,"מ":40,"ם":40,"נ":50,"ן":50,
            "ס":60,"ע":70,"פ":80,"ף":80,"צ":90,"ץ":90,
            "ק":100,"ר":200,"ש":300,"ת":400}
    total = 0
    for ch in s:
        if ch not in vals:
            return None
        total += vals[ch]
    return total or None


def parse_hebrew_date(text: str) -> tuple[int | None, int | None, int | None] | None:
    """
    Try to parse "<day> <month> <year>" Hebrew-calendar date.
    Returns (year_g, month_g, day_g) Gregorian, or None.
    Conservative: requires both month name and an explicit Hebrew year-letters block.
    """
    if not text:
        return None
    s = re.sub(r"\s+", " ", text).strip()
    # Locate month
    month_h = None
    for name, num in HEB_MONTHS.items():
        if re.search(rf"(?<![א-ת]){re.escape(name)}(?![א-ת])", s):
            month_h = num
            month_name = name
            break
    if month_h is None:
        return None
    # Year: a token of Hebrew letters with gershayim, length suggesting a 4-digit year.
    # Common patterns: ת"ש, תרכ"ה, אתתמ"ד.
    yr_match = re.search(rf"([א-ת]+{GERSHAYIM}[א-ת])", s)
    if not yr_match:
        return None
    yr_tok = yr_match.group(1)
    yr_h = _heb_num(yr_tok)
    if yr_h is None:
        return None
    # Heuristic: year letters in literary use are typically 4xxx-5xxx (with implicit ה).
    # If yr_h < 1000, treat as missing-elef and add 5000.
    if yr_h < 1000:
        yr_h += 5000
    if yr_h < 4000 or yr_h > 6000:
        return None
    # Day (optional): Hebrew letters before month
    pre = s.split(month_name)[0]
    day_match = re.search(rf"([א-ת]+{GERSHAYIM}?[א-ת]?)\s*$", pre.strip())
    day_h = None
    if day_match:
        d = _heb_num(day_match.group(1))
        if d and 1 <= d <= 30:
            day_h = d
    try:
        if day_h is not None:
            g = pdates.HebrewDate(yr_h, month_h if month_h <= 12 else 12, day_h).to_pydate()
            return (g.year, g.month, g.day)
        # Month-year only: pick first of month
        g = pdates.HebrewDate(yr_h, month_h if month_h <= 12 else 12, 1).to_pydate()
        return (g.year, g.month, None)
    except Exception:
        return None


# ---------- DateISO parser (Tier A) ----------

DATEISO_FULL = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
DATEISO_YM = re.compile(r"^(\d{4})-(\d{2})$")
DATEISO_Y = re.compile(r"^(\d{4})$")
PARTIAL_XYMD = re.compile(r"^X{4}-(\d{2})-(\d{2})$")
PARTIAL_XYM = re.compile(r"^X{4}-(\d{2})$")
PARTIAL_DASHMD = re.compile(r"^--(\d{2})-(\d{2})$")
PARTIAL_DASHM = re.compile(r"^--(\d{2})$")
PARTIAL_CENT_YM = re.compile(r"^(\d{2})XX-(\d{2})(?:-(\d{2})|-XX)?$")
PARTIAL_CENT = re.compile(r"^(\d{2})XX$")
PARTIAL_DECADE = re.compile(r"^(\d{3})X$")
RANGE_YEARS = re.compile(r"^(\d{4})-(\d{4})$")


def parse_dateiso(s: str) -> dict | None:
    """Return component dict or None if unparseable."""
    if s is None:
        return None
    s = str(s).strip()
    if not s or s.lower() == "unknown":
        return None
    out = {"year": None, "month": None, "day": None,
           "year_min": None, "year_max": None,
           "tier": "tsv_dateiso_partial", "partial_policy": None}
    if m := DATEISO_FULL.match(s):
        y, mo, d = map(int, m.groups())
        out.update(year=y, month=mo, day=d, year_min=y, year_max=y, tier="tsv_dateiso_day")
        return out
    if m := DATEISO_YM.match(s):
        y, mo = map(int, m.groups())
        out.update(year=y, month=mo, year_min=y, year_max=y, tier="tsv_dateiso_month")
        return out
    if m := DATEISO_Y.match(s):
        y = int(m.group(1))
        out.update(year=y, year_min=y, year_max=y, tier="tsv_dateiso_year")
        return out
    # ----- partials -----
    if m := PARTIAL_XYMD.match(s):
        mo, d = map(int, m.groups())
        out.update(month=mo, day=d, partial_policy="month_day_only")
        return out
    if m := PARTIAL_XYM.match(s):
        out.update(month=int(m.group(1)), partial_policy="month_only")
        return out
    if m := PARTIAL_DASHMD.match(s):
        mo, d = map(int, m.groups())
        out.update(month=mo, day=d, partial_policy="month_day_only")
        return out
    if m := PARTIAL_DASHM.match(s):
        out.update(month=int(m.group(1)), partial_policy="month_only")
        return out
    if m := PARTIAL_CENT_YM.match(s):
        cent = int(m.group(1)) * 100
        mo = int(m.group(2))
        d = int(m.group(3)) if m.group(3) else None
        out.update(month=mo, day=d, year_min=cent, year_max=cent + 99,
                   partial_policy="century_known")
        return out
    if m := PARTIAL_CENT.match(s):
        cent = int(m.group(1)) * 100
        out.update(year_min=cent, year_max=cent + 99, partial_policy="century_known")
        return out
    if m := PARTIAL_DECADE.match(s):
        dec = int(m.group(1)) * 10
        out.update(year_min=dec, year_max=dec + 9, partial_policy="century_known")
        return out
    if m := RANGE_YEARS.match(s):
        y1, y2 = int(m.group(1)), int(m.group(2))
        out.update(year_min=min(y1, y2), year_max=max(y1, y2), partial_policy="range_span")
        return out
    # List of dates separated by "," or ";" — take min as date, span as range
    parts = re.split(r"[;,]\s*", s)
    parsed = []
    for p in parts:
        p = p.strip()
        if mm := DATEISO_FULL.match(p):
            parsed.append(tuple(map(int, mm.groups())))
    if parsed:
        parsed.sort()
        y, mo, d = parsed[0]
        ymax = parsed[-1][0]
        out.update(year=y, month=mo, day=d, year_min=y, year_max=ymax,
                   partial_policy="list_min")
        return out
    return None  # unrecognized


# ---------- Main ----------

def main():
    pp = pd.read_parquet(POOL)
    eg = pp[pp["source"] == "egeret"].copy()
    eg["row_idx"] = eg["doc_id"].str.extract(r"egeret_(\d+)").astype(int)
    tsv = pd.read_csv(TSV, sep="\t", encoding="utf-8-sig", low_memory=False)
    tsv["row_idx"] = range(len(tsv))
    m = eg.merge(tsv, on="row_idx", how="left", suffixes=("_pp", "_tsv"))

    rows = []
    for _, r in m.iterrows():
        rec = {
            "doc_id": r["doc_id"],
            "date": None,
            "year": None, "month": None, "day": None,
            "year_min": None, "year_max": None,
            "source_of_date": "none",
            "confidence": "none",
            "partial_policy": None,
            "raw_dateiso": r.get("DateISO"),
            "raw_date": r.get("Date"),
            "raw_orig_pub": r.get("origPublicationDate"),
            "author": r.get("authorString"),
        }
        # Tier A: DateISO
        parsed = parse_dateiso(r.get("DateISO"))
        if parsed:
            rec.update({k: parsed[k] for k in ("year","month","day","year_min","year_max","partial_policy")})
            rec["source_of_date"] = parsed["tier"]
            rec["confidence"] = {
                "tsv_dateiso_day": "high",
                "tsv_dateiso_month": "high",
                "tsv_dateiso_year": "medium",
                "tsv_dateiso_partial": "medium",
            }[parsed["tier"]]
            if rec["year"] and rec["month"] and rec["day"]:
                rec["date"] = f"{rec['year']:04d}-{rec['month']:02d}-{rec['day']:02d}"
            rows.append(rec)
            continue

        # Tier B: Hebrew-calendar parse of `Date`
        if isinstance(r.get("Date"), str):
            heb = parse_hebrew_date(r["Date"])
            if heb:
                y, mo, d = heb
                rec.update(year=y, month=mo, day=d, year_min=y, year_max=y,
                           source_of_date="tsv_date_hebrew", confidence="medium")
                if d:
                    rec["date"] = f"{y:04d}-{mo:02d}-{d:02d}"
                rows.append(rec)
                continue

        # Tier C: origPublicationDate as upper-bound year_max
        opd = r.get("origPublicationDate")
        if isinstance(opd, str) and (m_ := DATEISO_FULL.match(opd) or DATEISO_Y.match(opd)):
            opd_year = int(m_.group(1))
            ymin = None
            # Combine with author lifespan if available
            life = LIFESPANS.get(r.get("authorString"))
            if life:
                ymin = max(life[0] + 15, 1700)
                ymin = min(ymin, opd_year)
            rec.update(year_min=ymin, year_max=opd_year,
                       source_of_date="volume_metadata", confidence="low",
                       partial_policy="volume_upper_bound")
            rows.append(rec)
            continue

        # Tier D: author lifespan
        life = LIFESPANS.get(r.get("authorString"))
        if life:
            ymin = max(life[0] + 15, 1700) if life[0] >= 1500 else life[0] + 15
            ymax = life[1]
            rec.update(year_min=ymin, year_max=ymax,
                       source_of_date="author_lifespan", confidence="low",
                       partial_policy="lifespan_writing_window")
            rows.append(rec)
            continue

        rows.append(rec)

    out = pd.DataFrame(rows)
    out = out[["doc_id","date","year","month","day","year_min","year_max",
               "source_of_date","confidence","partial_policy",
               "raw_dateiso","raw_date","raw_orig_pub","author"]]
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT}  rows={len(out)}")

    # Report
    print("\n=== source_of_date ===")
    print(out["source_of_date"].value_counts().to_string())
    print("\n=== confidence ===")
    print(out["confidence"].value_counts().to_string())
    print("\n=== resolution ===")
    def res(r):
        has = lambda k: pd.notna(r[k])
        if r["date"]: return "exact_day"
        if has("year") and has("month"): return "year_month"
        if has("year"): return "year"
        if has("year_min") and has("year_max"):
            span = r["year_max"] - r["year_min"]
            if span <= 1: return "range_<=1y"
            if span <= 10: return "range_<=10y"
            if span <= 50: return "range_<=50y"
            return "range_>50y"
        if has("month"): return "month_only_no_year"
        if has("year_max") or has("year_min"): return "one_sided_bound"
        return "none"
    out["resolution"] = out.apply(res, axis=1)
    print(out["resolution"].value_counts().to_string())

    print("\n=== 20-row sample ===")
    sample = out.sample(20, random_state=42)[
        ["doc_id","date","year","month","day","year_min","year_max",
         "source_of_date","confidence","partial_policy","raw_dateiso","raw_date"]
    ]
    for _, r in sample.iterrows():
        print(f"  {r['doc_id']:<12} {str(r['date']):<12} y={r['year']} m={r['month']} d={r['day']} "
              f"[{r['year_min']}..{r['year_max']}] {r['source_of_date']}/{r['confidence']} "
              f"raw_iso={r['raw_dateiso']!r} raw_date={r['raw_date']!r}")

    print("\n=== failures (source_of_date=none) ===")
    fails = out[out["source_of_date"] == "none"]
    print(f"count: {len(fails)}")
    for _, r in fails.head(15).iterrows():
        print(f"  {r['doc_id']} author={r['author']!r} raw_iso={r['raw_dateiso']!r} "
              f"raw_date={r['raw_date']!r} raw_pub={r['raw_orig_pub']!r}")

if __name__ == "__main__":
    main()
