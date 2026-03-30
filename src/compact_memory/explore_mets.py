"""
Phase 1: Fetch METS metadata from Compact Memory OAI-PMH endpoint
for 13 target Hebrew periodicals. Extract structural hierarchy,
file availability, and page counts.
"""

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

OAI_BASE = "https://sammlungen.ub.uni-frankfurt.de/cm/oai/"

TARGETS = [
    {"cm_id": "4785731", "title": "Kokhve Yitsḥaḳ", "hebrew": "כוכבי יצחק",
     "dates": "1845-1873", "place": "Wien", "type": "Literary almanac",
     "date_filter": (1850, 1900)},
    {"cm_id": "9582285", "title": "Ōṣar neḥmād", "hebrew": "אוצר נחמד",
     "dates": "1856-1863", "place": "Wien", "type": "Letters on faith/wisdom",
     "date_filter": (1850, 1900)},
    {"cm_id": "3769475", "title": "Jeschurun (Hebr. Abth.)", "hebrew": "ישורון",
     "dates": "1857-1878", "place": "Lemberg", "type": "Hebrew academic journal",
     "date_filter": (1850, 1900)},
    # DROPPED: Jeschurun (Dt. Abth.) — German-language section, not useful for Hebrew corpus
    # DROPPED: Der ungarische Israelit — bilingual German-Hungarian, minimal Hebrew, partial digitization
    {"cm_id": "8003959", "title": "Pardes", "hebrew": "פרדס",
     "dates": "1892-1896", "place": "Odessa", "type": "Literary collection",
     "date_filter": (1850, 1900)},
    {"cm_id": "10719318", "title": "Aḥiasaf", "hebrew": "אחיאסף",
     "dates": "1893-1924", "place": "Warsaw", "type": "Literary almanac",
     "date_filter": (1850, 1900)},
    # DROPPED: Mim-mizraḥ ū-mim-maʿarāv — image-only PDF, no OCR text layer
    {"cm_id": "4789469", "title": "Luaḥ erets Yiśraʾel", "hebrew": "לוח ארץ ישראל",
     "dates": "1895-1916", "place": "Jerusalem", "type": "Practical+literary almanac",
     "date_filter": (1850, 1900)},
    {"cm_id": "3773345", "title": "Ha-Eshkol", "hebrew": "האשכול",
     "dates": "1898-1913", "place": "Krakau", "type": "Literary/scientific",
     "date_filter": (1850, 1900)},
    # DROPPED: Zionisten-Kongress Protokoll — no volume records in METS, primarily German
    # Pre-1850 Haskalah — no date filter, take all
    {"cm_id": "9582265", "title": "Ha-Meʾasef", "hebrew": "המאסף",
     "dates": "1783-1811", "place": "Berlin/Breslau/Königsberg",
     "type": "First Hebrew Haskalah journal", "date_filter": None},
    # DROPPED: Bikkūrē hā-'ittīm — image-only PDF, no OCR text layer
]

# Common METS/MODS namespaces
NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "mets": "http://www.loc.gov/METS/",
    "mods": "http://www.loc.gov/mods/v3",
    "xlink": "http://www.w3.org/1999/xlink",
}


def fetch_mets(cm_id: str) -> ET.Element:
    """Fetch METS record for a single CM identifier via OAI-PMH GetRecord."""
    params = {
        "verb": "GetRecord",
        "metadataPrefix": "mets",
        "identifier": f"oai:sammlungen.ub.uni-frankfurt.de/cm:{cm_id}",
    }
    resp = requests.get(OAI_BASE, params=params, timeout=30)
    resp.raise_for_status()
    return ET.fromstring(resp.content)


def parse_struct_map(root: ET.Element) -> list[dict]:
    """Extract structural divisions from METS structMap (logical or physical)."""
    divisions = []
    for struct_map in root.iter(f"{{{NS['mets']}}}structMap"):
        map_type = struct_map.get("TYPE", "unknown")
        for div in struct_map.iter(f"{{{NS['mets']}}}div"):
            div_info = {
                "map_type": map_type,
                "type": div.get("TYPE", ""),
                "label": div.get("LABEL", ""),
                "order": div.get("ORDER", ""),
                "id": div.get("ID", ""),
            }
            # Count file pointers under this div
            fptrs = div.findall(f"{{{NS['mets']}}}fptr")
            div_info["file_pointer_count"] = len(fptrs)
            divisions.append(div_info)
    return divisions


def parse_file_sec(root: ET.Element) -> dict:
    """Extract file groups and counts from METS fileSec."""
    file_groups = {}
    for fg in root.iter(f"{{{NS['mets']}}}fileGrp"):
        use = fg.get("USE", "unknown")
        files = fg.findall(f"{{{NS['mets']}}}file")
        urls = []
        for f in files[:3]:  # sample up to 3 URLs
            for loc in f.findall(f"{{{NS['mets']}}}FLocat"):
                url = loc.get(f"{{{NS['xlink']}}}href", "")
                if url:
                    urls.append(url)
        file_groups[use] = {
            "count": len(files),
            "sample_urls": urls,
        }
    return file_groups


def parse_mods_dates(root: ET.Element) -> list[str]:
    """Extract date information from embedded MODS metadata."""
    dates = []
    for date_el in root.iter(f"{{{NS['mods']}}}dateIssued"):
        if date_el.text:
            dates.append(date_el.text.strip())
    for date_el in root.iter(f"{{{NS['mods']}}}dateCreated"):
        if date_el.text:
            dates.append(date_el.text.strip())
    return dates


def count_pages(divisions: list[dict]) -> int:
    """Count page-level divisions in structMap."""
    return sum(1 for d in divisions if d["type"].lower() in ("page", "physicalpage"))


def count_issues(divisions: list[dict]) -> int:
    """Count issue/volume-level divisions."""
    return sum(
        1 for d in divisions
        if d["type"].lower() in ("issue", "volume", "year", "band", "heft", "jahrgang")
    )


def explore_all(output_path: str = "data/compact_memory/cm_catalog.json"):
    """Fetch and parse METS for all 13 target periodicals."""
    catalog = {}

    for target in TARGETS:
        cm_id = target["cm_id"]
        print(f"\n{'='*60}")
        print(f"Fetching: {target['title']} ({target['hebrew']}) — CM ID: {cm_id}")
        print(f"{'='*60}")

        try:
            root = fetch_mets(cm_id)

            # Check for OAI-PMH error
            error = root.find(f"{{{NS['oai']}}}error")
            if error is not None:
                print(f"  OAI ERROR: {error.get('code')} — {error.text}")
                catalog[cm_id] = {**target, "status": "error",
                                  "error": f"{error.get('code')}: {error.text}"}
                time.sleep(1)
                continue

            divisions = parse_struct_map(root)
            file_groups = parse_file_sec(root)
            dates = parse_mods_dates(root)

            entry = {
                **target,
                "status": "ok",
                "mods_dates": dates,
                "struct_map_summary": {
                    "total_divisions": len(divisions),
                    "page_count": count_pages(divisions),
                    "issue_count": count_issues(divisions),
                    "div_types": list(set(d["type"] for d in divisions if d["type"])),
                },
                "file_groups": file_groups,
                "top_level_divisions": [
                    d for d in divisions
                    if d["map_type"] == "LOGICAL"
                    and d["type"] not in ("page", "physicalpage", "")
                ][:20],  # first 20 logical divisions for review
            }

            # Print summary
            s = entry["struct_map_summary"]
            print(f"  Pages: {s['page_count']}, Issues: {s['issue_count']}")
            print(f"  Division types: {s['div_types']}")
            print(f"  File groups: {list(file_groups.keys())}")
            for fg_name, fg_info in file_groups.items():
                print(f"    {fg_name}: {fg_info['count']} files")
                for url in fg_info["sample_urls"][:1]:
                    print(f"      e.g. {url}")
            if dates:
                print(f"  MODS dates: {dates[:5]}")

            catalog[cm_id] = entry

        except requests.RequestException as e:
            print(f"  REQUEST ERROR: {e}")
            catalog[cm_id] = {**target, "status": "request_error", "error": str(e)}
        except ET.ParseError as e:
            print(f"  XML PARSE ERROR: {e}")
            catalog[cm_id] = {**target, "status": "parse_error", "error": str(e)}

        time.sleep(1)  # rate limit

    # Save catalog
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)

    print(f"\n\nCatalog saved to {output_path}")
    print(f"Total periodicals processed: {len(catalog)}")
    ok = sum(1 for v in catalog.values() if v.get('status') == 'ok')
    print(f"Successful: {ok}, Errors: {len(catalog) - ok}")

    return catalog


if __name__ == "__main__":
    explore_all()
