"""Regenerate volume_map.json for the retained periodicals."""

import json
import requests
import xml.etree.ElementTree as ET

NS_VL = "http://visuallibrary.net/vl"
NS_METS = "http://www.loc.gov/METS/"
NS_MODS = "http://www.loc.gov/mods/v3"
OAI_BASE = "https://sammlungen.ub.uni-frankfurt.de/cm/oai/"

with open("data/compact_memory/cm_catalog.json") as f:
    catalog = json.load(f)

volume_map = {}

for cm_id, entry in catalog.items():
    if entry.get("status") != "ok":
        continue
    print(f"{entry['title']} ({cm_id}):")

    resp = requests.get(OAI_BASE, params={
        "verb": "GetRecord", "metadataPrefix": "mets",
        "identifier": f"oai:sammlungen.ub.uni-frankfurt.de/cm:{cm_id}",
    }, timeout=30)
    root = ET.fromstring(resp.content)

    volumes = []
    for dmd_sec in root.iter(f"{{{NS_METS}}}dmdSec"):
        for si in dmd_sec.iter(f"{{{NS_VL}}}sourceinfo"):
            if si.get("type") == "volume":
                vid = si.get("id")
                caption = si.get("caption", "")
                date = ""
                for d in dmd_sec.iter(f"{{{NS_MODS}}}date"):
                    date = d.text or ""
                    break
                volumes.append({"id": vid, "caption": caption, "date": date})

        if not any(True for si in dmd_sec.iter(f"{{{NS_VL}}}sourceinfo")):
            for ri in dmd_sec.iter(f"{{{NS_MODS}}}recordIdentifier"):
                if ri.get("source") == "local" and ri.text:
                    vid = ri.text.replace("ubffm-server:", "")
                    date = ""
                    for d in dmd_sec.iter(f"{{{NS_MODS}}}date"):
                        date = d.text or ""
                        break
                    number = ""
                    for n in dmd_sec.iter(f"{{{NS_MODS}}}number"):
                        number = n.text or ""
                        break
                    if date or number:
                        volumes.append({"id": vid, "caption": f"vol {number}", "date": date})

    # Apply date filter
    date_filter = entry.get("date_filter")
    if date_filter:
        lo, hi = date_filter
        filtered = []
        for v in volumes:
            try:
                year = int(v["date"][:4])
                if lo <= year <= hi:
                    filtered.append(v)
            except (ValueError, IndexError):
                filtered.append(v)  # keep if date unparseable
        volumes = filtered

    volumes.sort(key=lambda x: x.get("date", ""))
    for v in volumes:
        print(f"  {v['id']}: {v['caption']} ({v['date']})")

    volume_map[cm_id] = {
        "title": entry["title"],
        "hebrew": entry.get("hebrew", ""),
        "date_filter": date_filter,
        "volumes": volumes,
    }
    import time; time.sleep(1)

with open("data/compact_memory/volume_map.json", "w") as f:
    json.dump(volume_map, f, ensure_ascii=False, indent=2)

total = sum(len(v["volumes"]) for v in volume_map.values())
print(f"\nSaved volume map: {len(volume_map)} periodicals, {total} volumes")
