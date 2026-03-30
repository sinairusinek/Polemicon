"""Quick probe: for each periodical, download first ~5 pages of one volume
and check if any text is extractable. Avoid downloading full PDFs."""

import json
import sys
import time
from pathlib import Path
from io import BytesIO

import pdfplumber
import requests

with open("data/compact_memory/volume_map.json") as f:
    volume_map = json.load(f)

results = []

for cm_id, info in volume_map.items():
    title = info["title"]
    volumes = info["volumes"]
    if not volumes:
        print(f"{title}: no volumes, skipping")
        results.append({"cm_id": cm_id, "title": title, "has_text": None, "note": "no volumes"})
        continue

    # Pick first volume
    vol = volumes[0]
    vid = vol["id"]
    print(f"\n{title} — vol {vol['caption']} ({vol['date']}), ID={vid}")

    url = f"https://sammlungen.ub.uni-frankfurt.de/download/pdf/{vid}"
    try:
        # Download with streaming, read only first chunk
        resp = requests.get(url, timeout=60, stream=True)
        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code}")
            results.append({"cm_id": cm_id, "title": title, "has_text": None,
                           "note": f"HTTP {resp.status_code}"})
            continue

        # Read full PDF (we need it for pdfplumber to parse)
        content = resp.content
        print(f"  Downloaded {len(content)/1e6:.1f} MB")

        with pdfplumber.open(BytesIO(content)) as pdf:
            total_pages = len(pdf.pages)
            text_found = False
            sample_text = ""
            for i in range(min(10, total_pages)):
                text = pdf.pages[i].extract_text() or ""
                if len(text) > 50:
                    text_found = True
                    sample_text = text[:150]
                    break

            print(f"  Pages: {total_pages}, Text found: {text_found}")
            if text_found:
                print(f"  Sample: {sample_text}")
            results.append({
                "cm_id": cm_id, "title": title, "volume_id": vid,
                "has_text": text_found, "pages": total_pages,
                "sample": sample_text if text_found else "",
            })

    except Exception as e:
        print(f"  Error: {e}")
        results.append({"cm_id": cm_id, "title": title, "has_text": None, "note": str(e)})

    time.sleep(2)

print("\n\n=== SUMMARY ===")
for r in results:
    status = "TEXT" if r.get("has_text") else "IMAGE-ONLY" if r.get("has_text") is False else "ERROR"
    print(f"  {status:12s} | {r['title']}")
