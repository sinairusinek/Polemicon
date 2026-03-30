# Phase 1 Report: Compact Memory METS Exploration

**Date:** 2026-03-29
**Method:** OAI-PMH GetRecord (metadataPrefix=mets) via `https://sammlungen.ub.uni-frankfurt.de/cm/oai/`

---

## Summary

13 Hebrew-tagged periodicals from Compact Memory were evaluated for inclusion in the Polemicon corpus. All 13 periodical-level METS records were fetched successfully. Volume-level IDs were extracted from embedded `vl:sourceinfo` elements, and text availability was tested via PDF download + pdfplumber extraction.

**Key finding:** Text is available exclusively through PDF download (`/download/pdf/{volume_id}`). No ALTO XML, plaintext, or full-text search endpoints are functional. The PDFs contain OCR text layers with legible but noisy Hebrew — comparable to JPRESS quality before cleaning.

**Result:** 8 periodicals retained (49 volumes in date window + 11 pre-1850). 5 periodicals dropped.

---

## All Periodicals Evaluated

### Retained (8 periodicals, 49 volumes in date window + 11 pre-1850)

| # | Title | Hebrew | Dates | Place | CM ID | Type | Volumes (in window) |
|---|-------|--------|-------|-------|-------|------|---------------------|
| 1 | Kokhve Yitsḥaḳ | כוכבי יצחק | 1845-1873 | Wien | 4785731 | Literary almanac | 20 (1850-1873) |
| 2 | Ōṣar neḥmād | אוצר נחמד | 1856-1863 | Wien | 9582285 | Letters on faith/wisdom | 4 |
| 3 | Jeschurun (Hebr. Abth.) | ישורון | 1857-1878 | Lemberg | 3769475 | Hebrew academic journal | 6 |
| 6 | Pardes | פרדס | 1892-1896 | Odessa | 8003959 | Literary collection | 3 |
| 7 | Aḥiasaf | אחיאסף | 1893-1924 | Warsaw | 10719318 | Literary almanac | 8 (1893-1900) |
| 9 | Luaḥ erets Yiśraʾel | לוח ארץ ישראל | 1895-1916 | Jerusalem | 4789469 | Practical+literary almanac | 6 (1895-1900) |
| 10 | Ha-Eshkol | האשכול | 1898-1913 | Krakau | 3773345 | Literary/scientific | 3 (1898-1900) |
| 12 | Ha-Meʾasef | המאסף | 1783-1811 | Berlin/Breslau/Königsberg | 9582265 | First Hebrew Haskalah journal | 11 (all, pre-1850) |
| 13 | Bikkūrē hā-'ittīm | בכורי העתים | 1820-1832 | Wien | 4782723 | Hebrew literary almanac | 12 (all, pre-1850) |

### Dropped (5 periodicals)

#### #4 — Jeschurun (Dt. Abth.) | CM ID 3768355

- **Dates:** 1857-1878 (Lemberg), 9 volumes
- **Reason for dropping:** This is the German-language section of Jeschurun. The Hebrew section (#3, CM 3769475) is retained separately. Since the Polemicon corpus targets Hebrew-language texts, the German section would not contribute usable material. Including it would introduce predominantly German text that would need to be filtered out during cleaning, adding complexity with no analytical payoff.

#### #5 — Der ungarische Israelit | CM ID 9570383

- **Dates:** 1874-1908 (Budapest), 5 volumes found (1878-1882)
- **Reason for dropping:** This is a bilingual German-Hungarian newspaper with limited Hebrew content. Only 5 volumes were found in the METS metadata, all within a narrow 1878-1882 window — far fewer than the full 1874-1908 run, suggesting incomplete digitization. The combination of minimal Hebrew content, partial coverage, and bilingual noise makes it a poor candidate for a Hebrew corpus.

#### #11 — Zionisten-Kongress Protokoll | CM ID 3476254

- **Dates:** 1897-1937 (Wien), 0 volumes found
- **Reason for dropping:** No volume-level records were found in the METS metadata. Unlike all other periodicals, the Zionisten-Kongress record had no `vl:sourceinfo` entries for individual volumes, meaning there are no downloadable units. This is likely because the congress proceedings are structured differently in the CM system (as a series of standalone documents rather than as a periodical with volumes). The proceedings are also primarily in German, with Hebrew limited to occasional addresses. Further investigation could potentially locate the individual protocol documents, but the effort is not justified given the marginal Hebrew content.

#### #8 — Mim-mizraḥ ū-mim-maʿarāv | CM ID 4861829

- **Dates:** 1893-1899 (Berlin/Wien), 4 volumes
- **Reason for dropping:** PDFs are image-only scans with no OCR text layer. Text extraction via pdfplumber returns zero characters across all 162 pages of vol. 1. Including this periodical would require running Hebrew OCR (e.g., Tesseract with Hebrew model or Kraken), which is a significant scope expansion. Deferred for potential future OCR phase. See `future_ocr_plan.md`.

#### #13 — Bikkūrē hā-'ittīm | CM ID 4782723

- **Dates:** 1820-1832 (Wien), 12 volumes
- **Reason for dropping:** PDFs are image-only scans with no OCR text layer. Text extraction via pdfplumber returns zero characters across all 210 pages of vol. 1. Same situation as Mim-mizraḥ. This is particularly unfortunate as Bikkūrē hā-'ittīm is a key early Haskalah literary almanac and would have been valuable for the pre-1850 portion of the corpus. Deferred for potential future OCR phase. See `future_ocr_plan.md`.

---

## Technical Findings

### Access Methods Tested

| Method | Endpoint | Result |
|--------|----------|--------|
| OAI-PMH (periodical METS) | `/cm/oai/?verb=GetRecord&metadataPrefix=mets&identifier=oai:...:{cm_id}` | Works. Returns volume-level structure with IDs, dates, labels. |
| OAI-PMH (volume METS) | Same endpoint with volume ID | Fails (`cannotDisseminateFormat`). Volume IDs are not valid OAI identifiers. |
| IIIF Manifest | `/i3f/v20/{volume_id}/manifest` | Works. Returns canvas (page) list with image service URLs. No text annotations. |
| PDF Download | `/download/pdf/{volume_id}` | Works. Full volume PDFs with OCR text layer. 18-75 MB per volume. |
| ALTO XML | `/download/webcache/350/{page_id}` | 404. Not available. |
| Fulltext endpoint | `/download/fulltext/{page_id}` | 404. Not available. |
| Full-text search | `/proto_ftsearch/{volume_id}` | 404. Not available. |
| Volume-level METS (non-CM OAI) | `/oai?verb=GetRecord&metadataPrefix=mets&identifier={volume_id}` | Returns empty file groups. |

### OCR Quality Samples

**Kokhve Yitsḥaḳ vol. 5 (1850), page 1:**
```
קחצי יבכוכ
ל ל ו כ ^ -
ימיענמו רודה ימכחמ הצילמ ידליו רקחמ יימ
```
- Hebrew text present but word order reversed (RTL not handled by OCR)
- Some garbled characters and stray Latin marks
- German sections on later pages are significantly cleaner

**Pardes vol. 1 (1892), page 11:**
```
ימיב ,הנושארב .התויה םוימ השדחה ונתורפס הרחב ןורחאה הזה ךררבו
המינפ וחורלו ,תיפוריא הרוצ םינוציחה ונמע ייחל תתל התמגמ לכ התיה "הלכשהה
```
- Longer Hebrew runs with better coherence
- Russian censor page detected (Cyrillic text on page 4)
- Later periodicals (1890s) generally have better OCR than earlier ones (1850s)

### Volume ID Extraction

Volume IDs are embedded in the periodical METS as `<vl:sourceinfo>` elements within `<mets:dmdSec>` sections:
```xml
<vl:sourceinfo caption="1850" type="volume" id="4786146"/>
```

Some volumes (especially earlier ones like vol. 1 of Kokhve Yitsḥaḳ) lack `vl:sourceinfo` but have `<mods:recordIdentifier source="local">ubffm-server:{id}</mods:recordIdentifier>` as fallback.

Full volume ID mapping saved to `data/compact_memory/volume_map.json`.

---

## Files Produced

| File | Description |
|------|-------------|
| `data/compact_memory/cm_catalog.json` | Full METS metadata for all 13 periodicals |
| `data/compact_memory/volume_map.json` | Volume IDs, dates, and captions for all 118 volumes |
| `data/compact_memory/sample_mets.xml` | Raw METS XML for Kokhve Yitsḥaḳ (reference) |
| `data/compact_memory/sample_manifest.json` | IIIF manifest for Kokhve Yitsḥaḳ vol. 5 (reference) |
| `data/compact_memory/sample_volume.pdf` | Sample PDF: Kokhve Yitsḥaḳ vol. 5 (1850) |
| `data/compact_memory/sample_pardes_8003960.pdf` | Sample PDF: Pardes vol. 1 (1892) |
| `data/compact_memory/sample_pardes_8004266.pdf` | Sample PDF: Pardes vol. 2 (1894) |

---

## Next Steps (Phase 2)

1. Download 2-3 volumes from diverse periodicals (early/late, different publishers)
2. Extract full text via pdfplumber
3. Run `compute_quality_score()` from `src/cleaning.py` and compare to JPRESS baseline
4. Present sample text for manual review
5. Set minimum quality thresholds for inclusion
