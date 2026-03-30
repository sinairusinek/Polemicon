# Future Plan: OCR for Image-Only CM Periodicals

**Status:** Deferred (as of 2026-03-29)
**Reason for deferral:** Scope management — the current CM acquisition phase focuses on periodicals with existing OCR text layers.

---

## Periodicals Requiring OCR

| Title | Hebrew | Dates | Place | CM ID | Volumes | Pages (sample) |
|-------|--------|-------|-------|-------|---------|----------------|
| Mim-mizraḥ ū-mim-maʿarāv | ממזרח וממערב | 1893-1899 | Berlin/Wien | 4861829 | 4 | 162 (vol 1) |
| Bikkūrē hā-'ittīm | בכורי העתים | 1820-1832 | Wien | 4782723 | 12 | 210 (vol 1) |

**Estimated total pages:** ~2,000-3,000 (16 volumes of literary almanacs, ~150-250 pages each)

## Why These Matter

- **Bikkūrē hā-'ittīm** (1820-1832) is one of the two foundational Haskalah Hebrew literary almanacs (alongside Ha-Meʾasef, which does have OCR). It was the primary Hebrew literary periodical in the post-Ha-Meʾasef generation. Without it, the pre-1850 corpus has only one journal voice.
- **Mim-mizraḥ ū-mim-maʿarāv** (1893-1899) was edited by scholars in Berlin and Wien, bridging Eastern and Western European Hebrew intellectual culture. It fills a geographic and intellectual gap in the 1890s coverage.

## Technical Approach

### Option A: Tesseract with Hebrew Model
- **Pros:** Free, local, well-documented
- **Cons:** Tesseract's Hebrew model struggles with 19th-century typefaces (especially Rashi script variants and dense typographical layouts). Would likely need fine-tuning or post-correction.
- **Steps:**
  1. Extract page images from PDFs (or download via IIIF image API)
  2. Pre-process: binarize, deskew, denoise
  3. Run Tesseract with `--oem 1 -l heb` (LSTM engine)
  4. Post-process: basic error correction patterns

### Option B: Kraken/eScriptorium
- **Pros:** Designed for historical scripts, trainable on specific typefaces, better handling of RTL and mixed scripts
- **Cons:** Requires training data (ground truth transcriptions), more setup
- **Steps:**
  1. Manually transcribe 20-50 pages as ground truth
  2. Train Kraken model on the specific typefaces used
  3. Run batch OCR
  4. Evaluate CER (character error rate)

### Option C: Cloud OCR (Google Vision, Azure)
- **Pros:** Best out-of-the-box accuracy for Hebrew, no local setup
- **Cons:** Cost (~$1.50/1000 pages for Google Vision), data leaves local machine
- **Steps:**
  1. Extract page images
  2. Submit to API in batches
  3. Parse responses
- **Estimated cost:** ~$3-5 for all pages

### Option D: LLM-assisted OCR
- **Pros:** Multimodal models (Claude, GPT-4V) can read historical Hebrew typefaces with context awareness
- **Cons:** Expensive at scale (~$0.05-0.10 per page with vision), slow
- **Steps:**
  1. Extract page images
  2. Submit to vision API with prompt for Hebrew transcription
  3. Post-process
- **Estimated cost:** ~$100-300 for all pages — not practical for bulk OCR, but could be used for spot-checking or ground truth creation

## Recommendation

If/when this phase is activated:
1. Start with **Option C (Google Vision)** for a quick, cheap baseline (~$5)
2. Evaluate quality against the OCR text we already have from other CM periodicals
3. If quality is insufficient, invest in **Option B (Kraken)** with ground truth from the Google Vision output + manual corrections

## Access

Page images can be retrieved via the IIIF Image API without downloading full PDFs:
```
https://sammlungen.ub.uni-frankfurt.de/i3f/v20/{page_id}/full/full/0/default.jpg
```
Page IDs are available from the IIIF manifest at:
```
https://sammlungen.ub.uni-frankfurt.de/i3f/v20/{volume_id}/manifest
```
