# Vectorization Log

## Original Plan (Phase A.4 from polemiconPlan.md)

**Dual representation approach:**
1. **Dense:** `intfloat/multilingual-e5-large` embeddings (1024-dim). Sliding window (512 tokens, stride 256) + mean pool for long texts. Batch size 32. Store as numpy memmap + FAISS `IndexFlatIP`.
2. **Sparse:** Character n-gram TF-IDF (`char_wb`, 3-5 grams, 50K features) + word-level TF-IDF (1-2 grams, 30K features).

**Estimated time:** ~30 min for embeddings on Apple Silicon.

**What happened:** The dense embedding step failed when run by VS Code agents. The `multilingual-e5-large` model (~1.3GB, 560M params) was too slow and memory-intensive for 33K texts averaging 2K words each. The code also had issues: chunking used `.split()` instead of the model tokenizer, no checkpointing, no batching across documents, and no `"passage: "` prefix required by E5 models.

---

## Revised 3-Tier Plan (2026-03-27)

After evaluating the corpus (33,514 texts, mean 2,037 words, one 317K-word outlier), the vectorization was restructured into three tiers of increasing cost:

### Tier 1: TF-IDF only (implemented)
- Char n-gram TF-IDF (3-5 grams, 50K features) — OCR-robust by design
- Word n-gram TF-IDF (1-2 grams, 30K features)
- TruncatedSVD to 300 dims on combined matrix for FAISS nearest-neighbor index
- **Runtime:** ~2-3 minutes
- **Cost:** zero (local CPU only)
- **Rationale:** Sufficient for clustering (UMAP + HDBSCAN), similarity search, and thread detection. Char n-grams handle OCR noise gracefully. This alone covers Phases B and C needs.

### Tier 2: Dense embeddings with a smaller model (deferred)
- Use `intfloat/multilingual-e5-base` (768-dim, ~550MB) or `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, ~470MB)
- Add proper tokenizer-based chunking, checkpointing every 1K docs, max 20 chunks per doc
- **When to use:** Only if TF-IDF clustering isn't semantically sharp enough (e.g., texts with different vocabulary but same meaning aren't grouping)

### Tier 3: API-based embeddings (deferred)
- OpenAI `text-embedding-3-small`: ~$1.30 for full corpus (66M tokens)
- **When to use:** If local models are too slow or quality is insufficient

---

## Tier 1 Results

**Date:** 2026-03-27

**Script:** `src/vectorize.py`

**Output files:**
| File | Description |
|------|-------------|
| `char_tfidf.npz` | Sparse char n-gram TF-IDF matrix (33514 x 50000) |
| `word_tfidf.npz` | Sparse word n-gram TF-IDF matrix (33514 x 30000) |
| `tfidf_svd_300.npy` | SVD-reduced dense matrix (33514 x 300) |
| `tfidf_faiss.index` | FAISS IndexFlatIP for nearest-neighbor search |
| `doc_ids.txt` | Ordered document IDs |
| `vectorizers.joblib` | Fitted TfidfVectorizer + TruncatedSVD objects |

**Runtime:** ~21 minutes (mostly char n-gram fitting on the 317K-word outlier)

**SVD explained variance:** 28.5% (300 components) — adequate for nearest-neighbor search

**Sanity check (nearest neighbors to press_1):**
| Neighbor | Cosine sim |
|----------|-----------|
| press_98 | 0.90 |
| press_65 | 0.90 |
| press_123 | 0.90 |
| press_17 | 0.89 |
| press_120 | 0.87 |

All nearest neighbors are from the same source (press), which makes sense — same newspaper OCR characteristics cluster together.

**Notes:**
- The char_tfidf.npz file is 1.5GB — the dominant cost. Could reduce `max_features` to 30K if storage is a concern.
- 28.5% explained variance in 300 SVD dims is typical for high-dimensional sparse text data. The full sparse matrices are preserved for exact cosine similarity when needed.

### Re-vectorization (without outlier)

**Date:** 2026-03-27

The 317K-word outlier (bypc_5539: "מבואות לספרות התנאים" by Ezra Zion Melamed — an entire academic book on Tannaitic literature) was dropped. Re-vectorized 33,513 texts.

---

## Keyword Baseline (Phase B.3)

**Date:** 2026-03-27

**Script:** `src/keyword_baseline.py`

Hebrew polemic indicator lexicon: debate markers, address markers, evaluative intensifiers, rhetorical question density, quotation density. Keywords normalized to match cleaned corpus (final form normalization).

**Results:**
- Mean polemic score: 0.1037, median: 0.1020
- Only 6 texts score > 0.3, zero > 0.5
- Weak discriminative power on its own — common Hebrew words (אבל, כתב) appear in non-polemic texts too
- Directionally correct: e-geret and polemic_candidates score slightly higher than press

**Conclusion:** Keyword baseline alone cannot identify polemics. Useful as a supplementary feature for stratified sampling and the eventual fine-tuned classifier. Confirms the need for LLM classification (B.2).

---

## Clustering (Phase C.1)

**Date:** 2026-03-27

**Script:** `src/cluster.py`

UMAP (300 → 50 dim, cosine) + HDBSCAN (min_cluster_size=10) on TF-IDF SVD vectors.

**Results:**
- **409 clusters**, 13,783 noise points (41.1%)
- Largest cluster: 621 texts (cluster 180)
- Top 5 clusters: 621, 570, 514, 496, 402 texts
- 158 clusters represented in the 200-text pilot sample

**Notes:**
- 41% noise is high but expected for a heterogeneous historical corpus — many texts are unique/miscellaneous
- Could reduce noise by lowering `min_cluster_size` to 5, but this risks spurious micro-clusters
- 2D UMAP coordinates saved for visualization

---

## Pilot Sample (B.2 Prep)

**Date:** 2026-03-27

**Script:** `src/sample_pilot.py`

Stratified 200-text sample for annotation pilot:
- 100 press, 50 polemic_candidates, 50 egeret
- Stratified by keyword score quartile and cluster diversity
- 158 clusters represented
- Saved to `data/pilot_sample.parquet`

---

## Phase B.2: 4-Model LLM Classification Pilot

**Date:** 2026-03-28

**Script:** `src/classify_pilot.py`

**Models:** Claude Opus 4.6, Claude Sonnet 4.6, Gemini 2.5 Pro, Gemini 2.5 Flash (stable, not 3.x preview).

Ran all 4 models on the 200-text pilot sample. Each model classifies: `is_polemic`, `confidence`, `polemic_type` (attack/defense/debate/satire/critique), `target`, `evidence`, `topic`. Texts truncated to 4,000 words for cost control.

**Results:**

| Model | Polemic Rate | Avg Confidence | Tier |
|-------|-------------|---------------|------|
| Claude Opus | 23.5% (47/200) | 0.87 | expensive |
| Claude Sonnet | 24.1% (48/199) | 0.88 | cheap |
| Gemini Pro | 70.0% (140/200) | 0.96 | expensive |
| Gemini Flash | 69.0% (136/197) | 0.97 | cheap |

**Inter-model agreement:**
- Claude Opus vs Sonnet: 91.8% (within-family)
- Gemini Pro vs Flash: 88.8% (within-family)
- Cross-family: ~53% — massive divergence
- All 4 agree polemic: 37 texts
- All 4 agree not polemic: 49 texts
- Expensive models disagree: 94 texts (priority for human review)
- Full 4-model agreement: 43.9% (86/196)
- Polemic type agreement (among unanimous polemic): 29.7%

**Key finding:** Gemini models have a much broader definition of "polemic" than Claude. The dominant disagreement pattern is Gemini=polemic + Claude=not-polemic. Human review of the 94 disagreement cases will determine the correct threshold.

**Output files:**
| File | Description |
|------|-------------|
| `data/pilot_classifications.parquet` | 800 rows (200 texts × 4 models) |
| `data/pilot_disagreements.parquet` | Agreement category per text |
| `data/agreement_report.txt` | Full inter-model agreement analysis |

**API errors:** 0. **Parse errors:** 4 (Gemini Flash, minor).

---

## Metadata Backfill

**Date:** 2026-03-28

**Script:** `src/backfill_metadata.py`

Enriched `corpus.parquet` with metadata from source files that was not carried through during initial corpus construction:

| Field | Coverage | Source |
|-------|----------|--------|
| newspaper | 23,444 (70%) | Press: HZF, MGD, HZT, HLB, HMZ |
| author | 9,811 (29%) | E-geret: 100%, polemic candidates: 96.5%, press: 0% |
| recipient | 2,359 (7%) | E-geret letters |
| headline | 17,319 (52%) | Press articles |
| intertextual_reference | 1,622 (5%) | Press articles (pre-annotated) |

Top authors: Ahad Ha'am (456), Berl Katznelson (455), David Yellin (398), Bialik (354), Shadal (312), Y.L. Gordon (278).

Corpus now has 15 columns (was 12).

---

## Reference Extraction (Pilot)

**Date:** 2026-03-28

**Script:** `src/extract_references.py`

Two-layer intertextual reference extraction on pilot texts:

**Layer 1 — Mechanical (all 200 texts):**
- Newspaper name regex (13 newspapers + abbreviations): 6,128 hits
- Attribution pattern matching (כתב/אמר + name): 829 hits
- Footnote markers (↩ in Ben-Yehuda texts): 2 hits (low because footnote *content* needs LLM to parse)

**Layer 2 — LLM/Sonnet (56 texts with 3+ polemic model votes):**
- 562 categorized references (~10/text)
- Categories: biblical (197), contemporary person (112), Talmudic (76), other/medieval (65), contemporary publication (46), contemporary text (44), scholarly (22)
- Reference types: allusion (260), attribution (200), explicit citation (86), response_to (8), footnote (8)
- The 8 `response_to` references are direct thread signals for C.2

**Design decision:** Extract ALL reference types (not just contemporary/dialogical), categorize each. Biblical/Talmudic citations reveal rhetorical strategies — who marshals which sources in polemic arguments. This is a finding in its own right.

**Ben-Yehuda footnote patterns:** 3,311/7,457 polemic candidate texts (44%) contain footnotes (↩ markers with `&nbsp;`). Many contain publication references ("נדפס בהמליצ"), editorial notes (הערת פב"י), and scholarly citations. Rich source for full-corpus extraction later.

**Output:** `data/pilot_references.parquet` (7,521 rows). Displayed in Streamlit app with contemporary references highlighted and biblical/Talmudic in expandable section.

---

## Cluster Characterization

**Date:** 2026-03-29

**Script:** `src/cluster_top_terms.py`

For each of 409 clusters, computed top 10 distinctive TF-IDF terms using the word-level TF-IDF matrix (30K features). "Distinctive" = cluster mean TF-IDF minus corpus-wide mean — terms that characterize this cluster relative to the whole corpus, not just frequent terms.

**Output:** `data/cluster_labels.parquet` (409 rows)
- `cluster_id`, `top_terms` (JSON list of 10 terms), `n_texts`, `mean_polemic_score` (keyword baseline, not a classification label)

**Note:** `mean_polemic_score` is the keyword baseline score (Phase B.3), not a human or LLM polemic label. It reflects lexicon density, not true polemic classification.

---

## Cluster Visualization

**Date:** 2026-03-29

**Page:** `src/pages/Cluster_Map.py`

Interactive UMAP scatter plot of all 33,513 corpus texts using Plotly Scattergl (WebGL). Added as a second Streamlit page.

**Features:**
- Color modes: cluster membership, source (press/egeret/polemic_candidates), keyword score heatmap
- Cluster selector: highlights selected cluster in red, shows detail panel with top terms, size, source breakdown
- Source filter to isolate one dataset
- Sortable table of all 409 clusters with top 5 terms each
- All Hebrew terms displayed with `restore_final_forms()`
