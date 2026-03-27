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
