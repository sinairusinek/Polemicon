# Polemicon: Hebrew Polemic Corpus Analysis Pipeline

## Status Update (2026-03-29)

### Completed
- **Phase A (Corpus + Vectorization):** TF-IDF vectorization (char 3-5 grams + word 1-2 grams, 50K+30K features), TruncatedSVD to 300 dims, FAISS index. Outlier bypc_5539 (317K-word academic book) dropped. 33,513 texts.
- **Phase B.3 (Keyword baseline):** Hebrew polemic lexicon scoring. Weak signal alone (mean 0.10, only 6 texts >0.3). Useful as supplementary feature.
- **Phase C.1 (Clustering):** UMAP + HDBSCAN → 409 clusters, 41% noise. 2D UMAP coords saved for future visualization.
- **Pilot sample:** 200 texts stratified by source/score/cluster → `data/pilot_sample.parquet`
- **Streamlit app:** Rewritten with real corpus data, navigation, filters, annotation UI. Deployed on Streamlit Cloud.
- **Phase B.2 (4-model LLM classification pilot):** All 4 models (Claude Opus, Sonnet, Gemini Pro 2.5, Flash 2.5) ran on 200 pilot texts. 0 API errors, 4 minor parse errors. Key finding: Claude models (~24% polemic rate) and Gemini models (~70%) diverge sharply — within-family agreement ~90%, cross-family ~53%. 37 unanimous polemic, 49 unanimous not, 94 expensive-model disagreements queued for human review. See `data/agreement_report.txt`.
- **Metadata backfill:** Enriched `corpus.parquet` with newspaper codes (23,444 press texts), authors (9,811 texts: 100% e-geret, 96.5% polemic candidates), recipients (2,359 e-geret), headlines (17,319 press), and 1,622 press intertextual references.
- **Reference extraction:** Two-layer approach on pilot texts. Mechanical: 6,959 references (newspaper mentions, attribution patterns, footnotes) across all 200 texts. LLM (Sonnet): 562 categorized references across 56 texts with 3+ polemic model votes — biblical (197), contemporary persons (112), Talmudic (76), contemporary publications (46), contemporary texts (44), scholarly (22). 8 `response_to` references are direct thread signals. All displayed in Streamlit app.
- **Display fix:** `restore_final_forms()` in cleaning.py reverses final-form normalization (ם→מ etc.) for readable Hebrew display. Applied in Streamlit app.
- **Vocab extraction (2026-03-29):** Sonnet re-ran on 94 disagreement texts with prompt asking for 3-5 polemic marker words/phrases per text. 460 markers extracted (459 unique). Stored in `data/pilot_vocab.parquet`. Displayed in Streamlit app with per-marker approve/reject buttons for reviewer curation.
- **Reviewer comments (2026-03-29):** Free-text comment field added to Streamlit annotation panel. Comments stored per doc_id, exported in annotations CSV.
- **Keyword export updated (2026-03-29):** CSV export now includes human keyword suggestions + model-suggested vocabulary with approved/rejected status.
- **Metadata display in Streamlit (2026-03-29):** Full metadata (author, recipient, headline, newspaper, title) merged from `corpus.parquet` into pilot texts at load time. Two-row metadata bar shows only non-null fields per text. Ben-Yehuda Project links generated for `bypc_*` doc IDs.
- **Cluster characterization (2026-03-29):** Top 10 distinctive TF-IDF terms per cluster computed (cluster mean minus corpus mean). 409 clusters labeled in `data/cluster_labels.parquet`. Terms displayed in annotation app metadata bar.
- **Cluster visualization (2026-03-29):** Interactive UMAP scatter plot page (`src/pages/Cluster_Map.py`) showing all 33,513 texts. Color by cluster/source/keyword score. Cluster selector highlights points and shows top terms, size, source breakdown. Sortable table of all 409 clusters.

### Next: Human review + calibration
- **Priority 1:** Researcher reviews the 94 disagreement cases (now with vocab markers, metadata, cluster terms, and comments) to establish gold labels.
- **Priority 2:** Based on review, decide model selection for 2K calibration run.
- **After review:** Phase B.2 calibration run (2,000 texts) with the chosen model(s).
- **Optional:** Sonnet-generated topic labels for top 20 clusters (beyond keyword-based labels).

### Deferred
- Dense embeddings (Tier 2/3) — only if TF-IDF clustering proves insufficient
- Phase C.2 (thread detection) — approach discussed: tiered edge signals (strong/medium/weak), expect loose topical clusters with partial name/vocab overlap rather than strict reply chains. Will be more effective after polemic labels are established.
- Full-corpus reference extraction — current approach validated on 56 polemic pilot texts; scale to all polemic texts after B.4 classification.

## Context

The project contains ~80K Hebrew texts across 4 datasets (19th-century press articles, letters, polemic candidates) with near-zero polemic labels. The goal is to build a vectorized corpus from the overlap period, classify polemic texts, and detect debate threads. The main challenges are: OCR noise in the press dataset, absence of training labels, and linking texts across sources/media.

---

## Phase 0: Environment Setup

Create a dedicated conda environment with all required packages:

```
conda create -n polemicon python=3.11 -y && conda activate polemicon
pip install pandas numpy tqdm matplotlib seaborn plotly
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentence-transformers
pip install scikit-learn hdbscan umap-learn faiss-cpu
pip install anthropic jupyter ipykernel
```

**Key model choices:**
| Model | Purpose |
|-------|---------|
| `intfloat/multilingual-e5-large` | Dense sentence embeddings (1024-dim, strong Hebrew support) |
| `onlplab/alephbert-base` | Hebrew-specific fine-tuning if we bootstrap enough labels |
| Claude Sonnet via API | Zero-shot polemic classification (critical path given no labels) |
| TF-IDF char n-grams (3-5) | OCR-robust sparse features |

---

## Phase A: Corpus Construction & Vectorization

### A.1 Data loading (`src/loaders.py`)

Each dataset needs a dedicated loader:

- **Press** (`MGD-LBN-MLZ-HZF-HZTfull2021-08-14-(1)-tsv.csv`): 69,391 articles, 1862-1888, 5 newspapers. CSV with large text fields -- needs `csv.field_size_limit` adjustment.
- **E-Geret** (`e-geret-batch-export.tsv`): 4,720 letters with full text, sender/recipient metadata. TSV, BOM-encoded. ~544 fall in the 1862-1888 overlap window.
- **Polemic candidates** (`Ben-Yehuda-Project-polemic-candidates.csv`): 8,044 texts with full text in "Column 1". 34 items have explicit press cross-references (`אזכור מכ״ע` column). Dates are sparse (only 11 have dates).
- **Ben-Yehuda metadata** (`benyehuda-full-metadata.csv`): 60,207 items, metadata only (no text). Used to cross-reference and recover dates for polemic candidates.

**Date inference for polemic candidates:** Cross-reference the `File` column (e.g., `polemic-candidates/p819/m40830.txt`) with Ben-Yehuda metadata by extracting the numeric ID (e.g., 40830) and matching against the `id` column in `benyehuda-full-metadata.csv`. This should recover dates for most of the 8,044 candidates. Only texts confirmed in the 1862-1888 overlap window (or with no recoverable date but strong textual evidence of the period) will enter the corpus.

### A.2 Text cleaning (`src/cleaning.py`)

1. Strip Ben-Yehuda project footer boilerplate from e-geret and candidate texts
2. Normalize Hebrew: remove nikkud (`[\u0591-\u05C7]`), normalize final forms, standardize punctuation
3. Detect and flag non-Hebrew segments (Latin/German mastheads in press OCR)
4. Compute per-text quality score (Hebrew character ratio, average word length)
5. Minimum length filter: discard texts < 200 Hebrew words

### A.3 Overlap identification & unified corpus (`src/corpus.py`)

**Overlap window: 1862-1888** (the press dataset's full range).

Build a single DataFrame with unified schema:

| Column | Description |
|--------|-------------|
| `doc_id` | Prefixed unique ID (`press_`, `egeret_`, `bypc_`) |
| `source` | `press` / `egeret` / `polemic_candidates` |
| `text` | Cleaned full text |
| `date`, `year` | Publication date (nullable for candidates) |
| `author`, `title`, `genre` | Metadata |
| `newspaper` | Newspaper code (press only) |
| `quality_score` | OCR quality estimate |
| `in_overlap` | Whether year falls in 1862-1888 |
| `press_crossref` | Newspaper reference from polemic candidates |

Save as `corpus.parquet`. Expected size: ~77K-78K texts.

### A.4 Vectorization (`src/vectorize.py`)

**Dual representation:**

1. **Dense:** `multilingual-e5-large` embeddings. Sliding window (512 tokens, stride 256) + mean pool for long texts. Batch size 32. Store as numpy memmap + FAISS `IndexFlatIP`.
2. **Sparse:** Character n-gram TF-IDF (`char_wb`, 3-5 grams, 50K features) -- deliberately character-level for OCR robustness. Also word-level TF-IDF (1-2 grams, 30K features).

**Estimated time:** ~30 min for embeddings on Apple Silicon.

### A.5 Verification
- Spot-check 20 texts per source after cleaning
- Verify all press dates in 1862-1900; e-geret dates in expected range
- Cosine similarity between same-author texts should be > 0.7
- The 34 cross-referenced polemic candidates should have high similarity to press articles from the same newspaper

---

## Phase B: Polemic Classification

### B.1 Define "polemic" operationally

A polemic text: engages in argumentative debate, responds to or attacks another writer's position, defends a stance against criticism, or participates in a public intellectual dispute. Key markers: direct address to opponents, rhetorical questions, evaluative language, intertextual references.

### B.2 Claude API zero-shot classification (bootstrap labels)

Since there are effectively zero labels, Claude bootstraps the initial labeled set.

**Staged rollout:**
1. **Pilot (200 texts):** 50 from each source. Researcher manually reviews Claude's output. Iterate on prompt until Cohen's kappa > 0.6.
2. **Calibration (2,000 texts):** Stratified sample (~$18 with Sonnet). Estimate polemic prevalence, set confidence thresholds. Researcher reviews a subset to create gold labels.

**Classification output per text:** `is_polemic`, `confidence`, `polemic_type` (attack/defense/debate/satire/critique), `target`, `evidence`, `topic`.

### B.3 Hebrew keyword baseline (parallel with B.2)

Rule-based scoring using a polemic indicator lexicon:
- Debate markers: אך, אבל, אולם, להפך, חלילה
- Address markers: השיב, ענה, טען, כתב
- negative superlatives: איום, נורא, מרעיש
- sarcastive superlatives: נאצל, נכבד, גבוה, רם
- Evaluative intensifiers: שקר, כזב, הבל, טעות, סכלות, טמא, עוון, חטא, חשוך, רפה, זוהמה, מרעיש, שערוריה, צבוע, מתחסד
- Rhetorical question density (`?`, `?!`) and quotation density ('')
- Named references to other writers/newspapers

Use as comparison baseline and supplementary feature for fine-tuning.

### B.4 Fine-tune AlephBERT (primary classifier for full corpus)

This is the core classification step -- Claude bootstraps labels, AlephBERT scales:

1. Combine Claude's 2K silver labels + researcher's gold labels (target: 500-1000 human-reviewed)
2. Fine-tune `onlplab/alephbert-base` with a classification head for binary polemic detection
3. Add keyword baseline scores as auxiliary features (hybrid model)
4. Train with 5-fold cross-validation; evaluate on gold set (target F1 > 0.8)
5. **Run fine-tuned model on the full corpus** -- fast and free compared to API calls
6. For uncertain predictions (0.3-0.7 confidence), optionally send to Claude for a second opinion

**Cost:** ~$18 for Claude bootstrap + compute time for fine-tuning (minutes on CPU). Much cheaper than $700 for full API classification.

### B.5 Verification
- Researcher labels 100 texts, compare to Claude: target kappa > 0.6
- Polemic prevalence in press should be ~10-30%
- The 34 press cross-reference items should be flagged as polemic at high rates

---

## Phase C: Clustering & Thread Detection

### C.1 Topic clustering

1. Filter to polemic + uncertain texts
2. UMAP reduction (1024-dim -> 50-dim, cosine metric)
3. HDBSCAN clustering (`min_cluster_size=10`) -- no need to specify K
4. Characterize clusters: top TF-IDF terms + Claude-generated summaries of central texts
5. 2D UMAP visualization colored by cluster

### C.2 Thread detection (polemic chains)

Build a directed graph within each cluster:
1. **Explicit references:** Parse `intertextual reference` column; detect newspaper name mentions in letter content
2. **Author-pair interleaving:** If A and B both write in the same cluster with alternating dates -> debate signal
3. **Semantic reply detection:** Nearest neighbors in same cluster, published earlier, by different author, cosine sim > 0.85
4. **Cross-source linking:** Use the 34 known cross-references as seeds; detect newspaper mentions in e-geret letters

**Thread construction:** Connected components in the graph, ordered by date. Score threads by: participant count, temporal span, number of exchanges.

### C.3 Visualization
- Timeline of polemic threads (Plotly)
- Author network graph (nodes=authors, edges=polemic exchanges)
- Topic evolution over 1862-1888 (stacked area chart)

### C.4 Verification
- Cluster silhouette score; manual review of 5 largest clusters
- Known cross-references recovered as thread links (target > 50%)
- Temporal gaps within threads: median < 6 months
- Threads involve small author groups (2-5), not random noise

---

## Project Structure

```
Polemicon/
  src/
    __init__.py
    config.py          # paths, model names, hyperparameters
    loaders.py         # dataset-specific loading
    cleaning.py        # OCR cleanup, normalization
    corpus.py          # unified corpus construction + date inference
    vectorize.py       # embedding pipeline
    classify.py        # polemic classification (Claude + AlephBERT)
    cluster.py         # clustering + thread detection
  notebooks/
    01_exploration.ipynb       # data exploration & quality assessment
    02_corpus.ipynb            # corpus construction & vectorization
    03_classification.ipynb    # polemic classification pipeline
    04_clustering.ipynb        # clustering & thread visualization
  tests/
    test_loaders.py
    test_cleaning.py
  environment.yml
```

Core logic lives in `src/` modules; notebooks import from `src/` for interactive exploration and visualization.

---

## Implementation Order

```
Phase 0  ->  A.1-A.2 (load/clean)  ->  A.3 (corpus + date inference)  ->  A.4 (vectorize)
         ->  B.2 pilot (200 texts)  ->  B.2 calibration (2K texts)  ->  B.4 fine-tune AlephBERT
              B.3 keyword baseline (parallel)                         ->  B.4 classify full corpus
         ->  C.1 (clustering)  ->  C.2 (thread detection)  ->  C.3 (visualization)
```
