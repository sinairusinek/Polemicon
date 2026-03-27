# Polemicon: Hebrew Polemic Corpus Analysis Pipeline

## Status Update (2026-03-27)
- Repository reset and pushed to GitHub with only code/config files (no large data)
- .gitignore expanded to exclude all large/model files
- Streamlit app deployed from src/streamlit_app.py
- Ready for further development, annotation, or deployment steps

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
