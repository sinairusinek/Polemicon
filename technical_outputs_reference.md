# Polemicon technical outputs reference

This file summarizes the main technical outputs produced during the project, where they are located, and what each artifact was used for.

## 1. Core data products

- `data/calibration_v2.parquet`  
  Classified calibration dataset used for model validation and distribution checks.

- `data/pilot_classifications.parquet`  
  Earlier pilot classification outputs used for comparison and review.

- `data/ra_gold_labels.parquet`  
  Human-reviewed gold labels used as benchmark/reference annotations.

- `data/cluster_labels.parquet`  
  Cluster labels summarizing the main topic groupings discovered in the corpus.

- `data/cluster_assignments.parquet`  
  Assignment of documents to cluster IDs.

- `data/pilot_sample.parquet`  
  Initial pilot sample used for exploratory review and annotation.

- `data/pilot_vocab.parquet`  
  Vocabulary and keyword candidates extracted during the pilot stage.

- `data/pilot_references.parquet`  
  Reference extraction outputs for sample texts.

- `data/polemic_pool.parquet`  
  Working corpus of candidate polemical texts and related metadata.

- `data/full_corpus_predictions.parquet`  
  Prediction outputs across the broader corpus.

- `data/thread_llm_summaries.parquet`  
  Model-generated summaries for threads and debates.

- `data/threads.parquet`  
  Thread-level structure produced from clustered and linked text data.

- `data/egeret_dates.parquet`  
  Date recovery and provenance information for e-geret material.

- `data/egeret_threads.parquet`  
  Thread candidate outputs specific to e-geret data.

- `data/egeret_press_dual_publication.parquet`  
  Output documenting overlapping or duplicate publication patterns between e-geret and press sources.

- `data/thread_browser_corpus.parquet`  
  Corpus prepared for thread browsing and inspection.

- `data/thread_doc_summaries.parquet`  
  Per-document summary output used to inspect debate components.

- `data/keyword_scores.parquet`  
  Keyword-based scoring outputs used as a baseline or feature layer.

## 2. Model and analysis outputs

- `data/b4a_model_comparison.json`  
  Comparison results between candidate classification models.

- `data/acceptance_test_v2.parquet`  
  Acceptance testing results for the v2 model and label scheme.

- `data/agreement_report.txt`  
  Agreement and disagreement summary across models and outputs.

- `data/bypc_chunk_validation.parquet`  
  Validation output for chunking on Ben-Yehuda project candidate texts.

- `data/vocab_baseline_eval.parquet`  
  Evaluation of the vocabulary baseline model.

- `data/c2_edges_explicit.parquet`  
  Explicit thread-link edges.

- `data/c2_edges_interleave.parquet`  
  Interleave-based linkage edges.

- `data/c2_edges_semantic.parquet`  
  Semantic similarity-based linkage edges.

- `data/c2_edges_xref_seeds.parquet`  
  Seed edges for cross-reference-driven linking.

- `data/press_polemic_clusters.parquet`  
  Press-specific clusters flagged as polemic-related.

- `data/egeret_polemic_clusters.parquet`  
  E-geret-specific clusters/threads examined for polemic relevance.

- `data/press_polemic_cluster_labels.parquet`  
  Labels generated for press polemic clusters.

- `data/egeret_polemic_cluster_labels.parquet`  
  Labels generated for e-geret polemic clusters.

## 3. Source code and tooling

- `src/streamlit_app.py`  
  Main interactive application for browsing the corpus, cluster outputs, and pilot labels.

- `src/classify_pilot.py`  
  Pilot classifier and calibration pipeline.

- `src/finetune_compare.py`  
  Model comparison and fine-tuning benchmark script.

- `src/corpus.py`  
  Unified corpus construction and metadata joins.

- `src/vectorize.py`  
  Vectorization and indexing pipeline.

- `src/cluster.py`  
  Clustering logic and cluster generation.

- `src/cluster_top_terms.py`  
  Cluster-level word extraction and top terms analysis.

- `src/extract_references.py`  
  Reference extraction pipeline.

- `src/extract_vocab.py`  
  Vocabulary extraction for polemic-related markers and phrases.

- `src/keyword_baseline.py`  
  Baseline keyword-based detection model.

- `src/thread_summaries.py`  
  Thread summary generation and model-based narrative extraction.

- `src/ingest_ra_gold.py`  
  Loader for human-reviewed RA gold labels.

- `src/loaders.py`  
  Data loading utilities for the different source corpora.

- `src/cleaning.py`  
  Text cleaning, normalization, and quality control routines.

- `src/egeret_literature_review.py`  
  Literature review generation for e-geret-related threads.

- `src/egeret_literature_review_gemini.py`  
  Alternative literature review pipeline using Gemini.

## 4. Project validation and evidence logs

- `logs/pilot_validation_report.md`  
  Main validation report for the pilot and classification quality.

- `logs/egeret_threading_summary.md`  
  Summary of e-geret thread experiments and results.

- `logs/egeret_threading_close_read.md`  
  Close-reading validation notes for key e-geret thread candidates.

- `logs/c2_summary.md`  
  Summary of the C2 phase.

- `logs/c2_verification.md`  
  Verification notes for C2 outputs.

- `logs/overnight_summary.md`  
  Overnight run summary and broader project status notes.

- `logs/arbitration_targets.md`  
  Documentation of arbitration targets and review cases.

- `logs/prompt_fix_delta.md`  
  Prompt iteration and debugging history.

## 5. Overall interpretation

The main technical outputs are not only the models themselves, but the full stack of artifacts generated around them:

- cleaned and merged corpus data,
- vectorized and cluster-annotated representations,
- classification and calibration outputs,
- thread and debate candidate outputs,
- evaluation reports,
- and the interactive app used to inspect and review the results.

These outputs together form the technical basis for the project’s current status: a working prototype and evidence base, with the main unresolved issue being the decision on the project direction rather than lack of technical progress.
