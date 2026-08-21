# Polemicon

This repository contains the technical work for the Polemicon project: building a corpus, cleaning and normalizing historical Hebrew texts, clustering by topic, classifying polemical content, and exploring thread-level connections across source materials.

## Project purpose

The project examines whether historical Hebrew texts can be analyzed to identify polemical discourse, debate structures, and related source clusters in a way that is useful for research and human-guided review.

## Main technical outputs

The project generates structured data products, evaluation artifacts, and analysis tools in the following locations:

- `data/` – processed datasets and model outputs
- `src/` – ingestion, cleaning, vectorization, clustering, classification, and app code
- `logs/` – validation reports, run logs, and summary notes

For a structured catalog of the technical outputs, see:

- [technical_outputs_reference.md](technical_outputs_reference.md)

For a Hebrew project summary and report version, see:

- [hebrew_report.md](hebrew_report.md)

## Repository structure

- `data/` — processed corpus files, model outputs, thread outputs, and validation artifacts
- `src/` — main project codebase
- `logs/` — validation reports, run summaries, and debugging notes
- `polemiconPlan.md` — project plan and status notes
- `technical_outputs_reference.md` — catalog of technical outputs
- `hebrew_report.md` — Hebrew status report

## Key outputs

Some of the primary technical artifacts include:

- `data/calibration_v2.parquet`
- `data/pilot_classifications.parquet`
- `data/ra_gold_labels.parquet`
- `data/cluster_labels.parquet`
- `data/threads.parquet`
- `data/thread_llm_summaries.parquet`
- `data/egeret_dates.parquet`
- `data/full_corpus_predictions.parquet`
- `data/polemic_pool.parquet`

These files support the core workflow of:

1. corpus construction,
2. data cleaning and metadata enrichment,
3. vectorization and clustering,
4. classification and calibration,
5. thread exploration and review,
6. validation and documentation.

## Main application entry points

- `src/streamlit_app.py` — interactive browser for exploring corpus and cluster data
- `src/classify_pilot.py` — classification and calibration pipeline
- `src/finetune_compare.py` — model comparison workflow
- `src/corpus.py` — corpus construction logic
- `src/vectorize.py` — vectorization and indexing logic

## Validation and project notes

Validation and project reflections are stored in:

- `logs/pilot_validation_report.md`
- `logs/egeret_threading_summary.md`
- `logs/c2_summary.md`
- `logs/overnight_summary.md`
- `polemiconPlan.md`

## Notes

The current project status reflects substantial technical progress and working outputs, while the unresolved issue is the strategic direction of the next phase. In other words, the project has produced a working technical foundation, but the continuation depends on deciding what the next phase is meant to accomplish.
