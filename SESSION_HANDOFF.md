# Session Handoff Prompt

Copy-paste this to start the next session:

---

Continuing the Polemicon project. Read `polemiconPlan.md` (Status Update section) for full context.

**Where we left off:** The Sonnet v2 classification prompt (4-tier label scheme) has been built, tested, and accepted with documented limitations. The acceptance test scored 65.2% (15/23) against RA gold labels — the formal 78% threshold was not met, but the decision was made to proceed rather than continue prompt tuning. The failure patterns are documented in the plan and are considered acceptable limitations, not prompt bugs.

## Task for this session: 2K calibration run + Streamlit v2 display

### Step 1: Run the 2K calibration

```bash
cd /Users/sinairusinek/Documents/GitHub/Polemicon
source .venv/bin/activate
python src/classify_pilot.py --v2 --calibration --calibration-n 2000
```

This runs Sonnet-v2 on a stratified 2,000-text sample (source × cluster × keyword score). Output → `data/pilot_classifications_v2.parquet`. Cost ~$20–25 (Hebrew is 2–3× more token-heavy than English; earlier $5 estimate was wrong). ~30–60 min. The script already has checkpointing — if it interrupts, rerun the same command to resume.

### Step 2: Review the output distribution

After the run completes, check:
- Per-source polemic-rate: press should be 10–30% combined-polemic (explicit + implicit); polemic_candidates much higher.
- `broader_polemic_link=clear` cases — these are candidates for RA spot-check.
- Any parse errors or unexpected label values in the output.

### Step 3: Update Streamlit app

The app (`src/streamlit_app.py`) already has stub loaders for `pilot_classifications_v2.parquet` and `ra_gold_labels.parquet`, and a display section for the v2 label and `broader_polemic_link`. Verify these work correctly once the v2 parquet exists. The annotation panel for each text should show:
- **Sonnet v2 label** (color-coded by tier: explicit=red, implicit=orange, meta=blue, non=green)
- **broader_polemic_link** (none/suspected/clear + justification text)
- **RA gold label** (where available from `ra_gold_labels.parquet`) as an `st.info` box
- The old v1 binary labels (from `pilot_classifications.parquet`) should remain visible for comparison

---

## Key files

| File | Purpose |
|------|---------|
| `src/classify_pilot.py` | Main classification script. `--v2 --calibration --calibration-n 2000` for the next run. |
| `src/streamlit_app.py` | Annotation + review app. Deployed on Streamlit Cloud (auto-deploys from main). |
| `src/ingest_ra_gold.py` | Parses RA annotation sources → `data/ra_gold_labels.parquet`. Already run; no need to re-run unless RA delivers new annotations. |
| `data/pilot_classifications_v2.parquet` | **Does not exist yet** — created by the calibration run. |
| `data/ra_gold_labels.parquet` | 102 RA gold labels across 4 sources. Already generated. |
| `data/pilot_classifications.parquet` | Original v1 binary labels (4 models, 200 texts). Keep for comparison. |
| `data/pilot_sample.parquet` | 200-text pilot sample used for v1. The 2K calibration uses a separate stratified sample. |

---

## v2 prompt schema (for reference)

`CLASSIFICATION_PROMPT_V2` in `src/classify_pilot.py` outputs:

```json
{
  "polemic_label": "non-polemic | implicit polemic | explicit polemic | meta-polemic (descriptive)",
  "confidence": 0.0-1.0,
  "polemic_type": "attack | defense | debate | satire | critique | description | none",
  "broader_polemic_link": "none | suspected | clear",
  "broader_polemic_justification": "one-line explanation",
  "target": "name of target or null",
  "evidence": "quoted or paraphrased supporting text",
  "topic": "one-sentence topic summary"
}
```

The 4-tier label maps to RA Hebrew labels: `כן`→explicit, `לדיון`→implicit, `לא`+describes=`כן`→meta, `לא`+`לא`→non-polemic.

---

## Known v2 limitations (do not re-fix these)

Three failure patterns identified across 3 prompt iterations — accepted as-is:
1. Texts that rhetorically claim "universal agreement" while actually advancing a contested position get classified as non-polemic (the `broader_polemic_link` field captures the signal, but not the label).
2. Texts defending a position against *historical* opposition (past generations' views) are occasionally over-classified as implicit polemic.
3. Truncated texts in the pilot sample — 3 of the 8 mismatches involve texts where polemic content is in a continuation section not included in the excerpt.

---

## Important reminders

- Always use `restore_final_forms()` from `src/cleaning.py` when displaying Hebrew text.
- The Streamlit app is deployed on Streamlit Cloud; auto-deploys from `main` branch. Test locally with `streamlit run src/streamlit_app.py` before pushing.
- Discuss approach before implementing non-trivial changes.

---
