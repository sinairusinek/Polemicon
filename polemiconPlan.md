# Polemicon: Hebrew Polemic Corpus Analysis Pipeline

## Status Update (2026-05-14)

### B.4a fine-tune results
- DictaBERT 6 epochs: macro F1 **0.526** (3 epochs: 0.518). Per-class F1: non 0.88 / implicit 0.49 / explicit 0.45 / meta 0.26.
- heBERT 6 epochs: macro F1 **0.466**. Winner: **dicta-il/dictabert**.
- Save-pass retrain hit 0.467 — fine-tuning variance on 1,599 train examples is ±0.05; saved checkpoint is the 0.467 version. Could be reproduced with a deterministic seed if needed.
- 4-class is at the ceiling: Sonnet teacher labels matched RA gold at only 65%, so a student model can't exceed that noise floor regardless of base model or epochs.

### Pilot-stage decision: use binary `is_polemic`
- Collapsing the 4-class predictions to binary (polemic vs. non-polemic) gives extrapolated macro F1 ≈ **0.82** — usable as a triage layer.
- Pilot work (Phase C.2 threading) will use the binary signal only. Subtype labels remain in the parquet for inspection but are not load-bearing.

### Polemic direction (internal vs. external_defense) (2026-05-14)
- New `polemic_direction` field on `data/thread_llm_summaries.parquet` distinguishes intra-Jewish disputes (`internal`) from cross-paper apologetics against external antisemitism / blood libel / missionaries / hostile non-Jewish press (`external_defense`); also `mixed` and `n/a`. Motivation: thread 406 (Tiszaeszlár-type) — Opus rates it `external_defense`/meta-polemic while Gemini-Flash3 and Sonnet rate it as topical-only. The field makes that disagreement structured rather than buried in narrative text.
- Pipeline: `src/thread_summaries.py` Stage B prompt now requests the field; new threads get it natively.
- Backfill: `scripts/backfill_polemic_direction.py` re-prompts each (thread, model) row with a lightweight narrative-only prompt (no excerpts), preserving per-model verdicts so cross-model disagreement remains visible. Non-polemic rows are set to `n/a` without an LLM call. All 64 existing rows backfilled (1 retry needed for a CLI JSON parse error).
- Distribution on top-32 pilot threads: 43 `internal`, 9 `mixed`, 1 `external_defense` (thread 406, Opus only), 11 `n/a` — consistent with a Haskalah-era corpus where most fights are intra-Jewish.
- UI: thread inspector in `src/streamlit_app.py` shows the field as a colored badge (blue=internal, red=external_defense, purple=mixed, grey=n/a) next to `polemic_type`.

### Egeret date backfill sidecar (2026-05-14)
- `data/egeret_dates.parquet` — composition dates for all 2,613 egeret rows in `polemic_pool`. `polemic_pool.parquet` not mutated.
- Built by `scripts/extract_egeret_dates.py`. No LLM calls — reuses the `DateISO` field from the prior NER pass on `e-geret-batch-export.tsv` (mapping: `egeret_N` = TSV row N, verified by title).
- Tiered provenance per-row in `source_of_date` (+ `confidence`, `partial_policy`):
  - `tsv_dateiso_day` 1894 / `_month` 179 / `_year` 100 / `_partial` 33 — direct parse of normalized field.
  - `tsv_date_hebrew` 10 — pyluach parse of raw Hebrew `Date` text (Tier B fallback).
  - `volume_metadata` 273 — `origPublicationDate` as `year_max` upper bound.
  - `author_lifespan` 124 — last-resort birth+15..death window. 30 authors hand-coded in `LIFESPANS` dict.
- Coverage: 73% exact day, 80% day-or-month, 100% has some bound. 2,073 high / 143 medium / 397 low confidence.
- **Why this exists:** cross-source threading needs month-level contemporaneity to distinguish a March-1880 letter reacting to a January-1880 article from a year-apart coincidence. Year-only bounds (the `polemic_pool.year` column was 0% populated for egeret) are too coarse.
- **Use:** join on `doc_id`. The `partial_policy` column documents how each derived bound was produced; raw fields preserved for audit.

### B.5 verification results (2026-05-14)
- ✓ Press polemic prevalence: 25.2% (target 10–30%).
- ⚠ Cross-reference lift: manual (n=33) flags 36% polemic vs 28.7% baseline (1.27×). Mechanical (n=1,622) 45.4% (1.58×). Weaker than plan implied but real.
- ✗ Kappa vs RA gold: aggregate 0.091 (target >0.6). **But concentrated in one source**: kappa is 0.31–0.46 on press-derived hard cases, **-0.20 on BenYehuda**, +0.22 on egeret (n=7). Press threading remains defensible; BenYehuda threading is not.

### BenYehuda diagnostic (chunking-validation, n=22)
- Length-truncation hypothesis tested: paragraph-aware chunking + `max(prob_polemic)` aggregation.
- Polemic recall **30% → 70%** — length artifact is real for recall.
- Non-polemic precision unchanged — model over-predicts polemic on BenYehuda regardless of chunking. Best kappa across all aggregation rules ≈ -0.05.
- Conclusion: chunking helps but is not sufficient. The real problem is that the model learned to associate BenYehuda surface features (literary Hebrew register, biblical allusion density, argumentative style) with polemic. **Press-only threading is the correct pilot scope.**

### Pilot limitations & post-pilot decisions to revisit
*(material for the pilot report — every entry here is a pilot-stage choice that should be reopened later, with the reason it was made and how to revisit it)*

#### Scope decisions (what the pilot deliberately excluded)
- **Press-only threading.** BenYehuda (`polemic_candidates`) and egeret (letters) were excluded from C.1/C.2 because (a) BenYehuda labels had kappa −0.20 vs RA gold, (b) egeret had too few polemic texts (n=565) and only 7 RA gold cases. **Revisit when:** RA gold expansion delivers ≥100 BenYehuda non-polemic and ≥50 egeret cases; threading can then extend to all sources.
- **Binary `is_polemic` only.** Pilot uses `prob_non_polemic < 0.5` collapsed from the 4-class predictions (extrapolated binary macro F1 ≈ 0.82). Subtype labels (implicit / explicit / meta) stay in the parquet but are not load-bearing because 4-class kappa was 0.09. **Revisit when:** A retrained model on expanded RA gold pushes 4-class kappa above 0.5.

#### Label-quality decisions (what we accepted as good-enough)
- **Sonnet-distilled labels accepted at ~65% RA agreement.** This is the noise floor; the fine-tuned model can't exceed it. Pilot accepted this rather than blocking on Sonnet prompt re-engineering. **Revisit when:** Either Sonnet prompts are revised against the documented failure patterns (rhetorical denial, historical-vs-current opposition, truncated texts) or Sonnet is replaced by direct RA labels.
- **Saved B.4a model is the save-retrain version (macro F1 0.467), not the best observed (0.526).** Differs by ~0.06 F1 due to fine-tuning variance on 1,599 train examples. **Revisit when:** B.4a is re-run with a deterministic seed + held-out validation split + early stopping.
- **B.5 verification failed kappa target** (0.091 aggregate, target >0.6). Pilot proceeded with documented per-source breakdown rather than blocking. **Revisit when:** Per-source kappa is re-measured after gold expansion.

#### Methodological substitutions (changes from the original plan)
- **Newspaper-pair edges substituted for author-pair edges.** Press has zero filled-in authors; the plan's "different-author" filter was unbuildable. Substituted "different-newspaper" — historically defensible since 19th-c Hebrew papers were themselves polemic actors with distinct ideological lines. **Revisit when:** Author extraction from press text (bylines, pseudonym signatures) makes per-author analysis feasible; can then run author-pair edges alongside newspaper-pair edges.
- **Same-newspaper edges retained** (not filtered out) — enables identifying topics a single paper runs internally. Threads are tagged `internal` / `engaged` / `co-occurrence` so the consumer can filter. **Revisit when:** A specific research question requires only cross-paper engagement; the current data already supports both views.
- **Greedy date-cut split for over-span threads.** When a thread's span exceeds 730 days, the largest internal temporal gap is identified and edges across that cut are removed. **Limitation:** densely cross-connected threads (e.g., thread 374, MGD-dominated, 1114-day span) aren't fragmented because no single cut disconnects them. **Revisit with:** Graph-theoretic splitting (edge betweenness, modularity) instead of a temporal cut.

#### Tuning decisions (thresholds picked from data, not from historical validation)
- **Edge windows: 90d for interleave/semantic, 180d for explicit references.** Asymmetric tuning informed by the post-v3 sensitivity table (median edge gap was 39d, 90d captures ~59% of original-window edges, 180d preserves explicit-ref resolution at 40%). **Revisit by:** Sampling top engaged threads (especially #412, #408, #406) and comparing their span/edge structure to documented historical polemics. If the windows are systematically too tight or too wide, retune.
- **Cosine threshold 0.85 for semantic edges.** Plan default, not tuned. **Revisit by:** Calibrating against thread examples where historians have already identified explicit text-to-text correspondence.
- **HDBSCAN min_cluster_size=10** and UMAP n_neighbors=15. Plan defaults. **Revisit by:** Running a coarser clustering (min_cluster_size=20–30) to see whether the 98 clusters consolidate into a smaller set of meaningful topic families.
- **730-day span cap.** Chosen as "≥2 years feels too long for one debate." Not validated against history. **Revisit by:** Looking at thread 374 (1114-day MGD thread on Jewish peoplehood) — is it really one running discourse or two distinct phases?

#### Diagnosed-but-unfixed issues
- **BenYehuda over-prediction is genre/register-driven, not just length.** Chunking validation (n=22) lifted polemic recall 30%→70% but precision stayed bad: the model flags literary Hebrew with biblical allusion density as polemic regardless of actual rhetorical function. **Fix path:** Expand RA gold with explicit non-polemic BenYehuda examples (target ≥100); train chunk-level rather than document-level on long works.
- **compact_memory source (n=142): 0% polemic.** Not investigated. May be a length/register artifact or a genuine signal that this small periodical subset isn't polemic. **Revisit by:** Sampling 10 compact_memory texts and reading them.
- **Cross-reference probe is weaker than plan implied.** Manual cross-refs (n=33) lift polemic rate to 36% (1.27× baseline) — plan expected "high rates." Mechanical refs do better (1.58×) but most are simple newspaper mentions. **Revisit when:** Better gold labels exist; re-test on a curated polemic-mention vs non-polemic-mention split.
- **Explicit-reference resolution 40%.** Of 406 references in clustered docs, only 162 found a same-cluster target from the named newspaper within 180d. Unresolved cases may include cross-cluster references, references to non-pilot newspapers, or noise. **Revisit by:** Inspecting a sample of unresolved cases to characterize the failure modes.
- **Actor-name vocabulary baseline (Q.B) under-recalls because of spelling variance, not prompt bugs (2026-05-14).** After fixing the Stage B prompt to emit actors in original Hebrew (`פינס (Pines)` rather than transliterated only), actor-name retrieval recall@N on top-30 threads rose only from ~0.6% (substring) to 6.7% (TF-IDF cosine) — well below the 30–50% target. Root cause: the LLM emits normalized modern Hebrew (`פינס`, `סמולנסקין`), but the OCR'd 19th-c corpus uses period spellings, abbreviations, and Yiddish-influenced variants (`פינעס`, `סמאלענסקין`, `יל"ג`). Compounding factors: newspaper-name actors (`המגיד`, `המליץ`) saturate cosine similarity, and recall@N truncates the candidate pool too tightly. **Why this matters for the post-pilot project:** entity linking is not just a retrieval optimization — it is the missing infrastructural layer between named-entity mentions and threading. The pilot's TF-IDF-cosine baseline (38% recall on topic words) outperforms actor-name retrieval (6.7%) precisely because topic-word matching is bag-of-vocab and forgives spelling, while name matching demands string identity that the OCR corpus cannot provide. **Revisit by:** Treating NER + entity linking as a Phase-D infrastructural goal in its own right. Hebrew Haskalah-era authority files (e.g. National Library of Israel name authority, Wikidata entries for `סמולנסקין`, `פינס`, `ליליענבלום`, etc.) cover the major polemic actors though not lesser-known correspondents. An enriched corpus with linked-data person/place/organization mentions would (a) raise the actor-name baseline to a level where it can fairly be compared to the C.2+LLM pipeline, (b) let threading edges include co-mention-of-same-entity as a first-class signal alongside reference/interleave/semantic, and (c) be a corpus-level deliverable independent of polemic threading. The pilot is not the place to build this, but the limitation justifies the investment.

#### Verification gaps (what we didn't check)
- **No historical validation.** Threads have not been cross-checked against documented 19th-c Hebrew press polemics in the secondary literature. The pilot's threading is a *candidate generator*, not a confirmed inventory.
- **Q.A bibliography curation deferred (2026-05-14).** URL spot-check of 10/99 cited sources in `data/thread_literature_review.parquet` (logged in `logs/pilot_validation_report.md` → "URL spot-check") found ~10% byline/role error rate — including one fully misattributed author (Penslar→Halperin on tandfonline `10.1080/00263206.2014.886574`). The pipeline's web-search bibliographies are usable as candidate lists for the pilot finding ("0/20 well-documented") but **need a two-stage curation pass before any publication-facing use**: (1) a second LLM verification pass that re-fetches each source / Crossref-looks-up DOIs and flags author/year/role mismatches, (2) a human reviewer audit on the flag list. The pilot does not run these passes; they are part of the post-pilot writing workflow.
- **Hebrew-language secondary literature underrepresented.** Web-search-grounded LLMs over-index on Google-indexed English scholarship. RAMBI (Index of Articles on Jewish Studies, NLI) and similar Hebrew/Israeli catalogues likely hold press-polemic literature not surfaced by the current Q.A pipeline. **Revisit by:** investigating whether RAMBI is queryable programmatically and how Hebrew-titled articles can be matched against thread topics/actors. May overturn the "0/20 well-documented" finding for some threads.
- **Cross-source threading blocked by press-corpus scope (2026-05-14).** Concentrated effort (logged in `logs/pilot_validation_report.md` → "Question C") to find threads bridging press / egeret / Ben-Yehuda corpora yielded only 3 in-window egeret↔press mention candidates after filtering 6128 raw `mechanical_newspaper` mentions on (a) target must be in our 5-paper press corpus — most cited papers (העברי, הכרמל, השחר, חבצלת) are 19th-c. but not ingested; (b) common-word "דבר" false positives inflate raw counts ~4666/5180; (c) mentioning doc must fall in 1862–1888 — most egeret mentioners write retrospectively in the 1900s–1930s. **A separate finding emerged**: the two remaining in-window egeret bridges resolved not as cross-corpus *threads* but as **dual publication** — `egeret_2136` (Y.Y. Rivlin, May 1883) is the same signed text as `press_68753` (Ha-Maggid 1883-06-27). A fingerprint scan confirmed at least 5 such dual-publication pairs in 1881–83 (Smolenskin ×2, Rivlin ×2, Ben-Yehuda ×1; lower bound — punctuation/normalization differences produce false negatives). **Implication:** the egeret and press corpora are not independent for select 1880s writer-editors; any analytic treating them as such is double-counting. **Also surfaced:** threading internally uses day-resolution dates (`corpus.parquet` retains full `date`, edges built on `pub_date` with day windows); the year-only collapse is only in `polemic_pool.parquet`'s display layer. Saved: `data/egeret_press_dual_publication.parquet`.
- **No researcher review of cluster topics.** Cluster top-terms are interpretable but haven't been confirmed by a Haskalah-press historian.
- **No held-out test set for the classifier.** B.4a used 80/20 stratified split but the same split was used for model selection and final report; technically the F1 numbers are over-optimistic by a small margin.
- **No silhouette calibration against alternative cluster counts.** Silhouette 0.615 looks good but wasn't compared to other clustering parameter choices.

### Post-pilot backlog (deferred until press threading is exercised)
- **Expand RA gold to ~500–1000 cases**, including **100+ BenYehuda non-polemic** examples specifically to teach the model that argumentative literary Hebrew ≠ polemic. Highest-leverage path.
- **Chunk-level labeling, not document-level**, for long works during gold expansion.
- Dedicated binary classifier head (small expected gain over post-hoc collapse, ~+0.02–0.05 F1).
- Sonnet-at-scale as the classifier (no student) — accept cost, gain ~0.65+ F1 against RA gold.
- Re-run B.4a with deterministic seeds and a held-out val split for early stopping.
- Full BenYehuda chunking only after the gold expansion; pre-gold chunking lifts recall but tanks precision.
- **NER + entity linking as Phase-D infrastructure.** Build a corpus-wide person/place/organization layer linked to authority files (NLI name authority, Wikidata) for the major Haskalah-era actors. Justification carried by the actor-baseline finding above. Deliverable is corpus-level (enriched linked-data layer) and benefits threading as a downstream consumer: co-mention-of-same-entity becomes a first-class edge type alongside reference/interleave/semantic, and actor-name retrieval becomes a fair comparison baseline.
- **Cross-source threading expansion → public-sphere framing (post-pilot).** Three coordinated workstreams justified by the Q.C finding above: (a) **broaden press ingest** to the 19th-c. papers egeret writers actually cite — Ha-Ivri, Ha-Karmel, Ha-Shahar, Havatzelet, Ha-Pisgah — without which most cross-source mention bridges have no target to resolve to; (b) **date `polemic_candidates`** (currently undated in `polemic_pool`) to expand the bridge pool beyond egeret; (c) **refine the dual-publication scan** with fuzzy / character-n-gram matching against the present fingerprint method's known false-negative rate. Strategic reframe: dating polemic_candidates extends the project's unit of analysis from polemic-detection to *the circulating text across press / letters / Ben-Yehuda open-publication*, i.e. a public-sphere study rather than a polemic-search task. Adjacent extractor fix: the `mechanical_newspaper` matcher must suppress the common-word "דבר" false-positive class before any of this scales.
- **Bibliography curation workflow + Hebrew-source retrieval (post-pilot writing stage).** When the pilot is written up: (a) build a second-pass verifier over `data/thread_literature_review.parquet` that re-fetches each URL / Crossref-looks-up each DOI and flags author/year/role mismatches against the original LLM claim, (b) human-review the flag list, (c) explore RAMBI (NLI's Index of Articles on Jewish Studies) and equivalent Hebrew academic indexes for press-polemic literature the English-biased web search missed. Spot-check on 2026-05-14 found ~10% byline error rate with at least one fully-wrong author — workflow is non-optional for publication.

## Status Update (2026-05-06)

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
- **RA gold labels ingested (2026-05-05):** 102 gold labels from 4 RA sources (16 cheap-diverge CSV, 7 disagree CSV, 50 Letters Excel, 29 BenYehudaProject Excel) → `data/ra_gold_labels.parquet`. Label distribution: non-polemic=53, implicit=15, uncertain=15, meta-polemic=10, explicit=9.
- **Sonnet v2 prompt + schema (2026-05-05):** `CLASSIFICATION_PROMPT_V2` in `src/classify_pilot.py` replaces binary label with RA's 4-tier scheme (non-polemic / implicit polemic / explicit polemic / meta-polemic (descriptive)). Adds `broader_polemic_link` field (none/suspected/clear + justification), metadata injection (year, author, headline, recipient), continuation detection regex (`is_continuation`). Run via `--v2` flag.
- **Acceptance test (2026-05-05):** Sonnet-v2 scored 65.2% (15/23) on the 23 RA-annotated cases. Formal threshold (78%) not met; decision is to proceed with documented limitations rather than further prompt tuning.
- **2K calibration run (2026-05-06):** Sonnet-v2 via `claude -p` CLI (Max plan) classified 1,999/2,000 stratified texts → `data/calibration_v2.parquet`. 4 runs required due to per-session rate limits on Max plan (~1,000–1,200 calls per session); resume logic handled automatically. Final distribution: non-polemic 67.8% (1,355), implicit polemic 17.6% (351), explicit polemic 10.7% (214), meta-polemic 4.0% (79). Combined polemic rate 32.2%. `broader_polemic_link`: clear 32.6%, suspected 26.5%, none 41.0%. Confidence highest for non-polemic (0.90) and explicit (0.87), lower for implicit (0.76) and meta (0.73).
- **Streamlit calibration browser (2026-05-06):** App updated with (1) sidebar stats panel showing label and broader_polemic_link distribution, (2) "Calibration Browser" view mode — browse explicit/implicit/meta-polemic texts with metadata, confidence, topic, broader_polemic_link badge, and RTL text preview; (3) distribution expander with per-source breakdown table and per-year polemic rate line chart by source. Text and metadata embedded directly in calibration_v2.parquet (21MB) so deployed app works without corpus.parquet.
- **B.4a comparison script (2026-05-06):** `src/finetune_compare.py` written. Compares `dicta-il/dictabert` vs `avichr/heBERT` on 4-tier classification using calibration_v2 labels. Balanced class weights, AdamW lr=2e-5, 3 epochs, stratified 80/20 split. Run with `python src/finetune_compare.py`.

### Known Sonnet v2 limitations (documented, not fixed)
Three systematic failure patterns identified across 3 prompt iterations:
1. **Rhetorical denial of controversy** — texts that open with "everyone agrees on X" while actually advancing a contested position (e.g., press_34463). Sonnet takes the claim at face value; the polemic signal is in a reference to a prior article, which Sonnet flags as `broader_polemic_link=suspected` but doesn't escalate to the label.
2. **Historical vs. current opposition** — texts defending a position against past/generational criticism (e.g., bypc_7349, defending Jewish chosenness against historical mockery) get classified as implicit polemic when they should be non-polemic.
3. **Truncated texts** — 3 of the 8 mismatches (bypc_988, bypc_1286, press_49005) involve texts where the RA's own notes say the polemical content is in the continuation not included in the pilot sample. These are not model errors.
The remaining 2 failures (egeret_3213, bypc_5681) are edge cases where the RA notes themselves express uncertainty.

### Next: B.4a model selection
- **Run `python src/finetune_compare.py`** — fine-tunes `dicta-il/dictabert` vs `avichr/heBERT` on calibration_v2 labels, prints macro-F1 comparison, saves `data/b4a_model_comparison.json`. Update this section with results.
- **Then B.4:** Fine-tune chosen model on full calibration set, classify all 33K corpus texts.

### Deferred
- Dense embeddings (Tier 2/3) — only if TF-IDF clustering proves insufficient
- Phase C.2 (thread detection) — approach discussed: tiered edge signals (strong/medium/weak), expect loose topical clusters with partial name/vocab overlap rather than strict reply chains. Will be more effective after polemic labels are established.
- Full-corpus reference extraction — current approach validated on 56 polemic pilot texts; scale to all polemic texts after B.4 classification.
- Sonnet-generated topic labels for top 20 clusters.

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

**Current implementation (Phase A, completed):**
- Sparse: Character n-gram TF-IDF (`char_wb`, 3-5 grams, 50K features) + word-level TF-IDF (1-2 grams, 30K features). Chosen for OCR robustness on press texts.
- TruncatedSVD to 300 dims, FAISS index.

**Decision point (revisit at B.4a): unified dense vs. hybrid strategy**

The OCR robustness argument applies only to the press corpus. Ben-Yehuda Project and e-geret texts are clean and would benefit from a stronger semantic model. Two strategies:

| Strategy | When to use | Trade-off |
|----------|-------------|-----------|
| **Unified dense** | If the chosen model (B.4a) tokenizes at sub-word/char level and degrades gracefully on OCR noise | Simpler; cross-source similarity works natively |
| **Hybrid per-task** | If dense model performs poorly on noisy press text | Better per-source quality, but cross-source thread detection (C.2) requires a separate alignment step |

**Test to run at B.4a:** Take 20 high-noise press texts + their manually corrected equivalents (or clean same-author texts). Compute cosine similarity under the candidate dense model. If mean similarity > 0.7, the model is OCR-robust enough to unify. If not, use hybrid: dense for BenYehuda/e-geret classification, TF-IDF retained for cross-source linking.

**Dense model plan (if unified):** Sliding window (512 tokens, stride 256) + mean pool for long texts. Batch size 32. Store as numpy memmap + FAISS `IndexFlatIP`. Estimated ~30 min on Apple Silicon.

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
2. **Calibration (2,000 texts):** Stratified sample (~$20–25 with Sonnet; Hebrew is 2–3× more token-heavy than English — earlier $5/$18 estimates were wrong). Estimate polemic prevalence, set confidence thresholds. Researcher reviews a subset to create gold labels.

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

### B.4a Hebrew model selection (do before fine-tuning)

Before committing to AlephBERT, survey available Hebrew-capable models on HuggingFace and recent literature. Criteria:

| Criterion | Notes |
|-----------|-------|
| Hebrew pre-training | Must be trained on Hebrew text (not just multilingual) |
| Text length | Corpus texts average several hundred words — models with >512 token limit preferred |
| Classification track record | Any reported results on Hebrew NLP tasks (sentiment, NER, classification) |
| License | Must allow research use |
| Size / CPU feasibility | Fine-tuning must be feasible without a GPU (or document GPU requirement) |

**Candidates evaluated:**
- `dicta-il/dictabert` — chosen candidate: DictaBERT base, trained on modern + historical Hebrew (19th-century register appropriate). Use base model, not dictabert-sentiment (sentiment fine-tuning biases intermediate layers toward emotional tone markers, hurting polemic/non-polemic distinction).
- `avichr/heBERT` — trained on a larger, more varied Hebrew corpus. Second candidate.
- Ruled out: `onlplab/alephbert-base` (modern Hebrew only, Wikipedia/news), `dicta-il/MsBERT` (first-millennium manuscripts — wrong era), `dictabert-sentiment` (sentiment bias).

**Comparison script:** `src/finetune_compare.py`. Run `python src/finetune_compare.py` (both models) or `--model dictabert` (one). Results saved to `data/b4a_model_comparison.json`.

**Decision output:** *(pending run)* Update with chosen model and macro-F1. Revisit if F1 < 0.75 after fine-tuning.

### B.4 Fine-tune chosen Hebrew model (primary classifier for full corpus)

This is the core classification step -- Claude bootstraps labels, the fine-tuned model scales:

1. Combine Claude's 2K silver labels + researcher's gold labels (target: 500-1000 human-reviewed)
2. Fine-tune chosen model (see B.4a) with a classification head for 4-tier polemic detection
3. Add keyword baseline scores as auxiliary features (hybrid model)
4. Train with 5-fold cross-validation; evaluate on gold set (target F1 > 0.8)
5. **Run fine-tuned model on the full corpus** -- fast and free compared to API calls
6. For uncertain predictions (0.3-0.7 confidence), optionally send to Claude for a second opinion

**Cost:** ~$20–25 for Claude bootstrap (2K calibration) + compute time for fine-tuning (minutes on CPU). Much cheaper than ~$300–350 for full API classification of all 33K texts at the same per-token rate.

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
         ->  B.2 pilot (200 texts)  ->  B.2 calibration (2K texts)  ->  B.4a model selection  ->  B.4 fine-tune
              B.3 keyword baseline (parallel)                                                  ->  B.4 classify full corpus
         ->  C.1 (clustering)  ->  C.2 (thread detection)  ->  C.3 (visualization)
```
