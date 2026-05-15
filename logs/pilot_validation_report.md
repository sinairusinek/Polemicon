# Pilot validation report — Questions A and B
*2026-05-14*

Two tests run against the top-30 engaged threads from C.2 to validate the threading + LLM pipeline:

- **Question A** — Are these polemic threads documented in secondary literature? (`src/thread_literature_review.py`)
- **Question B** — Could vocabulary search alone find the same documents? (`src/vocab_baseline.py`)

## TL;DR

| Result | Value |
|---|---|
| Best vocab-only baseline recall@N (top-30 mean) | **38%** |
| Threads with scholarly press-history treatment | **0 / 20** |
| Threads "mentioned-in-passing" | 16 / 20 |
| Threads with no scholarly attention found | 4 / 20 |
| Threads with canonical *underlying* event | 11 / 20 |

→ The pipeline reliably surfaces **either** canonical events whose press dynamics scholars haven't analyzed, **or** entirely under-studied polemics. Either way, the threading approach is adding ~62% of each thread that a vocabulary search cannot reach.

---

## Question B — Vocabulary baseline

Four configurations × 30 threads = 120 evaluation runs. Output: [`data/vocab_baseline_eval.parquet`](../data/vocab_baseline_eval.parquet), [`data/vocab_baseline_missing.parquet`](../data/vocab_baseline_missing.parquet).

### Aggregate metrics (mean across 30 threads)

| Strategy | Engine | Recall@N | Precision@N | mAP | Docs found | Docs missed |
|---|---|---:|---:|---:|---:|---:|
| `tfidf_terms` | `tfidf_cos` | **0.380** | 0.380 | 0.358 | 22.4 | 2.9 |
| `actor_names` | `tfidf_cos` | 0.046 | 0.046 | 0.021 | 4.3 | 21.1 |
| `actor_names` | `substring` | 0.041 | 0.108 | 0.029 | 2.1 | 23.2 |
| `tfidf_terms` | `substring` | 0.006 | 0.006 | 0.009 | 7.5 | 17.8 |

### Key findings

1. **Best vocab-only baseline recovers 38% of a thread on average.** The remaining 62% is what the threading pipeline (reference edges + interleave + semantic similarity + heuristic clustering) adds.

2. **Actor-name baselines essentially fail (~4-5%) — but for a fixable reason.** The LLM returned actors in English/transliterated form (e.g. `"BILU members"`, `"Ha-Maggid"`, `"Dr. Lehmann"`) that don't substring-match the Hebrew text. If we ask the Stage B prompt for actors *in their original Hebrew form*, this baseline should rise substantially (estimated to 30-50%, possibly the strongest baseline).

3. **Per-thread variance is large** (12% – 75%). Hypothesis: high-recall threads are vocabulary-driven (named-entity disputes, specific terms), low-recall are reference/semantic-driven (thematic exchanges, varied terminology). The lowest-recall threads are the ones where the threading pipeline adds the most:
    - Thread 432 (12%): Inter-Press Rivalry HaLevanon vs HaMagid
    - Thread 408 (13%): Palestine-vs-America emigration
    - Thread 411 (18%)
    - Thread 381 (20%): She'eilot ha-Hayyim
   And the highest:
    - Thread 399 (75%)
    - Thread 391 (64%)

4. **TF-IDF terms baseline picks generic Hebrew function words** (e.g. `"עלינו", "אנחנו", "אבותינו"`) as the top thread-vs-corpus terms in some threads. That makes substring retrieval useless but cosine retrieval still works because the full vector captures specificity. A stop-word-aware TF-IDF (or just a Hebrew stopword list) would lift this baseline a few points.

---

## Question A — Secondary literature review

20 of 30 threads reviewed before the Anthropic API credit balance was exhausted (~$9.05 spent, ~$0.45/thread — higher than my $0.10-0.30 estimate, mostly due to web-search results being expensive context tokens). 10 threads remain (407, 426, 377, 385, 141, 375, 193, 376, 418, 391). Output: [`data/thread_literature_review.parquet`](../data/thread_literature_review.parquet).

### Aggregate

- **20 threads reviewed**, 99 sources cited total, **85% of sources include direct URLs**
- Source types: 32 articles, 31 encyclopedia entries, 20 books, 9 web resources, 6 chapters, 1 thesis
- **0 / 20** rated `well-documented` — striking
- 16 / 20 rated `mentioned-in-passing`
- 4 / 20 rated `not-found`
- 11 / 20 underlying events are canonical (BILU/1881, Tiszaeszlár, Hibbat Zion, She'eilot ha-Hayyim, etc.)

### Interpretation

The model is rigorously distinguishing two things:
- **Is the underlying historical event canonical?** Often yes (BILU, Tiszaeszlár, post-1881 emigration debate are all heavily studied)
- **Has the Hebrew-press coverage of this event been studied as a discrete press polemic?** Almost never.

That gap is the project's research opportunity in concrete form. The threading pipeline is surfacing press dynamics that historiography has noticed at the event level but not analyzed at the medium level.

### The four "not-found" threads — most-novel candidates

| Thread | Topic | Notes |
|---|---|---|
| **395** | First Aliyah press polemics (missionary threats, colony governance) | Events canonical, press exchange not studied |
| **404** | 1878-79 bibliographical disputes on medieval Jewish literature (Gastfreund/Halberstam/Kaufmann/Buber/Harkavy across Ha-Magid, Ha-Melitz, Magazin für die Wissenschaft) | No secondary literature found at all |
| **433** | Falcz divorce scandal + traditional-scholarship defense (1875-76) | Confirms Opus's earlier reading as syndicated shaming; nobody has documented it |
| **397** | Khashkes' Graetz-translation controversy | No scholarly attention found |

### Example: Thread 412 (BILU/Old Yishuv, the top-scored thread)

Status: `mentioned-in-passing`, `is_canonical_event=True`. The model's notes:

> "The underlying events (1881-1882 pogroms, BILU founding, Chovevei Zion formation, America-vs.-Palestine emigration debate) are thoroughly canonical in Jewish historiography. However, the specific multi-newspaper polemical exchange across Ha-Melitz, Ha-Maggid, Ha-Zefirah, Ha-Levanon, and Ha-Tzfira — including the charges of heresy against BILU, the role of Yechiel Michal Pines, and the figures of Lehmann, Renan, and Rabbi Diskin — is treated in the scholarly literature primarily as context or background rather than as a discrete, named polemic subject to close press-history analysis."

Sources include Yosef Salmon's work on Ha-Maggid, Feiner's *Haskalah and History*, Jonathan Frankel's *Crisis and Modernity* chapter, and at least one prior DH paper. All with URLs.

---

## URL spot-check (2026-05-14)

Sampled 10 of the 99 cited sources (8 mentioned-in-passing, 2 not-found) and verified each via WebFetch + Crossref. Result: **6 fully verified, 2 with minor inaccuracies (year/role mischaracterization), 1 misattributed author, 1 unverifiable from this tooling but real on inspection.**

Notable findings:

- **Author misattribution (1/10).** Tandfonline `10.1080/00263206.2014.886574` was cited as Derek J. Penslar. Crossref returns **Liora R. Halperin** as the actual author; DOI/title/year/journal are correct. The web-search-grounded LLM correctly retrieved the paper but invented a plausible-sounding wrong byline.
- **Minor role/year errors (2/10).** Deinard's *Toledot Even Reshef* year cited as 1875 but the Deinard encyclopedia entry says 1879 (the Firkovich entry bibliography uses 1875 — sources themselves disagree); Firkovich entry was cited as naming "Deinard" among authenticity-respondents, but the entry actually lists Harkavy, Strack, Frankl, Kunik (Deinard appears only as a biographer in the bibliography).
- **No fabricated URLs in this sample.** All 10 URLs point to a real page on the cited domain (academia.edu links require login and were verified separately by the researcher).

### Implication: two-stage citation curation is required

The pipeline's web-search-backed bibliographies are **good candidate generators but not publication-ready**. Author bylines and per-source roles need a second pass. The accepted workflow for any downstream publication-facing use of these citations is:

1. **LLM pass (this pipeline)** — gather URLs and a one-line claim about each source's relevance.
2. **Second LLM verification pass** — for each cited source, re-fetch (or Crossref/DOI-lookup) and check author + title + year + role-in-claim. Flag mismatches.
3. **Human curation** — a domain-aware reviewer audits the flag list and resolves Hebrew-script and transliteration edge cases that even the verifier won't catch.

The pilot does not need to perform passes 2–3 — the bibliographies are already strong enough to support the "0/20 well-documented" headline finding and the four "most-novel-candidate" threads. The curation workflow is a deliverable for the post-pilot writing stage.

### Post-pilot exploration: Hebrew-language secondary literature

Web-search-grounded LLMs over-index on English-language scholarship indexed by Google. Important Hebrew-language secondary literature on Haskalah-era polemics is catalogued in **RAMBI** (The Index of Articles on Jewish Studies, National Library of Israel) and similar Hebrew/Israeli academic indexes, which may not be discoverable from a generic web search. Post-pilot exploration item: investigate whether RAMBI is queryable programmatically (or via structured search), and how Hebrew-titled articles can be matched against thread topics/actors. This is a research question in its own right — both a methodological one (multilingual bibliography retrieval) and a substantive one (it may overturn the "0/20 well-documented" finding for some threads).

---

## What's still pending

1. **10 unreviewed threads** (Q.A): 407, 426, 377, 385, 141, 375, 193, 376, 418, 391. Options:
   - **Recommended**: switch to Gemini 2.5 Flash with grounding (~$0.50 total for all 10) — same web-search capability, ~10× cheaper. Some loss in citation reliability but acceptable for a "second pass."
   - Top up Anthropic credits and continue with Sonnet (~$4.50 for remaining 10).
   - Stop at 20 — we already have strong signal.

2. ~~**Citation reliability spot-check**~~ — done 2026-05-14, see "URL spot-check" section above. Flag rate ~10% (1 misattributed author + 2 minor role/year errors out of 10 sampled). Two-stage curation workflow defined.

3. **Improved actor baseline** — re-run Q.B with Hebrew-form actor names in `actors` field of the LLM summary. Likely lifts that baseline from 5% to 30-50%, narrowing the pipeline's value-add headline.

4. **Streamlit integration**: surface the literature review verdict + sources panel next to the LLM verdict in the Thread Browser. ~30 LOC.

## Recommended next move

Switch to Gemini Flash to finish the remaining 10 reviews (~$0.50). Then spot-check ~10 sources. Then take the combined result back to the user for what to write up / publish first — the "not-found" threads are the strongest research bid.

---

## Methodological decision: the COUNTERWEIGHT clause in Stage B

*Added 2026-05-15.* Before launching the 141-thread production run, an audit ([logs/audit_implementation_results.md](audit_implementation_results.md)) surfaced a regression introduced by an earlier prompt fix.

### Background

The Stage B `STAGE_B_PROMPT` in [src/thread_summaries.py](../src/thread_summaries.py) widened the POLEMICAL EXCHANGE definition to include `external_defense` (commit `62dc9b4`), so that cross-paper Hebrew apologetics against the Tiszaeszlár blood libel would correctly count as polemic. That fix worked for thread 406 — but it had a side effect: it loosened the definition too far elsewhere. Thread 433 (a 14-doc cluster of *unrelated* 1875 disputes — Faltz divorce shaming, Brüll Mishnah critique, Tripoli rabbi defense, Pesach kashrut, Goldberg agunah — all sharing polemical *vocabulary* but with no shared adversary, target, or proposition) flipped under Gemini from a previous "topical-only basket" verdict to **0.85 internal explicit** post-widening. The earlier Opus + Gemini consensus on 433 was that it is a basket of unrelated polemics; the post-widening Gemini narrative absorbed them into "several distinct but related internal Jewish polemics," collapsing the very distinction that made the earlier verdict useful.

### The clause

A COUNTERWEIGHT clause was added to the prompt:

> *Polemical register alone (תועבה, חרפה, רהיטות, sharp rhetoric) is NOT a polemical exchange. A "polemical exchange" REQUIRES a shared dispute referent: a common adversary, a common proposition being attacked/defended, or a chain of articles responding to each other. If the cluster contains polemical articles that do not share an adversary, target, or proposition, prefer is_polemic_thread=false with polemic_type=topical-only — even when the articles are individually polemical in tone. Syndicated public shaming of a single named individual across papers (no defender, no rejoinder) is one-sided denunciation, NOT a polemical exchange. Apply this counterweight especially when n_docs is large and span_days is wide.*

### What it changed on the top-30

Re-running Gemini + Sonnet Stage B and Opus arbitration on the 13 disagreement targets after adding the clause:

| Thread | Before (post-widening) | After (counterweight) |
|---|---|---|
| 433 | Gemini 0.85 internal explicit | **Gemini 0.20 n/a topical-only ✓**; Opus 0.35 n/a; Sonnet 0.25 + sub_thread_signal=true |
| 54 | Sonnet true / Gemini false (binary flip) | All three: n/a topical-only |
| 437 | All three: internal polemic | Opus 0.15 topical-only; cheap models still polemic |
| 385 | Direction disagreement | Opus 0.15 topical-only; cheap models still polemic |
| 392 | 0.49 score gap | Opus 0.15 n/a; Sonnet 0.38 implicit; Gemini 0.78 mixed |

Arbitration targets dropped **13 → 4**. Thread 432 (the 18-vs-7 outlier disagreement) now has all three models agreeing internal-polemic with sub_thread_signal=true.

### The trade-off, and why this is also a future-project decision

The counterweight produces what we call a **strict reading** of polemic: a thread is polemical iff it has an identifiable shared referent — a named adversary, a defended proposition, or a chain of named cross-paper responses. Implicit cross-paper engagement on the same broad question, without a specific named exchange, is treated as topical-only.

This is a definitional choice, not a fact about the corpus. Threads 437, 385, and 392 sit on the borderline: Opus reads them as topical-only under the strict reading; Sonnet and Gemini still see implicit polemic. Whether implicit engagement counts as polemic depends on what the analytical unit is supposed to support:

- **Strict reading** (current production setting): every thread labeled polemic can answer "what was the dispute, who was disputing what?" Cleaner unit, more conservative count, better suited to claims of the form "we found undocumented press polemics."
- **Loose reading** (without counterweight): polemical register + cross-paper participation is sufficient. More inclusive, more diluted, better suited to claims of the form "we found patterns of public sphere engagement on shared political/religious questions."

The pilot adopts the **strict reading** because the project's claim is the first kind: we want every "polemic thread" to have a verifiable referent that a reader can interrogate. This is a defensible editorial choice for *this* pilot; it is **not** the only defensible choice for future iterations of the project. Specifically:

1. **A loose-reading complement is worth building post-pilot.** Run both prompts on the same threads, surface both verdicts, and let downstream analysis choose. The Stage B parquet schema is already model-keyed; a `prompt_variant` column would let both readings co-exist.
2. **The "borderline" threads (437, 385, 392) are themselves interesting.** They are the cases where the analytical definition meaningfully changes the count — flagging them is a finding, not a defect. The pilot report should name them explicitly.
3. **The sub_thread_signal field is an empirical proxy for "the prompt is being stressed."** Threads where Stage B raises sub_thread_signal=true are exactly the threads where the strict/loose distinction matters most. Future iterations can use this signal to route specific threads to a different prompt variant.

### What this means for downstream interpretation

When the pilot or any writeup reports thread counts (e.g., "X of the 171 engaged threads are polemic per the LLM pipeline"), the count is **conditional on the strict reading**. A reader who prefers the loose reading would, depending on the rate of borderline threads, see a count 10–20% higher.

---

## Methodological decision: mega-threads and sub-clustering

*Added 2026-05-15.* Four engaged threads (374 / 115 docs, 412 / 89 docs, 407 / 47 docs, 141 / 41 docs) are an order of magnitude larger than the median (4 docs). These threads survived the C.2 span-cap (730d threshold) because their internal edge density was high enough that the date-cut did not fragment them. Thread 374 in particular spans 1114 days. This raises a structural question the pilot does not fully resolve.

### What the LLM verdicts say about the mega-threads

Stage B (with counterweight) on 374 and 412:

- **374** (Hibbat Zion era MGD-heavy): Sonnet 0.68 mixed with 12 outliers and sub_thread_signal=true. Gemini 0.90 internal with 4 outliers. Opus did not arbitrate (not in trigger set, but is in remaining-arbitration list).
- **412** (post-1881 BILU / Old Yishuv): Sonnet 0.62 mixed with 6 outliers, Gemini 0.90 mixed with 5 outliers.

Both models flag both threads as polemic but raise `sub_thread_signal=true` aggressively: the models are *telling us* the cluster contains multiple sub-disputes. Reading the narratives, the disputes within 374 include (a) internal Jewish nationalism vs cosmopolitanism, (b) Palestine vs America emigration, (c) external-defensive responses to Tiszaeszlár / Stoecker / Rohling. Within 412: (a) Palestine-vs-America, (b) attacks on indifferent Jewish magnates, (c) defense against Renan's racial theories. The C.2 threading correctly identified that these are *connected* (shared post-1881 polemical moment, recurring actors), but the connection is "this was the dominant ideological moment in HaMagid for three years," not "this is one polemic."

### Three available ways to handle this

| Option | Where the split happens | Effect on the 141-run |
|---|---|---|
| **A. Upstream graph split** | Modify [scripts/c1_c2_pipeline.py](../scripts/c1_c2_pipeline.py) to apply community detection (Louvain modularity, edge-betweenness) on dense clusters before threading. Changes the set of threads. | Yes — rewrites the 141 set; requires re-running C.2 → re-run Stage A → re-run Stage B. ~1 day plus Stage-A cost on the changed scope. |
| **B. LLM-level sub_thread_signal + post-hoc split** | Keep C.2 threads as the analytical unit. Use the new `sub_thread_signal` field to flag mega-threads, then run a second-pass LLM that enumerates each sub-dispute and assigns each doc to one. | No — the 141 run already emits sub_thread_signal; sub-clustering becomes a downstream pass. |
| **C. Treat as-is for pilot, document the mega-thread cases, defer split to post-pilot** | Stage B narratives already describe the sub-disputes inside each mega-thread. The pilot accepts that "thread" means "cluster" not "single polemic" for these specific cases. | No — but the pilot report must be explicit about which threads are mega-threads and what `sub_thread_signal` raised means. |

### Why the pilot chose Option C (with a path to Option B)

The decision is to **launch the 141-thread production run with the current C.2 threads**, document the mega-thread issue explicitly in the pilot report, and treat Option B as a post-pilot deliverable. Reasons:

1. **The C.2 threading is defensible as an analytical unit even when n_docs is large.** The mega-threads are not "wrong" — they correctly identify dense, sustained, multi-paper engagement on a clustered set of related questions. The question is whether they should be presented as one thread or as a tree of sub-threads.
2. **"What is a polemic boundary?" is itself a research question, not an implementation detail.** Option A bakes a graph-theoretic answer into the pipeline before we have studied the question. Modularity-detected sub-clusters are not the same thing as polemic-boundary-detected sub-clusters; the alignment between them is what we would actually want to measure.
3. **Stage B already produces the data we'd need to do Option B properly.** The narratives describe the sub-disputes; `sub_thread_signal` flags which threads have them; `outlier_docs` removes the genuinely off-topic items. A second-pass LLM ("enumerate the disputes inside this thread and assign each doc to one") is straightforward to add when we are ready.
4. **The 141 run does not commit us to one approach.** All threads continue to live in `data/threads.parquet`; the LLM verdicts live in `data/thread_llm_summaries.parquet`. Sub-clustering, when it happens, will be a derived artifact, not a rewrite.

### What the pilot report must explicitly say

- **Which threads are mega-threads** (374, 412, 407, 141 in the top 30; the production run will likely add 2-4 more at this scale).
- **What `sub_thread_signal=true` means** in the parquet — that the LLM flagged the cluster as containing 2+ distinct disputes.
- **That the pilot's "polemic thread" count includes these mega-threads as single units**, even though the LLM narrative for each enumerates multiple sub-disputes.
- **That a sub-clustering pass is a post-pilot deliverable**, not an open methodological hole. The data already exists to do it; what is missing is a decision about whether sub-clusters should be modularity-defined, narrative-defined, or rebuttal-edge-defined.

### Items flagged for future human / RA review (not blockers for the production run)

These are "would benefit from domain-expert eyes" cases that should be listed in the pilot report as examples of *the kind of work that needs to happen post-pilot*, rather than fixed within the pilot:

- **Mega-thread sub-clustering review.** For threads 374 and 412 specifically, an expert read of the LLM narratives + the corresponding outlier flags would let us validate whether the cheap models' "mixed" verdicts and Sonnet's larger outlier counts actually reflect 2-3 distinct polemics that should be reported separately in any writeup that names these threads.
- **The "not-found" / canonical-event distinction in Q.A.** The 4 not-found threads (395, 404, 397, 433) are the strongest novelty bid but were never spot-checked against Hebrew-language indexes (RAMBI). Documented in the pilot as a research opportunity; not a verification gap for the production run itself.
- **Borderline-polemic threads under the strict reading (437, 385, 392).** Whether these should remain "polemic" under a future loose-reading complement is a definitional decision that should be made explicit when the pilot is written up, not buried in the data.

---

## Question C — Cross-source threading: do threads cross press / egeret / Ben-Yehuda boundaries?

*Added 2026-05-14.* All 438 threads in `data/threads.parquet` are press-only by design (threading is restricted to `source == "press"` in `scripts/c1_c2_pipeline.py`). A concentrated effort was made to find source-crossing threads via two avenues: (1) leveraging prior `mechanical_newspaper` detections of press mentions in non-press docs and looking for thread attachments, and (2) considering whether to extend threading to the dated non-press subcorpus.

### Finding C.1 — Cross-source bridges are sparse and most candidates are out-of-period or out-of-corpus

Of 6128 `mechanical_newspaper` mentions in `data/pilot_references.parquet`, 5180 originate in non-press docs (94 unique mentioners). After two filters:

- **Target newspaper must be in our press corpus** (HLB, HMZ, HZF, HZT, MGD). Most mentioned papers (העברי, הכרמל, השחר, חבצלת, הפסגה) are 19th-century Hebrew papers we did not ingest. Only ~23 mention rows survive.
- **"דבר" inflates raw counts massively** (4666/5180). The extractor matches the common Hebrew word *davar* ("matter, thing") and not only the 20th-c. *Davar* newspaper, which founded only in 1925 anyway. This is an extractor false-positive class worth fixing.
- **Mentioning doc must be in the press window 1862–1888** (using `data/egeret_dates.parquet` for egeret dating, which the polemic_pool's `year` field does not carry). Most egeret mentioners wrote in the 1900s–1930s, recollecting rather than reacting. **Only 3 in-window egeret bridges survive** (2 unique docs).

### Finding C.2 — The two in-window egeret bridges resolve in surprising ways

- **egeret_3439** (Y.L. Gordon, 1868-06-21, private letter to R. Joshua Steinberg): mentions reading Steinberg's letter in Ha-Maggid. Cannot thread because **our press corpus contains zero MGD docs from 1868** — the target year isn't ingested.
- **egeret_2136** (Y.Y. Rivlin, dated 25 Iyar 5643): an open letter to the editor of Ha-Maggid reporting the Jerusalem celebration of Tsar Alexander III's coronation. The Ben-Yehuda igrot corpus carries an editorial annotation "המגיד, כ״ב בסיון תרמ״ג" (22 Sivan 5643 = 1883-06-27) marking the publication venue. **The cited press article is in our corpus**: `press_68753` from the 1883-06-27 MGD issue contains all six distinctive proper nouns of Rivlin's letter (Meir Panigel, Shmuel Salant, Beit Yaakov synagogue, Alexander III, prayer-text gloss). The press text is signed "יוספ ריבלינ סוה״כ".

**egeret_2136 IS press_68753 — same text, two corpora.** This reframes "cross-source threading" as a question about *dual publication*, not bridging between distinct documents.

### Finding C.3 — Dual publication is a real phenomenon, not a one-off

A fingerprint scan (six 60-char windows per egeret doc, substring search against press) over 401 in-window egeret docs found **5 dual-publication pairs** (lower bound — the method has known false negatives from punctuation/normalization differences; the Rivlin pair only matched on some windows). Output: `data/egeret_press_dual_publication.parquet`.

| author | n_pairs |
|---|---|
| Peretz Smolenskin | 2 |
| Yosef Yehoshua Rivlin | 2 |
| Eliezer Ben-Yehuda | 1 |

The matched authors — Smolenskin, Rivlin, Ben-Yehuda — are exactly the writer-editors of the 1880s maskilic public sphere for whom dual public/private epistolary practice would be expected. A confirmed ~1% rate (lower bound) on the right authors is more informative than a high random rate: it indicates **open-letter publication as a selective, deliberate practice by specific writers**, not a wholesale phenomenon.

### Finding C.4 — Methodological clarifications surfaced during this investigation

1. **Threading uses day-resolution dates internally; the year-only collapse is only in `polemic_pool.parquet`.** `scripts/c1_c2_pipeline.py` reads from `corpus.parquet` (which retains full `date`, 23444/23444 coverage on press), parses `pub_date`, and builds edges with day-level windows. `threads.parquet`'s `span_days` is real day math. Analyses and bridge searches should join on `corpus.parquet`, not `polemic_pool`.
2. **A press doc_id is not an authored unit.** `press_68753` is a 26K-character column bundling multiple letters from different correspondents (Rivlin's segment is preceded by another signed by Dov Baumgarten). Any text-overlap or attribution analysis must operate at the segment level, or unit mismatch will mask matches.
3. **The `mechanical_newspaper` extractor over-matches "דבר".** Headline mention counts in `data/pilot_references.parquet` are inflated and should be filtered.

### Implications for the pilot writeup

- Cross-source threading at scale is currently blocked by (a) narrow press corpus — only 5 papers, mostly missing what egeret writers actually cite; (b) genre mismatch — engaged-polemic threading would not cluster celebratory/reportorial letters-to-the-editor anyway.
- But the dual-publication phenomenon is a publishable side finding: **the egeret corpus and the press corpus are not independent** for select 1880s authors. Any analytic that treats them as such is double-counting.
- The Rivlin → press_68753 identification is concrete evidence that egeret entries by prolific press contributors merit a systematic cross-check.

### Post-pilot expansions queued

- **Broaden press ingest** to Ha-Ivri, Ha-Karmel, Ha-Shahar, Havatzelet — these are what 19th-c. egeret writers actually cite.
- **Date `polemic_candidates`** to expand the bridge pool beyond egeret. Framing this dating effort more broadly: it extends the project from a polemic-search task into a **public-sphere study**, where the unit of analysis is the circulating text across press / letters / Ben-Yehuda's open-publication corpus.
- **Refine the dual-publication scan** with fuzzy / character-n-gram matching for a better rate estimate.
- **Fix the `mechanical_newspaper` extractor** to suppress the common-word "דבר" false-positive class.
- **Cross-check whether other egeret entries by prolific press contributors** (Gordon, Smolenskin, Ben-Yehuda) were themselves published in press, beyond the five already found.

---

## Synthesis — why one mixed pipeline produced thin results on both sides

*Draft, 2026-05-15. To be elaborated in a later session.*

Read together, the pilot's findings point one direction: **polemic detection and Ben-Yehuda-corpus enrichment are different tasks, and running them through one pipeline gave a thin result on each side.**

Evidence already in this report:

- **Threading is press-shaped.** All 438 threads in `threads.parquet` are press-only by construction; the pipeline's reference edges + interleave + semantic similarity were tuned to press dynamics (Q.C.1).
- **Cross-source bridges are sparse and largely out-of-period.** After filtering, only 3 in-window egeret→press bridges survived from 5180 candidate mentions, and the strongest of those collapsed into a *dual-publication* identity rather than a thread edge (Q.C.1–C.2).
- **Egeret-internal threading works on its own terms but is not polemic-shaped.** 17 author-pair threads, 18% polemic yield, all internal — a useful pilot but a different object than the press polemics (see [egeret_threading_close_read.md](egeret_threading_close_read.md), [project_egeret_threading](../memory/project_egeret_threading.md)).
- **Actor recall stays at 6.7% even with the Hebrew-prompt fix** — i.e. the entity layer is not yet load-bearing enough to anchor cross-corpus linking, which is exactly the layer a BY-centric project would need first (see [project_ner_entity_linking](../memory/project_ner_entity_linking.md)).
- **0/20 threads have direct press-polemic scholarship**, but 11/20 underlying events are canonical (Q.A). Press-polemic scholarship is the gap; BY enrichment doesn't fill it.

The two tasks pull in opposite directions: press-polemic work wants *dispute structure + secondary literature + RAMBI*, while BY work wants *entities + outgoing references + cross-genre linking* with polemic at most a tag. A single pipeline that tries both produces a press-polemic output that under-uses BY material and a BY output that doesn't really exist.

### Proposed follow-up: two projects, one shared infrastructure layer

1. **Press-polemic project.** Primary axis: lit-review + Wikidata + RAMBI on the polemic threads. BY items appear as a **hybrid** secondary enrichment — retrieval surfaces ~20 candidate BY items per thread, an LLM pass + light human review trims to 3–5 displayed. This makes the press-polemic corpus *browsable* while keeping the displayed BY links curation-grade.
2. **BY-corpus project.** Primary axis: NER + reference extraction. Polemic drops out as the organizing question; the unit of analysis is either (a) a single BY item plus its outgoing references, or (b) a cross-genre topic cluster spanning letters/poems/essays/press. Both unit-of-analysis variants are carried into the demos rather than chosen now.

Shared infrastructure between the two: the entity layer (currently 6.7% recall) and the reference-extraction layer, which both projects need but neither would build alone.

Demos illustrating each branch follow in separate files (`logs/demos/`).

