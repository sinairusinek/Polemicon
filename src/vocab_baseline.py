"""
vocab_baseline.py — Test Question B from the pilot conclusion.

For each top-N engaged thread, ask: would simple vocabulary search alone
recover the same set of documents the C.2+LLM pipeline identified?

Two query strategies × two retrieval engines = 4 baselines:
  - query = top-K TF-IDF terms from thread docs vs background corpus
  - query = actor names extracted from LLM `actors` field
  - engine = union of substring matches over (text + headline)
  - engine = TF-IDF cosine similarity over full corpus (char 3-5 grams, OCR-robust)

Outputs:
  data/vocab_baseline_eval.parquet   per-thread per-(strategy,engine) metrics
  data/vocab_baseline_missing.parquet thread docs not retrieved at all → most interesting cases

Usage:
  python src/vocab_baseline.py --top 30 --topk-terms 12
  python src/vocab_baseline.py --thread-ids 412,381
"""
import os
import re
import json
import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
from cleaning import restore_final_forms  # noqa: E402

DATA = ROOT / "data"

# Minimal Hebrew stopword list — pronouns, common function words, generic
# corpus-frequent terms surfaced by the unfiltered TF-IDF as top thread terms.
HEBREW_STOPWORDS = {
    "עלינו", "אנחנו", "אתמ", "לכמ", "אבותינו", "עמנו", "כלנו", "אחינו", "בנו",
    "בני", "בני ישראל", "עת", "המה", "נא", "אשר", "כי", "אנו", "אני", "הוא",
    "היא", "הם", "הן", "אתה", "את", "אתם", "אתן", "אלה", "אלו", "זה", "זאת",
    "כל", "גם", "רק", "אם", "או", "כמו", "כן", "לא", "לו", "לה", "להם", "להן",
    "של", "אל", "על", "עם", "מן", "מ", "ב", "ל", "ה", "ו", "ש", "כ",
    "יש", "אין", "היה", "היתה", "היו", "יהיה", "תהיה", "יהיו",
    "אמר", "אמרה", "אמרו", "ויאמר", "ויהי", "אך", "אבל", "כאשר",
    "עוד", "כבר", "אז", "פה", "שם", "מה", "מי", "איך", "למה", "מתי",
}


def load_inputs():
    threads = pd.read_parquet(DATA / "threads.parquet")
    corpus = pd.read_parquet(ROOT / "corpus.parquet",
                             columns=["doc_id", "text", "headline", "newspaper"])
    corpus["text"] = corpus["text"].fillna("")
    corpus["headline"] = corpus["headline"].fillna("")
    corpus["doc_text"] = corpus["headline"] + " " + corpus["text"]
    llm = pd.read_parquet(DATA / "thread_llm_summaries.parquet")
    return threads, corpus, llm


def select_threads(threads, top_n, thread_ids):
    if thread_ids:
        return threads[threads["thread_id"].isin(thread_ids)].copy()
    return (threads[threads["thread_type"] == "engaged"]
            .sort_values("score", ascending=False)
            .head(top_n)
            .copy())


def extract_actors(llm: pd.DataFrame, thread_id: int) -> list[str]:
    """Union of actor names across all models for the thread."""
    rows = llm[llm["thread_id"] == thread_id]
    names = set()
    for _, r in rows.iterrows():
        raw = r.get("actors")
        if not raw:
            continue
        try:
            lst = json.loads(raw) if isinstance(raw, str) else list(raw)
        except Exception:
            continue
        for a in lst:
            a = str(a).strip()
            # strip parenthetical English transliteration if any: "פינס (Pines)" → "פינס"
            for sep in (" (", " - "):
                if sep in a:
                    a = a.split(sep, 1)[0].strip()
            if a and len(a) >= 2:
                names.add(a)
    return sorted(names)


def tfidf_query_terms(thread_docs: pd.DataFrame, corpus_vec, vectorizer,
                      thread_corpus_indices: list[int], topk: int) -> list[str]:
    """Top-K word terms whose TF-IDF is most concentrated in the thread vs corpus."""
    # Use the already-fit vectorizer to score thread docs vs background
    thread_mat = corpus_vec[thread_corpus_indices]
    bg_mat = corpus_vec  # background is the whole corpus (good enough as denominator)
    thread_mean = np.asarray(thread_mat.mean(axis=0)).ravel()
    bg_mean = np.asarray(bg_mat.mean(axis=0)).ravel()
    # Score = thread mean - bg mean (concentration)
    score = thread_mean - bg_mean
    feats = vectorizer.get_feature_names_out()
    ranked = np.argsort(-score)
    terms = []
    for i in ranked:
        if score[i] <= 0:
            break
        term = feats[i]
        if term in HEBREW_STOPWORDS:
            continue
        terms.append(term)
        if len(terms) >= topk:
            break
    return terms


def substring_retrieve(corpus: pd.DataFrame, query_terms: list[str], max_results: int = 1000):
    """Return ranked doc_ids by count of distinct query terms matched in text+headline."""
    if not query_terms:
        return []
    terms = [re.escape(t) for t in query_terms if t]
    if not terms:
        return []
    pattern = re.compile("|".join(terms))
    hits = corpus["doc_text"].apply(lambda x: len(pattern.findall(x)))
    out = pd.DataFrame({"doc_id": corpus["doc_id"], "score": hits})
    out = out[out["score"] > 0].sort_values("score", ascending=False).head(max_results)
    return out["doc_id"].tolist()


def tfidf_search_retrieve(query: str, corpus_vec, vectorizer, corpus_ids: list[str],
                          max_results: int = 1000):
    """Cosine similarity between query string and corpus."""
    if not query.strip():
        return []
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, corpus_vec).ravel()
    top = np.argsort(-sims)[:max_results]
    return [(corpus_ids[i], float(sims[i])) for i in top if sims[i] > 0]


def metrics(retrieved_ids: list[str], gold_ids: set[str], n_gold: int) -> dict:
    """Recall and precision at several K-values plus AP."""
    ks = [n_gold, 2 * n_gold, 5 * n_gold, 100, 500]
    out = {}
    retrieved_set = set()
    hits = 0
    ap_terms = []
    for rank, did in enumerate(retrieved_ids, 1):
        retrieved_set.add(did)
        if did in gold_ids:
            hits += 1
            ap_terms.append(hits / rank)
        for k in ks:
            tag = f"k{k}"
            if rank == k:
                out[f"recall@{tag}"] = round(hits / max(1, n_gold), 3)
                out[f"precision@{tag}"] = round(hits / k, 3)
    # Fill in missing K (if retrieved list shorter than K)
    for k in ks:
        tag = f"k{k}"
        if f"recall@{tag}" not in out:
            out[f"recall@{tag}"] = round(hits / max(1, n_gold), 3)
            out[f"precision@{tag}"] = round(hits / max(1, len(retrieved_ids)), 3) if retrieved_ids else 0.0
    out["ap"] = round(sum(ap_terms) / max(1, n_gold), 3)
    out["found"] = hits
    out["missed"] = sorted(gold_ids - retrieved_set)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--thread-ids", type=str, default="")
    ap.add_argument("--topk-terms", type=int, default=12)
    ap.add_argument("--core-only", action="store_true",
                    help="Restrict gold set to non-outlier docs (union across models from thread_llm_summaries.outlier_docs).")
    args = ap.parse_args()

    threads, corpus, llm = load_inputs()
    thread_ids = [int(x) for x in args.thread_ids.split(",")] if args.thread_ids else None
    thread_view = select_threads(threads, args.top, thread_ids)
    print(f"Evaluating {len(thread_view)} threads against {len(corpus):,}-doc corpus...")

    # Fit one vectorizer for both term-extraction and search retrieval.
    # Word 1-2grams: good for actor names and topic words; not too OCR-fragile when the corpus is large.
    print("Fitting word 1-2gram TF-IDF over corpus...")
    word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=80_000,
                               min_df=3, max_df=0.4)
    corpus_word_mat = word_vec.fit_transform(corpus["doc_text"].values)
    corpus_ids = corpus["doc_id"].tolist()
    id_to_idx = {did: i for i, did in enumerate(corpus_ids)}

    # Build per-thread outlier set (union across all models) if --core-only.
    outlier_by_thread: dict = {}
    if args.core_only:
        for _, lr in llm.iterrows():
            raw = lr.get("outlier_docs")
            if not raw:
                continue
            try:
                items = json.loads(raw) if isinstance(raw, str) else list(raw)
            except Exception:
                continue
            s = outlier_by_thread.setdefault(int(lr["thread_id"]), set())
            for it in items:
                did = (it.get("doc_id") if isinstance(it, dict) else str(it)) or ""
                if did:
                    s.add(did.strip())

    eval_rows = []
    missing_rows = []
    for _, tr in thread_view.iterrows():
        tid = int(tr["thread_id"])
        gold = [d.strip() for d in str(tr["doc_ids"]).split(",") if d.strip()]
        if args.core_only:
            drop = outlier_by_thread.get(tid, set())
            gold = [d for d in gold if d not in drop]
        gold_ids = set(gold)
        if not gold_ids:
            continue
        thread_indices = [id_to_idx[d] for d in gold if d in id_to_idx]
        n_gold = len(gold)

        # Query strategy 1: TF-IDF top terms
        terms = tfidf_query_terms(None, corpus_word_mat, word_vec, thread_indices,
                                  topk=args.topk_terms)
        # Query strategy 2: actors from LLM summary
        actors = extract_actors(llm, tid)
        actor_query = " ".join(actors)

        # Engines
        sub_tfidf = substring_retrieve(corpus, terms, max_results=1000)
        sub_actors = substring_retrieve(corpus, actors, max_results=1000)
        ret_tfidf = tfidf_search_retrieve(" ".join(terms), corpus_word_mat, word_vec,
                                          corpus_ids, max_results=1000)
        ret_actors = tfidf_search_retrieve(actor_query, corpus_word_mat, word_vec,
                                           corpus_ids, max_results=1000)

        configs = [
            ("tfidf_terms",  "substring", sub_tfidf,                            terms),
            ("tfidf_terms",  "tfidf_cos", [d for d, _ in ret_tfidf],            terms),
            ("actor_names",  "substring", sub_actors,                           actors),
            ("actor_names",  "tfidf_cos", [d for d, _ in ret_actors],           actors),
        ]
        for strategy, engine, retrieved_ids, query_used in configs:
            m = metrics(retrieved_ids, gold_ids, n_gold)
            eval_rows.append({
                "thread_id": tid,
                "n_thread": n_gold,
                "strategy": strategy,
                "engine": engine,
                "query": json.dumps(query_used[:args.topk_terms], ensure_ascii=False),
                **{k: v for k, v in m.items() if k != "missed"},
                "missed_count": len(m["missed"]),
            })
            for d in m["missed"]:
                missing_rows.append({
                    "thread_id": tid,
                    "doc_id": d,
                    "strategy": strategy,
                    "engine": engine,
                })

    eval_df = pd.DataFrame(eval_rows)
    miss_df = pd.DataFrame(missing_rows)

    suffix = "_core" if args.core_only else ""
    eval_path = DATA / f"vocab_baseline_eval{suffix}.parquet"
    miss_path = DATA / f"vocab_baseline_missing{suffix}.parquet"
    eval_df.to_parquet(eval_path, index=False)
    miss_df.to_parquet(miss_path, index=False)
    print(f"\nWrote {eval_path} ({len(eval_df)} rows)")
    print(f"Wrote {miss_path} ({len(miss_df)} rows)")

    # Quick summary printout
    def _val_at_N(row, prefix):
        col = f"{prefix}@k{int(row['n_thread'])}"
        v = row.get(col)
        return float(v) if pd.notna(v) else None

    eval_df["recall_at_N"] = eval_df.apply(lambda r: _val_at_N(r, "recall"), axis=1)
    eval_df["precision_at_N"] = eval_df.apply(lambda r: _val_at_N(r, "precision"), axis=1)

    print("\n=== Mean metrics by (strategy, engine) ===")
    print(eval_df.groupby(["strategy", "engine"])
                 [["recall_at_N", "precision_at_N", "ap", "found", "missed_count"]]
                 .mean().round(3).to_string())

    print("\n=== Per-thread recall@N (rows=threads, cols=strategy/engine) ===")
    print(eval_df.pivot_table(index="thread_id", columns=["strategy", "engine"],
                              values="recall_at_N").round(2).to_string())


if __name__ == "__main__":
    main()
