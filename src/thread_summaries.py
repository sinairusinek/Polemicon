"""
thread_summaries.py — LLM summarization + polemic re-evaluation of C.2 threads.

Two-stage map-reduce so large threads (89-115 docs) fit in context:

  Stage A  per-doc mini-summary (cached by doc_id+model)
            output: data/thread_doc_summaries.parquet
  Stage B  thread-level summary + polemic verdict
            output: data/thread_llm_summaries.parquet

Bake-off lineup:
  - gemini_flash3  : paid API, Gemini 3 Flash         (~$0.25 for top-10 threads)
  - cli_sonnet     : `claude -p --model claude-sonnet-4-6`   (Max plan, $0)
  - cli_opus       : `claude -p --model claude-opus-4-7`     (tiebreaker, $0)

Usage:
    python src/thread_summaries.py --top 10 --models gemini_flash3,cli_sonnet
    python src/thread_summaries.py --top 10 --models cli_opus   # tiebreaker
    python src/thread_summaries.py --thread-ids 412,408         # specific threads
"""
import os
import sys
import json
import time
import fcntl
import asyncio
import argparse
import subprocess
from contextlib import contextmanager
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
load_dotenv(ROOT / ".env")

DATA = ROOT / "data"
DOC_SUMMARIES_PATH = DATA / "thread_doc_summaries.parquet"
THREAD_SUMMARIES_PATH = DATA / "thread_llm_summaries.parquet"

# --- Model configs ---

MODELS = {
    "gemini_flash3": {
        "provider": "google",
        "model_id": os.getenv("GEMINI_FLASH3_MODEL", "gemini-3-flash-preview"),
        "tier": "cheap",
    },
    "cli_sonnet": {
        "provider": "claude_cli",
        "model_id": os.getenv("CLAUDE_SONNET_MODEL", "claude-sonnet-4-6"),
        "tier": "free",
    },
    "cli_opus": {
        "provider": "claude_cli",
        "model_id": os.getenv("CLAUDE_OPUS_MODEL", "claude-opus-4-7"),
        "tier": "free",
    },
}

DOC_TRUNC_CHARS = 4000           # per-doc input cap for Stage A
EXCERPT_CHARS = 1200             # length of representative excerpts in Stage B
N_EXCERPTS = 4                   # how many full excerpts in Stage B
BATCH_DELAY = 0.3                # between API calls

# --- Prompts ---

STAGE_A_PROMPT = """You are a research assistant reading a 19th-century Hebrew press article (Haskalah era, 1862-1888).

Article metadata: {metadata}

Article text (truncated):
---
{text}
---

Produce a JSON object (no markdown fences, no prose outside JSON) with EXACTLY these keys:
- "summary_he": one Hebrew sentence (≤ 30 words) stating the main claim or topic.
- "is_polemical": true if the article attacks/defends a position, responds to another writer, or participates in a public dispute; false if descriptive/informational/news/literary.
- "key_actors": JSON list of person names, organizations, or newspaper names mentioned as participants (Hebrew or transliterated; empty list if none clear).
- "stance_marker": one short Hebrew phrase quoted from the text that best signals stance (or "" if none).

Respond with ONLY the JSON object.
"""

STAGE_B_PROMPT = """You are evaluating whether a cluster of 19th-century Hebrew press articles, grouped algorithmically by reference + co-occurrence + semantic similarity, represents an actual POLEMICAL EXCHANGE (multi-author dispute, debate, controversy) — or merely TOPICAL CO-OCCURRENCE (same subject discussed without dispute).

Cluster metadata:
- {n_docs} articles across {n_papers} newspapers
- Span: {span_days} days
- Newspapers: {newspapers}
- Edge types detected: {edge_types}
- Heuristic threading score: {heuristic_score:.1f}

Per-article mini-summaries (Hebrew, one line each). Format: [date · paper · author? · doc_id] "headline" summary_he.
Use the dates to detect temporal escalation/lulls; use the paper attribution to judge whether this is a cross-paper exchange or one paper dominating; use authors to identify recurring polemicists.

{summaries_block}

Representative excerpts (most polemic-likely articles):
{excerpts_block}

Produce a JSON object (no markdown fences, no prose outside JSON) with EXACTLY these keys:
- "topic_label": 5-10 word English topic label
- "topic_label_he": short Hebrew topic label (≤ 8 words)
- "narrative": 2-3 sentence English description of WHO is arguing WHAT, and across which papers
- "actors": JSON list of the principal persons/papers/groups in the dispute. CRITICAL: each name MUST be given in its original Hebrew form exactly as it appears in the source text (e.g. "פינס", "המגיד", "סמולנסקין"), since these names will be matched against the Hebrew corpus. You may append an English transliteration in parentheses for clarity (e.g. "פינס (Pines)"), but the Hebrew form must come first. Do NOT return English-only or transliterated-only names.
- "is_polemic_thread": true if this is a genuine polemical exchange, false if it is topical-only / news / non-polemic
- "polemic_score": float 0.0-1.0 expressing how strongly this is a real polemical exchange (0 = pure co-occurrence, 1 = clear sustained polemic)
- "polemic_type": one of "explicit" | "implicit" | "meta-polemic" | "topical-only" | "none"
- "polemic_direction": one of "internal" | "external_defense" | "mixed" | "n/a".
    * "internal"         = intra-Jewish dispute (e.g. Hasidim vs Mitnagdim, Haskalah vs Orthodoxy, Reform vs traditionalists, paper-vs-paper, rabbi-vs-rabbi).
    * "external_defense" = cross-paper apologetics responding to external antisemitism / blood libel (Tiszaeszlár, Damascus affair) / missionary attacks / hostile non-Jewish press, where Jewish writers across papers close ranks against an outside accuser.
    * "mixed"            = both registers genuinely active in the same thread.
    * "n/a"              = thread is not polemical (use whenever is_polemic_thread is false).
- "evidence": 1-2 sentence English justification, citing concrete signals from the summaries
- "rebuttal_edges": JSON list of [doc_a, doc_b, relation] triples where doc_a explicitly responds to / attacks / defends-against doc_b (or where summaries strongly indicate this). `relation` is one of "responds-to", "attacks", "defends-against", "cites". Empty list if no specific cross-doc edges are identifiable from the summaries. Use the doc_id strings exactly as they appear in the summary lines.

Respond with ONLY the JSON object.
"""

# --- Helpers ---

def parse_json(raw: str) -> dict:
    raw = raw.strip()
    # strip code fences if present
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0]
    # extract outermost {...}
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start:end + 1]
    return json.loads(raw)


def build_doc_metadata(row) -> str:
    parts = []
    for k, label in [("date", "date"), ("newspaper", "paper"), ("author", "author"), ("headline", "headline")]:
        v = row.get(k)
        if pd.notna(v) and str(v) not in ("", "nan"):
            parts.append(f"{label}={str(v)[:80]}")
    return "; ".join(parts) if parts else "(no metadata)"


def truncate(text: str, n_chars: int) -> str:
    if not isinstance(text, str):
        return ""
    return text[:n_chars]


# --- Stage A: per-doc summaries ---

async def stage_a_gemini(doc_rows: pd.DataFrame, model_id: str) -> list[dict]:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    client = genai.GenerativeModel(model_id)
    results = []
    for i, (_, row) in enumerate(doc_rows.iterrows()):
        prompt = STAGE_A_PROMPT.format(
            metadata=build_doc_metadata(row),
            text=truncate(row.get("text", ""), DOC_TRUNC_CHARS),
        )
        t0 = time.time()
        try:
            resp = await client.generate_content_async(prompt)
            parsed = parse_json(resp.text)
            usage = getattr(resp, "usage_metadata", None)
            in_tok = getattr(usage, "prompt_token_count", 0) if usage else 0
            out_tok = getattr(usage, "candidates_token_count", 0) if usage else 0
            results.append({
                "doc_id": row["doc_id"],
                "summary_he": parsed.get("summary_he", ""),
                "is_polemical": bool(parsed.get("is_polemical", False)),
                "key_actors": json.dumps(parsed.get("key_actors", []), ensure_ascii=False),
                "stance_marker": parsed.get("stance_marker", ""),
                "_input_tokens": in_tok,
                "_output_tokens": out_tok,
                "_wall_seconds": time.time() - t0,
                "_error": None,
            })
        except Exception as e:
            results.append({
                "doc_id": row["doc_id"], "summary_he": "", "is_polemical": False,
                "key_actors": "[]", "stance_marker": "",
                "_input_tokens": 0, "_output_tokens": 0,
                "_wall_seconds": time.time() - t0, "_error": str(e)[:200],
            })
        print(f"  [{i+1}/{len(doc_rows)}] {row['doc_id']}  {results[-1].get('_error') or 'ok'}")
        await asyncio.sleep(BATCH_DELAY)
    return results


def stage_a_cli(doc_rows: pd.DataFrame, model_id: str) -> list[dict]:
    results = []
    for i, (_, row) in enumerate(doc_rows.iterrows()):
        prompt = STAGE_A_PROMPT.format(
            metadata=build_doc_metadata(row),
            text=truncate(row.get("text", ""), DOC_TRUNC_CHARS),
        )
        t0 = time.time()
        try:
            proc = subprocess.run(
                ["claude", "-p", "--model", model_id, prompt],
                capture_output=True, text=True, timeout=180,
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.strip()[:200] or f"exit {proc.returncode}")
            parsed = parse_json(proc.stdout)
            results.append({
                "doc_id": row["doc_id"],
                "summary_he": parsed.get("summary_he", ""),
                "is_polemical": bool(parsed.get("is_polemical", False)),
                "key_actors": json.dumps(parsed.get("key_actors", []), ensure_ascii=False),
                "stance_marker": parsed.get("stance_marker", ""),
                "_input_tokens": 0, "_output_tokens": 0,
                "_wall_seconds": time.time() - t0, "_error": None,
            })
        except Exception as e:
            results.append({
                "doc_id": row["doc_id"], "summary_he": "", "is_polemical": False,
                "key_actors": "[]", "stance_marker": "",
                "_input_tokens": 0, "_output_tokens": 0,
                "_wall_seconds": time.time() - t0, "_error": str(e)[:200],
            })
        print(f"  [{i+1}/{len(doc_rows)}] {row['doc_id']}  {results[-1].get('_error') or 'ok'}")
    return results


# --- Stage B: thread-level summary + verdict ---

def pick_excerpt_docs(doc_ids: list[str], preds: pd.DataFrame, corpus: pd.DataFrame) -> list[str]:
    """Pick N docs most likely to be polemic for full-text inclusion in Stage B."""
    if preds is None:
        return doc_ids[:N_EXCERPTS]
    have = preds.loc[preds.index.intersection(doc_ids)].copy()
    if have.empty:
        return doc_ids[:N_EXCERPTS]
    have["polemic_prob"] = 1.0 - have["prob_non_polemic"]
    return have.sort_values("polemic_prob", ascending=False).head(N_EXCERPTS).index.tolist()


def build_stage_b_prompt(thread_row, doc_summaries: pd.DataFrame, corpus: pd.DataFrame, preds: pd.DataFrame) -> str:
    doc_ids = [d.strip() for d in str(thread_row["doc_ids"]).split(",") if d.strip()]
    summaries_block_lines = []
    for did in doc_ids:
        s = doc_summaries[doc_summaries["doc_id"] == did]
        if s.empty:
            continue
        sm = str(s.iloc[0]["summary_he"]) or "(no summary)"
        meta = corpus.loc[did] if did in corpus.index else None
        def _m(k):
            if meta is None or k not in meta or pd.isna(meta.get(k)):
                return ""
            v = str(meta.get(k))
            return v if v not in ("nan", "") else ""
        date = _m("date")[:10] or "?"
        paper = _m("newspaper") or "?"
        author = _m("author")
        headline = _m("headline")[:60]
        bits = [date, paper]
        if author:
            bits.append(author)
        bits.append(did)
        head = " · ".join(bits)
        tail = f' "{headline}"' if headline else ""
        summaries_block_lines.append(f"- [{head}]{tail} {sm}")
    summaries_block = "\n".join(summaries_block_lines) or "(no summaries)"

    excerpt_ids = pick_excerpt_docs(doc_ids, preds, corpus)
    excerpts = []
    for did in excerpt_ids:
        if did not in corpus.index:
            continue
        r = corpus.loc[did]
        excerpts.append(
            f"--- [{str(r.get('date',''))[:10]} · {r.get('newspaper','?')} · {did}] ---\n"
            f"{truncate(str(r.get('text','')), EXCERPT_CHARS)}"
        )
    excerpts_block = "\n\n".join(excerpts) or "(no excerpts)"

    return STAGE_B_PROMPT.format(
        n_docs=int(thread_row["n_docs"]),
        n_papers=int(thread_row["n_newspapers"]),
        span_days=int(thread_row["span_days"]),
        newspapers=thread_row["newspapers"],
        edge_types=thread_row["edge_types"],
        heuristic_score=float(thread_row["score"]),
        summaries_block=summaries_block,
        excerpts_block=excerpts_block,
    )


async def stage_b_gemini(prompts: list[tuple[int, str]], model_id: str) -> list[dict]:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    client = genai.GenerativeModel(model_id)
    out = []
    for thread_id, prompt in prompts:
        t0 = time.time()
        try:
            resp = await client.generate_content_async(prompt)
            parsed = parse_json(resp.text)
            usage = getattr(resp, "usage_metadata", None)
            out.append({
                "thread_id": thread_id,
                **{k: parsed.get(k) for k in ("topic_label", "topic_label_he", "narrative",
                                              "is_polemic_thread", "polemic_score",
                                              "polemic_type", "polemic_direction", "evidence")},
                "actors": json.dumps(parsed.get("actors", []), ensure_ascii=False),
                "rebuttal_edges": json.dumps(parsed.get("rebuttal_edges", []), ensure_ascii=False),
                "_input_tokens": getattr(usage, "prompt_token_count", 0) if usage else 0,
                "_output_tokens": getattr(usage, "candidates_token_count", 0) if usage else 0,
                "_wall_seconds": time.time() - t0,
                "_error": None,
            })
        except Exception as e:
            out.append({"thread_id": thread_id, "_error": str(e)[:200],
                        "_wall_seconds": time.time() - t0})
        print(f"  thread {thread_id}: {out[-1].get('_error') or 'ok'}")
        await asyncio.sleep(BATCH_DELAY)
    return out


def stage_b_cli(prompts: list[tuple[int, str]], model_id: str) -> list[dict]:
    out = []
    for thread_id, prompt in prompts:
        t0 = time.time()
        try:
            proc = subprocess.run(
                ["claude", "-p", "--model", model_id, prompt],
                capture_output=True, text=True, timeout=300,
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.strip()[:200] or f"exit {proc.returncode}")
            parsed = parse_json(proc.stdout)
            out.append({
                "thread_id": thread_id,
                **{k: parsed.get(k) for k in ("topic_label", "topic_label_he", "narrative",
                                              "is_polemic_thread", "polemic_score",
                                              "polemic_type", "polemic_direction", "evidence")},
                "actors": json.dumps(parsed.get("actors", []), ensure_ascii=False),
                "rebuttal_edges": json.dumps(parsed.get("rebuttal_edges", []), ensure_ascii=False),
                "_input_tokens": 0, "_output_tokens": 0,
                "_wall_seconds": time.time() - t0, "_error": None,
            })
        except Exception as e:
            out.append({"thread_id": thread_id, "_error": str(e)[:200],
                        "_wall_seconds": time.time() - t0})
        print(f"  thread {thread_id}: {out[-1].get('_error') or 'ok'}")
    return out


# --- Persistence ---

def load_existing(path: Path, key_cols: list[str]) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=key_cols)


def upsert(df_old: pd.DataFrame, df_new: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    if df_old.empty:
        return df_new
    mask = ~df_old.set_index(key_cols).index.isin(df_new.set_index(key_cols).index)
    return pd.concat([df_old[mask], df_new], ignore_index=True)


@contextmanager
def file_lock(path: Path):
    """Cross-process exclusive lock on a sidecar lock file."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def locked_upsert_write(path: Path, df_new: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    """Read parquet under lock, upsert df_new, write back, return merged frame."""
    with file_lock(path):
        df_old = load_existing(path, key_cols)
        merged = upsert(df_old, df_new, key_cols)
        merged.to_parquet(path, index=False)
        return merged


# --- Main ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10, help="Top-N engaged threads to process")
    ap.add_argument("--thread-ids", type=str, default="", help="Comma-sep specific thread_ids (overrides --top)")
    ap.add_argument("--models", type=str, default="gemini_flash3,cli_sonnet",
                    help="Comma-sep model keys: gemini_flash3,cli_sonnet,cli_opus")
    ap.add_argument("--engaged-only", action="store_true", default=True)
    ap.add_argument("--skip-stage-a", action="store_true",
                    help="Reuse cached Stage A summaries; only re-run Stage B")
    args = ap.parse_args()

    threads = pd.read_parquet(DATA / "threads.parquet")
    if args.engaged_only:
        threads = threads[threads["thread_type"] == "engaged"]
    if args.thread_ids:
        ids = [int(x) for x in args.thread_ids.split(",")]
        threads = threads[threads["thread_id"].isin(ids)]
    else:
        threads = threads.sort_values("score", ascending=False).head(args.top)
    threads = threads.reset_index(drop=True)
    print(f"Processing {len(threads)} threads: {threads['thread_id'].tolist()}")

    corpus_cols = ["doc_id", "date", "year", "newspaper", "headline", "author", "text"]
    corpus = pd.read_parquet(ROOT / "corpus.parquet", columns=corpus_cols).set_index("doc_id")
    preds_path = DATA / "full_corpus_predictions.parquet"
    preds = pd.read_parquet(preds_path).set_index("doc_id") if preds_path.exists() else None

    # Union of all doc_ids in selected threads
    all_doc_ids: set = set()
    for _, tr in threads.iterrows():
        all_doc_ids.update(d.strip() for d in str(tr["doc_ids"]).split(",") if d.strip())
    all_doc_ids = [d for d in all_doc_ids if d in corpus.index]
    print(f"Stage A scope: {len(all_doc_ids)} unique docs")

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    for mk in model_keys:
        if mk not in MODELS:
            raise SystemExit(f"unknown model key: {mk}")

    # --- Stage A per model ---
    doc_sum_old = load_existing(DOC_SUMMARIES_PATH, ["doc_id", "model"])
    if not args.skip_stage_a:
        for mk in model_keys:
            cfg = MODELS[mk]
            done_ids = set(doc_sum_old[doc_sum_old.get("model", "") == mk]["doc_id"]) if not doc_sum_old.empty else set()
            todo_ids = [d for d in all_doc_ids if d not in done_ids]
            print(f"\n=== Stage A · {mk} ({cfg['model_id']}) · {len(todo_ids)}/{len(all_doc_ids)} pending ===")
            if not todo_ids:
                continue
            doc_rows = corpus.loc[todo_ids].reset_index().rename(columns={"index": "doc_id"})
            if cfg["provider"] == "google":
                rows = asyncio.run(stage_a_gemini(doc_rows, cfg["model_id"]))
            else:
                rows = stage_a_cli(doc_rows, cfg["model_id"])
            df_new = pd.DataFrame(rows)
            df_new["model"] = mk
            doc_sum_old = locked_upsert_write(DOC_SUMMARIES_PATH, df_new, ["doc_id", "model"])
            print(f"  → wrote {DOC_SUMMARIES_PATH} ({len(doc_sum_old)} rows total)")

    # --- Stage B per model ---
    thread_sum_old = load_existing(THREAD_SUMMARIES_PATH, ["thread_id", "model"])
    for mk in model_keys:
        cfg = MODELS[mk]
        doc_summ_mk = doc_sum_old[doc_sum_old["model"] == mk] if not doc_sum_old.empty else pd.DataFrame()
        prompts = []
        for _, tr in threads.iterrows():
            already = (not thread_sum_old.empty and
                       ((thread_sum_old["thread_id"] == tr["thread_id"]) &
                        (thread_sum_old["model"] == mk)).any())
            if already:
                continue
            prompts.append((int(tr["thread_id"]),
                            build_stage_b_prompt(tr, doc_summ_mk, corpus, preds)))
        print(f"\n=== Stage B · {mk} · {len(prompts)} thread prompts ===")
        if not prompts:
            continue
        if cfg["provider"] == "google":
            rows = asyncio.run(stage_b_gemini(prompts, cfg["model_id"]))
        else:
            rows = stage_b_cli(prompts, cfg["model_id"])
        df_new = pd.DataFrame(rows)
        df_new["model"] = mk
        thread_sum_old = locked_upsert_write(THREAD_SUMMARIES_PATH, df_new, ["thread_id", "model"])
        print(f"  → wrote {THREAD_SUMMARIES_PATH} ({len(thread_sum_old)} rows total)")

    print("\nDone.")


if __name__ == "__main__":
    main()
