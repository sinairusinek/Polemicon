"""
thread_literature_review.py — Test Question A from the pilot conclusion.

For each top-N engaged thread, ask Claude (with web search) whether the
underlying polemic is documented in secondary scholarly literature,
and what specific sources discuss it.

Output:
  data/thread_literature_review.parquet — per-thread documentation verdict + citations with URLs

Usage:
  python src/thread_literature_review.py --top 30 --model claude-sonnet-4-6
  python src/thread_literature_review.py --thread-ids 412,381
"""
import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
load_dotenv(ROOT / ".env")

DATA = ROOT / "data"
OUT_PATH = DATA / "thread_literature_review.parquet"


PROMPT = """You are a research assistant for a digital humanities project on 19th-century Hebrew press polemics (Haskalah and post-Haskalah eras, 1862-1888). I have identified a polemical thread from primary press sources and want to know whether scholars have documented this specific polemic.

Thread metadata:
- Topic (EN): {topic_label}
- Topic (HE): {topic_label_he}
- Date range: {date_range}
- Newspapers involved: {newspapers}
- Principal actors: {actors}
- Narrative: {narrative}

Please search the web for secondary scholarly literature that discusses this SPECIFIC polemical exchange (not just the broader period or general Haskalah history). Prioritize academic books, peer-reviewed articles, encyclopedia entries (Encyclopaedia Judaica, YIVO Encyclopedia, etc.), and reputable digital humanities resources. Hebrew, English, German, and other-language sources are all acceptable.

Return a JSON object (no markdown fences, no prose outside JSON) with EXACTLY these keys:

- "is_documented": one of "well-documented" (multiple sources discuss this specific debate), "mentioned-in-passing" (referenced in works on the period but not analyzed in depth), or "not-found" (no sources discovered).
- "is_canonical_event": true if the UNDERLYING event/dispute (e.g. the Tiszaeszlár trial, the BILU emigration, the She'eilot ha-Hayyim debate) is canonical in Jewish/Haskalah historiography, regardless of whether the Hebrew-press coverage specifically is analyzed.
- "key_sources": JSON list of up to 6 source objects, each with EXACTLY these fields:
    - "author": author or organization name (or "" if unknown)
    - "title": work title
    - "year": publication year as integer or 0
    - "type": one of "book" | "article" | "chapter" | "encyclopedia" | "thesis" | "web"
    - "url": direct URL to the source if available from your search (or "" if no URL found)
    - "where_discussed": one sentence describing where/how this work discusses the polemic
- "notes": 1-3 sentences of additional context — e.g. whether the press coverage specifically is examined vs. the underlying event only, related polemics, or scholarly gaps.

CRITICAL: Only cite sources you actually found in your web search. Do NOT fabricate citations. If a source has no URL in your search results, set url="". If you find fewer than 6 high-quality sources, return fewer — quality over quantity. If literature is genuinely sparse, set is_documented="not-found" with empty key_sources.

Respond with ONLY the JSON object.
"""


def parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0]
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start:end + 1]
    return json.loads(raw)


def build_thread_inputs():
    """Merge top-N thread metadata + LLM summaries into review-ready rows."""
    threads = pd.read_parquet(DATA / "threads.parquet")
    llm = pd.read_parquet(DATA / "thread_llm_summaries.parquet")
    # Prefer gemini_flash3 summary if available, else any model
    pref_order = ["gemini_flash3", "cli_opus", "cli_sonnet"]
    llm["_pref"] = llm["model"].map({m: i for i, m in enumerate(pref_order)}).fillna(99)
    llm_sorted = llm.sort_values(["thread_id", "_pref"])
    llm_first = llm_sorted.drop_duplicates("thread_id", keep="first").set_index("thread_id")

    # Need date range — pull from corpus
    corpus = pd.read_parquet(ROOT / "corpus.parquet", columns=["doc_id", "date"])
    corpus_idx = corpus.set_index("doc_id")

    rows = []
    for _, tr in threads.iterrows():
        tid = int(tr["thread_id"])
        if tid not in llm_first.index:
            continue
        lr = llm_first.loc[tid]
        gold = [d.strip() for d in str(tr["doc_ids"]).split(",") if d.strip()]
        dates = corpus_idx.loc[corpus_idx.index.intersection(gold), "date"].dropna()
        date_range = (f"{str(dates.min())[:10]} to {str(dates.max())[:10]}"
                      if len(dates) else "unknown")
        actors_raw = lr.get("actors") or "[]"
        try:
            actors_list = json.loads(actors_raw) if isinstance(actors_raw, str) else list(actors_raw)
        except Exception:
            actors_list = []
        rows.append({
            "thread_id": tid,
            "topic_label": lr.get("topic_label") or "",
            "topic_label_he": lr.get("topic_label_he") or "",
            "narrative": lr.get("narrative") or "",
            "actors": ", ".join(str(a) for a in actors_list[:12]),
            "newspapers": tr["newspapers"],
            "date_range": date_range,
        })
    return pd.DataFrame(rows)


def review_thread_cli(thread_row: pd.Series, model_id: str) -> dict:
    """Route the review through `claude -p` (Max plan, no API credits).

    The Claude Code CLI has WebSearch and WebFetch built in, so the prompt
    asking the model to search the web works without any API tool plumbing.
    """
    prompt = PROMPT.format(
        topic_label=thread_row["topic_label"],
        topic_label_he=thread_row["topic_label_he"],
        narrative=thread_row["narrative"],
        actors=thread_row["actors"],
        newspapers=thread_row["newspapers"],
        date_range=thread_row["date_range"],
    )
    # Tell the CLI explicitly that we want it to use WebSearch.
    prompt = ("Use the WebSearch tool as needed to find scholarly sources. "
              "Up to 6 searches. Then respond with the JSON only.\n\n" + prompt)
    t0 = time.time()
    try:
        proc = subprocess.run(
            ["claude", "-p", "--model", model_id, prompt],
            capture_output=True, text=True, timeout=600,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip()[:300] or f"exit {proc.returncode}")
        raw = proc.stdout
        parsed = parse_json(raw)
    except Exception as e:
        return {
            "thread_id": int(thread_row["thread_id"]),
            "_error": str(e)[:300],
            "_n_searches": 0,
            "_wall_seconds": time.time() - t0,
            "_input_tokens": 0,
            "_output_tokens": 0,
        }
    return {
        "thread_id": int(thread_row["thread_id"]),
        "is_documented": parsed.get("is_documented"),
        "is_canonical_event": bool(parsed.get("is_canonical_event", False)),
        "key_sources": json.dumps(parsed.get("key_sources", []), ensure_ascii=False),
        "notes": parsed.get("notes", ""),
        "_n_searches": 0,  # CLI doesn't expose this; left as 0
        "_wall_seconds": time.time() - t0,
        "_input_tokens": 0,
        "_output_tokens": 0,
        "_error": None,
    }


def review_thread(client, thread_row: pd.Series, model_id: str, max_searches: int) -> dict:
    prompt = PROMPT.format(
        topic_label=thread_row["topic_label"],
        topic_label_he=thread_row["topic_label_he"],
        narrative=thread_row["narrative"],
        actors=thread_row["actors"],
        newspapers=thread_row["newspapers"],
        date_range=thread_row["date_range"],
    )
    t0 = time.time()
    resp = client.messages.create(
        model=model_id,
        max_tokens=2048,
        tools=[{
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": max_searches,
        }],
        messages=[{"role": "user", "content": prompt}],
    )
    # Final text block is the model's structured answer
    text_blocks = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    raw = "\n".join(text_blocks).strip()
    n_searches = sum(1 for b in resp.content if getattr(b, "type", None) == "server_tool_use")
    try:
        parsed = parse_json(raw)
    except Exception as e:
        return {
            "thread_id": int(thread_row["thread_id"]),
            "_error": f"parse: {str(e)[:200]} | raw: {raw[:300]}",
            "_n_searches": n_searches,
            "_wall_seconds": time.time() - t0,
            "_input_tokens": resp.usage.input_tokens,
            "_output_tokens": resp.usage.output_tokens,
        }
    return {
        "thread_id": int(thread_row["thread_id"]),
        "is_documented": parsed.get("is_documented"),
        "is_canonical_event": bool(parsed.get("is_canonical_event", False)),
        "key_sources": json.dumps(parsed.get("key_sources", []), ensure_ascii=False),
        "notes": parsed.get("notes", ""),
        "_n_searches": n_searches,
        "_wall_seconds": time.time() - t0,
        "_input_tokens": resp.usage.input_tokens,
        "_output_tokens": resp.usage.output_tokens,
        "_error": None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--thread-ids", type=str, default="")
    ap.add_argument("--model", type=str, default=os.getenv("CLAUDE_SONNET_MODEL", "claude-sonnet-4-6"))
    ap.add_argument("--max-searches", type=int, default=6)
    ap.add_argument("--use-cli", action="store_true",
                    help="Route through `claude -p` CLI (Max plan, no API credits). Inherits WebSearch tool.")
    args = ap.parse_args()

    client = None
    if not args.use_cli:
        import anthropic
        client = anthropic.Anthropic()

    threads_input = build_thread_inputs()
    if args.thread_ids:
        ids = [int(x) for x in args.thread_ids.split(",")]
        threads_input = threads_input[threads_input["thread_id"].isin(ids)]
    else:
        # Restrict to the same set as the bake-off (top-N engaged by score)
        threads_table = pd.read_parquet(DATA / "threads.parquet")
        top_ids = (threads_table[threads_table["thread_type"] == "engaged"]
                   .sort_values("score", ascending=False)
                   .head(args.top)["thread_id"].tolist())
        threads_input = threads_input[threads_input["thread_id"].isin(top_ids)]

    existing = pd.read_parquet(OUT_PATH) if OUT_PATH.exists() else pd.DataFrame(columns=["thread_id"])
    # Skip rows that already succeeded (have is_documented). Re-run rows that errored.
    if "is_documented" in existing.columns:
        good = existing[existing["is_documented"].notna()]
    else:
        good = existing.iloc[:0]
    bad = existing[~existing.index.isin(good.index)]
    done_ids = set(good["thread_id"].tolist())
    todo = threads_input[~threads_input["thread_id"].isin(done_ids)]
    route = "claude CLI (Max plan)" if args.use_cli else args.model + " (API)"
    print(f"Reviewing {len(todo)} threads (skipping {len(threads_input) - len(todo)} already done). route={route}")

    # Start from the existing successful rows so we don't lose them
    existing = good

    results = []
    for i, (_, row) in enumerate(todo.iterrows()):
        try:
            if args.use_cli:
                res = review_thread_cli(row, args.model)
            else:
                res = review_thread(client, row, args.model, args.max_searches)
        except Exception as e:
            res = {"thread_id": int(row["thread_id"]),
                   "_error": str(e)[:200], "_wall_seconds": 0.0}
        results.append(res)
        msg = (f"  [{i+1}/{len(todo)}] thread {res['thread_id']}: "
               f"{res.get('is_documented') or res.get('_error', 'err')[:60]} "
               f"({res.get('_n_searches', 0)} searches, "
               f"{res.get('_wall_seconds', 0):.1f}s)")
        print(msg)
        # Persist after each thread — web search is expensive, don't lose progress
        if results:
            merged = pd.concat([existing, pd.DataFrame(results)], ignore_index=True)
            merged.to_parquet(OUT_PATH, index=False)

    print(f"\n→ wrote {OUT_PATH} ({len(merged) if results else len(existing)} rows total)")


if __name__ == "__main__":
    main()
