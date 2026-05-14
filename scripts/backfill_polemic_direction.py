"""
backfill_polemic_direction.py — add polemic_direction to existing thread_llm_summaries rows.

Lightweight: re-prompts each model with only narrative + actors + topic_label (no excerpts),
asking only for polemic_direction. Non-polemic rows (is_polemic_thread != True) are set to
"n/a" without an LLM call.

Usage:
    python scripts/backfill_polemic_direction.py             # all models, all rows missing field
    python scripts/backfill_polemic_direction.py --models cli_opus
    python scripts/backfill_polemic_direction.py --dry-run
"""
import os
import sys
import json
import time
import asyncio
import argparse
import subprocess
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from thread_summaries import (  # noqa: E402
    MODELS, THREAD_SUMMARIES_PATH, parse_json, locked_upsert_write,
)

PROMPT = """You are classifying the DIRECTION of a 19th-century Hebrew press polemic.

Topic: {topic_label}
Topic (Hebrew): {topic_label_he}
Principal actors: {actors}

Narrative:
{narrative}

Prior evidence: {evidence}

Decide which register this polemic belongs to:
- "internal"         = intra-Jewish dispute (Hasidim vs Mitnagdim, Haskalah vs Orthodoxy, Reform vs traditionalists, paper-vs-paper, rabbi-vs-rabbi, intra-communal politics).
- "external_defense" = cross-paper apologetics responding to external antisemitism / blood libel (Tiszaeszlár, Damascus affair) / missionary attacks / hostile non-Jewish press, where Jewish writers close ranks against an outside accuser.
- "mixed"            = both registers genuinely active.

Respond with ONLY a JSON object: {{"polemic_direction": "internal" | "external_defense" | "mixed", "rationale": "one short English sentence"}}
"""


async def call_gemini(prompt: str, model_id: str) -> dict:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    client = genai.GenerativeModel(model_id)
    resp = await client.generate_content_async(prompt)
    return parse_json(resp.text)


def call_cli(prompt: str, model_id: str) -> dict:
    proc = subprocess.run(
        ["claude", "-p", "--model", model_id, prompt],
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[:200] or f"exit {proc.returncode}")
    return parse_json(proc.stdout)


def build_prompt(row) -> str:
    actors = row.get("actors") or "[]"
    try:
        actors_list = json.loads(actors) if isinstance(actors, str) else list(actors)
        actors_str = ", ".join(str(a) for a in actors_list) or "(none listed)"
    except Exception:
        actors_str = str(actors)
    return PROMPT.format(
        topic_label=row.get("topic_label") or "(none)",
        topic_label_he=row.get("topic_label_he") or "(none)",
        actors=actors_str,
        narrative=row.get("narrative") or "(none)",
        evidence=row.get("evidence") or "(none)",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, default="",
                    help="Comma-sep model keys to backfill (default: all present in file)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not THREAD_SUMMARIES_PATH.exists():
        raise SystemExit(f"missing {THREAD_SUMMARIES_PATH}")
    df = pd.read_parquet(THREAD_SUMMARIES_PATH)
    if "polemic_direction" not in df.columns:
        df["polemic_direction"] = pd.NA

    needs = df["polemic_direction"].isna() | (df["polemic_direction"].astype(str) == "")
    target = df[needs].copy()
    if args.models:
        wanted = {m.strip() for m in args.models.split(",") if m.strip()}
        target = target[target["model"].isin(wanted)]
    print(f"Rows missing polemic_direction: {len(target)}")
    if target.empty:
        return

    # Non-polemic rows → "n/a" without an LLM call.
    is_polemic = target["is_polemic_thread"].apply(lambda v: v is True or v == "true" or v == True)  # noqa: E712
    na_rows = target[~is_polemic]
    llm_rows = target[is_polemic]
    print(f"  → {len(na_rows)} non-polemic (set to n/a), {len(llm_rows)} need LLM")

    updates = []
    for _, r in na_rows.iterrows():
        updates.append({"thread_id": int(r["thread_id"]), "model": r["model"],
                        "polemic_direction": "n/a"})

    for _, r in llm_rows.iterrows():
        mk = r["model"]
        if mk not in MODELS:
            print(f"  skip thread {r['thread_id']} model={mk} (unknown model key)")
            continue
        cfg = MODELS[mk]
        prompt = build_prompt(r)
        if args.dry_run:
            print(f"  [dry-run] thread {r['thread_id']} model={mk}")
            continue
        t0 = time.time()
        try:
            if cfg["provider"] == "google":
                parsed = asyncio.run(call_gemini(prompt, cfg["model_id"]))
            else:
                parsed = call_cli(prompt, cfg["model_id"])
            direction = parsed.get("polemic_direction", "")
            if direction not in ("internal", "external_defense", "mixed"):
                print(f"  thread {r['thread_id']} {mk}: unexpected direction={direction!r}")
                continue
            updates.append({"thread_id": int(r["thread_id"]), "model": mk,
                            "polemic_direction": direction})
            print(f"  thread {r['thread_id']} {mk}: {direction}  ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  thread {r['thread_id']} {mk}: ERROR {str(e)[:160]}")
        time.sleep(0.3)

    if args.dry_run or not updates:
        print(f"Done (dry-run={args.dry_run}, {len(updates)} updates).")
        return

    upd_df = pd.DataFrame(updates).set_index(["thread_id", "model"])
    df_indexed = df.set_index(["thread_id", "model"])
    df_indexed.loc[upd_df.index, "polemic_direction"] = upd_df["polemic_direction"]
    merged = df_indexed.reset_index()
    locked_upsert_write(THREAD_SUMMARIES_PATH, merged, ["thread_id", "model"])
    print(f"Wrote {len(updates)} updates to {THREAD_SUMMARIES_PATH}")


if __name__ == "__main__":
    main()
