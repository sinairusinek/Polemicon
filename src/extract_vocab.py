"""
extract_vocab.py - Extract polemic vocabulary markers from disagreement texts

Re-runs Sonnet on the 94 expensive-model disagreement texts with a prompt
that asks the model to identify specific Hebrew words/phrases that serve as
polemic markers. These become candidates for expanding the B.3 keyword lexicon.

Output: data/pilot_vocab.parquet (one row per text, with suggested markers)

Usage:
    python src/extract_vocab.py
    python src/extract_vocab.py --model sonnet     # default
    python src/extract_vocab.py --model opus        # use Opus instead
    python src/extract_vocab.py --all-polemic       # run on all texts with >=3 polemic votes
"""
import os
import sys
import json
import asyncio
import argparse
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
load_dotenv(ROOT / ".env")

MAX_TEXT_WORDS = 4000
BATCH_DELAY_SECONDS = 1.0

VOCAB_PROMPT = """You are an expert in 19th-century Hebrew literature (Haskalah era, 1862-1888).
Analyze the following Hebrew text and identify polemic vocabulary.

A polemic text: engages in argumentative debate, responds to or attacks another writer's position,
defends a stance against criticism, or participates in a public intellectual dispute.

Respond with a JSON object (no markdown, no explanation outside the JSON) with exactly these fields:
- "is_polemic": boolean (true if the text is polemic)
- "confidence": float 0.0-1.0 (your confidence in the is_polemic judgment)
- "polemic_type": one of "attack", "defense", "debate", "satire", "critique", or "none"
- "polemic_markers": list of 3-5 specific Hebrew words or short phrases FROM THE TEXT that are the strongest polemic signals. Each entry should be a single word or 2-3 word phrase in Hebrew, exactly as it appears in the text. If the text is not polemic, return the 3-5 most argumentative/evaluative words you can find anyway.
- "marker_explanations": list of brief English explanations (same length as polemic_markers), explaining why each word/phrase is a polemic signal
- "evidence": string (brief description of the key polemic markers found)
- "topic": string (the main subject of the text, in English)

TEXT:
{text}

JSON:"""

MODEL_CONFIGS = {
    "sonnet": {
        "provider": "anthropic",
        "model_id": os.getenv("CLAUDE_SONNET_MODEL", "claude-sonnet-4-6"),
        "display_name": "Claude Sonnet",
    },
    "opus": {
        "provider": "anthropic",
        "model_id": os.getenv("CLAUDE_OPUS_MODEL", "claude-opus-4-6"),
        "display_name": "Claude Opus",
    },
}


def truncate_text(text: str, max_words: int = MAX_TEXT_WORDS) -> str:
    words = str(text).split()
    if len(words) <= max_words:
        return str(text)
    return " ".join(words[:max_words]) + " [...]"


def parse_json_response(raw: str) -> dict:
    raw = raw.strip()
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                raw = part
                break
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        return {"_parse_error": raw[:200]}
    try:
        return json.loads(raw[start:end])
    except json.JSONDecodeError:
        return {"_parse_error": raw[start:end][:200]}


def validate_result(result: dict) -> dict:
    if "_parse_error" in result:
        return {
            "is_polemic": None, "confidence": None, "polemic_type": None,
            "polemic_markers": [], "marker_explanations": [],
            "evidence": "", "topic": "",
            "_parse_error": result["_parse_error"],
        }

    is_pol = result.get("is_polemic")
    if isinstance(is_pol, str):
        is_pol = is_pol.lower() in ("true", "yes", "1")
    result["is_polemic"] = bool(is_pol) if is_pol is not None else None

    conf = result.get("confidence")
    if conf is not None:
        try:
            conf = max(0.0, min(1.0, float(conf)))
        except (ValueError, TypeError):
            conf = None
    result["confidence"] = conf

    ptype = str(result.get("polemic_type", "none")).lower().strip()
    valid_types = {"attack", "defense", "debate", "satire", "critique", "none"}
    result["polemic_type"] = ptype if ptype in valid_types else "none"

    # Vocabulary fields
    markers = result.get("polemic_markers", [])
    if not isinstance(markers, list):
        markers = []
    result["polemic_markers"] = markers

    explanations = result.get("marker_explanations", [])
    if not isinstance(explanations, list):
        explanations = []
    # Pad or truncate to match markers length
    while len(explanations) < len(markers):
        explanations.append("")
    result["marker_explanations"] = explanations[:len(markers)]

    result["evidence"] = str(result.get("evidence", ""))
    result["topic"] = str(result.get("topic", ""))

    return result


async def classify_text(text: str, model_id: str, client) -> dict:
    prompt = VOCAB_PROMPT.format(text=truncate_text(text))
    message = await client.messages.create(
        model=model_id,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text
    return validate_result(parse_json_response(raw))


async def run_extraction(texts_df: pd.DataFrame, model_key: str, output_path: Path):
    config = MODEL_CONFIGS[model_key]
    model_id = config["model_id"]
    display = config["display_name"]

    # Load existing results for resumability
    existing_ids = set()
    all_results = []
    if output_path.exists():
        prev = pd.read_parquet(output_path)
        all_results = prev.to_dict("records")
        existing_ids = set(prev["doc_id"].tolist())
        print(f"  Loaded {len(prev)} existing results from checkpoint.")

    pending = texts_df[~texts_df["doc_id"].isin(existing_ids)]
    if len(pending) == 0:
        print(f"  All {len(texts_df)} texts already processed.")
        return pd.DataFrame(all_results)

    print(f"  {display} ({model_id}): {len(pending)} texts to process...")

    import anthropic
    client = anthropic.AsyncAnthropic()

    errors = 0
    for i, (_, row) in enumerate(pending.iterrows()):
        try:
            result = await classify_text(row["text"], model_id, client)
            result["doc_id"] = row["doc_id"]
            result["model"] = model_key
            # Serialize lists as JSON strings for parquet compatibility
            result["polemic_markers_json"] = json.dumps(result.pop("polemic_markers"), ensure_ascii=False)
            result["marker_explanations_json"] = json.dumps(result.pop("marker_explanations"), ensure_ascii=False)
            all_results.append(result)
        except Exception as e:
            errors += 1
            all_results.append({
                "doc_id": row["doc_id"], "model": model_key,
                "is_polemic": None, "confidence": None, "polemic_type": None,
                "polemic_markers_json": "[]", "marker_explanations_json": "[]",
                "evidence": "", "topic": "", "_error": str(e)[:200],
            })
            print(f"    ERROR on {row['doc_id']}: {str(e)[:100]}")

        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(pending)} done...")
            # Checkpoint
            df_out = pd.DataFrame(all_results)
            df_out.to_parquet(output_path, index=False)

        await asyncio.sleep(BATCH_DELAY_SECONDS)

    # Final save
    df_out = pd.DataFrame(all_results)
    df_out.to_parquet(output_path, index=False)
    print(f"  Done: {len(df_out)} total results, {errors} errors.")
    return df_out


def main():
    parser = argparse.ArgumentParser(description="Extract polemic vocabulary markers")
    parser.add_argument("--model", default="sonnet", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--all-polemic", action="store_true",
                        help="Run on all texts with >=3 polemic votes (not just disagreements)")
    args = parser.parse_args()

    output_path = ROOT / "data" / "pilot_vocab.parquet"

    # Load data
    print("Loading data...")
    pilot_df = pd.read_parquet(ROOT / "data" / "pilot_sample.parquet")
    disagree_df = pd.read_parquet(ROOT / "data" / "pilot_disagreements.parquet")

    if args.all_polemic:
        # Run on all texts with >=3 polemic votes
        clf_df = pd.read_parquet(ROOT / "data" / "pilot_classifications.parquet")
        polemic_votes = clf_df[clf_df["is_polemic"] == True].groupby("doc_id").size()
        target_ids = set(polemic_votes[polemic_votes >= 3].index)
        # Also include disagreement texts
        target_ids |= set(disagree_df[disagree_df["agreement_category"] == "expensive_disagree"]["doc_id"])
        texts_df = pilot_df[pilot_df["doc_id"].isin(target_ids)]
        print(f"  {len(texts_df)} texts (>=3 polemic votes + disagreements)")
    else:
        # Default: just the 94 disagreement texts
        target_ids = set(disagree_df[disagree_df["agreement_category"] == "expensive_disagree"]["doc_id"])
        texts_df = pilot_df[pilot_df["doc_id"].isin(target_ids)]
        print(f"  {len(texts_df)} disagreement texts")

    print(f"\nRunning vocabulary extraction with {args.model}...")
    print(f"  Estimated: {len(texts_df)} API calls\n")

    df_out = asyncio.run(run_extraction(texts_df, args.model, output_path))

    # Summary of extracted vocabulary
    if len(df_out) > 0:
        all_markers = []
        for _, row in df_out.iterrows():
            try:
                markers = json.loads(row.get("polemic_markers_json", "[]"))
                all_markers.extend(markers)
            except (json.JSONDecodeError, TypeError):
                pass

        print(f"\n--- Vocabulary Summary ---")
        print(f"  Total marker phrases extracted: {len(all_markers)}")
        print(f"  Unique markers: {len(set(all_markers))}")

        # Most common markers
        from collections import Counter
        counts = Counter(all_markers)
        print(f"\n  Top 20 most frequent markers:")
        for marker, count in counts.most_common(20):
            print(f"    {marker}: {count}")

    print(f"\nOutput saved to {output_path}")
    print("Next: review suggested markers in the Streamlit app")


if __name__ == "__main__":
    main()
