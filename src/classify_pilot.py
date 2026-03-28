"""
classify_pilot.py - Phase B.2: Dual-model LLM classification pilot

Runs 4 models (Claude Opus, Claude Sonnet, Gemini Pro, Gemini Flash) on the
200-text pilot sample. Each model classifies each text for polemic characteristics.

Output: data/pilot_classifications.parquet (one row per text per model)
        data/agreement_report.txt (inter-model agreement analysis)

Usage:
    python src/classify_pilot.py                  # run all 4 models
    python src/classify_pilot.py --models opus sonnet  # run specific models
    python src/classify_pilot.py --report-only    # just regenerate the report
"""
import os
import sys
import json
import asyncio
import argparse
import time
from pathlib import Path

# Force unbuffered output for background runs
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Project root
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
load_dotenv(ROOT / ".env")

# --- Configuration ---

MODEL_CONFIGS = {
    "opus": {
        "provider": "anthropic",
        "model_id": os.getenv("CLAUDE_OPUS_MODEL", "claude-opus-4-6"),
        "display_name": "Claude Opus",
        "tier": "expensive",
    },
    "sonnet": {
        "provider": "anthropic",
        "model_id": os.getenv("CLAUDE_SONNET_MODEL", "claude-sonnet-4-6"),
        "display_name": "Claude Sonnet",
        "tier": "cheap",
    },
    "gemini_pro": {
        "provider": "google",
        "model_id": os.getenv("GEMINI_PRO_MODEL", "gemini-2.5-pro"),
        "display_name": "Gemini Pro",
        "tier": "expensive",
    },
    "gemini_flash": {
        "provider": "google",
        "model_id": os.getenv("GEMINI_FLASH_MODEL", "gemini-2.5-flash"),
        "display_name": "Gemini Flash",
        "tier": "cheap",
    },
}

MAX_TEXT_WORDS = 4000  # truncate long texts to control cost
BATCH_DELAY_SECONDS = 1.0  # delay between API calls to avoid rate limits

CLASSIFICATION_PROMPT = """You are an expert in 19th-century Hebrew literature (Haskalah era, 1862-1888).
Analyze the following Hebrew text and classify whether it is polemic.

A polemic text: engages in argumentative debate, responds to or attacks another writer's position,
defends a stance against criticism, or participates in a public intellectual dispute.
Key markers: direct address to opponents, rhetorical questions, evaluative language, intertextual references.

Respond with a JSON object (no markdown, no explanation outside the JSON) with exactly these fields:
- "is_polemic": boolean (true if the text is polemic)
- "confidence": float 0.0-1.0 (your confidence in the is_polemic judgment)
- "polemic_type": one of "attack", "defense", "debate", "satire", "critique", or "none"
- "target": string (who or what is being argued against; empty string if not polemic)
- "evidence": string (brief quote or description of the key polemic markers found, in Hebrew if quoting)
- "topic": string (the main subject of the text, in English)

TEXT:
{text}

JSON:"""


def truncate_text(text: str, max_words: int = MAX_TEXT_WORDS) -> str:
    words = str(text).split()
    if len(words) <= max_words:
        return str(text)
    return " ".join(words[:max_words]) + " [...]"


def parse_json_response(raw: str) -> dict:
    """Extract JSON from model response, handling markdown fences and preamble."""
    raw = raw.strip()
    # Strip markdown code fences
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                raw = part
                break
    # Find the JSON object
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        return {"_parse_error": raw[:200]}
    try:
        return json.loads(raw[start:end])
    except json.JSONDecodeError:
        return {"_parse_error": raw[start:end][:200]}


EXPECTED_FIELDS = {"is_polemic", "confidence", "polemic_type", "target", "evidence", "topic"}
VALID_TYPES = {"attack", "defense", "debate", "satire", "critique", "none"}


def validate_classification(result: dict) -> dict:
    """Normalize and validate a classification result."""
    if "_parse_error" in result:
        return {
            "is_polemic": None, "confidence": None, "polemic_type": None,
            "target": "", "evidence": "", "topic": "",
            "_parse_error": result["_parse_error"],
        }
    # Normalize is_polemic
    is_pol = result.get("is_polemic")
    if isinstance(is_pol, str):
        is_pol = is_pol.lower() in ("true", "yes", "1")
    result["is_polemic"] = bool(is_pol) if is_pol is not None else None

    # Normalize confidence
    conf = result.get("confidence")
    if conf is not None:
        try:
            conf = float(conf)
            conf = max(0.0, min(1.0, conf))
        except (ValueError, TypeError):
            conf = None
    result["confidence"] = conf

    # Normalize polemic_type
    ptype = str(result.get("polemic_type", "none")).lower().strip()
    if ptype not in VALID_TYPES:
        ptype = "none"
    result["polemic_type"] = ptype

    # String fields
    for field in ("target", "evidence", "topic"):
        result[field] = str(result.get(field, ""))

    return {k: result.get(k) for k in EXPECTED_FIELDS}


# --- API Callers ---

async def classify_anthropic(text: str, model_id: str, client) -> dict:
    """Classify a single text using the Anthropic API."""
    prompt = CLASSIFICATION_PROMPT.format(text=truncate_text(text))
    message = await client.messages.create(
        model=model_id,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text
    return validate_classification(parse_json_response(raw))


async def classify_google(text: str, model_id: str, client) -> dict:
    """Classify a single text using the Google Gemini API."""
    prompt = CLASSIFICATION_PROMPT.format(text=truncate_text(text))
    response = await client.generate_content_async(prompt)
    raw = response.text
    return validate_classification(parse_json_response(raw))


# --- Main pipeline ---

async def run_model(model_key: str, texts_df: pd.DataFrame, existing: set) -> list[dict]:
    """Run a single model on all pilot texts. Returns list of result dicts."""
    config = MODEL_CONFIGS[model_key]
    provider = config["provider"]
    model_id = config["model_id"]
    display = config["display_name"]

    # Skip texts already classified by this model
    pending = texts_df[~texts_df["doc_id"].isin(existing)]
    if len(pending) == 0:
        print(f"  {display}: all {len(texts_df)} texts already classified, skipping.")
        return []

    print(f"  {display} ({model_id}): {len(pending)} texts to classify...")

    results = []
    errors = 0

    if provider == "anthropic":
        import anthropic
        client = anthropic.AsyncAnthropic()
        classify_fn = lambda text: classify_anthropic(text, model_id, client)
    else:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        client = genai.GenerativeModel(model_id)
        classify_fn = lambda text: classify_google(text, model_id, client)

    for i, (_, row) in enumerate(pending.iterrows()):
        try:
            result = await classify_fn(row["text"])
            result["doc_id"] = row["doc_id"]
            result["model"] = model_key
            result["model_display"] = display
            result["tier"] = config["tier"]
            results.append(result)
        except Exception as e:
            errors += 1
            results.append({
                "doc_id": row["doc_id"], "model": model_key,
                "model_display": display, "tier": config["tier"],
                "is_polemic": None, "confidence": None, "polemic_type": None,
                "target": "", "evidence": "", "topic": "",
                "_error": str(e)[:200],
            })
            print(f"    ERROR on {row['doc_id']}: {str(e)[:100]}")

        if (i + 1) % 20 == 0:
            print(f"    {i + 1}/{len(pending)} done...")
        await asyncio.sleep(BATCH_DELAY_SECONDS)

    print(f"  {display}: {len(results)} classified, {errors} errors.")
    return results


async def run_all_models(models: list[str], texts_df: pd.DataFrame, output_path: Path):
    """Run selected models sequentially (to respect rate limits), with checkpointing."""
    # Load existing results for resumability
    all_results = []
    existing_by_model = {}
    if output_path.exists():
        prev = pd.read_parquet(output_path)
        all_results = prev.to_dict("records")
        for model_key in MODEL_CONFIGS:
            model_rows = prev[prev["model"] == model_key]
            existing_by_model[model_key] = set(model_rows["doc_id"].tolist())
        print(f"  Loaded {len(prev)} existing classifications from checkpoint.")
    else:
        for model_key in MODEL_CONFIGS:
            existing_by_model[model_key] = set()

    for model_key in models:
        if model_key not in MODEL_CONFIGS:
            print(f"  Unknown model: {model_key}, skipping.")
            continue
        new_results = await run_model(model_key, texts_df, existing_by_model.get(model_key, set()))
        all_results.extend(new_results)

        # Checkpoint after each model
        if new_results:
            df_out = pd.DataFrame(all_results)
            df_out.to_parquet(output_path, index=False)
            print(f"  Checkpointed {len(df_out)} total results.")

    return pd.DataFrame(all_results) if all_results else pd.DataFrame()


# --- Agreement Report ---

def generate_agreement_report(clf_df: pd.DataFrame, output_path: Path):
    """Analyze inter-model agreement and generate a report."""
    # Filter to successful classifications only
    valid = clf_df[clf_df["is_polemic"].notna()].copy()
    models = sorted(valid["model"].unique())
    n_models = len(models)

    # Pivot: one row per doc_id, columns = model judgments
    pivot = valid.pivot_table(
        index="doc_id", columns="model", values="is_polemic", aggfunc="first"
    )
    # Only analyze texts classified by all models
    complete = pivot.dropna()

    lines = []
    lines.append("=" * 70)
    lines.append("POLEMICON PILOT — INTER-MODEL AGREEMENT REPORT")
    lines.append("=" * 70)
    lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Models: {', '.join(models)}")
    lines.append(f"Total texts: {len(clf_df['doc_id'].unique())}")
    lines.append(f"Texts with all {n_models} models: {len(complete)}")
    lines.append("")

    # Per-model polemic rate
    lines.append("--- Per-Model Polemic Rate ---")
    for model in models:
        model_data = valid[valid["model"] == model]
        rate = model_data["is_polemic"].mean()
        display = MODEL_CONFIGS.get(model, {}).get("display_name", model)
        lines.append(f"  {display}: {rate:.1%} ({int(model_data['is_polemic'].sum())}/{len(model_data)})")
    lines.append("")

    if len(complete) == 0:
        lines.append("Not enough data for agreement analysis.")
        report = "\n".join(lines)
        output_path.write_text(report)
        print(report)
        return pd.DataFrame()

    # Agreement categories
    all_agree_polemic = []
    all_agree_not = []
    expensive_agree_cheap_diverge = []
    expensive_disagree = []

    expensive_models = [m for m in models if MODEL_CONFIGS.get(m, {}).get("tier") == "expensive"]
    cheap_models = [m for m in models if MODEL_CONFIGS.get(m, {}).get("tier") == "cheap"]

    for doc_id, row in complete.iterrows():
        votes = {m: bool(row[m]) for m in models if m in row.index}
        all_true = all(votes.values())
        all_false = not any(votes.values())

        expensive_votes = [votes[m] for m in expensive_models if m in votes]
        cheap_votes = [votes[m] for m in cheap_models if m in votes]

        if all_true:
            all_agree_polemic.append(doc_id)
        elif all_false:
            all_agree_not.append(doc_id)
        elif len(expensive_votes) >= 2 and len(set(expensive_votes)) == 1:
            # Expensive models agree with each other
            if cheap_votes and set(cheap_votes) != set(expensive_votes[:1]):
                expensive_agree_cheap_diverge.append(doc_id)
            else:
                expensive_agree_cheap_diverge.append(doc_id)
        else:
            expensive_disagree.append(doc_id)

    lines.append("--- Agreement Categories ---")
    lines.append(f"  All {n_models} agree POLEMIC:     {len(all_agree_polemic)}")
    lines.append(f"  All {n_models} agree NOT polemic:  {len(all_agree_not)}")
    lines.append(f"  Expensive agree, cheap diverge:  {len(expensive_agree_cheap_diverge)}")
    lines.append(f"  Expensive models DISAGREE:        {len(expensive_disagree)} ← PRIORITY FOR REVIEW")
    lines.append("")

    # Overall agreement rate
    agree_count = len(all_agree_polemic) + len(all_agree_not)
    lines.append(f"  Full agreement rate: {agree_count}/{len(complete)} ({agree_count/len(complete):.1%})")
    lines.append("")

    # Pairwise agreement (Cohen's kappa placeholder — raw agreement)
    lines.append("--- Pairwise Agreement (% same is_polemic) ---")
    for i, m1 in enumerate(models):
        for m2 in models[i + 1:]:
            if m1 in complete.columns and m2 in complete.columns:
                agree = (complete[m1] == complete[m2]).mean()
                d1 = MODEL_CONFIGS.get(m1, {}).get("display_name", m1)
                d2 = MODEL_CONFIGS.get(m2, {}).get("display_name", m2)
                lines.append(f"  {d1} vs {d2}: {agree:.1%}")
    lines.append("")

    # Polemic type agreement for texts all agree are polemic
    if all_agree_polemic:
        lines.append("--- Polemic Type Agreement (texts all agree are polemic) ---")
        type_pivot = valid[valid["doc_id"].isin(all_agree_polemic)].pivot_table(
            index="doc_id", columns="model", values="polemic_type", aggfunc="first"
        )
        type_complete = type_pivot.dropna()
        if len(type_complete) > 0:
            type_agree = type_complete.apply(lambda r: len(set(r)) == 1, axis=1).mean()
            lines.append(f"  Type agreement rate: {type_agree:.1%}")
        lines.append("")

    # Confidence summary
    lines.append("--- Confidence Distribution ---")
    for model in models:
        model_data = valid[valid["model"] == model]
        conf = model_data["confidence"].dropna()
        if len(conf) > 0:
            display = MODEL_CONFIGS.get(model, {}).get("display_name", model)
            lines.append(f"  {display}: mean={conf.mean():.2f}, median={conf.median():.2f}, "
                         f"min={conf.min():.2f}, max={conf.max():.2f}")
    lines.append("")

    # Priority review list
    lines.append("--- Priority Review Queue (expensive models disagree) ---")
    for doc_id in expensive_disagree[:20]:
        doc_votes = valid[valid["doc_id"] == doc_id][["model_display", "is_polemic", "confidence", "polemic_type"]]
        lines.append(f"  {doc_id}:")
        for _, v in doc_votes.iterrows():
            lines.append(f"    {v['model_display']}: polemic={v['is_polemic']}, "
                         f"conf={v['confidence']}, type={v['polemic_type']}")
    if len(expensive_disagree) > 20:
        lines.append(f"  ... and {len(expensive_disagree) - 20} more")
    lines.append("")
    lines.append("=" * 70)

    report = "\n".join(lines)
    output_path.write_text(report)
    print(report)

    # Return disagreement dataframe for Streamlit
    disagreement_df = pd.DataFrame({
        "doc_id": (all_agree_polemic + all_agree_not +
                   expensive_agree_cheap_diverge + expensive_disagree),
        "agreement_category": (
            ["all_agree_polemic"] * len(all_agree_polemic) +
            ["all_agree_not_polemic"] * len(all_agree_not) +
            ["expensive_agree_cheap_diverge"] * len(expensive_agree_cheap_diverge) +
            ["expensive_disagree"] * len(expensive_disagree)
        ),
        "review_priority": (
            [3] * len(all_agree_polemic) +     # low — all agree
            [3] * len(all_agree_not) +          # low — all agree
            [2] * len(expensive_agree_cheap_diverge) +  # medium
            [1] * len(expensive_disagree)       # HIGH — needs human review
        ),
    })
    return disagreement_df


# --- Entry point ---

def main():
    parser = argparse.ArgumentParser(description="Phase B.2: LLM classification pilot")
    parser.add_argument("--models", nargs="+", default=list(MODEL_CONFIGS.keys()),
                        help="Models to run (opus, sonnet, gemini_pro, gemini_flash)")
    parser.add_argument("--report-only", action="store_true",
                        help="Skip classification, just regenerate the agreement report")
    parser.add_argument("--sample-path", default="data/pilot_sample.parquet",
                        help="Path to pilot sample")
    args = parser.parse_args()

    output_path = ROOT / "data" / "pilot_classifications.parquet"
    report_path = ROOT / "data" / "agreement_report.txt"
    disagreement_path = ROOT / "data" / "pilot_disagreements.parquet"

    # Load pilot sample
    print("Loading pilot sample...")
    texts_df = pd.read_parquet(ROOT / args.sample_path)
    print(f"  {len(texts_df)} texts loaded.")

    if not args.report_only:
        print(f"\nRunning classification with models: {args.models}")
        print(f"  Estimated: {len(texts_df)} texts × {len(args.models)} models "
              f"= {len(texts_df) * len(args.models)} API calls\n")

        clf_df = asyncio.run(run_all_models(args.models, texts_df, output_path))

        if len(clf_df) == 0:
            print("No classifications produced.")
            return
        print(f"\nTotal classifications: {len(clf_df)}")
    else:
        if not output_path.exists():
            print("No classifications found. Run without --report-only first.")
            return
        clf_df = pd.read_parquet(output_path)
        print(f"Loaded {len(clf_df)} existing classifications.")

    # Generate agreement report
    print("\n" + "=" * 70)
    disagreement_df = generate_agreement_report(clf_df, report_path)

    if len(disagreement_df) > 0:
        disagreement_df.to_parquet(disagreement_path, index=False)
        print(f"\nSaved disagreement categories to {disagreement_path}")

    # Summary
    errors = clf_df["_error"].notna().sum() if "_error" in clf_df.columns else 0
    parse_errors = clf_df["_parse_error"].notna().sum() if "_parse_error" in clf_df.columns else 0
    if errors > 0 or parse_errors > 0:
        print(f"\nWarnings: {errors} API errors, {parse_errors} parse errors")

    print("\nDone. Next steps:")
    print("  1. Review data/agreement_report.txt")
    print("  2. Open Streamlit app — disagreements are loaded as priority review queue")
    print("  3. Run: streamlit run src/streamlit_app.py")


if __name__ == "__main__":
    main()
