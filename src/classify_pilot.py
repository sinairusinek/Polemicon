"""
classify_pilot.py - Phase B.2 / B.2-v2: LLM classification

v1  Ran 4 models (Claude Opus, Sonnet, Gemini Pro, Flash) on 200-text pilot.
    Output: data/pilot_classifications.parquet

v2  Sonnet-only, 4-tier label scheme, metadata-aware, continuation detection.
    Output: data/pilot_classifications_v2.parquet  (--v2 flag)
    Acceptance test on RA gold labels:              (--acceptance-test flag)
    2K calibration:                                 (--calibration flag)

Usage:
    python src/classify_pilot.py                             # v1: all 4 models on pilot
    python src/classify_pilot.py --models sonnet             # v1: sonnet only
    python src/classify_pilot.py --report-only               # regenerate agreement report
    python src/classify_pilot.py --acceptance-test           # v2: run on 23 RA gold cases
    python src/classify_pilot.py --v2 --models sonnet        # v2: run on pilot sample
    python src/classify_pilot.py --calibration               # v2: run on 2K calibration set
"""
import os
import re
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
BATCH_DELAY_SECONDS = 0.4  # delay between API calls to avoid rate limits

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

# ── v2 prompt ─────────────────────────────────────────────────────────────────

CLASSIFICATION_PROMPT_V2 = """You are an expert in 19th-century Hebrew literature (Haskalah era, 1862–1888).
Analyze the following Hebrew text and assign it one of four polemic labels.

LABEL DEFINITIONS
-----------------
Assign exactly one of the four labels below based on what the TEXT itself does, not just its topic.

"non-polemic"               — No debate function: the text has no argumentative dispute, does not defend
                              a position against an adversary, and does not report on a controversy.
                              IMPORTANT: Critical tone, expressions of sorrow or disappointment, minor
                              qualifications in a celebratory piece, and laments do NOT make a text polemic.
                              Use this label for: neutral news reports, biographical sketches, scientific/
                              historical reviews, travel writing, calls to action without a named adversary,
                              personal letters discussing plans or feelings.

"implicit polemic"          — The text takes a side in an ongoing public debate WITHOUT naming a specific
                              opponent. The polemic stance must be SUBSTANTIVE and CENTRAL, not incidental.
                              Signals: the author devotes significant space to defending a contested position
                              (e.g., on Halukka reform, Hebrew education, aliya, Haskalah, Zionism) against
                              implied opposition; OR explicitly frames the text as a response to critics
                              ("those who say…", "some claim…") without identifying them by name.
                              Do NOT use this label when criticism is marginal (one sentence in a long report),
                              when the text merely mentions a controversy in passing, or when a positive
                              stance is taken without any implied opposition.

"explicit polemic"          — The text directly attacks, refutes, or responds to a NAMED or clearly
                              identifiable opponent, publication, or article. The dispute is the primary
                              purpose of the text, and the adversary is explicitly present.

"meta-polemic (descriptive)"— The text DESCRIBES, SUMMARIZES, or REPORTS ON a controversy or polemic that
                              is taking place among others — the author is a journalist, chronicler, or
                              analyst, not a combatant. The controversy must be the PRIMARY FOCUS of at
                              least a significant portion of the text (not a passing mention).
                              Key signal: the text recounts "there was a dispute about X," summarizes what
                              different parties argued, or analyzes the rhetoric of a debate without taking
                              a side.
                              Examples: a report on a Rabbinical Committee controversy; a diary entry whose
                              main subject is describing disputes at meetings; a biographical account where
                              the author's polemical activities are a central theme.
                              Do NOT use this label when the text merely states in passing that someone
                              "attacked" or "criticized" something — such a passing mention is non-polemic.

WHEN IN DOUBT between "non-polemic" and "implicit polemic": ask whether the text is actively
arguing for or against a contested position. If yes → implicit polemic. If it merely reports,
describes, or expresses feeling without arguing → non-polemic.

WHEN IN DOUBT between "non-polemic" and "meta-polemic (descriptive)": ask whether the text's
primary purpose is to describe a controversy. If yes → meta-polemic (descriptive).

NOTE ON HEBREW IDIOMS: The phrase "לא כתבו ידם" (lit. "their hand did not write") means they
did not sign/endorse something — it is NOT a criticism implying they "did not contribute."
Do not treat this as a polemic marker.

BROADER DEBATE LINK
-------------------
Separately, note whether this text appears to be connected to a known broader public debate
(even if the text itself is non-polemic). Signals: reference to a controversy publicly debated
in the period, a known polemicist or newspaper, or contested subjects (Halukka reform, Hebrew
education, Haskalah vs. Orthodoxy, Hibbat Zion / Zionism, emigration, Hebrew language revival).
Confidence should reflect certainty of this connection.

METADATA (use for context, not as a substitute for reading the text)
--------
{metadata_block}

TEXT
----
{text}

Respond with a JSON object (no markdown, no explanation outside the JSON):
- "polemic_label":            one of the four labels above (string)
- "confidence":               float 0.0–1.0 — confidence in polemic_label
- "polemic_type":             one of "attack", "defense", "debate", "satire", "critique", "none"
- "target":                   who/what is opposed; empty string if non-polemic
- "evidence":                 brief quote or description of key markers (Hebrew ok for quotes)
- "topic":                    main subject in English
- "broader_polemic_link":     one of "none", "suspected", "clear"
- "broader_polemic_justification": one sentence explaining the link (empty if "none")

JSON:"""

# ── continuation detection ────────────────────────────────────────────────────

_CONTINUATION_PATTERNS = re.compile(
    r"(המשך|סוף|חלק\s+[א-ת]|part\s+[IVXivx]+|\(continued\)|\(continuation\)|"
    r"\bII\b|\bIII\b|\bIV\b|\(\d+\)$)",
    re.IGNORECASE | re.UNICODE,
)

def detect_continuation(title: str = "", headline: str = "") -> bool:
    """Return True if title or headline signals a multi-part continuation."""
    combined = f"{title or ''} {headline or ''}".strip()
    return bool(_CONTINUATION_PATTERNS.search(combined))


def build_metadata_block(row: pd.Series) -> str:
    """Format available metadata fields for prompt injection."""
    fields = [
        ("Source",     row.get("source")),
        ("Year",       row.get("year")),
        ("Author",     row.get("author")),
        ("Recipient",  row.get("recipient")),
        ("Newspaper",  row.get("newspaper")),
        ("Headline",   row.get("headline")),
        ("Title",      row.get("title")),
        ("Is continuation of a multi-part series",
         "yes" if detect_continuation(str(row.get("title","")), str(row.get("headline",""))) else "no"),
    ]
    lines = [f"  {k}: {v}" for k, v in fields if v and str(v).strip() not in ("", "nan", "None")]
    return "\n".join(lines) if lines else "  (no metadata available)"


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

# v2 schema
VALID_LABELS_V2      = {"non-polemic", "implicit polemic", "explicit polemic",
                         "meta-polemic (descriptive)", "uncertain", "unlabeled"}
VALID_BROADER_LINK   = {"none", "suspected", "clear"}
EXPECTED_FIELDS_V2   = {"polemic_label", "confidence", "polemic_type", "target",
                         "evidence", "topic", "broader_polemic_link",
                         "broader_polemic_justification"}


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


def validate_classification_v2(result: dict) -> dict:
    """Normalize and validate a v2 classification result."""
    if "_parse_error" in result:
        return {
            "polemic_label": None, "confidence": None, "polemic_type": None,
            "target": "", "evidence": "", "topic": "",
            "broader_polemic_link": None, "broader_polemic_justification": "",
            "_parse_error": result["_parse_error"],
        }
    # polemic_label
    label = str(result.get("polemic_label", "")).strip().lower()
    # Normalize common variants
    if label in ("non polemic", "non-polemic"):
        label = "non-polemic"
    elif label in ("implicit", "implicit polemic"):
        label = "implicit polemic"
    elif label in ("explicit", "explicit polemic"):
        label = "explicit polemic"
    elif "meta" in label or "descriptive" in label:
        label = "meta-polemic (descriptive)"
    if label not in VALID_LABELS_V2:
        label = "unlabeled"
    result["polemic_label"] = label

    # confidence
    conf = result.get("confidence")
    try:
        conf = float(conf)
        conf = max(0.0, min(1.0, conf))
    except (ValueError, TypeError):
        conf = None
    result["confidence"] = conf

    # polemic_type
    ptype = str(result.get("polemic_type", "none")).lower().strip()
    if ptype not in VALID_TYPES:
        ptype = "none"
    result["polemic_type"] = ptype

    # broader_polemic_link
    blink = str(result.get("broader_polemic_link", "none")).lower().strip()
    if blink not in VALID_BROADER_LINK:
        blink = "none"
    result["broader_polemic_link"] = blink

    # string fields
    for field in ("target", "evidence", "topic", "broader_polemic_justification"):
        result[field] = str(result.get(field, ""))

    return {k: result.get(k) for k in EXPECTED_FIELDS_V2}


# --- API Callers ---

async def classify_anthropic(text: str, model_id: str, client) -> dict:
    """Classify a single text using the Anthropic API (v1 schema)."""
    prompt = CLASSIFICATION_PROMPT.format(text=truncate_text(text))
    message = await client.messages.create(
        model=model_id,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text
    return validate_classification(parse_json_response(raw))


async def classify_anthropic_v2(row: pd.Series, model_id: str, client) -> dict:
    """Classify a single text using the Anthropic API (v2 schema with metadata)."""
    metadata_block = build_metadata_block(row)
    prompt = CLASSIFICATION_PROMPT_V2.format(
        metadata_block=metadata_block,
        text=truncate_text(str(row.get("text", ""))),
    )
    message = await client.messages.create(
        model=model_id,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text
    result = validate_classification_v2(parse_json_response(raw))
    result["_input_tokens"]  = message.usage.input_tokens
    result["_output_tokens"] = message.usage.output_tokens
    return result


def classify_cli_v2(row: pd.Series) -> dict:
    """Classify a single text via `claude -p` (uses Max plan, no API credits)."""
    import subprocess
    metadata_block = build_metadata_block(row)
    prompt = CLASSIFICATION_PROMPT_V2.format(
        metadata_block=metadata_block,
        text=truncate_text(str(row.get("text", ""))),
    )
    proc = subprocess.run(
        ["claude", "-p", prompt],
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[:200] or f"exit code {proc.returncode}")
    return validate_classification_v2(parse_json_response(proc.stdout))


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


# --- v2 pipeline ---

async def run_sonnet_v2(texts_df: pd.DataFrame, output_path: Path,
                         model_id: str = None, label: str = "v2 run",
                         use_cli: bool = False) -> pd.DataFrame:
    """Run Sonnet v2 on texts_df. Resumes from output_path if it exists.

    use_cli=True routes through `claude -p` (Max plan, no API credits).
    use_cli=False uses the Anthropic API directly.
    """
    if not use_cli:
        import anthropic
        client = anthropic.AsyncAnthropic()
    else:
        client = None
    if model_id is None:
        model_id = os.getenv("CLAUDE_SONNET_MODEL", "claude-sonnet-4-6")

    # Resume support
    existing_ids = set()
    all_results = []
    if output_path.exists():
        prev = pd.read_parquet(output_path)
        # Only keep successfully classified rows; error rows will be retried and re-appended
        done = prev[prev["polemic_label"].notna() & prev["_error"].isna()] if "_error" in prev.columns else prev[prev["polemic_label"].notna()]
        existing_ids = set(done["doc_id"].tolist())
        all_results = done.to_dict("records")
        print(f"  Resuming: {len(existing_ids)} successfully classified, {len(prev) - len(existing_ids)} errors will be retried.")

    pending = texts_df[~texts_df["doc_id"].isin(existing_ids)]
    route = "claude CLI (Max plan)" if use_cli else model_id
    print(f"  {label}: {len(pending)} texts remaining ({route})")

    # Sonnet 4.6 pricing ($/million tokens) — only meaningful for API path
    INPUT_COST_PER_M  = 3.00
    OUTPUT_COST_PER_M = 15.00

    errors = 0
    total_input_tokens  = 0
    total_output_tokens = 0
    for i, (_, row) in enumerate(pending.iterrows()):
        try:
            if use_cli:
                result = classify_cli_v2(row)
            else:
                result = await classify_anthropic_v2(row, model_id, client)
                total_input_tokens  += result.pop("_input_tokens",  0)
                total_output_tokens += result.pop("_output_tokens", 0)
            result["doc_id"]        = row["doc_id"]
            result["model"]         = "sonnet"
            result["model_display"] = "Claude Sonnet"
            result["tier"]          = "cheap"
            all_results.append(result)
        except Exception as e:
            errors += 1
            all_results.append({
                "doc_id": row["doc_id"], "model": "sonnet",
                "model_display": "Claude Sonnet", "tier": "cheap",
                "polemic_label": None, "confidence": None, "polemic_type": None,
                "target": "", "evidence": "", "topic": "",
                "broader_polemic_link": None, "broader_polemic_justification": "",
                "_error": str(e)[:200],
            })
            print(f"  ERROR on {row['doc_id']}: {str(e)[:100]}")

        if (i + 1) % 20 == 0 or (i + 1) == len(pending):
            if use_cli:
                print(f"  {i + 1}/{len(pending)} done...")
            else:
                cost = (total_input_tokens * INPUT_COST_PER_M
                        + total_output_tokens * OUTPUT_COST_PER_M) / 1_000_000
                print(f"  {i + 1}/{len(pending)} done... "
                      f"(tokens: {total_input_tokens:,} in / {total_output_tokens:,} out, "
                      f"cost so far: ${cost:.2f})")
            df_out = pd.DataFrame(all_results)
            output_path.parent.mkdir(exist_ok=True)
            df_out.to_parquet(output_path, index=False)

        await asyncio.sleep(BATCH_DELAY_SECONDS)

    total_cost = (total_input_tokens * INPUT_COST_PER_M
                  + total_output_tokens * OUTPUT_COST_PER_M) / 1_000_000
    print(f"  Done. {len(all_results)} total, {errors} errors.")
    print(f"  Tokens: {total_input_tokens:,} input / {total_output_tokens:,} output")
    print(f"  Estimated cost: ${total_cost:.2f}")
    return pd.DataFrame(all_results)


def run_acceptance_test(output_path: Path):
    """Run v2 Sonnet on the 23 RA-reviewed annotation-CSV cases, then report agreement."""
    gold_path = ROOT / "data" / "ra_gold_labels.parquet"
    if not gold_path.exists():
        print("ERROR: ra_gold_labels.parquet not found. Run ingest_ra_gold.py first.")
        return

    gold = pd.read_parquet(gold_path)
    # Use only the 23 cases from the annotation CSVs (reviewed pilot cases)
    csv_gold = gold[gold["source"].isin(["cheap_diverge_csv", "disagree_csv"])].copy()
    print(f"Acceptance test: {len(csv_gold)} RA-reviewed cases")

    # Load their texts from corpus
    corpus = pd.read_parquet(ROOT / "corpus.parquet", columns=["doc_id", "text",
                "source", "year", "author", "recipient", "newspaper", "headline", "title"])
    texts = corpus[corpus["doc_id"].isin(csv_gold["doc_id"])].copy()
    missing = set(csv_gold["doc_id"]) - set(texts["doc_id"])
    if missing:
        print(f"  Warning: {len(missing)} gold doc_ids not found in corpus: {missing}")

    if len(texts) == 0:
        print("ERROR: no texts found for gold cases.")
        return

    clf = asyncio.run(run_sonnet_v2(texts, output_path, label="acceptance test"))

    # Compare with gold
    merged = clf.merge(csv_gold[["doc_id", "ra_label_4tier"]], on="doc_id", how="inner")
    match = (merged["polemic_label"] == merged["ra_label_4tier"]).sum()
    total = len(merged)
    pct   = match / total if total > 0 else 0

    print(f"\n=== ACCEPTANCE TEST RESULTS ===")
    print(f"Agreement: {match}/{total} ({pct:.1%})")
    print(f"Threshold: 78% (18/23)")
    print(f"{'✓ PASSED' if pct >= 0.78 else '✗ FAILED — revise prompt before scaling'}")
    print()
    # Per-label breakdown
    print("Label distribution (model vs RA gold):")
    for label in sorted(VALID_LABELS_V2):
        gold_n  = (merged["ra_label_4tier"] == label).sum()
        model_n = (merged["polemic_label"]  == label).sum()
        if gold_n > 0 or model_n > 0:
            print(f"  {label:35s}  gold={gold_n}  model={model_n}")
    print()

    # Show mismatches
    mismatches = merged[merged["polemic_label"] != merged["ra_label_4tier"]]
    if len(mismatches) > 0:
        print("Mismatches:")
        for _, row in mismatches.iterrows():
            print(f"  {row['doc_id']}: model={row['polemic_label']}  gold={row['ra_label_4tier']}")
    print(f"\nFull results saved to {output_path}")


# --- Entry point ---

def main():
    parser = argparse.ArgumentParser(description="Phase B.2: LLM classification")
    parser.add_argument("--models", nargs="+", default=list(MODEL_CONFIGS.keys()),
                        help="Models to run (opus, sonnet, gemini_pro, gemini_flash)")
    parser.add_argument("--report-only", action="store_true",
                        help="Skip classification, just regenerate the agreement report")
    parser.add_argument("--sample-path", default="data/pilot_sample.parquet",
                        help="Path to pilot sample")
    # v2 flags
    parser.add_argument("--v2", action="store_true",
                        help="Use v2 schema (4-tier label, metadata, broader-link)")
    parser.add_argument("--acceptance-test", action="store_true",
                        help="v2: run Sonnet on 23 RA-reviewed gold cases and report agreement")
    parser.add_argument("--calibration", action="store_true",
                        help="v2: run Sonnet on a 2K calibration sample")
    parser.add_argument("--calibration-n", type=int, default=2000,
                        help="Number of texts for calibration run (default 2000)")
    parser.add_argument("--claude-cli", action="store_true",
                        help="Route v2 calls through `claude -p` (Max plan) instead of Anthropic API")
    args = parser.parse_args()

    output_path       = ROOT / "data" / "pilot_classifications.parquet"
    output_path_v2    = ROOT / "data" / "pilot_classifications_v2.parquet"
    acceptance_path   = ROOT / "data" / "acceptance_test_v2.parquet"
    calibration_path  = ROOT / "data" / "calibration_v2.parquet"
    report_path       = ROOT / "data" / "agreement_report.txt"
    disagreement_path = ROOT / "data" / "pilot_disagreements.parquet"

    # ── v2 modes ──────────────────────────────────────────────────────────────
    if args.acceptance_test:
        run_acceptance_test(acceptance_path)
        return

    if args.calibration or args.v2:
        texts_df = pd.read_parquet(ROOT / args.sample_path)
        if args.calibration:
            # Use a stratified 2K sample drawn from full corpus
            corpus = pd.read_parquet(ROOT / "corpus.parquet",
                                     columns=["doc_id","text","source","year","author",
                                              "recipient","newspaper","headline","title"])
            kws = pd.read_parquet(ROOT / "keyword_scores.parquet")
            corpus = corpus.merge(kws[["doc_id","polemic_score"]], on="doc_id", how="left")
            # Keep only overlap window: known in-range OR undated (can't confirm out-of-range)
            # Exclude texts with a confirmed year outside 1862-1888
            corpus = corpus[corpus["year"].isna() | corpus["year"].between(1862, 1888)]
            print(f"Corpus after date filter: {len(corpus)} texts")
            # Stratify by source (proportional) and take up to args.calibration_n
            n = args.calibration_n
            sampled = (corpus.groupby("source", group_keys=False)
                       .apply(lambda g: g.sample(
                           min(len(g), max(1, int(n * len(g) / len(corpus)))),
                           random_state=42)))
            sampled = sampled.sample(min(n, len(sampled)), random_state=42)
            print(f"Calibration sample: {len(sampled)} texts")
            asyncio.run(run_sonnet_v2(sampled, calibration_path, label="calibration", use_cli=args.claude_cli))
        else:
            # --v2 on pilot sample
            asyncio.run(run_sonnet_v2(texts_df, output_path_v2, label="v2 pilot", use_cli=args.claude_cli))
        return

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
