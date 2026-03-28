"""
extract_references.py - Intertextual reference extraction from pilot texts

Two-layer approach:
1. Mechanical extraction: footnotes (↩ markers), newspaper names, quoted attributions
2. LLM extraction (Sonnet): structured reference detection with categorization

Categories:
- biblical: Torah, Prophets, Writings citations
- talmudic: Mishnah, Talmud, Midrash citations
- contemporary_person: reference to a contemporary author/intellectual
- contemporary_publication: reference to a newspaper, journal, or book
- contemporary_text: reference to a specific article or letter
- scholarly: reference to academic/scholarly works (non-Jewish)
- other: anything else

Output: data/pilot_references.parquet
"""
import os
import sys
import json
import asyncio
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
load_dotenv(ROOT / ".env")

# Force unbuffered output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

# --- Mechanical extraction ---

# Known newspaper names (with variants)
NEWSPAPER_PATTERNS = {
    "המגיד": "HaMagid",
    "המליץ": "HaMelitz",
    "הצפירה": "HaTzfira",
    "חבצלת": "Havatzelet",
    "הלבנון": "HaLevanon",
    "השחר": "HaShachar",
    "הכרמל": "HaCarmel",
    "הגרן": "HaGoren",
    "הפסגה": "HaPisgah",
    "העברי": "HaIvri",
    "הצופה": "HaTzofe",
    "דבר": "Davar",
    "לוח ארץ ישראל": "Luach Eretz Israel",
}

# Common abbreviation patterns for newspapers
NEWSPAPER_ABBREV = {
    "הצפ'": "HaTzfira",
    "המל'": "HaMelitz",
    "המג'": "HaMagid",
}

# Biblical book names (Hebrew)
BIBLICAL_BOOKS = [
    "בראשית", "שמות", "ויקרא", "במדבר", "דברימ",
    "יהושע", "שופטימ", "שמואל", "מלכימ",
    "ישעיה", "ירמיה", "יחזקאל",
    "הושע", "יואל", "עמוס", "עובדיה", "יונה", "מיכה", "נחומ", "חבקוק", "צפניה", "חגי", "זכריה", "מלאכי",
    "תהלימ", "משלי", "איוב", "שיר השירימ", "רות", "איכה", "קהלת", "אסתר", "דניאל", "עזרא", "נחמיה", "דברי הימימ",
]

# Talmudic tractate names
TALMUDIC_TRACTATES = [
    "ברכות", "שבת", "עירובינ", "פסחימ", "שקלימ", "יומא", "סוכה", "ביצה", "ראש השנה",
    "תענית", "מגילה", "מועד קטנ", "חגיגה", "יבמות", "כתובות", "נדרימ", "נזיר", "סוטה",
    "גיטינ", "קידושינ", "בבא קמא", "בבא מציעא", "בבא בתרא", "סנהדרינ", "מכות", "שבועות",
    "עבודה זרה", "הוריות", "זבחימ", "מנחות", "חולינ", "בכורות", "ערכינ", "תמורה", "כריתות",
    "מעילה", "תמיד", "מדות", "קנימ", "נדה", "אבות",
]


def extract_footnotes(text: str) -> list[dict]:
    """Extract footnotes marked with ↩ from Ben-Yehuda texts."""
    refs = []
    # Split on ↩ to get footnote segments
    parts = text.split("↩")
    if len(parts) <= 1:
        return refs

    for i, part in enumerate(parts[:-1]):  # last part is after final ↩
        # Get the text before ↩ (the footnote content is before the marker)
        # Take last ~200 chars before ↩ as footnote context
        fn_text = part.strip()
        if "&nbsp;" in fn_text:
            # Footnote content is typically between &nbsp; markers
            segments = fn_text.split("&nbsp;")
            fn_content = segments[-1].strip() if segments else fn_text[-200:]
        else:
            fn_content = fn_text[-200:] if len(fn_text) > 200 else fn_text

        if len(fn_content) > 5:  # skip empty footnotes
            refs.append({
                "method": "mechanical_footnote",
                "raw_text": fn_content[:300],
                "footnote_index": i + 1,
            })
    return refs


def extract_newspaper_mentions(text: str) -> list[dict]:
    """Find newspaper/journal names in text."""
    refs = []
    for heb_name, eng_name in {**NEWSPAPER_PATTERNS, **NEWSPAPER_ABBREV}.items():
        for m in re.finditer(re.escape(heb_name), text):
            start = max(0, m.start() - 40)
            end = min(len(text), m.end() + 40)
            context = text[start:end]
            refs.append({
                "method": "mechanical_newspaper",
                "category": "contemporary_publication",
                "target_name": heb_name,
                "target_name_eng": eng_name,
                "raw_text": context,
                "position": m.start(),
            })
    return refs


def extract_quoted_attributions(text: str) -> list[dict]:
    """Find attribution patterns: כתב/אמר/טען + name."""
    refs = []
    # Pattern: attribution verb + name-like phrase
    pattern = r'(כתב|אמר|טענ|השיב|ענה|הודיע|הראה|העיר|הוכיח|ציינ|זכר)\s+(\S+(?:\s+\S+)?)\s+(כי|ש|:|,|\.)'
    for m in re.finditer(pattern, text):
        refs.append({
            "method": "mechanical_attribution",
            "raw_text": text[max(0, m.start()-20):m.end()+20],
            "verb": m.group(1),
            "attributed_to": m.group(2),
            "position": m.start(),
        })
    return refs


def mechanical_extraction(text: str, doc_id: str) -> list[dict]:
    """Run all mechanical extraction methods on a text."""
    all_refs = []

    footnotes = extract_footnotes(text)
    for fn in footnotes:
        fn["doc_id"] = doc_id
        all_refs.append(fn)

    newspapers = extract_newspaper_mentions(text)
    for np in newspapers:
        np["doc_id"] = doc_id
        all_refs.append(np)

    attributions = extract_quoted_attributions(text)
    for attr in attributions:
        attr["doc_id"] = doc_id
        all_refs.append(attr)

    return all_refs


# --- LLM extraction ---

MAX_TEXT_WORDS = 4000

REFERENCE_EXTRACTION_PROMPT = """You are an expert in 19th-century Hebrew literature (Haskalah era, 1862-1888).
Analyze the following Hebrew text and extract ALL intertextual references — every mention of, citation from, or allusion to another text, person, or source.

For each reference found, classify it into one of these categories:
- "biblical": Torah, Prophets, Writings (Tanakh) citations or allusions
- "talmudic": Mishnah, Talmud, Midrash, or other rabbinic literature
- "contemporary_person": reference to a contemporary (19th-century) author, intellectual, rabbi, or public figure
- "contemporary_publication": reference to a newspaper, journal, periodical, or book from the era
- "contemporary_text": reference to a specific article, letter, essay, or speech
- "scholarly": reference to academic/scholarly works (including non-Jewish European scholarship)
- "other": anything else (medieval commentators, Kabbalistic works, etc.)

Respond with a JSON array (no markdown fences, no explanation outside the JSON). Each element should have:
- "category": one of the categories above
- "target_name": the name of the referenced person, text, or source (in Hebrew if Hebrew, transliterated if not)
- "target_type": "person", "publication", "text", "book", "verse", "tractate", or "concept"
- "context": the Hebrew phrase/sentence where the reference appears (brief, max 50 words)
- "reference_type": "explicit_citation" (direct quote), "attribution" (X said/wrote), "allusion" (indirect), "response_to" (argues against), "footnote" (editorial/footnote reference)
- "confidence": float 0.0-1.0

If no references are found, return an empty array: []

TEXT:
{text}

JSON:"""


def truncate_text(text: str, max_words: int = MAX_TEXT_WORDS) -> str:
    words = str(text).split()
    if len(words) <= max_words:
        return str(text)
    return " ".join(words[:max_words]) + " [...]"


def parse_json_array(raw: str) -> list:
    """Extract JSON array from model response."""
    raw = raw.strip()
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("["):
                raw = part
                break
    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start == -1 or end == 0:
        return []
    try:
        return json.loads(raw[start:end])
    except json.JSONDecodeError:
        return []


VALID_CATEGORIES = {"biblical", "talmudic", "contemporary_person", "contemporary_publication",
                    "contemporary_text", "scholarly", "other"}


def validate_reference(ref: dict) -> dict:
    """Normalize a single extracted reference."""
    cat = str(ref.get("category", "other")).lower().strip()
    if cat not in VALID_CATEGORIES:
        cat = "other"
    ref["category"] = cat
    ref["target_name"] = str(ref.get("target_name", ""))
    ref["target_type"] = str(ref.get("target_type", ""))
    ref["context"] = str(ref.get("context", ""))[:300]
    ref["reference_type"] = str(ref.get("reference_type", ""))
    conf = ref.get("confidence")
    try:
        conf = max(0.0, min(1.0, float(conf)))
    except (ValueError, TypeError):
        conf = 0.5
    ref["confidence"] = conf
    return ref


async def extract_references_llm(text: str, doc_id: str, client) -> list[dict]:
    """Extract references from a single text using Claude Sonnet."""
    prompt = REFERENCE_EXTRACTION_PROMPT.format(text=truncate_text(text))
    model_id = os.getenv("CLAUDE_SONNET_MODEL", "claude-sonnet-4-6")

    message = await client.messages.create(
        model=model_id,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text
    refs = parse_json_array(raw)

    validated = []
    for ref in refs:
        ref = validate_reference(ref)
        ref["doc_id"] = doc_id
        ref["method"] = "llm_sonnet"
        validated.append(ref)
    return validated


# --- Main pipeline ---

async def run_extraction(texts_df: pd.DataFrame, llm_doc_ids: set, output_path: Path):
    """Run mechanical extraction on all texts, LLM only on selected texts."""
    import anthropic
    client = anthropic.AsyncAnthropic()

    all_refs = []

    # Phase 1: Mechanical extraction on ALL texts (instant)
    print("Phase 1: Mechanical extraction (all texts)...")
    for _, row in texts_df.iterrows():
        mech_refs = mechanical_extraction(str(row["text"]), row["doc_id"])
        all_refs.extend(mech_refs)
    print(f"  {len(all_refs)} mechanical references extracted.")

    # Phase 2: LLM extraction on selected texts only
    llm_texts = texts_df[texts_df["doc_id"].isin(llm_doc_ids)]
    print(f"\nPhase 2: LLM extraction (Sonnet) on {len(llm_texts)} polemic texts "
          f"(3+ model agreement)...")
    errors = 0
    for i, (_, row) in enumerate(llm_texts.iterrows()):
        try:
            llm_refs = await extract_references_llm(str(row["text"]), row["doc_id"], client)
            all_refs.extend(llm_refs)
        except Exception as e:
            errors += 1
            print(f"  ERROR on {row['doc_id']}: {str(e)[:100]}")

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(llm_texts)} done...")
            # Checkpoint
            df_out = pd.DataFrame(all_refs)
            df_out.to_parquet(output_path, index=False)

        await asyncio.sleep(1.0)

    print(f"  LLM extraction done. {errors} errors.")

    # Save final results
    df_out = pd.DataFrame(all_refs)
    df_out.to_parquet(output_path, index=False)
    return df_out


def print_summary(refs_df: pd.DataFrame):
    """Print extraction summary."""
    print("\n" + "=" * 60)
    print("REFERENCE EXTRACTION SUMMARY")
    print("=" * 60)

    print(f"\nTotal references: {len(refs_df)}")
    print(f"Unique texts with references: {refs_df['doc_id'].nunique()}")

    print(f"\nBy method:")
    for method, count in refs_df["method"].value_counts().items():
        print(f"  {method}: {count}")

    if "category" in refs_df.columns:
        llm_refs = refs_df[refs_df["method"] == "llm_sonnet"]
        if len(llm_refs) > 0:
            print(f"\nLLM references by category:")
            for cat, count in llm_refs["category"].value_counts().items():
                print(f"  {cat}: {count}")

            print(f"\nLLM references by reference_type:")
            for rt, count in llm_refs["reference_type"].value_counts().items():
                print(f"  {rt}: {count}")

            # Top referenced targets
            print(f"\nTop referenced targets (contemporary):")
            contemp = llm_refs[llm_refs["category"].str.startswith("contemporary")]
            for target, count in contemp["target_name"].value_counts().head(15).items():
                print(f"  {target}: {count}")

    print("\n" + "=" * 60)


def get_polemic_doc_ids(min_votes: int = 3) -> set:
    """Get doc_ids where >= min_votes models classified as polemic."""
    clf_path = ROOT / "data" / "pilot_classifications.parquet"
    if not clf_path.exists():
        return set()
    clf = pd.read_parquet(clf_path)
    votes = clf.groupby("doc_id")["is_polemic"].sum()
    return set(votes[votes >= min_votes].index)


def main():
    output_path = ROOT / "data" / "pilot_references.parquet"

    print("Loading pilot sample...")
    texts_df = pd.read_parquet(ROOT / "data" / "pilot_sample.parquet")
    print(f"  {len(texts_df)} texts loaded.")

    # LLM extraction only on texts with 3+ polemic votes
    llm_doc_ids = get_polemic_doc_ids(min_votes=3)
    print(f"  {len(llm_doc_ids)} texts with 3+ polemic model votes (LLM target).")

    refs_df = asyncio.run(run_extraction(texts_df, llm_doc_ids, output_path))

    print(f"\nSaved {len(refs_df)} references to {output_path}")
    print_summary(refs_df)

    print("\nDone. References are ready for Streamlit display.")


if __name__ == "__main__":
    main()
