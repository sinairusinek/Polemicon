"""Literature review for egeret_3442 using Gemini 2.5 Flash with Google-Search grounding.

Uses the new google-genai SDK (which supports the `google_search` tool for Gemini 2.5).
Output: data/egeret_literature_review.parquet
"""
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
load_dotenv(ROOT / ".env")

DATA = ROOT / "data"
OUT_PATH = DATA / "egeret_literature_review.parquet"

PROMPT = """You are a research assistant for a digital humanities project on 19th-century Hebrew correspondence. I need scholarly bibliography for a specific 1870 letter by Y.L. Gordon (יהודה ליב גורדון).

CONTEXT:
- Author: Y.L. Gordon (יל"ג), 1830–1892, Hebrew Haskalah poet
- Date: ער"ח אייר תר"ל ≈ April 30, 1870
- Place: Telz / Telšiai, Lithuania
- Recipient: unidentified friend in Šiauliai. Manuscript-editor annotation: "כמדומה לי ר"א אפרתי בשאוועל". The corresponding HaMelitz article (issue 13/1870, 11 April) is signed "א.ל." from Šiauliai.

LETTER CONTENT (summary):
Yechiel Bril (יחיאל בריל), editor of HaLevanon (הלבנון), was running a sustained press campaign accusing Gordon's Haskalah poetry of crypto-apostasy — calling him "עוכר השכלת ישראל". The recipient just published a piece in HaMelitz №13 (1870-04-11) signed "א.ל." defending Gordon and attacking Bril. Gordon:
1) thanks the recipient;
2) pushes back on the recipient's hesitation ("מה לנו להתעבר על ריב לא לנו?") — arguing this is everyone's fight;
3) reports that he held back his own response on advice of friends (including "המגיד מסלוצק"), fell ill, and before Passover sent to R. Moshe David Wolfssohn at HaMelitz his satirical reply titled "ודוי הגדול לדניאל באג'ר" ("The Great Confession of Daniel Bager"); he fears the HaMelitz editor (whose own piece "משיב מלחמה שערה" he finds wishy-washy) will decline to print it;
4) also mentions: Senior Zaks of Paris sent Gordon his book on Ibn Gabirol's Shir HaShirim for review; Yosef Horowitz of Grodno commissioned annotations on a Russian-translation prayerbook.

KEY TOPICS to search for:
- The 1870 Gordon–Bril press polemic (specifically)
- Gordon's polemical poetry of late 1860s–early 1870s and reactions to it
- Yechiel Bril and HaLevanon as a traditionalist organ
- The Haskalah–orthodoxy press wars of 1869–1871
- The pseudonym "Daniel Bager" / Gordon's piece "ודוי הגדול לדניאל באג'ר"
- The "א.ל." signature in HaMelitz №13/1870 from Šiauliai (possible identity)

Note: Michael Stanislawski's *For Whom Do I Toil? Judah Leib Gordon and the Crisis of Russian Jewry* (Oxford UP, 1988) is the central English-language biography — verify what it covers about this controversy. Search both English and Hebrew sources.

Return ONLY a JSON object (no markdown fences) with EXACTLY these keys:
- "is_documented": "well-documented" | "mentioned-in-passing" | "not-found"
- "topic_canonical": true|false
- "key_sources": list of up to 8 sources, each with: "author","title","year"(int or 0),"type"("book"|"article"|"chapter"|"encyclopedia"|"thesis"|"web"),"url","where_discussed"(1 sentence),"relevance"("central"|"biographical"|"contextual"|"pseudonym"|"other")
- "pseudonym_leads": list (possibly empty) of leads on "א.ל." from Šiauliai ~1870
- "notes": 2–4 sentences on coverage gaps and findings

Only cite sources you actually find via Google Search grounding. Do not fabricate."""


def parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0]
    s = raw.find("{")
    e = raw.rfind("}")
    if s >= 0 and e > s:
        raw = raw[s:e + 1]
    return json.loads(raw)


def main():
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    model_id = os.getenv("GEMINI_FLASH_MODEL", "gemini-2.5-flash")
    print(f"egeret_3442 lit review · model={model_id}", flush=True)

    t0 = time.time()
    resp = client.models.generate_content(
        model=model_id,
        contents=PROMPT,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )
    wall = time.time() - t0
    raw = resp.text.strip()

    grounding = {}
    try:
        cand = resp.candidates[0]
        gm = getattr(cand, "grounding_metadata", None)
        if gm:
            grounding["web_search_queries"] = list(getattr(gm, "web_search_queries", []) or [])
            chunks = getattr(gm, "grounding_chunks", []) or []
            grounding["chunk_urls"] = [getattr(getattr(c, "web", None), "uri", "") for c in chunks]
    except Exception as e:
        grounding["_err"] = str(e)[:200]

    try:
        parsed = parse_json(raw)
    except Exception as e:
        print(f"PARSE ERROR: {e}\nRAW:\n{raw[:3000]}", file=sys.stderr)
        sys.exit(1)

    usage = getattr(resp, "usage_metadata", None)
    in_tok = getattr(usage, "prompt_token_count", 0) if usage else 0
    out_tok = getattr(usage, "candidates_token_count", 0) if usage else 0

    row = {
        "doc_id": "egeret_3442",
        "is_documented": parsed.get("is_documented"),
        "topic_canonical": bool(parsed.get("topic_canonical", False)),
        "key_sources": json.dumps(parsed.get("key_sources", []), ensure_ascii=False),
        "pseudonym_leads": json.dumps(parsed.get("pseudonym_leads", []), ensure_ascii=False),
        "notes": parsed.get("notes", ""),
        "grounding": json.dumps(grounding, ensure_ascii=False),
        "_wall_seconds": wall,
        "_input_tokens": in_tok,
        "_output_tokens": out_tok,
        "_model": model_id,
    }
    print(json.dumps(row, ensure_ascii=False, indent=2))
    pd.DataFrame([row]).to_parquet(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH} · wall={wall:.1f}s · in={in_tok} out={out_tok}", flush=True)


if __name__ == "__main__":
    main()
