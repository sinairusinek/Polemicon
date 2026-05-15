"""One-off literature-review for a single egeret item (egeret_3442).

Mirrors thread_literature_review.py but adapts the prompt to a private 1870
Hebrew letter rather than a press thread. Output: JSON written to
data/egeret_literature_review.parquet.
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

PROMPT = """You are a research assistant for a digital humanities project on 19th-century Hebrew correspondence and press polemics. I need scholarly bibliography for a specific 1870 letter by Y.L. Gordon (יהודה ליב גורדון, יל"ג).

LETTER CONTEXT:
- Author: Yehuda Leib Gordon (יל"ג), 1830–1892, Hebrew poet and central Haskalah figure
- Date: ער"ח אייר תר"ל ≈ April 30, 1870
- Place written: Telz / Telšiai (Lithuania)
- Recipient: an unidentified friend in Šiauliai (Lithuania); the manuscript editor's annotation reads "כמדומה לי ר"א אפרתי בשאוועל"; the corresponding article in HaMelitz is signed "א.ל."
- Letter ID in Project Ben-Yehuda's letters collection: letter №77 within אגרון יל"ג (collection 9991)

LETTER CONTENT (in summary):
The letter is Gordon's tactical response to a specific public controversy: Yechiel Bril (יחיאל בריל), the editor of *HaLevanon* (הלבנון), was running a sustained campaign attacking Gordon's poetry as crypto-apostasy, calling Gordon "עוכר השכלת ישראל" — a destroyer of Jewish faith. The recipient had just published an article in *HaMelitz* issue №13 of 1870 (April 11), signed "א.ל." from Šiauliai, defending Gordon and counter-attacking Bril.

In the letter, Gordon:
1. Thanks the recipient for the defense, which he hadn't expected.
2. Pushes back on the recipient's hesitation ("מה לנו להתעבר על ריב לא לנו?") — arguing the orthodox attack on him is an attack on the whole maskilic project.
3. Reports that he held back from publishing his own response, having been advised silence by friends including "המגיד מסלוצק" (the preacher of Slutsk), who promised to fight on his behalf. He then fell ill, but before Passover sent his own satirical defense to R. Moshe David Wolfssohn at *HaMelitz* — a piece titled "ודוי הגדול לדניאל באג'ר" ("The Great Confession of Daniel Bager") — though he fears the *HaMelitz* editor (whose article "משיב מלחמה שערה" looks wishy-washy to him) may decline to print it.
4. The letter also briefly references: Senior (שניאור) Zaks in Paris who sent Gordon his book on Ibn Gabirol's Shir HaShirim for review; R. Yosef Horowitz of Grodno who commissioned annotations on his Russian-translation prayerbook; and a rabbinic assembly with a *shochet* dispute.

KEY TOPICS for which I want secondary literature:
- The 1870 Gordon–Bril press polemic specifically (this is the central event)
- Y.L. Gordon's biography, his polemical poetry of the late 1860s and early 1870s, and the controversies it provoked
- Yechiel Bril and *HaLevanon* as a traditionalist press organ
- The Haskalah–orthodoxy press wars of 1869–1871 more broadly
- The pseudonym "Daniel Bager" / "ודוי הגדול לדניאל באג'ר" — Gordon's satirical persona; any secondary literature on this specific piece
- The recipient's pseudonym "א.ל." or possible identity (Russian-Jewish maskil in Šiauliai, ~1870)

Please search the web for SCHOLARLY SECONDARY literature. Prioritize: peer-reviewed books and articles (English or Hebrew), encyclopedia entries (Encyclopaedia Judaica, YIVO, etc.), academic biographies of Gordon and Bril, and Hebrew-press history works. Note that Michael Stanislawski's *For Whom Do I Toil? Judah Leib Gordon and the Crisis of Russian Jewry* (Oxford UP, 1988) is the central English-language biography and should be checked for coverage of this specific controversy.

Return a JSON object (no markdown fences, no prose outside JSON) with EXACTLY these keys:

- "is_documented": one of "well-documented" (multiple scholarly sources discuss this 1870 controversy), "mentioned-in-passing" (referenced in works on Gordon or HaLevanon but not analyzed in depth), or "not-found".
- "topic_canonical": true if the broader Gordon/Bril/Haskalah-orthodoxy clash is canonical in Jewish historiography, false otherwise.
- "key_sources": JSON list of up to 8 source objects, each with EXACTLY:
    - "author": author or organization (or "")
    - "title": work title
    - "year": integer publication year or 0
    - "type": "book" | "article" | "chapter" | "encyclopedia" | "thesis" | "web"
    - "url": direct URL or ""
    - "where_discussed": one sentence on how the source relates to this letter's themes
    - "relevance": one of "central" (directly covers Gordon–Bril 1870), "biographical" (Gordon biography touching this period), "contextual" (broader period/Haskalah press history), "pseudonym" (resolves "א.ל." or "Daniel Bager"), "other"
- "pseudonym_leads": JSON list (possibly empty) of any leads on the "א.ל." pseudonym from Šiauliai, ~1870 — including educated guesses with reasoning.
- "notes": 2–4 sentences of additional context — what is well-covered, what is missing, and any noteworthy findings.

CRITICAL: cite only sources you actually found via web search. Do not fabricate citations or URLs. If a source has no URL, leave url="". If literature is sparse, return fewer sources or "not-found".

Respond with ONLY the JSON object.
"""


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
    import anthropic
    client = anthropic.Anthropic()
    model_id = os.getenv("CLAUDE_SONNET_MODEL", "claude-sonnet-4-6")

    print(f"egeret_3442 literature review · model={model_id}", flush=True)
    t0 = time.time()
    resp = client.messages.create(
        model=model_id,
        max_tokens=4096,
        tools=[{
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 8,
        }],
        messages=[{"role": "user", "content": PROMPT}],
    )
    text_blocks = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    raw = "\n".join(text_blocks).strip()
    n_searches = sum(1 for b in resp.content if getattr(b, "type", None) == "server_tool_use")
    wall = time.time() - t0
    in_tok = resp.usage.input_tokens
    out_tok = resp.usage.output_tokens

    try:
        parsed = parse_json(raw)
    except Exception as e:
        print(f"PARSE ERROR: {e}\nRAW:\n{raw[:2000]}", file=sys.stderr)
        sys.exit(1)

    row = {
        "doc_id": "egeret_3442",
        "is_documented": parsed.get("is_documented"),
        "topic_canonical": bool(parsed.get("topic_canonical", False)),
        "key_sources": json.dumps(parsed.get("key_sources", []), ensure_ascii=False),
        "pseudonym_leads": json.dumps(parsed.get("pseudonym_leads", []), ensure_ascii=False),
        "notes": parsed.get("notes", ""),
        "_n_searches": n_searches,
        "_wall_seconds": wall,
        "_input_tokens": in_tok,
        "_output_tokens": out_tok,
    }
    print(json.dumps(row, ensure_ascii=False, indent=2))
    df = pd.DataFrame([row])
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH} · searches={n_searches} · wall={wall:.1f}s · in={in_tok} out={out_tok}", flush=True)


if __name__ == "__main__":
    main()
