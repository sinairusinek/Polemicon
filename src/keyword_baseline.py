"""
keyword_baseline.py - Rule-based polemic scoring for the Polemicon project (Phase B.3)

Scores each text using a Hebrew polemic indicator lexicon:
- Debate markers, address markers, evaluative intensifiers
- Rhetorical question density, quotation density
- Outputs keyword_scores.parquet with per-text scores
"""
import os
import re
import pandas as pd
import numpy as np


# Final-form normalization (must match cleaning.py)
FINALS = {'ך': 'כ', 'ם': 'מ', 'ן': 'נ', 'ף': 'פ', 'ץ': 'צ'}


def normalize_finals(word):
    for final, norm in FINALS.items():
        word = word.replace(final, norm)
    return word


# Polemic indicator lexicon (normalized to match cleaned corpus)
DEBATE_MARKERS = [normalize_finals(w) for w in ["אך", "אבל", "אולם", "להפך", "חלילה"]]
ADDRESS_MARKERS = [normalize_finals(w) for w in ["השיב", "ענה", "טען", "כתב"]]
NEGATIVE_SUPERLATIVES = [normalize_finals(w) for w in ["איום", "נורא", "מרעיש"]]
SARCASTIC_SUPERLATIVES = [normalize_finals(w) for w in ["נאצל", "נכבד", "גבוה", "רם"]]
EVALUATIVE_INTENSIFIERS = [normalize_finals(w) for w in [
    "שקר", "כזב", "הבל", "טעות", "סכלות", "טמא", "עוון", "חטא",
    "חשוך", "רפה", "זוהמה", "מרעיש", "שערוריה", "צבוע", "מתחסד",
]]

ALL_KEYWORDS = (
    DEBATE_MARKERS + ADDRESS_MARKERS + NEGATIVE_SUPERLATIVES +
    SARCASTIC_SUPERLATIVES + EVALUATIVE_INTENSIFIERS
)


def count_keywords(text, keywords):
    """Count occurrences of keyword list in text."""
    count = 0
    for kw in keywords:
        count += len(re.findall(r'\b' + re.escape(kw) + r'\b', text))
    return count


def rhetorical_question_density(text):
    """Fraction of sentences ending with ? or ?!"""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    questions = len(re.findall(r'\?[!]?', text))
    return questions / len(sentences)


def quotation_density(text):
    """Fraction of text within quotation marks."""
    # Count characters inside Hebrew-style quotes or standard quotes
    quoted = re.findall(r'["\u201c\u201d\u05f4](.*?)["\u201c\u201d\u05f4]', text, re.DOTALL)
    quoted_chars = sum(len(q) for q in quoted)
    return quoted_chars / len(text) if text else 0.0


def score_text(text):
    """Compute polemic indicator scores for a single text."""
    text = str(text)
    word_count = len(text.split())
    if word_count == 0:
        return {
            "debate_markers": 0, "address_markers": 0,
            "negative_superlatives": 0, "sarcastic_superlatives": 0,
            "evaluative_intensifiers": 0, "total_keyword_count": 0,
            "keyword_density": 0.0, "rhetorical_q_density": 0.0,
            "quotation_density": 0.0, "polemic_score": 0.0,
        }

    debate = count_keywords(text, DEBATE_MARKERS)
    address = count_keywords(text, ADDRESS_MARKERS)
    negative = count_keywords(text, NEGATIVE_SUPERLATIVES)
    sarcastic = count_keywords(text, SARCASTIC_SUPERLATIVES)
    evaluative = count_keywords(text, EVALUATIVE_INTENSIFIERS)
    total_kw = debate + address + negative + sarcastic + evaluative
    kw_density = total_kw / word_count
    rq_density = rhetorical_question_density(text)
    q_density = quotation_density(text)

    # Composite polemic score: weighted sum, normalized to 0-1 range
    # Weights reflect relative importance of each signal
    raw_score = (
        kw_density * 50 +       # keyword density (most important)
        rq_density * 20 +        # rhetorical questions
        q_density * 10 +         # quotation density
        min(debate / word_count * 100, 1) * 10 +  # debate marker boost
        min(evaluative / word_count * 100, 1) * 10  # evaluative boost
    )
    polemic_score = min(raw_score / 100, 1.0)  # cap at 1.0

    return {
        "debate_markers": debate,
        "address_markers": address,
        "negative_superlatives": negative,
        "sarcastic_superlatives": sarcastic,
        "evaluative_intensifiers": evaluative,
        "total_keyword_count": total_kw,
        "keyword_density": round(kw_density, 6),
        "rhetorical_q_density": round(rq_density, 4),
        "quotation_density": round(q_density, 4),
        "polemic_score": round(polemic_score, 4),
    }


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("Loading corpus...")
    corpus = pd.read_parquet("corpus.parquet")
    corpus = corpus[corpus["doc_id"] != "bypc_5539"].reset_index(drop=True)
    print(f"  {len(corpus)} texts.")

    print("Scoring texts...")
    scores = corpus["text"].apply(score_text).apply(pd.Series)
    result = pd.concat([corpus[["doc_id", "source"]], scores], axis=1)

    result.to_parquet("keyword_scores.parquet", index=False)
    print(f"Saved keyword_scores.parquet ({len(result)} rows)")

    # Summary stats
    print(f"\nPolemic score distribution:")
    print(f"  mean:   {result['polemic_score'].mean():.4f}")
    print(f"  median: {result['polemic_score'].median():.4f}")
    print(f"  >0.3:   {(result['polemic_score'] > 0.3).sum()} texts ({(result['polemic_score'] > 0.3).mean():.1%})")
    print(f"  >0.5:   {(result['polemic_score'] > 0.5).sum()} texts ({(result['polemic_score'] > 0.5).mean():.1%})")
    print(f"\nBy source:")
    print(result.groupby("source")["polemic_score"].describe().round(4).to_string())


if __name__ == "__main__":
    main()
