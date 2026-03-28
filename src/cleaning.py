"""
cleaning.py - Text cleaning utilities for the Polemicon project

Implements:
1. Ben-Yehuda footer removal
2. Hebrew normalization (remove nikkud, normalize final forms, standardize punctuation)
3. Non-Hebrew segment detection
4. Quality score computation
5. Minimum length filter
"""
import re
import pandas as pd

# 1. Ben-Yehuda footer removal
def remove_by_footer(text):
    # Example: Remove common Ben-Yehuda footer patterns
    footer_pattern = r"\n?--\s*הועתק מ\S*פרויקט בן-יהודה.*"
    return re.sub(footer_pattern, '', text, flags=re.DOTALL)

# 2. Hebrew normalization
def normalize_hebrew(text):
    # Remove nikkud
    text = re.sub(r'[\u0591-\u05C7]', '', text)
    # Normalize final forms (example: ך > כ, ם > מ, ן > נ, ף > פ, ץ > צ)
    finals = {'ך': 'כ', 'ם': 'מ', 'ן': 'נ', 'ף': 'פ', 'ץ': 'צ'}
    for final, norm in finals.items():
        text = text.replace(final, norm)
    # Standardize punctuation (example: convert smart quotes, dashes, etc.)
    text = re.sub(r'[“”]', '"', text)
    text = re.sub(r'[‘’]', "'", text)
    text = re.sub(r'[–—]', '-', text)
    return text

# 2b. Restore final forms for display (reverse of normalize step)
def restore_final_forms(text):
    """Replace word-final כ מ נ פ צ with their final forms ך ם ן ף ץ."""
    # Map: regular form -> final form
    to_final = {'כ': 'ך', 'מ': 'ם', 'נ': 'ן', 'פ': 'ף', 'צ': 'ץ'}
    # Replace at end of word: Hebrew letter followed by non-Hebrew or end of string
    for regular, final in to_final.items():
        text = re.sub(regular + r'(?=\s|[^\u05D0-\u05EA]|$)', final, text)
    return text


# 3. Non-Hebrew segment detection
def detect_non_hebrew_segments(text):
    # Returns True if non-Hebrew (Latin, German) segments are detected
    return bool(re.search(r'[A-Za-zÄÖÜäöüß]', text))

# 4. Quality score computation
def compute_quality_score(text):
    heb_chars = re.findall(r'[\u05D0-\u05EA]', text)
    total_chars = len(text)
    heb_ratio = len(heb_chars) / total_chars if total_chars else 0
    words = re.findall(r'\w+', text)
    avg_word_len = sum(len(w) for w in words) / len(words) if words else 0
    return {'hebrew_ratio': heb_ratio, 'avg_word_len': avg_word_len}

# 5. Minimum length filter
def is_long_enough(text, min_words=200):
    heb_words = re.findall(r'[\u05D0-\u05EA]{2,}', text)
    return len(heb_words) >= min_words
