"""
corpus.py - Build unified corpus for the Polemicon project

- Overlap window: 1862-1900 (extended)
- Unifies press, e-geret, and polemic candidates datasets
- Applies cleaning and filtering
- Recovers dates for polemic candidates via Ben-Yehuda metadata
- Outputs corpus.parquet
"""
import pandas as pd
import os
from loaders import load_press_articles, load_egeret_letters, load_polemic_candidates
from cleaning import remove_by_footer, normalize_hebrew, detect_non_hebrew_segments, compute_quality_score, is_long_enough

# Paths
PRESS_PATH = 'MGD-LBN-MLZ-HZF-HZTfull2021-08-14-(1)-tsv.csv'
EGERET_PATH = 'e-geret-batch-export.tsv'
POLEMIC_PATH = 'Ben-Yehuda-Project-polemic-candidates.csv'
BY_METADATA_PATH = 'benyehuda-full-metadata.tsv'

# Overlap window (enlarged)
START_YEAR = 1850
END_YEAR = 1900

def clean_text(text):
    text = remove_by_footer(str(text))
    text = normalize_hebrew(text)
    return text

def filter_and_score(text):
    if not is_long_enough(text):
        return False, None
    quality = compute_quality_score(text)
    return True, quality

def recover_candidate_dates(candidates_df, by_meta_df):
    # Extract numeric ID from File column
    def extract_id(file_path):
        import re
        m = re.search(r'm(\d+)\\?.txt', str(file_path))
        return int(m.group(1)) if m else None
    candidates_df['by_id'] = candidates_df['File'].apply(extract_id)
    by_meta_df['id'] = pd.to_numeric(by_meta_df['id'], errors='coerce')
    merged = candidates_df.merge(by_meta_df[['id', 'orig_publication_date', 'title', 'author_string', 'genre']], left_on='by_id', right_on='id', how='left')
    return merged

def main():
    # Load datasets
    press = load_press_articles(PRESS_PATH)
    egeret = load_egeret_letters(EGERET_PATH)
    candidates = load_polemic_candidates(POLEMIC_PATH)
    by_meta = pd.read_csv(BY_METADATA_PATH, sep='\t', encoding='utf-8')

    # Clean and filter
    press['text'] = press['text'].apply(clean_text)
    egeret['text'] = egeret['Content'].apply(clean_text)
    candidates['text'] = candidates['Column 1'].apply(clean_text)

    press['keep'], press['quality_score'] = zip(*press['text'].map(filter_and_score))
    egeret['keep'], egeret['quality_score'] = zip(*egeret['text'].map(filter_and_score))
    candidates['keep'], candidates['quality_score'] = zip(*candidates['text'].map(filter_and_score))

    press = press[press['keep']]
    egeret = egeret[egeret['keep']]
    candidates = candidates[candidates['keep']]

    # Recover dates for candidates
    candidates = recover_candidate_dates(candidates, by_meta)

    # Build unified DataFrame
    def get_year(date):
        try:
            return int(str(date)[:4])
        except:
            return None

    press['year'] = press['date'].apply(get_year) if 'date' in press else None
    egeret['year'] = egeret['date'].apply(get_year) if 'date' in egeret else None
    # Use 'orig_publication_date' for candidates
    candidates['date'] = candidates['orig_publication_date'] if 'orig_publication_date' in candidates else None
    candidates['author'] = candidates['author_string'] if 'author_string' in candidates else None
    candidates['year'] = candidates['date'].apply(get_year) if 'date' in candidates else None

    press['in_overlap'] = press['year'].between(START_YEAR, END_YEAR, inclusive='both')
    egeret['in_overlap'] = egeret['year'].between(START_YEAR, END_YEAR, inclusive='both')
    candidates['in_overlap'] = candidates['year'].between(START_YEAR, END_YEAR, inclusive='both')

    press['doc_id'] = 'press_' + press.index.astype(str)
    egeret['doc_id'] = 'egeret_' + egeret.index.astype(str)
    candidates['doc_id'] = 'bypc_' + candidates.index.astype(str)

    press['source'] = 'press'
    egeret['source'] = 'egeret'
    candidates['source'] = 'polemic_candidates'

    # Standardize columns
    columns = ['doc_id', 'source', 'text', 'date', 'year', 'author', 'title', 'genre', 'newspaper', 'quality_score', 'in_overlap', 'אזכור מכ״ע']
    press['אזכור מכ״ע'] = None
    egeret['אזכור מכ״ע'] = None
    candidates['newspaper'] = None

    press = press.reindex(columns=columns, fill_value=None)
    egeret = egeret.reindex(columns=columns, fill_value=None)
    candidates = candidates.reindex(columns=columns, fill_value=None)

    corpus = pd.concat([press, egeret, candidates], ignore_index=True)
    corpus.to_parquet('corpus.parquet', index=False)
    print(f'Unified corpus saved: {len(corpus)} texts')

if __name__ == '__main__':
    main()
