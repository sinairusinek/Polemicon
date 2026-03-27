"""
loaders.py - Data loading utilities for the Polemicon project

Handles loading of:
- Press articles (CSV, large text fields)
- E-Geret letters (TSV, BOM-encoded)
- Polemic candidates (CSV, sparse dates)
"""
import pandas as pd
import csv

# Loader for Press articles
def load_press_articles(path):
    # Increase CSV field size limit for large text fields
    csv.field_size_limit(2**30)
    return pd.read_csv(path)

# Loader for E-Geret letters
def load_egeret_letters(path):
    # Handles BOM-encoded TSV
    return pd.read_csv(path, sep='\t', encoding='utf-8-sig')

# Loader for Polemic candidates
def load_polemic_candidates(path):
    return pd.read_csv(path)

# Example usage (uncomment to test)
# press_df = load_press_articles('MGD-LBN-MLZ-HZF-HZTfull2021-08-14-(1)-tsv.csv')
# egeret_df = load_egeret_letters('e-geret-batch-export.tsv')
# polemic_df = load_polemic_candidates('Ben-Yehuda-Project-polemic-candidates.csv')
