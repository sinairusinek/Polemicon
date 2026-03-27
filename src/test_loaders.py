"""
test_loaders.py - Test script for data loaders in the Polemicon project

Loads each dataset and prints basic info to verify loader functionality.
"""
from loaders import load_press_articles, load_egeret_letters, load_polemic_candidates

PRESS_PATH = '../MGD-LBN-MLZ-HZF-HZTfull2021-08-14-(1)-tsv.csv'
EGERET_PATH = '../e-geret-batch-export.tsv'
POLEMIC_PATH = '../Ben-Yehuda-Project-polemic-candidates.csv'

def main():
    print('Testing Press articles loader...')
    press_df = load_press_articles(PRESS_PATH)
    print('Shape:', press_df.shape)
    print('Columns:', press_df.columns.tolist())
    print(press_df.head(2))
    print('\n---\n')

    print('Testing E-Geret letters loader...')
    egeret_df = load_egeret_letters(EGERET_PATH)
    print('Shape:', egeret_df.shape)
    print('Columns:', egeret_df.columns.tolist())
    print(egeret_df.head(2))
    print('\n---\n')

    print('Testing Polemic candidates loader...')
    polemic_df = load_polemic_candidates(POLEMIC_PATH)
    print('Shape:', polemic_df.shape)
    print('Columns:', polemic_df.columns.tolist())
    print(polemic_df.head(2))

if __name__ == '__main__':
    main()
