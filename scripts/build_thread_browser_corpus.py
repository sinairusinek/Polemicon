"""Bake a slim doc-metadata parquet for the Streamlit Thread Browser.

corpus.parquet (327MB) is gitignored and not available on Streamlit Cloud.
This script extracts only the doc_ids that appear in any thread parquet,
keeps the columns the Thread Browser actually displays, and writes a
~20MB file that can ride along in the data/ dir.

Output: data/thread_browser_corpus.parquet
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

THREAD_FILES = [
    DATA / "threads.parquet",
    DATA / "egeret_threads.parquet",
]
COLS = ["doc_id", "date", "year", "newspaper", "headline", "title", "author", "source", "text"]


def main():
    ids: set[str] = set()
    for p in THREAD_FILES:
        if not p.exists():
            print(f"skip (missing): {p.name}")
            continue
        df = pd.read_parquet(p, columns=["doc_ids"])
        for d in df["doc_ids"]:
            ids.update(x.strip() for x in str(d).split(",") if x.strip())
        print(f"{p.name}: cumulative {len(ids)} unique doc_ids")

    corpus = pd.read_parquet(ROOT / "corpus.parquet", columns=COLS)
    sub = corpus[corpus["doc_id"].isin(ids)].reset_index(drop=True)
    out = DATA / "thread_browser_corpus.parquet"
    sub.to_parquet(out, index=False)
    print(f"wrote {out} — {len(sub)} docs, {out.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
