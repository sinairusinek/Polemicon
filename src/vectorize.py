"""
vectorize.py - TF-IDF vectorization for the Polemicon project

- Char n-gram TF-IDF (3-5 grams, 50K features) — OCR-robust
- Word n-gram TF-IDF (1-2 grams, 30K features)
- TruncatedSVD to 300 dims for FAISS nearest-neighbor search
- Saves sparse matrices, SVD-reduced dense matrix, FAISS index, and fitted vectorizers
"""
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy import sparse
import faiss
import joblib


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("Loading corpus...")
    corpus = pd.read_parquet("corpus.parquet")
    # Drop known outlier: bypc_5539 (317K-word academic book, not a polemic text)
    before = len(corpus)
    corpus = corpus[corpus["doc_id"] != "bypc_5539"].reset_index(drop=True)
    print(f"  {before} texts loaded, {before - len(corpus)} outlier(s) dropped, {len(corpus)} remaining.")
    texts = corpus["text"].fillna("").tolist()
    doc_ids = corpus["doc_id"].tolist()

    # Char n-gram TF-IDF
    print("Fitting char n-gram TF-IDF (3-5 grams, 50K features)...")
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=50000)
    char_tfidf = char_vec.fit_transform(texts)
    print(f"  char_tfidf shape: {char_tfidf.shape}")

    # Word n-gram TF-IDF
    print("Fitting word n-gram TF-IDF (1-2 grams, 30K features)...")
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=30000)
    word_tfidf = word_vec.fit_transform(texts)
    print(f"  word_tfidf shape: {word_tfidf.shape}")

    # Save sparse matrices
    print("Saving sparse matrices...")
    sparse.save_npz("char_tfidf.npz", char_tfidf)
    sparse.save_npz("word_tfidf.npz", word_tfidf)

    # Combine + SVD for FAISS
    print("Running TruncatedSVD (300 components) on combined TF-IDF...")
    combined = sparse.hstack([char_tfidf, word_tfidf])
    svd = TruncatedSVD(n_components=300, random_state=42)
    dense_reduced = svd.fit_transform(combined)
    explained = svd.explained_variance_ratio_.sum()
    print(f"  SVD explained variance: {explained:.1%}")
    np.save("tfidf_svd_300.npy", dense_reduced)

    # FAISS index on L2-normalized SVD vectors (cosine similarity via inner product)
    print("Building FAISS index...")
    dense_reduced = dense_reduced.astype("float32")
    faiss.normalize_L2(dense_reduced)
    index = faiss.IndexFlatIP(300)
    index.add(dense_reduced)
    faiss.write_index(index, "tfidf_faiss.index")
    print(f"  FAISS index: {index.ntotal} vectors")

    # Save vectorizers and doc IDs
    joblib.dump({"char_vec": char_vec, "word_vec": word_vec, "svd": svd}, "vectorizers.joblib")
    with open("doc_ids.txt", "w") as f:
        for did in doc_ids:
            f.write(f"{did}\n")

    print("Vectorization complete.")


if __name__ == "__main__":
    main()
