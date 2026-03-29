"""
cluster_top_terms.py - Extract top distinctive TF-IDF terms per cluster.

For each of 409 clusters, computes mean TF-IDF vector and ranks terms by
how much they exceed the corpus-wide mean (distinctive, not just frequent).

Output: data/cluster_labels.parquet
  - cluster_id, top_terms (JSON list of 10), n_texts, mean_polemic_score
"""
import json
import numpy as np
import pandas as pd
import joblib
from scipy import sparse

# --- Load data ---
print("Loading word TF-IDF matrix...")
word_tfidf = sparse.load_npz("word_tfidf.npz")
print(f"  Shape: {word_tfidf.shape}")

with open("doc_ids.txt") as f:
    doc_ids = [line.strip() for line in f if line.strip()]
assert len(doc_ids) == word_tfidf.shape[0], "doc_ids / matrix row mismatch"

print("Loading vectorizers...")
vecs = joblib.load("vectorizers.joblib")
feature_names = vecs["word_vec"].get_feature_names_out()

print("Loading cluster assignments...")
clusters = pd.read_parquet("cluster_assignments.parquet")
clusters = clusters.set_index("doc_id")

print("Loading keyword scores...")
scores = pd.read_parquet("keyword_scores.parquet", columns=["doc_id", "polemic_score"])
scores = scores.set_index("doc_id")

# Build doc_id -> row index mapping
id_to_row = {did: i for i, did in enumerate(doc_ids)}

# --- Compute corpus-wide mean TF-IDF ---
print("Computing corpus-wide mean TF-IDF...")
corpus_mean = np.asarray(word_tfidf.mean(axis=0)).ravel()

# --- Compute per-cluster top terms ---
cluster_ids_all = clusters["cluster_id"]
unique_clusters = sorted(cluster_ids_all[cluster_ids_all != -1].unique())
print(f"Processing {len(unique_clusters)} clusters...")

results = []
for cid in unique_clusters:
    # Get doc_ids in this cluster
    c_docs = cluster_ids_all[cluster_ids_all == cid].index.tolist()
    row_indices = [id_to_row[d] for d in c_docs if d in id_to_row]
    if not row_indices:
        continue

    # Cluster mean TF-IDF
    cluster_matrix = word_tfidf[row_indices]
    cluster_mean = np.asarray(cluster_matrix.mean(axis=0)).ravel()

    # Distinctive terms: cluster_mean - corpus_mean
    diff = cluster_mean - corpus_mean
    top_indices = diff.argsort()[-10:][::-1]
    top_terms = [feature_names[i] for i in top_indices]

    # Stats
    n_texts = len(row_indices)
    c_scores = scores.reindex(c_docs)["polemic_score"].dropna()
    mean_score = float(c_scores.mean()) if len(c_scores) > 0 else 0.0

    results.append({
        "cluster_id": int(cid),
        "top_terms": json.dumps(top_terms, ensure_ascii=False),
        "n_texts": n_texts,
        "mean_polemic_score": round(mean_score, 4),
    })

out = pd.DataFrame(results)
out.to_parquet("data/cluster_labels.parquet", index=False)
print(f"\nSaved data/cluster_labels.parquet ({len(out)} clusters)")

# Show a few examples
for _, row in out.head(5).iterrows():
    terms = json.loads(row["top_terms"])
    print(f"  Cluster {row['cluster_id']}: n={row['n_texts']}, "
          f"score={row['mean_polemic_score']:.3f}, terms={terms[:5]}")
