"""
cluster.py - UMAP + HDBSCAN clustering for the Polemicon project (Phase C.1)

- UMAP 300-dim → 50-dim for clustering
- HDBSCAN on 50-dim embedding
- UMAP 300-dim → 2-dim for visualization
- Outputs cluster_assignments.parquet and umap_2d.npy
"""
import os
import numpy as np
import pandas as pd
import umap
import hdbscan


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("Loading SVD vectors and doc IDs...")
    svd_vectors = np.load("tfidf_svd_300.npy")
    with open("doc_ids.txt") as f:
        doc_ids = [line.strip() for line in f]
    print(f"  {len(doc_ids)} documents, {svd_vectors.shape[1]} dimensions")

    # UMAP 50-dim for clustering
    print("UMAP reduction: 300 → 50 dimensions...")
    reducer_50 = umap.UMAP(n_components=50, metric="cosine", random_state=42, verbose=True)
    embedding_50 = reducer_50.fit_transform(svd_vectors)

    # HDBSCAN clustering
    print("Running HDBSCAN (min_cluster_size=10)...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, metric="euclidean", core_dist_n_jobs=-1)
    labels = clusterer.fit_predict(embedding_50)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"  {n_clusters} clusters, {n_noise} noise points ({n_noise/len(labels):.1%})")

    # UMAP 2-dim for visualization
    print("UMAP reduction: 300 → 2 dimensions...")
    reducer_2 = umap.UMAP(n_components=2, metric="cosine", random_state=42, verbose=True)
    embedding_2 = reducer_2.fit_transform(svd_vectors)

    # Save results
    result = pd.DataFrame({
        "doc_id": doc_ids,
        "cluster_id": labels,
        "umap_x": embedding_2[:, 0],
        "umap_y": embedding_2[:, 1],
    })
    result.to_parquet("cluster_assignments.parquet", index=False)
    np.save("umap_2d.npy", embedding_2)

    print(f"\nSaved cluster_assignments.parquet ({len(result)} rows)")
    print(f"Saved umap_2d.npy ({embedding_2.shape})")

    # Cluster size distribution
    print(f"\nCluster size distribution (top 20):")
    sizes = pd.Series(labels[labels >= 0]).value_counts().head(20)
    for cid, size in sizes.items():
        print(f"  cluster {cid}: {size} texts")


if __name__ == "__main__":
    main()
