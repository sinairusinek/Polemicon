"""
sample_pilot.py - Create stratified 200-text pilot sample for B.2 annotation

Stratifies by:
- Source (press ~100, polemic_candidates ~50, egeret ~50)
- Keyword score quartile (within each source)
- Cluster diversity
"""
import os
import pandas as pd
import numpy as np


# Target sample sizes per source
SOURCE_TARGETS = {
    "press": 100,
    "polemic_candidates": 50,
    "egeret": 50,
}


def stratified_sample(df, n, random_state=42):
    """Sample n texts stratified by keyword_score quartile."""
    df = df.copy()
    df["kw_quartile"] = pd.qcut(df["polemic_score"], q=4, labels=False, duplicates="drop")
    per_q = max(1, n // df["kw_quartile"].nunique())

    samples = []
    for q in df["kw_quartile"].unique():
        q_df = df[df["kw_quartile"] == q]
        take = min(per_q, len(q_df))
        # Within each quartile, prefer diverse clusters
        if "cluster_id" in q_df.columns and q_df["cluster_id"].nunique() > 1:
            sampled = q_df.groupby("cluster_id", group_keys=False).apply(
                lambda x: x.sample(min(len(x), max(1, take // q_df["cluster_id"].nunique())),
                                   random_state=random_state),
                include_groups=False,
            )
            if len(sampled) < take:
                remaining = q_df[~q_df.index.isin(sampled.index)]
                extra = remaining.sample(min(take - len(sampled), len(remaining)),
                                         random_state=random_state)
                sampled = pd.concat([sampled, extra])
        else:
            sampled = q_df.sample(take, random_state=random_state)
        samples.append(sampled)

    result = pd.concat(samples)
    # Trim or pad to exact target
    if len(result) > n:
        result = result.sample(n, random_state=random_state)
    elif len(result) < n:
        remaining = df[~df.index.isin(result.index)]
        extra = remaining.sample(min(n - len(result), len(remaining)),
                                  random_state=random_state)
        result = pd.concat([result, extra])
    return result


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("Loading data...")
    corpus = pd.read_parquet("corpus.parquet")
    corpus = corpus[corpus["doc_id"] != "bypc_5539"].reset_index(drop=True)

    kw_scores = pd.read_parquet("keyword_scores.parquet")[["doc_id", "polemic_score"]]
    clusters = pd.read_parquet("cluster_assignments.parquet")

    df = corpus.merge(kw_scores, on="doc_id", how="left")
    df = df.merge(clusters, on="doc_id", how="left")
    print(f"  {len(df)} texts with scores and clusters")

    # Sample per source
    all_samples = []
    for source, target_n in SOURCE_TARGETS.items():
        source_df = df[df["source"] == source]
        if len(source_df) == 0:
            print(f"  WARNING: no texts for source {source}")
            continue
        actual_n = min(target_n, len(source_df))
        sampled = stratified_sample(source_df, actual_n)
        all_samples.append(sampled)
        print(f"  {source}: sampled {len(sampled)} texts")

    pilot = pd.concat(all_samples, ignore_index=True)

    # Select columns for the pilot sample
    cols = ["doc_id", "source", "text", "date", "year", "author", "title",
            "polemic_score", "cluster_id", "umap_x", "umap_y"]
    pilot = pilot[[c for c in cols if c in pilot.columns]]

    # Save
    os.makedirs("data", exist_ok=True)
    pilot.to_parquet("data/pilot_sample.parquet", index=False)
    print(f"\nSaved data/pilot_sample.parquet ({len(pilot)} rows)")
    print(f"Sources: {pilot['source'].value_counts().to_dict()}")
    print(f"Polemic score range: {pilot['polemic_score'].min():.4f} - {pilot['polemic_score'].max():.4f}")
    print(f"Clusters represented: {pilot['cluster_id'].nunique()}")


if __name__ == "__main__":
    main()
