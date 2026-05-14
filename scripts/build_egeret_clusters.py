"""Slice the joint cluster space into Egeret-bearing clusters.

Outputs:
  data/egeret_polemic_clusters.parquet        — per-doc rows for clusters with ≥1 Egeret doc
  data/egeret_polemic_cluster_labels.parquet  — per-cluster labels with source mix
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

POLEMIC_LABELS = {"implicit polemic", "explicit polemic", "meta-polemic (descriptive)"}


def main():
    ca = pd.read_parquet(DATA / "cluster_assignments.parquet")
    ca = ca[ca["cluster_id"] != -1]

    corpus = pd.read_parquet(
        ROOT / "corpus.parquet",
        columns=["doc_id", "source", "author", "year", "title", "newspaper"],
    )
    preds = pd.read_parquet(
        DATA / "full_corpus_predictions.parquet",
        columns=["doc_id", "predicted_label", "confidence"],
    )
    eg_dates = pd.read_parquet(
        DATA / "egeret_dates.parquet",
        columns=["doc_id", "year", "author"],
    ).rename(columns={"year": "eg_year", "author": "eg_author"})

    df = ca.merge(corpus, on="doc_id", how="left").merge(preds, on="doc_id", how="left")
    df = df.merge(eg_dates, on="doc_id", how="left")

    # Use enriched Egeret dates/authors where available
    df["year"] = df["eg_year"].combine_first(df["year"])
    df["author"] = df["eg_author"].combine_first(df["author"])
    df = df.drop(columns=["eg_year", "eg_author"])

    df["is_polemic"] = df["predicted_label"].isin(POLEMIC_LABELS)

    egeret_clusters = df.loc[df["source"] == "egeret", "cluster_id"].unique()
    sliced = df[df["cluster_id"].isin(egeret_clusters)].copy()

    sliced.to_parquet(DATA / "egeret_polemic_clusters.parquet", index=False)
    print(f"egeret_polemic_clusters: {sliced.shape} ({sliced['cluster_id'].nunique()} clusters)")

    # Labels
    cluster_labels_path = DATA / "cluster_labels.parquet"
    base_labels = (
        pd.read_parquet(cluster_labels_path)
        if cluster_labels_path.exists()
        else pd.DataFrame(columns=["cluster_id", "top_terms"])
    )

    rows = []
    for cid, grp in sliced.groupby("cluster_id"):
        src_counts = grp["source"].value_counts().to_dict()
        n_egeret = int(src_counts.get("egeret", 0))
        sources_present = sorted(src_counts.keys())
        n_sources = len(sources_present)
        if n_sources == 1:
            topic_type = "egeret-only" if "egeret" in sources_present else "no-egeret"
        else:
            topic_type = "cross-source"

        eg_authors = grp.loc[grp["source"] == "egeret", "author"].dropna().unique()
        years = grp["year"].dropna()

        rows.append({
            "cluster_id": int(cid),
            "n_docs": len(grp),
            "n_egeret": n_egeret,
            "n_authors_egeret": len(eg_authors),
            "n_sources": n_sources,
            "sources": ",".join(sources_present),
            "source_mix": json.dumps(src_counts, ensure_ascii=False),
            "n_polemic": int(grp["is_polemic"].sum()),
            "n_polemic_egeret": int(((grp["source"] == "egeret") & grp["is_polemic"]).sum()),
            "year_min": float(years.min()) if len(years) else None,
            "year_max": float(years.max()) if len(years) else None,
            "topic_type": topic_type,
        })

    labels = pd.DataFrame(rows)
    if "top_terms" in base_labels.columns:
        labels = labels.merge(base_labels[["cluster_id", "top_terms"]], on="cluster_id", how="left")

    labels = labels.sort_values(["topic_type", "n_egeret"], ascending=[True, False]).reset_index(drop=True)
    labels.to_parquet(DATA / "egeret_polemic_cluster_labels.parquet", index=False)

    print(f"egeret_polemic_cluster_labels: {labels.shape}")
    print(labels["topic_type"].value_counts())


if __name__ == "__main__":
    main()
