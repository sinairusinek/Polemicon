"""Egeret-internal candidate threads via author-pair edges within clusters.

Pilot-scope: no recipient resolution, no cross-corpus mention edges.

Edges  : same cluster, both Egeret, different authors, both dates known
         and confidence ∈ {high, medium}, within TIME_WINDOW_YEARS.
Guard  : skip clusters with > MAX_AUTHORS_PER_CLUSTER distinct Egeret authors
         (topic-heterogeneous; flag for manual review).
Threads: connected components of the edge graph.
Output : data/egeret_threads.parquet (schema mirrors data/threads.parquet so
         thread_summaries.py can consume it with --threads-path).
         data/egeret_edges_authorpair.parquet (raw edges, for inspection).
"""
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

TIME_WINDOW_YEARS = 2
MAX_AUTHORS_PER_CLUSTER = 50
ACCEPTED_DATE_CONFIDENCE = {"high", "medium"}

# Post-CC filters
MAX_THREAD_SPAN_YEARS = 15        # drop threads whose docs span > this
MIN_EDGES_PER_DOC = 0.5           # drop low-density components
MAX_THREAD_DOCS = 30              # anything bigger is a cluster, not a thread
MIN_THREAD_DOCS = 3


def main():
    sliced = pd.read_parquet(DATA / "egeret_polemic_clusters.parquet")
    eg_dates = pd.read_parquet(
        DATA / "egeret_dates.parquet",
        columns=["doc_id", "date", "year", "confidence"],
    )

    # Restrict to Egeret docs with usable dates
    eg = sliced[sliced["source"] == "egeret"].merge(
        eg_dates.rename(columns={"date": "eg_date", "year": "eg_year_d", "confidence": "date_confidence"}),
        on="doc_id",
        how="left",
    )
    eg["year_for_edge"] = pd.to_numeric(eg["eg_year_d"].combine_first(eg["year"]), errors="coerce")

    usable = eg[
        eg["date_confidence"].isin(ACCEPTED_DATE_CONFIDENCE)
        & eg["year_for_edge"].notna()
        & eg["author"].notna()
    ].copy()

    print(f"Egeret docs in slice: {len(eg)}")
    print(f"Usable for edges (date conf ∈ {ACCEPTED_DATE_CONFIDENCE}, author known): {len(usable)}")

    # Filter mega-clusters
    cluster_author_counts = usable.groupby("cluster_id")["author"].nunique()
    keep_clusters = cluster_author_counts[
        (cluster_author_counts >= 2)
        & (cluster_author_counts <= MAX_AUTHORS_PER_CLUSTER)
    ].index
    skipped_mega = cluster_author_counts[cluster_author_counts > MAX_AUTHORS_PER_CLUSTER]
    if len(skipped_mega):
        print(f"Skipping {len(skipped_mega)} mega-clusters (> {MAX_AUTHORS_PER_CLUSTER} authors): "
              f"{skipped_mega.index.tolist()}")
    print(f"Clusters with ≥2 authors and ≤{MAX_AUTHORS_PER_CLUSTER}: {len(keep_clusters)}")

    usable = usable[usable["cluster_id"].isin(keep_clusters)].reset_index(drop=True)

    # Build edges per cluster
    edges = []
    for cid, grp in usable.groupby("cluster_id"):
        rows = grp[["doc_id", "author", "year_for_edge"]].to_records(index=False)
        n = len(rows)
        for i in range(n):
            d_i, a_i, y_i = rows[i]
            for j in range(i + 1, n):
                d_j, a_j, y_j = rows[j]
                if a_i == a_j:
                    continue
                if abs(y_i - y_j) > TIME_WINDOW_YEARS:
                    continue
                edges.append({
                    "cluster_id": int(cid),
                    "doc_a": d_i,
                    "doc_b": d_j,
                    "author_a": a_i,
                    "author_b": a_j,
                    "year_a": float(y_i),
                    "year_b": float(y_j),
                    "delta_years": float(abs(y_i - y_j)),
                })
    edges_df = pd.DataFrame(edges)
    edges_df.to_parquet(DATA / "egeret_edges_authorpair.parquet", index=False)
    print(f"Author-pair edges: {len(edges_df)}")

    # Connected components → threads (within each cluster)
    threads = []
    next_thread_id = 100_000  # avoid collision with data/threads.parquet ids
    for cid, grp_edges in edges_df.groupby("cluster_id"):
        G = nx.Graph()
        for _, e in grp_edges.iterrows():
            G.add_edge(e["doc_a"], e["doc_b"])
        for component in nx.connected_components(G):
            doc_ids = sorted(component)
            sub = usable[usable["doc_id"].isin(doc_ids)]
            authors = sorted(sub["author"].dropna().unique())
            years = sub["year_for_edge"].dropna()
            n_edges = sum(
                1 for _, e in grp_edges.iterrows()
                if e["doc_a"] in component and e["doc_b"] in component
            )
            span_years = float(years.max() - years.min()) if len(years) else 0.0
            threads.append({
                "thread_id": next_thread_id,
                "cluster_id": int(cid),
                "n_docs": len(doc_ids),
                "n_newspapers": len(authors),      # reused field — authors here
                "span_days": int(span_years * 365),
                "n_edges": int(n_edges),
                "cross_paper_edges": int(n_edges),  # all edges are cross-author by construction
                "same_paper_edges": 0,
                "edge_types": "author_pair",
                "thread_type": "engaged",          # all threads have ≥1 cross-author edge
                "score": float(n_edges) / max(len(doc_ids), 1),
                "doc_ids": ",".join(doc_ids),
                "newspapers": ",".join(authors),   # reused field — authors here
                "authors": ",".join(authors),
            })
            next_thread_id += 1

    threads_df = pd.DataFrame(threads)
    n_raw = len(threads_df)
    threads_df["span_years"] = threads_df["span_days"] / 365.0
    threads_df["edges_per_doc"] = threads_df["n_edges"] / threads_df["n_docs"].clip(lower=1)

    keep = (
        (threads_df["span_years"] <= MAX_THREAD_SPAN_YEARS)
        & (threads_df["edges_per_doc"] >= MIN_EDGES_PER_DOC)
        & (threads_df["n_docs"] >= MIN_THREAD_DOCS)
        & (threads_df["n_docs"] <= MAX_THREAD_DOCS)
    )
    dropped = threads_df[~keep]
    threads_df = threads_df[keep].drop(columns=["span_years", "edges_per_doc"])

    threads_df = threads_df.sort_values(["n_docs", "score"], ascending=[False, False])
    threads_df.to_parquet(DATA / "egeret_threads.parquet", index=False)
    print(f"Threads raw: {n_raw} → kept: {len(threads_df)} (dropped {len(dropped)} by filters)")
    if len(threads_df):
        print(threads_df[["thread_id", "cluster_id", "n_docs", "n_newspapers", "span_days", "n_edges", "score"]].head(15).to_string())


if __name__ == "__main__":
    main()
