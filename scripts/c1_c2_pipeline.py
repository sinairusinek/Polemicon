"""C.1 re-clustering + C.2 threading (press, newspaper-pair model) + verification + summary.

Press has no author metadata, so the plan's author-pair edges are recast as
newspaper-pair edges. Same-newspaper edges are NOT filtered out — they
become "internal" topic threads. Threads and clusters are classified by
how much cross-newspaper engagement they contain.

Outputs:
  data/press_polemic_clusters.parquet
  data/press_polemic_cluster_labels.parquet
  data/c2_edges_explicit.parquet
  data/c2_edges_interleave.parquet
  data/c2_edges_semantic.parquet
  data/threads.parquet
  logs/c2_verification.md
  logs/c2_summary.md
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LOGS = ROOT / "logs"

# ── Config ────────────────────────────────────────────────────────────────────
MIN_CLUSTER_SIZE   = 10
UMAP_N_COMPONENTS  = 50
COSINE_THRESHOLD   = 0.85
TIME_WINDOW_DAYS   = 90            # for interleave + semantic edges
EXPLICIT_REF_WINDOW_D = 180        # wider — explicit references often point to older articles
MAX_THREAD_SPAN_D  = 730           # split threads spanning > 2 years
RANDOM_SEED        = 42

# Newspaper name → code mapping (for explicit-reference resolution)
NEWSPAPER_ALIASES = {
    "hzt":      "HZT",
    "hzf":      "HZF",
    "hamagid":  "MGD", "mgd": "MGD", "המגיד": "MGD",
    "hamelitz": "HMZ", "hmz": "HMZ", "המליץ": "HMZ",
    "halevanon":"HLB", "hlb": "HLB", "הלבנון": "HLB",
    "hatzefira":"HZF", "hatzefirah":"HZF", "הצפירה": "HZF",
    "hatzofeh": "HZT", "הצופה": "HZT",
    "havatselet":"HVT", "hvt":"HVT", "חבצלת":"HVT",
}


def banner(s: str) -> None:
    print(f"\n{'=' * 60}\n{s}\n{'=' * 60}")


def resolve_newspaper_code(ref_string: str, valid_codes: set) -> str:
    """Map a reference string like 'reference to HZT + MGD' to one newspaper code."""
    if not isinstance(ref_string, str):
        return ""
    low = ref_string.lower()
    # Pull all alias hits, ordered as they appear
    hits = []
    for alias, code in NEWSPAPER_ALIASES.items():
        if alias in low and code in valid_codes:
            hits.append((low.index(alias), code))
    hits.sort()
    return hits[0][1] if hits else ""


def main():
    banner("Loading data")
    preds = pd.read_parquet(DATA / "full_corpus_predictions.parquet")
    preds["is_polemic"] = preds["prob_non_polemic"] < 0.5
    corpus = pd.read_parquet(
        ROOT / "corpus.parquet",
        columns=["doc_id", "source", "year", "date", "newspaper",
                 "headline", "title", "intertextual_reference"],
    )
    ca = pd.read_parquet(DATA / "cluster_assignments.parquet")[["doc_id"]]
    ca["vec_row"] = np.arange(len(ca))

    df = preds.merge(corpus, on="doc_id").merge(ca, on="doc_id")
    press_pol = df[(df["source"] == "press") & (df["is_polemic"])].copy()
    print(f"Press binary-polemic texts: {len(press_pol):,}")

    svd = np.load(ROOT / "tfidf_svd_300.npy")
    X = svd[press_pol["vec_row"].values]
    press_pol = press_pol.reset_index(drop=True)
    press_pol["sub_row"] = np.arange(len(press_pol))
    press_pol["pub_date"] = pd.to_datetime(press_pol["date"], errors="coerce")
    has_date = press_pol["pub_date"].notna()
    print(f"Press polemics with parseable date: {has_date.sum():,}/{len(press_pol):,}")

    valid_codes = set(press_pol["newspaper"].dropna().unique())
    print(f"Newspaper codes in pool: {sorted(valid_codes)}")

    # ── C.1 re-cluster ────────────────────────────────────────────────────────
    banner("C.1: UMAP-50 + HDBSCAN on press polemics")
    import umap, hdbscan

    reducer = umap.UMAP(
        n_components=UMAP_N_COMPONENTS, metric="cosine",
        n_neighbors=15, min_dist=0.0, random_state=RANDOM_SEED,
    )
    X50 = reducer.fit_transform(X)
    print(f"UMAP done: {X50.shape}")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        metric="euclidean",
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(X50)
    press_pol["cluster_id"] = labels
    n_clusters = int(labels.max() + 1) if (labels >= 0).any() else 0
    n_noise = int((labels == -1).sum())
    print(f"Clusters: {n_clusters}  Noise: {n_noise} ({n_noise/len(labels):.1%})")

    press_pol[["doc_id", "cluster_id", "newspaper", "year", "pub_date",
               "predicted_label", "confidence", "headline", "title"]].to_parquet(
        DATA / "press_polemic_clusters.parquet", index=False)

    # ── Cluster characterization ──────────────────────────────────────────────
    banner("Cluster characterization (distinctive TF-IDF terms)")
    word_tfidf = sp.load_npz(ROOT / "word_tfidf.npz")
    vocab_path = ROOT / "tfidf_vocab.json"
    if vocab_path.exists():
        import json
        vocab = json.loads(vocab_path.read_text())
        word_terms = vocab.get("word", [f"w{i}" for i in range(word_tfidf.shape[1])])
    else:
        word_terms = [f"w{i}" for i in range(word_tfidf.shape[1])]

    word_mat = word_tfidf[press_pol["vec_row"].values]
    corpus_mean = np.asarray(word_mat.mean(axis=0)).ravel()
    label_rows = []
    for cid in sorted(set(labels) - {-1}):
        sub = press_pol[press_pol["cluster_id"] == cid]
        cluster_mean = np.asarray(word_mat[sub["sub_row"].values].mean(axis=0)).ravel()
        diff = cluster_mean - corpus_mean
        top_idx = np.argsort(-diff)[:10]
        top_terms = [word_terms[i] for i in top_idx]
        np_breakdown = sub["newspaper"].value_counts(normalize=True).to_dict()
        top_np = max(np_breakdown.items(), key=lambda kv: kv[1])
        label_rows.append({
            "cluster_id":         int(cid),
            "n_docs":             int(len(sub)),
            "n_newspapers":       int(sub["newspaper"].nunique()),
            "top_newspaper":      top_np[0],
            "top_newspaper_share":float(top_np[1]),
            "year_min":           float(sub["year"].min()) if sub["year"].notna().any() else None,
            "year_max":           float(sub["year"].max()) if sub["year"].notna().any() else None,
            "top_terms":          top_terms,
        })
    cluster_labels = pd.DataFrame(label_rows)
    cluster_labels.to_parquet(DATA / "press_polemic_cluster_labels.parquet", index=False)
    print(f"Characterized {len(cluster_labels)} clusters")

    # ── Edges ─────────────────────────────────────────────────────────────────
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    # 1. Explicit references: HLB→HZT etc.
    banner("C.2 edge type 1: explicit references")
    explicit_edges, unresolved = [], 0
    refs = press_pol[press_pol["intertextual_reference"].notna() &
                     (press_pol["cluster_id"] != -1) &
                     press_pol["pub_date"].notna()].copy()
    print(f"  Source candidates (in cluster, has date, has ref): {len(refs)}")
    by_cluster_paper = {}
    for cid, sub in press_pol[press_pol["cluster_id"] != -1].groupby("cluster_id"):
        for paper, papsub in sub.dropna(subset=["pub_date"]).groupby("newspaper"):
            by_cluster_paper[(int(cid), paper)] = papsub.sort_values("pub_date")
    for _, r in refs.iterrows():
        target_code = resolve_newspaper_code(r["intertextual_reference"], valid_codes)
        if not target_code or target_code == r["newspaper"]:
            unresolved += 1
            continue
        key = (int(r["cluster_id"]), target_code)
        candidates = by_cluster_paper.get(key)
        if candidates is None or candidates.empty:
            unresolved += 1
            continue
        # Best candidate: most recent target before source date, within window
        window_start = r["pub_date"] - pd.Timedelta(days=EXPLICIT_REF_WINDOW_D)
        c = candidates[(candidates["pub_date"] < r["pub_date"]) &
                       (candidates["pub_date"] >= window_start)]
        if c.empty:
            # Fall back to nearest by cosine within window (any direction)
            c2 = candidates[(candidates["pub_date"] >= window_start) &
                            (candidates["pub_date"] <= r["pub_date"] + pd.Timedelta(days=EXPLICIT_REF_WINDOW_D))]
            if c2.empty:
                unresolved += 1
                continue
            # cosine to source
            src_vec = Xn[r["sub_row"]]
            sims = Xn[c2["sub_row"].values] @ src_vec
            best = c2.iloc[int(np.argmax(sims))]
        else:
            # Pick most cosine-similar within prior window
            src_vec = Xn[r["sub_row"]]
            sims = Xn[c["sub_row"].values] @ src_vec
            best = c.iloc[int(np.argmax(sims))]
        explicit_edges.append({
            "cluster_id": int(r["cluster_id"]),
            "doc_a":      best["doc_id"],         # earlier (target)
            "doc_b":      r["doc_id"],            # later (source/referrer)
            "newspaper_a":best["newspaper"],
            "newspaper_b":r["newspaper"],
            "date_a":     best["pub_date"],
            "date_b":     r["pub_date"],
            "gap_days":   int(abs((r["pub_date"] - best["pub_date"]).days)),
            "edge_type":  "explicit",
            "same_newspaper": False,  # always cross-paper by construction
            "ref_string": r["intertextual_reference"],
        })
    edges_exp = pd.DataFrame(explicit_edges)
    edges_exp.to_parquet(DATA / "c2_edges_explicit.parquet", index=False)
    print(f"  Resolved explicit edges: {len(edges_exp):,}; unresolved: {unresolved:,}")

    # 2. Newspaper-pair interleaving (kept generic: any pair in cluster within window)
    banner("C.2 edge type 2: newspaper-pair interleaving (same and cross)")
    interleave_edges = []
    for cid, sub in press_pol[press_pol["cluster_id"] != -1].groupby("cluster_id"):
        sub = sub.dropna(subset=["pub_date"]).sort_values("pub_date")
        if len(sub) < 2:
            continue
        recs = sub.to_dict("records")
        for i, a in enumerate(recs):
            for b in recs[i + 1:]:
                gap = (b["pub_date"] - a["pub_date"]).days
                if gap > TIME_WINDOW_DAYS:
                    break
                interleave_edges.append({
                    "cluster_id": int(cid),
                    "doc_a": a["doc_id"], "doc_b": b["doc_id"],
                    "newspaper_a": a["newspaper"], "newspaper_b": b["newspaper"],
                    "date_a": a["pub_date"], "date_b": b["pub_date"],
                    "gap_days": gap,
                    "edge_type": "interleave",
                    "same_newspaper": a["newspaper"] == b["newspaper"],
                })
    edges_int = pd.DataFrame(interleave_edges)
    edges_int.to_parquet(DATA / "c2_edges_interleave.parquet", index=False)
    n_same = int(edges_int["same_newspaper"].sum()) if len(edges_int) else 0
    n_cross = len(edges_int) - n_same
    print(f"  Interleave edges: {len(edges_int):,}  (same-paper {n_same:,} / cross-paper {n_cross:,})")

    # 3. Semantic-reply edges (cosine > 0.85, within window; same/cross paper tagged)
    banner("C.2 edge type 3: semantic-reply edges")
    semantic_edges = []
    for cid, sub in press_pol[press_pol["cluster_id"] != -1].groupby("cluster_id"):
        sub = sub.dropna(subset=["pub_date"]).copy()
        if len(sub) < 2:
            continue
        rows = sub["sub_row"].values
        sim = Xn[rows] @ Xn[rows].T
        recs = sub.reset_index(drop=True)
        for i in range(len(recs)):
            for j in range(len(recs)):
                if i == j or sim[i, j] < COSINE_THRESHOLD:
                    continue
                if recs.loc[j, "pub_date"] <= recs.loc[i, "pub_date"]:
                    continue
                gap = (recs.loc[j, "pub_date"] - recs.loc[i, "pub_date"]).days
                if gap > TIME_WINDOW_DAYS:
                    continue
                semantic_edges.append({
                    "cluster_id": int(cid),
                    "doc_a": recs.loc[i, "doc_id"], "doc_b": recs.loc[j, "doc_id"],
                    "newspaper_a": recs.loc[i, "newspaper"], "newspaper_b": recs.loc[j, "newspaper"],
                    "date_a": recs.loc[i, "pub_date"], "date_b": recs.loc[j, "pub_date"],
                    "gap_days": gap,
                    "cosine": float(sim[i, j]),
                    "edge_type": "semantic",
                    "same_newspaper": recs.loc[i, "newspaper"] == recs.loc[j, "newspaper"],
                })
    edges_sem = pd.DataFrame(semantic_edges)
    edges_sem.to_parquet(DATA / "c2_edges_semantic.parquet", index=False)
    n_same = int(edges_sem["same_newspaper"].sum()) if len(edges_sem) else 0
    n_cross = len(edges_sem) - n_same
    print(f"  Semantic edges: {len(edges_sem):,}  (same-paper {n_same:,} / cross-paper {n_cross:,})")

    # ── Thread construction ──────────────────────────────────────────────────
    banner("Thread construction")
    import networkx as nx
    all_edges = pd.concat(
        [df for df in [edges_exp, edges_int, edges_sem] if len(df)],
        ignore_index=True,
    )
    G = nx.Graph()
    for _, e in all_edges.iterrows():
        if G.has_edge(e["doc_a"], e["doc_b"]):
            G[e["doc_a"]][e["doc_b"]]["edge_types"].add(e["edge_type"])
            G[e["doc_a"]][e["doc_b"]]["same_newspaper"] = (
                G[e["doc_a"]][e["doc_b"]]["same_newspaper"] and e["same_newspaper"]
            )
        else:
            G.add_edge(e["doc_a"], e["doc_b"],
                       cluster_id=int(e["cluster_id"]),
                       edge_types={e["edge_type"]},
                       same_newspaper=bool(e["same_newspaper"]))
    raw_components = [list(c) for c in nx.connected_components(G) if len(c) >= 2]
    print(f"  Raw components (≥2 docs): {len(raw_components):,}")

    # Split components whose date span > MAX_THREAD_SPAN_D, by cutting at largest
    # temporal gap. Each cut may split into multiple subcomponents; recurse.
    date_map = press_pol.set_index("doc_id")["pub_date"].to_dict()
    def span_days(comp):
        ds = [date_map.get(d) for d in comp]
        ds = [d for d in ds if pd.notna(d)]
        return (max(ds) - min(ds)).days if len(ds) >= 2 else 0

    final_components, queue, splits = [], list(raw_components), 0
    while queue:
        comp = queue.pop()
        if span_days(comp) <= MAX_THREAD_SPAN_D or len(comp) <= 2:
            final_components.append(comp); continue
        # Find sorted dates and the largest consecutive gap inside this component
        sub = sorted([(date_map.get(d), d) for d in comp if pd.notna(date_map.get(d))])
        if len(sub) < 2:
            final_components.append(comp); continue
        gaps = [(sub[i+1][0] - sub[i][0]).days for i in range(len(sub) - 1)]
        cut_idx = int(np.argmax(gaps))
        cut_date = sub[cut_idx][0]   # last doc on the "early" side
        early_docs = {d for ts, d in sub if ts <= cut_date}
        # Remove all edges crossing this temporal cut, then re-component
        sg = G.subgraph(comp).copy()
        to_remove = [(u, v) for u, v in sg.edges()
                     if (u in early_docs) != (v in early_docs)]
        sg.remove_edges_from(to_remove)
        new_subs = [list(c) for c in nx.connected_components(sg) if len(c) >= 2]
        if not new_subs or len(new_subs) == 1:
            final_components.append(comp)   # cut didn't actually split; keep as-is
        else:
            splits += 1
            queue.extend(new_subs)
    components = final_components
    print(f"  Components after span-cap splits ({splits} splits applied): {len(components):,}")

    doc_meta = press_pol.set_index("doc_id")[["newspaper", "pub_date", "cluster_id",
                                              "headline", "title"]]
    thread_rows = []
    for tid, comp in enumerate(components):
        comp = list(comp)
        sub = doc_meta.loc[comp]
        newspapers = set(sub["newspaper"].dropna())
        dates = sub["pub_date"].dropna()
        span = int((dates.max() - dates.min()).days) if len(dates) >= 2 else 0
        sg = G.subgraph(comp)
        comp_edges = sg.number_of_edges()
        cross_paper_edges = sum(1 for _, _, d in sg.edges(data=True) if not d["same_newspaper"])
        same_paper_edges  = comp_edges - cross_paper_edges
        edge_types = set().union(*[d["edge_types"] for _, _, d in sg.edges(data=True)])
        cid_mode = sub["cluster_id"].mode().iloc[0] if not sub["cluster_id"].mode().empty else -1
        if len(newspapers) == 1:
            thread_type = "internal"
        elif cross_paper_edges > 0:
            thread_type = "engaged"
        else:
            thread_type = "co-occurrence"
        score = len(newspapers) * np.log1p(comp_edges) * np.log1p(span + 1)
        thread_rows.append({
            "thread_id":        tid,
            "cluster_id":       int(cid_mode),
            "n_docs":           len(comp),
            "n_newspapers":     len(newspapers),
            "span_days":        span,
            "n_edges":          comp_edges,
            "cross_paper_edges":cross_paper_edges,
            "same_paper_edges": same_paper_edges,
            "edge_types":       ",".join(sorted(edge_types)),
            "thread_type":      thread_type,
            "score":            float(score),
            "doc_ids":          ",".join(comp),
            "newspapers":       ",".join(sorted(n for n in newspapers if isinstance(n, str))),
        })
    threads = pd.DataFrame(thread_rows).sort_values("score", ascending=False)
    threads.to_parquet(DATA / "threads.parquet", index=False)
    print(f"  Threads: {len(threads):,}  "
          f"(internal {(threads.thread_type=='internal').sum()}, "
          f"engaged {(threads.thread_type=='engaged').sum()}, "
          f"co-occ {(threads.thread_type=='co-occurrence').sum()})")

    # Topic typing
    cluster_labels["topic_type"] = cluster_labels.apply(
        lambda r: "single-paper" if r["top_newspaper_share"] > 0.80
        else ("multi-paper-engaged"
              if (threads[(threads.cluster_id == r["cluster_id"]) &
                          (threads.thread_type == "engaged")]).shape[0] > 0
              else "multi-paper-unengaged"),
        axis=1,
    )
    cluster_labels.to_parquet(DATA / "press_polemic_cluster_labels.parquet", index=False)

    # ── Verification ──────────────────────────────────────────────────────────
    banner("C.4 verification")
    verif = ["# C.4 Verification (press-only, newspaper-pair model)", ""]
    if n_clusters >= 2 and (labels >= 0).sum() >= 100:
        idx = np.where(labels != -1)[0]
        if len(idx) > 5000:
            idx = np.random.RandomState(RANDOM_SEED).choice(idx, 5000, replace=False)
        sil = silhouette_score(X50[idx], labels[idx])
        verif.append(f"- **Silhouette score:** {sil:.3f}")
    if len(all_edges):
        verif.append(f"- **Median edge gap:** {int(all_edges['gap_days'].median())} days")
    if len(threads):
        in_range = ((threads["n_newspapers"] >= 2) & (threads["n_newspapers"] <= 5)).mean()
        verif.append(f"- **Threads with 2–5 newspapers:** {in_range:.1%}")
        verif.append(f"- **Internal / engaged / co-occurrence: {(threads.thread_type=='internal').sum()} / "
                     f"{(threads.thread_type=='engaged').sum()} / {(threads.thread_type=='co-occurrence').sum()}**")
        verif.append(f"- **Median thread size:** {int(threads['n_docs'].median())} docs, "
                     f"{int(threads['n_newspapers'].median())} newspapers, "
                     f"{int(threads['span_days'].median())} days")
    # Explicit-ref resolution rate
    n_ref_sources = len(press_pol[press_pol["intertextual_reference"].notna() &
                                   (press_pol["cluster_id"] != -1) &
                                   press_pol["pub_date"].notna()])
    if n_ref_sources:
        verif.append(f"- **Explicit-reference resolution:** {len(edges_exp)}/{n_ref_sources} "
                     f"({len(edges_exp)/n_ref_sources:.1%})")
    # Window sensitivity
    if len(all_edges):
        verif.append("")
        verif.append("### Window sensitivity (edge counts at narrower windows)")
        verif.append("")
        verif.append("| window | total | interleave | semantic | explicit |")
        verif.append("|---|---|---|---|---|")
        for w in [30, 60, 90, 180]:
            tot   = (all_edges["gap_days"] <= w).sum()
            ile   = ((all_edges["edge_type"]=="interleave") & (all_edges["gap_days"]<=w)).sum()
            sem   = ((all_edges["edge_type"]=="semantic")   & (all_edges["gap_days"]<=w)).sum()
            exp_  = ((all_edges["edge_type"]=="explicit")   & (all_edges["gap_days"]<=w)).sum()
            verif.append(f"| {w}d | {tot:,} | {ile:,} | {sem:,} | {exp_:,} |")
    (LOGS / "c2_verification.md").write_text("\n".join(verif) + "\n")
    print("\n".join(verif))

    # ── Summary ───────────────────────────────────────────────────────────────
    banner("Summary report")
    lines = [
        "# C.1 + C.2 pipeline summary (press, newspaper-pair model)", "",
        f"Pool: **{len(press_pol):,}** press binary-polemic texts",
        f"Clusters: **{n_clusters}** ({n_noise:,} noise, {n_noise/len(press_pol):.1%})",
        f"Edges: explicit **{len(edges_exp):,}**, interleave **{len(edges_int):,}**, semantic **{len(edges_sem):,}**",
        f"Threads: **{len(threads):,}** ≥2-docs  "
        f"(internal {(threads.thread_type=='internal').sum()}, "
        f"engaged {(threads.thread_type=='engaged').sum()}, "
        f"co-occurrence {(threads.thread_type=='co-occurrence').sum()})",
        "",
        "## Topic-type distribution (clusters)", "",
    ]
    if len(cluster_labels):
        for t, n in cluster_labels["topic_type"].value_counts().items():
            lines.append(f"- {t}: {n}")
        lines.append("")
    lines += ["## Top 20 threads by score", "",
              "| id | cluster | type | docs | papers | span(d) | edges | x-paper | same-paper | edge types |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for _, t in threads.head(20).iterrows():
        lines.append(
            f"| {int(t.thread_id)} | {int(t.cluster_id)} | {t.thread_type} | "
            f"{int(t.n_docs)} | {int(t.n_newspapers)} | {int(t.span_days)} | "
            f"{int(t.n_edges)} | {int(t.cross_paper_edges)} | {int(t.same_paper_edges)} | "
            f"{t.edge_types} |"
        )
    lines += ["", "## Top 10 engaged threads (cross-paper)", ""]
    eng = threads[threads.thread_type=="engaged"].head(10)
    if len(eng):
        lines.append("| id | cluster | docs | papers | span(d) | x-paper edges | newspapers |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, t in eng.iterrows():
            lines.append(
                f"| {int(t.thread_id)} | {int(t.cluster_id)} | {int(t.n_docs)} | "
                f"{int(t.n_newspapers)} | {int(t.span_days)} | {int(t.cross_paper_edges)} | "
                f"{t.newspapers} |"
            )
    lines += ["", "## Top 10 internal threads (single-paper running topic)", ""]
    intl = threads[threads.thread_type=="internal"].head(10)
    if len(intl):
        lines.append("| id | cluster | docs | span(d) | newspaper |")
        lines.append("|---|---|---|---|---|")
        for _, t in intl.iterrows():
            lines.append(
                f"| {int(t.thread_id)} | {int(t.cluster_id)} | {int(t.n_docs)} | "
                f"{int(t.span_days)} | {t.newspapers} |"
            )
    lines += ["", "## Top 10 clusters by size", "",
              "| cluster | docs | papers | top_paper | type | year_range | top terms |",
              "|---|---|---|---|---|---|---|"]
    for _, c in cluster_labels.sort_values("n_docs", ascending=False).head(10).iterrows():
        yr = f"{int(c.year_min)}–{int(c.year_max)}" if c.year_min else "?"
        terms = ", ".join(c.top_terms[:6])
        lines.append(
            f"| {int(c.cluster_id)} | {int(c.n_docs)} | {int(c.n_newspapers)} | "
            f"{c.top_newspaper} ({c.top_newspaper_share:.0%}) | {c.topic_type} | "
            f"{yr} | {terms} |"
        )
    (LOGS / "c2_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:30]))


if __name__ == "__main__":
    main()
