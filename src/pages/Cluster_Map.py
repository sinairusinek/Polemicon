"""
Cluster Map — Interactive UMAP visualization of 33,513 corpus texts.

Points colored by cluster membership. Select a cluster to see its
top distinctive terms and metadata. Noise points shown in light grey.
"""
import streamlit as st
import pandas as pd
import json
import os
import sys
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cleaning import restore_final_forms

st.set_page_config(page_title="Cluster Map", layout="wide")
st.title("Cluster Map")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


@st.cache_data
def load_cluster_data():
    ca = pd.read_parquet(os.path.join(DATA_DIR, "cluster_assignments.parquet"))
    ks = pd.read_parquet(
        os.path.join(DATA_DIR, "keyword_scores.parquet"),
        columns=["doc_id", "polemic_score", "source"],
    )
    ca = ca.merge(ks, on="doc_id", how="left")
    return ca


@st.cache_data
def load_cluster_labels():
    path = os.path.join(DATA_DIR, "cluster_labels.parquet")
    if os.path.exists(path):
        cl = pd.read_parquet(path)
        cl["top_terms_list"] = cl["top_terms"].apply(
            lambda x: json.loads(x) if pd.notna(x) else []
        )
        return cl
    return None


@st.cache_data
def load_calibration():
    path = os.path.join(DATA_DIR, "calibration_v2.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path, columns=["doc_id", "polemic_label", "confidence"])
    return None


ca = load_cluster_data()
cl = load_cluster_labels()
calib = load_calibration()

noise = ca[ca["cluster_id"] == -1]
clustered = ca[ca["cluster_id"] != -1]

# --- Sidebar controls ---
st.sidebar.header("Controls")

# Source filter
source_options = ["all"] + sorted(ca["source"].dropna().unique().tolist())
selected_source = st.sidebar.selectbox("Filter by source", source_options)

# Cluster selector
cluster_sizes = clustered.groupby("cluster_id").size().reset_index(name="n")
if cl is not None:
    cluster_sizes = cluster_sizes.merge(cl[["cluster_id", "mean_polemic_score"]], on="cluster_id", how="left")
cluster_sizes = cluster_sizes.sort_values("n", ascending=False)

cluster_options = ["None"] + [
    f"{int(r['cluster_id'])} (n={r['n']})" for _, r in cluster_sizes.iterrows()
]
selected_cluster_str = st.sidebar.selectbox("Highlight cluster", cluster_options)

if selected_cluster_str != "None":
    selected_cid = int(selected_cluster_str.split(" ")[0])
else:
    selected_cid = None

# Point size
point_size = st.sidebar.slider("Point size", 1, 8, 2)

# Color by
color_by_options = ["Cluster", "Source", "Polemic score"]
if calib is not None:
    color_by_options.append("Calibration labels")
color_by = st.sidebar.radio("Color by", color_by_options)

# --- Calibration sidebar summary ---
if calib is not None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Calibration (2K sample)**")
    label_counts = calib["polemic_label"].value_counts()
    total = len(calib)
    for label, count in label_counts.items():
        st.sidebar.markdown(f"- {label}: **{count}** ({count/total:.0%})")

# --- Apply source filter ---
if selected_source != "all":
    plot_noise = noise[noise["source"] == selected_source]
    plot_clustered = clustered[clustered["source"] == selected_source]
else:
    plot_noise = noise
    plot_clustered = clustered

# --- Build figure ---
fig = go.Figure()

# Noise layer
fig.add_trace(go.Scatter(
    x=plot_noise["umap_x"],
    y=plot_noise["umap_y"],
    mode="markers",
    marker=dict(size=point_size, color="lightgrey", opacity=0.3),
    text=plot_noise["doc_id"],
    hoverinfo="text",
    name=f"Noise ({len(plot_noise):,})",
))

if color_by == "Cluster":
    fig.add_trace(go.Scatter(
        x=plot_clustered["umap_x"],
        y=plot_clustered["umap_y"],
        mode="markers",
        marker=dict(size=point_size, color="steelblue", opacity=0.4),
        text=plot_clustered.apply(
            lambda r: f"{r['doc_id']}<br>cluster {int(r['cluster_id'])}", axis=1
        ),
        hoverinfo="text",
        name=f"Clustered ({len(plot_clustered):,})",
    ))

elif color_by == "Source":
    source_colors = {"press": "#1f77b4", "egeret": "#ff7f0e", "polemic_candidates": "#2ca02c", "compact_memory": "#9467bd"}
    for src in sorted(plot_clustered["source"].dropna().unique()):
        sub = plot_clustered[plot_clustered["source"] == src]
        fig.add_trace(go.Scatter(
            x=sub["umap_x"],
            y=sub["umap_y"],
            mode="markers",
            marker=dict(size=point_size, color=source_colors.get(src, "grey"), opacity=0.5),
            text=sub.apply(
                lambda r: f"{r['doc_id']}<br>cluster {int(r['cluster_id'])}<br>{r['source']}", axis=1
            ),
            hoverinfo="text",
            name=f"{src} ({len(sub):,})",
        ))

elif color_by == "Polemic score":
    fig.add_trace(go.Scatter(
        x=plot_clustered["umap_x"],
        y=plot_clustered["umap_y"],
        mode="markers",
        marker=dict(
            size=point_size,
            color=plot_clustered["polemic_score"],
            colorscale="YlOrRd",
            opacity=0.5,
            colorbar=dict(title="Score"),
        ),
        text=plot_clustered.apply(
            lambda r: f"{r['doc_id']}<br>cluster {int(r['cluster_id'])}<br>score {r['polemic_score']:.3f}",
            axis=1,
        ),
        hoverinfo="text",
        name=f"Clustered ({len(plot_clustered):,})",
    ))

else:  # Calibration labels
    # Unlabeled points in muted grey first
    labeled_ids = set(calib["doc_id"])
    unlabeled = plot_clustered[~plot_clustered["doc_id"].isin(labeled_ids)]
    fig.add_trace(go.Scatter(
        x=unlabeled["umap_x"],
        y=unlabeled["umap_y"],
        mode="markers",
        marker=dict(size=point_size, color="#cccccc", opacity=0.25),
        text=unlabeled["doc_id"],
        hoverinfo="text",
        name=f"Unlabeled ({len(unlabeled):,})",
    ))

    label_colors = {
        "non-polemic": "#4575b4",
        "implicit polemic": "#fee090",
        "explicit polemic": "#d73027",
        "meta-polemic (descriptive)": "#74add1",
    }
    merged = plot_clustered.merge(calib, on="doc_id", how="inner")
    for label in ["non-polemic", "implicit polemic", "explicit polemic", "meta-polemic (descriptive)"]:
        sub = merged[merged["polemic_label"] == label]
        if len(sub) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=sub["umap_x"],
            y=sub["umap_y"],
            mode="markers",
            marker=dict(
                size=point_size + 1,
                color=label_colors.get(label, "grey"),
                opacity=0.8,
                line=dict(width=0.5, color="white"),
            ),
            text=sub.apply(
                lambda r: f"{r['doc_id']}<br>{r['polemic_label']}<br>cluster {int(r['cluster_id'])}",
                axis=1,
            ),
            hoverinfo="text",
            name=f"{label} ({len(sub):,})",
        ))

# Highlighted cluster on top
if selected_cid is not None:
    highlight = ca[ca["cluster_id"] == selected_cid]
    if selected_source != "all":
        highlight = highlight[highlight["source"] == selected_source]
    fig.add_trace(go.Scatter(
        x=highlight["umap_x"],
        y=highlight["umap_y"],
        mode="markers",
        marker=dict(size=point_size + 4, color="red", opacity=0.9, line=dict(width=1, color="darkred")),
        text=highlight.apply(
            lambda r: f"{r['doc_id']}<br>cluster {int(r['cluster_id'])}", axis=1
        ),
        hoverinfo="text",
        name=f"Cluster {selected_cid} ({len(highlight)})",
    ))

fig.update_layout(
    height=700,
    margin=dict(l=0, r=0, t=30, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
)

st.plotly_chart(fig, use_container_width=True)

# --- Calibration label breakdown by cluster ---
if color_by == "Calibration labels" and calib is not None and selected_cid is not None:
    cluster_calib = ca[ca["cluster_id"] == selected_cid].merge(calib, on="doc_id", how="inner")
    if len(cluster_calib) > 0:
        st.subheader(f"Calibration labels in cluster {selected_cid}")
        lc = cluster_calib["polemic_label"].value_counts().reset_index()
        lc.columns = ["label", "count"]
        lc["pct"] = (lc["count"] / lc["count"].sum() * 100).round(1)
        st.dataframe(lc, use_container_width=False, hide_index=True)
    else:
        st.info(f"No calibration-labeled docs in cluster {selected_cid}.")

# --- Cluster detail panel ---
if selected_cid is not None and cl is not None:
    cl_row = cl[cl["cluster_id"] == selected_cid]
    if len(cl_row) > 0:
        info = cl_row.iloc[0]
        st.subheader(f"Cluster {selected_cid}")

        cols = st.columns(3)
        cols[0].metric("Texts", info["n_texts"])
        cols[1].metric("Mean polemic score", f"{info['mean_polemic_score']:.3f}")

        c_sources = ca[ca["cluster_id"] == selected_cid]["source"].value_counts()
        cols[2].metric("Sources", ", ".join(f"{s}: {n}" for s, n in c_sources.items()))

        terms = info["top_terms_list"]
        st.markdown("**Top distinctive terms:**")
        term_display = " &nbsp;|&nbsp; ".join(
            f'<span dir="rtl" style="font-size:18px;">{restore_final_forms(t)}</span>'
            for t in terms
        )
        st.markdown(term_display, unsafe_allow_html=True)
elif selected_cid is not None:
    st.info(f"Cluster {selected_cid} selected but no label data available.")

# --- Cluster table ---
if cl is not None:
    with st.expander("All clusters (sortable table)"):
        table = cl[["cluster_id", "n_texts", "mean_polemic_score"]].copy()
        table["top_terms"] = cl["top_terms_list"].apply(
            lambda ts: ", ".join(restore_final_forms(t) for t in ts[:5])
        )
        table = table.sort_values("n_texts", ascending=False).reset_index(drop=True)
        st.dataframe(table, use_container_width=True, height=400)

# --- Stats ---
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Total:** {len(ca):,} texts  \n"
    f"**Clustered:** {len(clustered):,} ({len(clustered)/len(ca):.0%})  \n"
    f"**Noise:** {len(noise):,} ({len(noise)/len(ca):.0%})  \n"
    f"**Clusters:** {clustered['cluster_id'].nunique()}"
)
