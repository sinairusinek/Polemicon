"""Egeret clusters — slices of the joint cluster space that contain Egeret docs.

Three tabs:
  • Cross-source  — clusters mixing Egeret with press / bypc / cm
  • Egeret-only   — clusters with only Egeret docs (within-corpus cross-author)
  • All           — every cluster containing ≥1 Egeret doc
"""
import json
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cleaning import restore_final_forms

st.set_page_config(page_title="Egeret Clusters", layout="wide")
st.title("Egeret Clusters")
st.caption(
    "Slices of the joint cluster space (corpus.parquet, 438 clusters) "
    "that contain at least one Egeret letter. Surfaces cross-source threads."
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


@st.cache_data
def load_clusters():
    docs = pd.read_parquet(os.path.join(DATA_DIR, "egeret_polemic_clusters.parquet"))
    labels = pd.read_parquet(os.path.join(DATA_DIR, "egeret_polemic_cluster_labels.parquet"))
    labels["top_terms_list"] = labels["top_terms"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else (list(x) if x is not None else [])
    )
    return docs, labels


@st.cache_data
def load_threads():
    threads_path = os.path.join(DATA_DIR, "egeret_threads.parquet")
    summaries_path = os.path.join(DATA_DIR, "thread_llm_summaries.parquet")
    if not os.path.exists(threads_path):
        return None, None
    threads = pd.read_parquet(threads_path)
    summaries = None
    if os.path.exists(summaries_path):
        all_sum = pd.read_parquet(summaries_path)
        summaries = all_sum[all_sum["thread_id"].isin(threads["thread_id"])].copy()
    return threads, summaries


docs, labels = load_clusters()
threads, thread_summaries = load_threads()


def render_labels_table(lbl: pd.DataFrame) -> pd.DataFrame:
    show = lbl[[
        "cluster_id", "n_docs", "n_egeret", "n_polemic_egeret",
        "n_authors_egeret", "n_sources", "sources",
        "year_min", "year_max", "topic_type",
    ]].copy()
    show["top_terms"] = lbl["top_terms_list"].apply(
        lambda ts: ", ".join(restore_final_forms(t) for t in ts[:5])
    )
    return show


def render_cluster_detail(cid: int):
    sub = docs[docs["cluster_id"] == cid].copy()
    info = labels[labels["cluster_id"] == cid].iloc[0]

    cols = st.columns(4)
    cols[0].metric("Docs", info["n_docs"])
    cols[1].metric("Egeret docs", info["n_egeret"])
    cols[2].metric("Polemic (Egeret)", info["n_polemic_egeret"])
    cols[3].metric("Sources", info["sources"])

    terms = info["top_terms_list"]
    if terms:
        st.markdown("**Top distinctive terms:**")
        st.markdown(
            " &nbsp;|&nbsp; ".join(
                f'<span dir="rtl" style="font-size:18px;">{restore_final_forms(t)}</span>'
                for t in terms[:15]
            ),
            unsafe_allow_html=True,
        )

    st.markdown(f"**Source mix:** `{info['source_mix']}`")

    st.markdown("**Documents:**")
    cols_to_show = [
        "doc_id", "source", "author", "year", "title", "newspaper",
        "predicted_label", "confidence",
    ]
    table = sub[cols_to_show].sort_values(["source", "year"], na_position="last")
    st.dataframe(table, use_container_width=True, height=420)


def render_tab(subset_labels: pd.DataFrame, default_sort: str, key_prefix: str):
    if len(subset_labels) == 0:
        st.info("No clusters in this slice.")
        return

    sort_options = {
        "n_egeret (desc)": ("n_egeret", False),
        "n_polemic_egeret (desc)": ("n_polemic_egeret", False),
        "n_sources (desc)": ("n_sources", False),
        "n_authors_egeret (desc)": ("n_authors_egeret", False),
        "n_docs (desc)": ("n_docs", False),
        "year_min (asc)": ("year_min", True),
    }
    sort_choice = st.selectbox(
        "Sort by", list(sort_options.keys()),
        index=list(sort_options.keys()).index(default_sort),
        key=f"{key_prefix}_sort",
    )
    col, asc = sort_options[sort_choice]
    sorted_lbl = subset_labels.sort_values(col, ascending=asc).reset_index(drop=True)

    st.dataframe(render_labels_table(sorted_lbl), use_container_width=True, height=300)

    cluster_pick = st.selectbox(
        "Inspect cluster",
        ["None"] + [f"{int(r.cluster_id)} (n={r.n_docs}, egeret={r.n_egeret})"
                    for r in sorted_lbl.itertuples()],
        key=f"{key_prefix}_pick",
    )
    if cluster_pick != "None":
        cid = int(cluster_pick.split(" ")[0])
        st.markdown("---")
        render_cluster_detail(cid)


n_threads = 0 if threads is None else len(threads)
n_polemic_threads = 0
if thread_summaries is not None and len(thread_summaries):
    n_polemic_threads = int(thread_summaries["is_polemic_thread"].fillna(False).astype(bool).sum())

tab1, tab2, tab3, tab4 = st.tabs([
    f"Cross-source ({(labels['topic_type']=='cross-source').sum()})",
    f"Egeret-only ({(labels['topic_type']=='egeret-only').sum()})",
    f"All Egeret-containing ({len(labels)})",
    f"Threads ({n_threads} • {n_polemic_threads} polemic)",
])

with tab1:
    st.caption("Clusters mixing Egeret with press / bypc / cm — candidate cross-corpus threads.")
    render_tab(
        labels[labels["topic_type"] == "cross-source"],
        default_sort="n_sources (desc)",
        key_prefix="cross",
    )

with tab2:
    st.caption("Clusters with only Egeret docs — within-corpus cross-author exchanges.")
    render_tab(
        labels[labels["topic_type"] == "egeret-only"],
        default_sort="n_authors_egeret (desc)",
        key_prefix="egonly",
    )

with tab3:
    render_tab(labels, default_sort="n_egeret (desc)", key_prefix="all")


def render_threads_tab():
    st.caption(
        "Egeret-internal candidate threads from `egeret_threads.parquet` "
        "(author-pair edges, 2-yr window, date-confidence gated, span ≤ 15 yr, 3–30 docs). "
        "Stage B LLM verdicts where available."
    )
    if threads is None or len(threads) == 0:
        st.info("No Egeret threads found. Run `scripts/build_egeret_threads.py` to generate.")
        return

    tbl = threads[[
        "thread_id", "cluster_id", "n_docs", "n_newspapers",
        "span_days", "n_edges", "score", "authors",
    ]].rename(columns={"n_newspapers": "n_authors"}).copy()
    tbl["span_yrs"] = (tbl["span_days"] / 365).round(1)
    tbl = tbl.drop(columns=["span_days"])

    if thread_summaries is not None and len(thread_summaries):
        s = thread_summaries[[
            "thread_id", "is_polemic_thread", "polemic_score",
            "polemic_type", "polemic_direction", "topic_label", "model",
        ]]
        tbl = tbl.merge(s, on="thread_id", how="left")

    polemic_filter = st.radio(
        "Filter", ["all", "polemic only", "topical-only"], horizontal=True, key="threads_filter"
    )
    if polemic_filter == "polemic only" and "is_polemic_thread" in tbl.columns:
        tbl = tbl[tbl["is_polemic_thread"].fillna(False).astype(bool)]
    elif polemic_filter == "topical-only" and "is_polemic_thread" in tbl.columns:
        tbl = tbl[~tbl["is_polemic_thread"].fillna(False).astype(bool)]

    sort_col = "polemic_score" if "polemic_score" in tbl.columns else "n_docs"
    tbl = tbl.sort_values(sort_col, ascending=False, na_position="last").reset_index(drop=True)
    st.dataframe(tbl, use_container_width=True, height=350)

    pick = st.selectbox(
        "Inspect thread",
        ["None"] + [
            f"{int(r.thread_id)} — {(r.topic_label if 'topic_label' in tbl.columns and pd.notna(r.topic_label) else 'no verdict')}"
            for r in tbl.itertuples()
        ],
        key="thread_pick",
    )
    if pick == "None":
        return

    tid = int(pick.split(" ")[0])
    trow = threads[threads["thread_id"] == tid].iloc[0]
    st.markdown("---")
    st.subheader(f"Thread {tid} · cluster {int(trow['cluster_id'])}")

    cols = st.columns(4)
    cols[0].metric("Letters", int(trow["n_docs"]))
    cols[1].metric("Authors", int(trow["n_newspapers"]))
    cols[2].metric("Span (yrs)", round(trow["span_days"] / 365, 1))
    cols[3].metric("Edges", int(trow["n_edges"]))
    st.markdown(f"**Authors:** {trow['authors']}")

    if thread_summaries is not None:
        verdict = thread_summaries[thread_summaries["thread_id"] == tid]
        if len(verdict):
            v = verdict.iloc[0]
            badge = "🟥 POLEMIC" if bool(v.get("is_polemic_thread")) else "⬜ topical-only"
            st.markdown(
                f"### {badge} · {v.get('polemic_type','?')} · {v.get('polemic_direction','?')} "
                f"· score {v.get('polemic_score', 0):.2f}"
            )
            st.markdown(f"**Topic:** {v.get('topic_label','')}")
            st.markdown(f"**Narrative.** {v.get('narrative','')}")
            st.markdown(f"**Evidence.** {v.get('evidence','')}")
            actors = v.get("actors")
            if isinstance(actors, (list, tuple)) and len(actors):
                st.markdown(f"**Actors:** {', '.join(str(a) for a in actors)}")
            elif isinstance(actors, str) and actors:
                st.markdown(f"**Actors:** {actors}")

    doc_ids = [d.strip() for d in str(trow["doc_ids"]).split(",") if d.strip()]
    thread_docs = docs[docs["doc_id"].isin(doc_ids)].copy()
    st.markdown(f"**Letters in thread ({len(thread_docs)}):**")
    st.dataframe(
        thread_docs[["doc_id", "author", "year", "title", "predicted_label", "confidence"]]
        .sort_values(["year", "author"], na_position="last"),
        use_container_width=True,
        height=300,
    )


with tab4:
    render_threads_tab()

st.sidebar.markdown(
    f"**Clusters with Egeret:** {len(labels)}  \n"
    f"**Cross-source:** {(labels['topic_type']=='cross-source').sum()}  \n"
    f"**Egeret-only:** {(labels['topic_type']=='egeret-only').sum()}  \n"
    f"**Egeret docs in slice:** {(docs['source']=='egeret').sum():,}  \n"
    f"**Total docs in slice:** {len(docs):,}"
)
