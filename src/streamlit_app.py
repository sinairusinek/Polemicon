"""
streamlit_app.py - Polemicon Annotation & Keyword Discovery

Real corpus-backed annotation tool for Phase B.2 pilot.
- Loads 200-text stratified pilot sample
- LLM classification results from 4-model pilot (Phase B.2)
- Priority review queue for inter-model disagreements
- Multi-class polemic labeling
- Metadata display (source, year, keyword score, cluster)
- Human keyword suggestion
- CSV export of annotations
"""
import streamlit as st
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from cleaning import restore_final_forms

# --- Data loading ---

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


@st.cache_data
def load_pilot_sample():
    path = os.path.join(DATA_DIR, "pilot_sample.parquet")
    if not os.path.exists(path):
        st.error("Pilot sample not found. Run src/sample_pilot.py first.")
        st.stop()
    pilot = pd.read_parquet(path)
    # Merge full metadata from corpus
    corpus_path = os.path.join(DATA_DIR, "..", "corpus.parquet")
    if os.path.exists(corpus_path):
        meta_cols = ["doc_id", "author", "recipient", "headline", "newspaper", "title"]
        corpus_meta = pd.read_parquet(corpus_path, columns=meta_cols)
        # Drop pilot's own author/title (incomplete), replace with corpus versions
        pilot = pilot.drop(columns=["author", "title"], errors="ignore")
        pilot = pilot.merge(corpus_meta, on="doc_id", how="left")
    return pilot


@st.cache_data
def load_cluster_labels():
    path = os.path.join(DATA_DIR, "cluster_labels.parquet")
    if os.path.exists(path):
        import json
        cl = pd.read_parquet(path)
        cl["top_terms_list"] = cl["top_terms"].apply(
            lambda x: json.loads(x) if pd.notna(x) else []
        )
        return cl
    return None


@st.cache_data(ttl=3600)
def load_classifications():
    path = os.path.join(DATA_DIR, "pilot_classifications.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


@st.cache_data(ttl=3600)
def load_disagreements():
    path = os.path.join(DATA_DIR, "pilot_disagreements.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


@st.cache_data(ttl=3600)
def load_references():
    path = os.path.join(DATA_DIR, "pilot_references.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


@st.cache_data(ttl=3600)
def load_classifications_v2():
    # Prefer the largest available dataset in order: calibration (2K) > pilot v2 > acceptance test
    for fname in ("calibration_v2.parquet", "pilot_classifications_v2.parquet", "acceptance_test_v2.parquet"):
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            return pd.read_parquet(path)
    return None


@st.cache_data(ttl=3600)
def load_ra_gold():
    path = os.path.join(DATA_DIR, "ra_gold_labels.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


@st.cache_data(ttl=3600)
def load_calibration_with_corpus():
    cal_path = os.path.join(DATA_DIR, "calibration_v2.parquet")
    if not os.path.exists(cal_path):
        return None
    cal = pd.read_parquet(cal_path)
    if "_error" in cal.columns:
        cal = cal[cal["polemic_label"].notna() & cal["_error"].isna()].copy()
    else:
        cal = cal[cal["polemic_label"].notna()].copy()
    # Join with corpus only if text not already embedded in the parquet
    if "text" not in cal.columns:
        corpus_path = os.path.join(DATA_DIR, "..", "corpus.parquet")
        if os.path.exists(corpus_path):
            meta = pd.read_parquet(
                corpus_path,
                columns=["doc_id", "source", "year", "author", "title", "newspaper", "headline", "text"],
            )
            cal = cal.merge(meta, on="doc_id", how="left")
    return cal


@st.cache_data(ttl=3600)
def load_threads():
    path = os.path.join(DATA_DIR, "threads.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


@st.cache_data(ttl=3600)
def load_corpus_for_threads():
    corpus_path = os.path.join(DATA_DIR, "..", "corpus.parquet")
    if not os.path.exists(corpus_path):
        return None
    cols = ["doc_id", "date", "year", "newspaper", "headline", "title", "author", "source", "text"]
    return pd.read_parquet(corpus_path, columns=cols).set_index("doc_id")


@st.cache_data(ttl=3600)
def load_full_predictions():
    path = os.path.join(DATA_DIR, "full_corpus_predictions.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path).set_index("doc_id")
    return None


@st.cache_data(ttl=600)
def load_thread_llm_summaries():
    path = os.path.join(DATA_DIR, "thread_llm_summaries.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


@st.cache_data(ttl=600)
def load_thread_doc_summaries():
    path = os.path.join(DATA_DIR, "thread_doc_summaries.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


@st.cache_data(ttl=600)
def load_thread_literature_review():
    path = os.path.join(DATA_DIR, "thread_literature_review.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


@st.cache_data(ttl=3600)
def load_vocab():
    path = os.path.join(DATA_DIR, "pilot_vocab.parquet")
    if os.path.exists(path):
        import json
        vdf = pd.read_parquet(path)
        # Parse JSON-encoded marker lists
        vdf["polemic_markers"] = vdf["polemic_markers_json"].apply(
            lambda x: json.loads(x) if pd.notna(x) else []
        )
        vdf["marker_explanations"] = vdf["marker_explanations_json"].apply(
            lambda x: json.loads(x) if pd.notna(x) else []
        )
        return vdf
    return None


# --- App config ---

st.set_page_config(page_title="Polemicon Annotation", layout="wide")
st.title("Polemicon Annotation & Keyword Discovery")

df = load_pilot_sample()
clf_df = load_classifications()
clf_v2 = load_classifications_v2()
disagree_df = load_disagreements()
refs_df = load_references()
vocab_df = load_vocab()
cluster_labels_df = load_cluster_labels()
ra_gold_df = load_ra_gold()
cal_corpus_df = load_calibration_with_corpus()

# --- Session state ---

if "annotations" not in st.session_state:
    st.session_state["annotations"] = {}
if "comments" not in st.session_state:
    st.session_state["comments"] = {}
if "keyword_suggestions" not in st.session_state:
    st.session_state["keyword_suggestions"] = []
if "vocab_approvals" not in st.session_state:
    st.session_state["vocab_approvals"] = {}  # {doc_id: {marker: "approved"/"rejected"}}
if "current_idx" not in st.session_state:
    st.session_state["current_idx"] = 0

# --- Sidebar: filters and navigation ---

view_mode = st.sidebar.radio("View", ["Annotation Tool", "Calibration Browser", "Thread Browser"], horizontal=True)

st.sidebar.header("Filters")

sources = ["all"] + sorted(df["source"].unique().tolist())
selected_source = st.sidebar.selectbox("Source", sources)

if "cluster_id" in df.columns:
    cluster_ids = sorted(df["cluster_id"].dropna().unique().tolist())
    cluster_options = ["all"] + [str(int(c)) for c in cluster_ids]
    selected_cluster = st.sidebar.selectbox("Cluster", cluster_options)
else:
    selected_cluster = "all"

score_min, score_max = float(df["polemic_score"].min()), float(df["polemic_score"].max())
score_range = st.sidebar.slider(
    "Keyword score range",
    min_value=score_min, max_value=score_max,
    value=(score_min, score_max), step=0.01,
)

# Review priority filter (from LLM disagreements)
PRIORITY_OPTIONS = {
    "all": "All texts",
    "1": "🔴 Expensive models disagree (priority)",
    "2": "🟡 Expensive agree, cheap diverge",
    "3_polemic": "🟢 All agree polemic",
    "3_not": "⚪ All agree not polemic",
}
if disagree_df is not None:
    selected_priority = st.sidebar.selectbox("Review priority", list(PRIORITY_OPTIONS.keys()),
                                              format_func=lambda x: PRIORITY_OPTIONS[x])
else:
    selected_priority = "all"

# Show only unannotated
show_unannotated_only = st.sidebar.checkbox("Show unannotated only", value=False)

# Apply filters
filtered = df.copy()
if selected_source != "all":
    filtered = filtered[filtered["source"] == selected_source]
if selected_cluster != "all":
    filtered = filtered[filtered["cluster_id"] == int(selected_cluster)]
filtered = filtered[
    (filtered["polemic_score"] >= score_range[0]) &
    (filtered["polemic_score"] <= score_range[1])
]
if show_unannotated_only:
    annotated_ids = set(st.session_state["annotations"].keys())
    filtered = filtered[~filtered["doc_id"].isin(annotated_ids)]

# Apply review priority filter
if disagree_df is not None and selected_priority != "all":
    if selected_priority == "3_polemic":
        priority_ids = disagree_df[disagree_df["agreement_category"] == "all_agree_polemic"]["doc_id"]
    elif selected_priority == "3_not":
        priority_ids = disagree_df[disagree_df["agreement_category"] == "all_agree_not_polemic"]["doc_id"]
    elif selected_priority == "1":
        priority_ids = disagree_df[disagree_df["agreement_category"] == "expensive_disagree"]["doc_id"]
    elif selected_priority == "2":
        priority_ids = disagree_df[disagree_df["agreement_category"] == "expensive_agree_cheap_diverge"]["doc_id"]
    else:
        priority_ids = disagree_df["doc_id"]
    filtered = filtered[filtered["doc_id"].isin(priority_ids)]

filtered = filtered.reset_index(drop=True)

st.sidebar.markdown(f"**{len(filtered)}** texts match filters")
st.sidebar.markdown(f"**{len(st.session_state['annotations'])}** / {len(df)} annotated")

# LLM classification summary
if disagree_df is not None:
    st.sidebar.header("LLM Agreement")
    cats = disagree_df["agreement_category"].value_counts()
    st.sidebar.markdown(
        f"🔴 Disagree: **{cats.get('expensive_disagree', 0)}**  \n"
        f"🟡 Cheap diverge: **{cats.get('expensive_agree_cheap_diverge', 0)}**  \n"
        f"🟢 All polemic: **{cats.get('all_agree_polemic', 0)}**  \n"
        f"⚪ All not: **{cats.get('all_agree_not_polemic', 0)}**"
    )

# Calibration v2 stats
if cal_corpus_df is not None:
    st.sidebar.header("Calibration v2")
    _total = len(cal_corpus_df)
    _COLORS = {
        "explicit polemic":           "#d62728",
        "implicit polemic":           "#ff7f0e",
        "meta-polemic (descriptive)": "#1f77b4",
        "non-polemic":                "#2ca02c",
    }
    for _lbl, _col in _COLORS.items():
        _n = (cal_corpus_df["polemic_label"] == _lbl).sum()
        st.sidebar.markdown(
            f'<span style="background:{_col};color:white;padding:1px 6px;border-radius:3px;font-size:11px;">{_lbl}</span>'
            f' **{_n}** ({_n/_total:.1%})',
            unsafe_allow_html=True,
        )
    if "broader_polemic_link" in cal_corpus_df.columns:
        st.sidebar.markdown("**Broader debate link:**")
        for _val in ["clear", "suspected", "none"]:
            _n = (cal_corpus_df["broader_polemic_link"] == _val).sum()
            st.sidebar.markdown(f"  `{_val}`: {_n} ({_n/_total:.1%})")

# --- Navigation ---

st.sidebar.header("Navigation")

if len(filtered) == 0:
    st.info("No texts match the current filters.")
    st.stop()

# ── Thread Browser ─────────────────────────────────────────────────────────────

if view_mode == "Thread Browser":
    threads_df = load_threads()
    if threads_df is None:
        st.warning("threads.parquet not found. Run C.2 threading pipeline first.")
        st.stop()

    corpus_idx = load_corpus_for_threads()
    if corpus_idx is None:
        st.warning("corpus.parquet not found at project root.")
        st.stop()
    preds_idx = load_full_predictions()
    llm_threads = load_thread_llm_summaries()
    llm_docs = load_thread_doc_summaries()
    lit_review = load_thread_literature_review()

    st.subheader("Thread Browser")
    st.caption(
        f"{len(threads_df)} threads — engaged threads span multiple newspapers; "
        "internal threads are within-paper sequences."
    )

    tcol1, tcol2, tcol3 = st.columns([2, 2, 2])
    with tcol1:
        type_choice = st.selectbox("Type", ["engaged", "internal", "all"], index=0)
    with tcol2:
        min_papers = st.number_input("Min newspapers", min_value=1, max_value=10, value=2, step=1)
    with tcol3:
        sort_choice = st.selectbox(
            "Sort by",
            ["score", "n_docs", "n_edges", "cross_paper_edges", "span_days"],
        )

    tview = threads_df.copy()
    if type_choice != "all":
        tview = tview[tview["thread_type"] == type_choice]
    tview = tview[tview["n_newspapers"] >= int(min_papers)]
    tview = tview.sort_values(sort_choice, ascending=False).reset_index(drop=True)

    if len(tview) == 0:
        st.info("No threads match these filters.")
        st.stop()

    summary_cols = ["thread_id", "cluster_id", "n_docs", "n_newspapers", "span_days",
                    "n_edges", "cross_paper_edges", "same_paper_edges", "edge_types",
                    "thread_type", "score", "newspapers"]
    top_table = tview[summary_cols].head(50).copy()
    if llm_threads is not None and not llm_threads.empty:
        # Pick first model per thread for the summary column
        llm_first = (llm_threads.sort_values("model")
                     .drop_duplicates("thread_id", keep="first")
                     .set_index("thread_id"))
        top_table["llm_verdict"] = top_table["thread_id"].map(
            lambda t: ("✓" if llm_first.loc[t]["is_polemic_thread"] else "✗")
            if t in llm_first.index else "—"
        )
        top_table["llm_score"] = top_table["thread_id"].map(
            lambda t: round(float(llm_first.loc[t]["polemic_score"]), 2)
            if t in llm_first.index and pd.notna(llm_first.loc[t].get("polemic_score")) else None
        )
        top_table["llm_type"] = top_table["thread_id"].map(
            lambda t: llm_first.loc[t]["polemic_type"] if t in llm_first.index else None
        )
    if lit_review is not None and not lit_review.empty:
        lit_idx = lit_review.set_index("thread_id")
        top_table["lit_status"] = top_table["thread_id"].map(
            lambda t: str(lit_idx.loc[t].get("is_documented") or "—")
            if t in lit_idx.index else "—"
        )
        top_table["lit_canonical"] = top_table["thread_id"].map(
            lambda t: "✓" if (t in lit_idx.index and bool(lit_idx.loc[t].get("is_canonical_event")))
            else "" if t in lit_idx.index else "—"
        )
    st.markdown("**Top threads**")
    st.dataframe(top_table, use_container_width=True, height=280)

    def _thread_label(tid):
        r = tview[tview["thread_id"] == tid].iloc[0]
        return (f"#{int(tid)} — {int(r['n_docs'])} docs · {int(r['n_newspapers'])} papers "
                f"· {int(r['span_days'])}d · score {r['score']:.1f} · {r['newspapers']}")

    thread_id_sel = st.selectbox(
        "Select a thread to inspect",
        tview["thread_id"].head(50).tolist(),
        format_func=_thread_label,
    )

    trow = tview[tview["thread_id"] == thread_id_sel].iloc[0]
    doc_ids = [d.strip() for d in str(trow["doc_ids"]).split(",") if d.strip()]
    avail = [d for d in doc_ids if d in corpus_idx.index]
    docs = corpus_idx.loc[avail].copy()
    docs["doc_id"] = docs.index
    if "date" in docs.columns:
        docs = docs.sort_values("date")

    LABEL_COLORS = {
        "non-polemic": "#2ca02c",
        "implicit": "#ff7f0e",
        "explicit": "#d62728",
        "meta-polemic": "#1f77b4",
    }

    st.markdown(
        f"### Thread #{int(thread_id_sel)} — cluster {int(trow['cluster_id'])} · "
        f"{int(trow['n_docs'])} docs · {int(trow['n_newspapers'])} papers · "
        f"{int(trow['span_days'])} days · edges: {trow['edge_types']}"
    )

    # --- LLM verdict panel (if available) ---
    if llm_threads is not None:
        llm_rows = llm_threads[llm_threads["thread_id"] == int(thread_id_sel)]
        if not llm_rows.empty:
            model_choices = llm_rows["model"].unique().tolist()
            sel_model = st.selectbox("LLM model", model_choices, key="llm_model_select")
            lr = llm_rows[llm_rows["model"] == sel_model].iloc[0]
            verdict = lr.get("is_polemic_thread")
            score = lr.get("polemic_score")
            ptype = lr.get("polemic_type", "")
            verdict_color = "#d62728" if verdict else "#888"
            try:
                score_f = float(score) if score is not None else None
            except Exception:
                score_f = None
            score_s = f"{score_f:.2f}" if score_f is not None else "—"
            heur = float(trow["score"])
            verdict_txt = "POLEMIC THREAD" if verdict else ("not polemic" if verdict is False else "?")

            with st.container(border=True):
                top_cols = st.columns([3, 2, 2])
                with top_cols[0]:
                    st.markdown(
                        f'<span style="background:{verdict_color};color:white;'
                        f'padding:3px 10px;border-radius:4px;font-weight:bold;">{verdict_txt}</span> '
                        f'&nbsp;<b>LLM score:</b> {score_s} '
                        f'&nbsp;<b>heuristic:</b> {heur:.1f} '
                        f'&nbsp;<b>type:</b> <code>{ptype}</code>',
                        unsafe_allow_html=True,
                    )
                with top_cols[1]:
                    label_en = lr.get("topic_label") or ""
                    if label_en:
                        st.markdown(f"**Topic (EN):** {label_en}")
                with top_cols[2]:
                    label_he = lr.get("topic_label_he") or ""
                    if label_he:
                        st.markdown(
                            f'<div dir="rtl" style="text-align:right;"><b>נושא:</b> {label_he}</div>',
                            unsafe_allow_html=True,
                        )

                narrative = lr.get("narrative") or ""
                if narrative:
                    st.markdown(f"**Narrative:** {narrative}")

                evidence = lr.get("evidence") or ""
                if evidence:
                    st.caption(f"Evidence: {evidence}")

                actors_raw = lr.get("actors")
                if actors_raw:
                    try:
                        import json as _json
                        actors_list = _json.loads(actors_raw) if isinstance(actors_raw, str) else list(actors_raw)
                        if actors_list:
                            st.markdown("**Actors:** " + ", ".join(str(a) for a in actors_list))
                    except Exception:
                        pass

                edges_raw = lr.get("rebuttal_edges") if "rebuttal_edges" in lr.index else None
                if edges_raw:
                    try:
                        import json as _json
                        edges = _json.loads(edges_raw) if isinstance(edges_raw, str) else list(edges_raw)
                        if edges:
                            st.markdown("**Rebuttal edges (LLM-identified):**")
                            for e in edges:
                                if isinstance(e, (list, tuple)) and len(e) >= 2:
                                    a, b = e[0], e[1]
                                    rel = e[2] if len(e) > 2 else "responds-to"
                                    st.markdown(f"&nbsp;&nbsp;`{a}` → *{rel}* → `{b}`")
                    except Exception:
                        pass

    # --- Secondary literature panel ---
    if lit_review is not None:
        lit_rows = lit_review[lit_review["thread_id"] == int(thread_id_sel)]
        if not lit_rows.empty:
            lit = lit_rows.iloc[0]
            doc_status = lit.get("is_documented")
            if pd.notna(doc_status) and doc_status:
                status_colors = {
                    "well-documented": "#2ca02c",
                    "mentioned-in-passing": "#ff7f0e",
                    "not-found": "#888",
                }
                color = status_colors.get(doc_status, "#666")
                canonical = bool(lit.get("is_canonical_event", False))
                with st.container(border=True):
                    head = st.columns([3, 2])
                    with head[0]:
                        st.markdown(
                            f'**Secondary literature** &nbsp; '
                            f'<span style="background:{color};color:white;'
                            f'padding:3px 10px;border-radius:4px;font-weight:bold;">'
                            f'{doc_status}</span>'
                            f' &nbsp; underlying event canonical: '
                            f'**{"yes" if canonical else "no"}**',
                            unsafe_allow_html=True,
                        )
                    notes = lit.get("notes") or ""
                    if notes:
                        st.caption(notes)
                    sources_raw = lit.get("key_sources")
                    if sources_raw:
                        try:
                            import json as _json
                            sources = _json.loads(sources_raw) if isinstance(sources_raw, str) else list(sources_raw)
                            if sources:
                                st.markdown("**Sources:**")
                                for s in sources:
                                    author = str(s.get("author") or "").strip()
                                    title = str(s.get("title") or "").strip()
                                    year = s.get("year") or ""
                                    stype = s.get("type") or ""
                                    url = str(s.get("url") or "").strip()
                                    where = str(s.get("where_discussed") or "").strip()
                                    head_str = ""
                                    if author:
                                        head_str += author
                                    if year and year != 0:
                                        head_str += f" ({year})" if head_str else f"({year})"
                                    if title:
                                        head_str += f" — *{title}*" if head_str else f"*{title}*"
                                    if url:
                                        line = f"- [{head_str}]({url})"
                                    else:
                                        line = f"- {head_str}"
                                    if stype:
                                        line += f" `{stype}`"
                                    st.markdown(line)
                                    if where:
                                        st.markdown(
                                            f"&nbsp;&nbsp;&nbsp;&nbsp;<small style='color:#888;'>{where}</small>",
                                            unsafe_allow_html=True,
                                        )
                        except Exception:
                            pass

    text_limit = st.slider("Text preview length", 200, 3000, 800, 100)

    for _, drow in docs.iterrows():
        did = drow["doc_id"]
        meta_parts = []
        date_v = drow.get("date")
        if pd.notna(date_v):
            meta_parts.append(str(date_v)[:10])
        np_v = drow.get("newspaper")
        if pd.notna(np_v) and str(np_v) not in ("", "nan"):
            meta_parts.append(restore_final_forms(str(np_v)))
        au_v = drow.get("author")
        if pd.notna(au_v) and str(au_v) not in ("", "nan"):
            meta_parts.append(restore_final_forms(str(au_v)))
        meta_parts.append(did)

        pred_badge = ""
        if preds_idx is not None and did in preds_idx.index:
            pr = preds_idx.loc[did]
            lbl = str(pr["predicted_label"])
            conf = float(pr["confidence"])
            col = LABEL_COLORS.get(lbl, "#666")
            pred_badge = (
                f'<span style="background:{col};color:white;padding:2px 8px;'
                f'border-radius:3px;font-size:12px;">{lbl} · {conf:.0%}</span>'
            )

        with st.container(border=True):
            head_cols = st.columns([4, 1])
            with head_cols[0]:
                st.markdown(
                    f"<small style='color:#888;'>{' · '.join(meta_parts)}</small>",
                    unsafe_allow_html=True,
                )
                title = drow.get("headline") or drow.get("title")
                if pd.notna(title) and str(title) not in ("", "nan"):
                    st.markdown(
                        f'<div dir="rtl" style="text-align:right;font-weight:bold;">'
                        f'{restore_final_forms(str(title))}</div>',
                        unsafe_allow_html=True,
                    )
            with head_cols[1]:
                if pred_badge:
                    st.markdown(pred_badge, unsafe_allow_html=True)

            if llm_docs is not None:
                doc_sum_rows = llm_docs[llm_docs["doc_id"] == did]
                if not doc_sum_rows.empty:
                    ds = doc_sum_rows.iloc[0]
                    pol_icon = "🔴" if bool(ds.get("is_polemical")) else "⚪"
                    sm = str(ds.get("summary_he") or "")
                    if sm:
                        st.markdown(
                            f'<div style="background:#f5f3ef;border-left:3px solid #999;'
                            f'padding:5px 10px;margin:4px 0;font-size:13px;">'
                            f'<small style="color:#888;">{pol_icon} LLM ({ds.get("model","?")}):</small> '
                            f'<span dir="rtl" style="text-align:right;">{restore_final_forms(sm)}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            text_val = str(drow.get("text", "") or "")
            preview = restore_final_forms(text_val[:text_limit])
            st.markdown(
                f'<div dir="rtl" style="text-align:right;font-size:14px;line-height:1.7;">'
                f'{preview}{"…" if len(text_val) > text_limit else ""}</div>',
                unsafe_allow_html=True,
            )
            if len(text_val) > text_limit:
                with st.expander("Full text"):
                    st.markdown(
                        f'<div dir="rtl" style="text-align:right;font-size:14px;line-height:1.7;">'
                        f'{restore_final_forms(text_val[:8000])}</div>',
                        unsafe_allow_html=True,
                    )

    st.stop()


# ── Calibration Browser ────────────────────────────────────────────────────────

if view_mode == "Calibration Browser":
    if cal_corpus_df is None or "text" not in cal_corpus_df.columns:
        st.warning("Calibration data not available. Run classify_pilot.py --v2 --calibration first.")
        st.stop()

    _t = len(cal_corpus_df)
    st.caption(
        f"Claude Sonnet labeled {_t:,} stratified corpus texts into four polemic categories. "
        "These labels serve as training data for the Hebrew classifier (B.4) that will process the full 33,000-text corpus."
    )
    _SOURCE_DISPLAY = {
        "polemic_candidates": "Ben Yehuda Project",
        "press":              "press",
        "egeret":             "egeret",
        "compact_memory":     "compact_memory",
    }

    with st.expander("Distribution breakdown"):
        if "source" in cal_corpus_df.columns:
            st.markdown("**Per-source polemic rate**")
            _src_rows = []
            for _src, _grp in cal_corpus_df.groupby("source"):
                _n = len(_grp)
                _pol = (_grp["polemic_label"] != "non-polemic").sum()
                _src_rows.append({
                    "source":      _SOURCE_DISPLAY.get(_src, _src),
                    "texts":       _n,
                    "polemic %":   f"{_pol/_n:.0%}",
                    "explicit":    f"{(_grp['polemic_label']=='explicit polemic').sum()/_n:.0%}",
                    "implicit":    f"{(_grp['polemic_label']=='implicit polemic').sum()/_n:.0%}",
                    "meta":        f"{(_grp['polemic_label']=='meta-polemic (descriptive)').sum()/_n:.0%}",
                    "non-polemic": f"{(_grp['polemic_label']=='non-polemic').sum()/_n:.0%}",
                })
            st.dataframe(pd.DataFrame(_src_rows).set_index("source"), use_container_width=True)

        if "year" in cal_corpus_df.columns:
            st.markdown("**Polemic rate by year (% of texts classified as polemic)**")
            _yr = cal_corpus_df[cal_corpus_df["year"].notna()].copy()
            _yr["source_display"] = _yr["source"].map(_SOURCE_DISPLAY).fillna(_yr["source"])
            _yr["is_polemic"] = (_yr["polemic_label"] != "non-polemic").astype(float)
            _yr["year_int"] = _yr["year"].astype(int)
            _pivot = (
                _yr.groupby(["year_int", "source_display"])["is_polemic"]
                .mean()
                .mul(100)
                .round(1)
                .unstack("source_display")
            )
            _pivot.index.name = "year"
            st.line_chart(_pivot, use_container_width=True)

    _strip_labels = [
        ("explicit polemic",           "#d62728"),
        ("implicit polemic",           "#ff7f0e"),
        ("meta-polemic (descriptive)", "#1f77b4"),
        ("non-polemic",                "#2ca02c"),
    ]
    _parts = []
    for _lbl, _col in _strip_labels:
        _n = (cal_corpus_df["polemic_label"] == _lbl).sum()
        _parts.append(
            f'<span style="background:{_col};color:white;padding:2px 10px;'
            f'border-radius:3px;font-size:13px;margin-right:6px;">'
            f'{_lbl} <b>{_n}</b> <span style="opacity:.85">({_n/_t:.0%})</span></span>'
        )
    _bpl_clear = (cal_corpus_df["broader_polemic_link"] == "clear").sum()
    _bpl_susp  = (cal_corpus_df["broader_polemic_link"] == "suspected").sum()
    _parts.append(
        f'<span style="color:#555;font-size:13px;margin-left:10px;">'
        f'broader debate link — 🔗 clear: <b>{_bpl_clear}</b> &nbsp; ❓ suspected: <b>{_bpl_susp}</b>'
        f'</span>'
    )
    st.markdown(
        '<div style="background:#f8f8f8;border:1px solid #e0e0e0;border-radius:5px;'
        f'padding:7px 14px;margin-bottom:12px;">' + "".join(_parts) + "</div>",
        unsafe_allow_html=True,
    )

    POLEMIC_LABELS = ["explicit polemic", "implicit polemic", "meta-polemic (descriptive)"]
    LABEL_COLORS_CAL = {
        "explicit polemic":           "#d62728",
        "implicit polemic":           "#ff7f0e",
        "meta-polemic (descriptive)": "#1f77b4",
    }

    col_filter, col_sort = st.columns([2, 2])
    with col_filter:
        selected_label = st.selectbox(
            "Polemic category",
            POLEMIC_LABELS,
            format_func=lambda x: x,
        )
    with col_sort:
        sort_by = st.selectbox("Sort by", ["confidence (high→low)", "year", "source"])

    cal_view = cal_corpus_df[cal_corpus_df["polemic_label"] == selected_label].copy()

    if sort_by == "confidence (high→low)":
        cal_view = cal_view.sort_values("confidence", ascending=False)
    elif sort_by == "year":
        cal_view = cal_view.sort_values("year", na_position="last")
    elif sort_by == "source":
        cal_view = cal_view.sort_values("source")

    cal_view = cal_view.reset_index(drop=True)

    PAGE_SIZE = 8
    total_pages = max(1, (len(cal_view) - 1) // PAGE_SIZE + 1)

    if "cal_page" not in st.session_state:
        st.session_state["cal_page"] = 0
    # Reset page when label changes
    if st.session_state.get("cal_last_label") != selected_label:
        st.session_state["cal_page"] = 0
        st.session_state["cal_last_label"] = selected_label

    pg = st.session_state["cal_page"]
    p_col1, p_col2, p_col3 = st.columns([1, 3, 1])
    with p_col1:
        if st.button("← Prev", key="cal_prev") and pg > 0:
            st.session_state["cal_page"] -= 1
            st.rerun()
    with p_col2:
        st.markdown(
            f"<div style='text-align:center;'><b>{selected_label}</b> — "
            f"{len(cal_view)} texts &nbsp;|&nbsp; page {pg+1} / {total_pages}</div>",
            unsafe_allow_html=True,
        )
    with p_col3:
        if st.button("Next →", key="cal_next") and pg < total_pages - 1:
            st.session_state["cal_page"] += 1
            st.rerun()

    page_df = cal_view.iloc[pg * PAGE_SIZE : (pg + 1) * PAGE_SIZE]
    color = LABEL_COLORS_CAL[selected_label]

    for _, row_c in page_df.iterrows():
        meta_parts = []
        if pd.notna(row_c.get("source")):
            meta_parts.append(str(row_c["source"]))
        if pd.notna(row_c.get("year")):
            meta_parts.append(str(int(row_c["year"])))
        if pd.notna(row_c.get("newspaper")) and str(row_c.get("newspaper")) not in ("", "nan"):
            meta_parts.append(restore_final_forms(str(row_c["newspaper"])))
        if pd.notna(row_c.get("author")) and str(row_c.get("author")) not in ("", "nan"):
            meta_parts.append(restore_final_forms(str(row_c["author"])))
        conf = row_c.get("confidence")
        conf_s = f"conf {conf:.0%}" if conf is not None else ""
        meta_line = " · ".join(meta_parts) + (f" · {conf_s}" if conf_s else "")

        with st.container(border=True):
            head_cols = st.columns([4, 1])
            with head_cols[0]:
                st.markdown(f"<small style='color:#888;'>{meta_line}</small>", unsafe_allow_html=True)
                title = row_c.get("headline") or row_c.get("title")
                if pd.notna(title) and str(title) not in ("", "nan"):
                    st.markdown(
                        f'<div dir="rtl" style="text-align:right;font-weight:bold;">'
                        f'{restore_final_forms(str(title))}</div>',
                        unsafe_allow_html=True,
                    )
            with head_cols[1]:
                blink = row_c.get("broader_polemic_link", "none") or "none"
                blink_icon = {"clear": "🔗", "suspected": "❓", "none": ""}.get(blink, "")
                st.markdown(
                    f'<span style="background:{color};color:white;padding:2px 8px;'
                    f'border-radius:3px;font-size:12px;">{blink_icon} {blink}</span>',
                    unsafe_allow_html=True,
                )

            topic = row_c.get("topic", "")
            if pd.notna(topic) and str(topic) not in ("", "nan"):
                st.caption(str(topic))

            text_val = str(row_c.get("text", ""))
            preview = restore_final_forms(text_val[:500])
            st.markdown(
                f'<div dir="rtl" style="text-align:right;font-size:14px;line-height:1.7;">'
                f'{preview}{"…" if len(text_val) > 500 else ""}</div>',
                unsafe_allow_html=True,
            )
            if len(text_val) > 500:
                with st.expander("Full text"):
                    st.markdown(
                        f'<div dir="rtl" style="text-align:right;font-size:14px;line-height:1.7;">'
                        f'{restore_final_forms(text_val[:5000])}</div>',
                        unsafe_allow_html=True,
                    )

    st.stop()

# Clamp index
if st.session_state["current_idx"] >= len(filtered):
    st.session_state["current_idx"] = 0

col_prev, col_num, col_next = st.sidebar.columns([1, 2, 1])
with col_prev:
    if st.button("← Prev"):
        st.session_state["current_idx"] = max(0, st.session_state["current_idx"] - 1)
with col_next:
    if st.button("Next →"):
        st.session_state["current_idx"] = min(len(filtered) - 1, st.session_state["current_idx"] + 1)
with col_num:
    st.markdown(f"**{st.session_state['current_idx'] + 1}** / {len(filtered)}")

# Jump to index
jump = st.sidebar.number_input(
    "Jump to #", min_value=1, max_value=len(filtered),
    value=st.session_state["current_idx"] + 1, step=1
)
if jump - 1 != st.session_state["current_idx"]:
    st.session_state["current_idx"] = jump - 1

# --- Main content ---

row = filtered.iloc[st.session_state["current_idx"]]
doc_id = row["doc_id"]

# Metadata bar — Row 1: key identifiers
row1 = st.columns(5)
row1[0].metric("Source", row["source"])
row1[1].metric("Year", int(row["year"]) if pd.notna(row.get("year")) else "N/A")
row1[2].metric("Initial vocab score", f"{row['polemic_score']:.3f}")
if "cluster_id" in row and pd.notna(row.get("cluster_id")):
    row1[3].metric("Cluster", int(row["cluster_id"]))
else:
    row1[3].metric("Cluster", "N/A")
row1[4].metric("Doc ID", doc_id)

# Metadata bar — Row 2: conditional fields (only non-null)
row2_items = []
if pd.notna(row.get("author")):
    row2_items.append(("Author", restore_final_forms(str(row["author"]))))
if pd.notna(row.get("newspaper")):
    row2_items.append(("Newspaper", restore_final_forms(str(row["newspaper"]))))
if pd.notna(row.get("headline")):
    row2_items.append(("Headline", restore_final_forms(str(row["headline"]))))
if pd.notna(row.get("title")):
    title_display = restore_final_forms(str(row["title"]))
    # Add Ben-Yehuda link for polemic_candidates
    if doc_id.startswith("bypc_"):
        byid = doc_id.replace("bypc_", "")
        title_display += f' <a href="https://benyehuda.org/read/{byid}" target="_blank">🔗</a>'
    row2_items.append(("Title", title_display))
if pd.notna(row.get("recipient")):
    row2_items.append(("Recipient", restore_final_forms(str(row["recipient"]))))

if row2_items:
    row2 = st.columns(len(row2_items))
    for i, (label, value) in enumerate(row2_items):
        row2[i].markdown(
            f'<div dir="rtl" style="text-align:right;">'
            f'<small style="color:#888;">{label}</small><br>'
            f'<b>{value}</b></div>',
            unsafe_allow_html=True,
        )

# Cluster top terms
if cluster_labels_df is not None and pd.notna(row.get("cluster_id")):
    cid = int(row["cluster_id"])
    cl_row = cluster_labels_df[cluster_labels_df["cluster_id"] == cid]
    if len(cl_row) > 0:
        terms = cl_row.iloc[0]["top_terms_list"]
        displayed = ", ".join(restore_final_forms(t) for t in terms[:7])
        st.caption(f"Cluster {cid} top terms: {displayed}")

# Text display
st.subheader("Text")
st.markdown(
    f'<div dir="rtl" style="text-align:right; font-size:16px; line-height:1.8; '
    f'max-height:400px; overflow-y:auto; border:1px solid #ddd; padding:12px; '
    f'border-radius:4px;">{restore_final_forms(str(row["text"])[:5000])}{"..." if len(str(row["text"])) > 5000 else ""}</div>',
    unsafe_allow_html=True,
)

# --- LLM Classification Results ---

if clf_df is not None:
    doc_clf = clf_df[clf_df["doc_id"] == doc_id]
    if len(doc_clf) > 0:
        st.subheader("LLM Classifications")

        # Show agreement category badge
        if disagree_df is not None:
            doc_disagree = disagree_df[disagree_df["doc_id"] == doc_id]
            if len(doc_disagree) > 0:
                cat = doc_disagree.iloc[0]["agreement_category"]
                cat_labels = {
                    "all_agree_polemic": "🟢 All models agree: POLEMIC",
                    "all_agree_not_polemic": "⚪ All models agree: NOT polemic",
                    "expensive_agree_cheap_diverge": "🟡 Expensive agree, cheap diverge",
                    "expensive_disagree": "🔴 Expensive models DISAGREE — needs review",
                }
                st.markdown(f"**{cat_labels.get(cat, cat)}**")

        # Show each model's classification in columns
        model_cols = st.columns(len(doc_clf))
        for i, (_, clf_row) in enumerate(doc_clf.iterrows()):
            with model_cols[i]:
                is_pol = clf_row.get("is_polemic")
                icon = "✅" if is_pol else "❌" if is_pol is False else "⚠️"
                st.markdown(f"**{clf_row.get('model_display', clf_row['model'])}**")
                st.markdown(f"{icon} Polemic: **{is_pol}**")
                conf = clf_row.get("confidence")
                if conf is not None:
                    st.markdown(f"Confidence: **{conf:.0%}**")
                ptype = clf_row.get("polemic_type", "")
                if ptype and ptype != "none":
                    st.markdown(f"Type: **{ptype}**")
                target = clf_row.get("target", "")
                if target:
                    st.markdown(f"Target: {target}")
                topic = clf_row.get("topic", "")
                if topic:
                    st.markdown(f"Topic: _{topic}_")

        # Expandable evidence section
        with st.expander("Model evidence/reasoning"):
            for _, clf_row in doc_clf.iterrows():
                evidence = clf_row.get("evidence", "")
                if evidence:
                    st.markdown(f"**{clf_row.get('model_display', clf_row['model'])}:** {evidence}")

# --- v2 Sonnet Classification (4-tier) ---

if clf_v2 is not None:
    doc_v2 = clf_v2[clf_v2["doc_id"] == doc_id]
    if len(doc_v2) > 0:
        v2row = doc_v2.iloc[0]
        st.subheader("Sonnet v2 Classification")

        LABEL_COLORS = {
            "explicit polemic":          "#d62728",
            "implicit polemic":          "#ff7f0e",
            "meta-polemic (descriptive)":"#1f77b4",
            "non-polemic":               "#2ca02c",
            "uncertain":                 "#9467bd",
            "unlabeled":                 "#7f7f7f",
        }
        plabel = v2row.get("polemic_label", "unlabeled") or "unlabeled"
        color  = LABEL_COLORS.get(plabel, "#888")
        conf   = v2row.get("confidence")
        conf_s = f" ({conf:.0%})" if conf is not None else ""

        st.markdown(
            f'<span style="background:{color}; color:white; padding:4px 10px; '
            f'border-radius:4px; font-weight:bold;">{plabel}</span>{conf_s}',
            unsafe_allow_html=True,
        )

        ptype = v2row.get("polemic_type", "")
        if ptype and ptype != "none":
            st.markdown(f"**Type:** {ptype}")

        topic = v2row.get("topic", "")
        if topic:
            st.markdown(f"**Topic:** _{topic}_")

        blink = v2row.get("broader_polemic_link", "none") or "none"
        if blink in ("suspected", "clear"):
            bj = v2row.get("broader_polemic_justification", "")
            icon = "🔗" if blink == "clear" else "❓"
            st.markdown(
                f"{icon} **Broader debate link:** `{blink}`"
                + (f" — {bj}" if bj else "")
            )

        with st.expander("v2 evidence"):
            ev = v2row.get("evidence", "")
            if ev:
                st.markdown(ev)

# --- RA Gold Label (for reviewed cases) ---

if ra_gold_df is not None:
    doc_gold = ra_gold_df[ra_gold_df["doc_id"] == doc_id]
    if len(doc_gold) > 0:
        grow = doc_gold.iloc[0]
        glabel = grow.get("ra_label_4tier", "")
        gsrc   = grow.get("source", "")
        gnotes = grow.get("ra_notes", "")
        gref   = grow.get("ra_reference_in_text", "")
        gcmt   = grow.get("ra_comment", "")

        st.info(
            f"**RA gold label:** {glabel}  \n"
            f"*(source: {gsrc})*"
            + (f"  \n📝 {gnotes}" if gnotes and str(gnotes) != "nan" else "")
            + (f"  \n🔍 Reference: {gref}" if gref and str(gref) != "nan" else "")
            + (f"  \n💬 {gcmt}" if gcmt and str(gcmt) != "nan" else "")
        )

# --- Intertextual References ---

if refs_df is not None:
    doc_refs = refs_df[refs_df["doc_id"] == doc_id]
    if len(doc_refs) > 0:
        st.subheader("Intertextual References")

        # Summary counts by category
        CATEGORY_ICONS = {
            "biblical": "📖", "talmudic": "📜", "contemporary_person": "👤",
            "contemporary_publication": "📰", "contemporary_text": "📄",
            "scholarly": "🎓", "other": "📌",
        }
        if "category" in doc_refs.columns:
            llm_refs = doc_refs[doc_refs["method"] == "llm_sonnet"]
            if len(llm_refs) > 0:
                cats = llm_refs["category"].value_counts()
                summary_parts = []
                for cat, count in cats.items():
                    icon = CATEGORY_ICONS.get(cat, "")
                    summary_parts.append(f"{icon} {cat}: **{count}**")
                st.markdown(" | ".join(summary_parts))

                # Contemporary references table (most relevant for threading)
                contemp = llm_refs[llm_refs["category"].str.startswith("contemporary")]
                if len(contemp) > 0:
                    st.markdown("**Contemporary references:**")
                    for _, ref in contemp.iterrows():
                        rtype = ref.get("reference_type", "")
                        target = ref.get("target_name", "")
                        context = restore_final_forms(str(ref.get("context", "")))
                        conf = ref.get("confidence", 0)
                        st.markdown(
                            f"- **{target}** ({ref['category'].replace('contemporary_', '')}, "
                            f"_{rtype}_, conf={conf:.0%}): {context}"
                        )

                # Expandable sections for other categories
                other_refs = llm_refs[~llm_refs["category"].str.startswith("contemporary")]
                if len(other_refs) > 0:
                    with st.expander(f"Biblical, Talmudic & other references ({len(other_refs)})"):
                        for _, ref in other_refs.iterrows():
                            icon = CATEGORY_ICONS.get(ref.get("category", ""), "")
                            target = ref.get("target_name", "")
                            context = restore_final_forms(str(ref.get("context", "")))
                            st.markdown(f"- {icon} **{target}** ({ref.get('reference_type', '')}): {context}")

        # Mechanical extraction results
        mech_refs = doc_refs[doc_refs["method"].str.startswith("mechanical")]
        if len(mech_refs) > 0:
            with st.expander(f"Mechanical extraction ({len(mech_refs)} hits)"):
                for method, group in mech_refs.groupby("method"):
                    st.markdown(f"**{method.replace('mechanical_', '').title()}** ({len(group)}):")
                    for _, ref in group.head(10).iterrows():
                        raw = restore_final_forms(str(ref.get("raw_text", "")))
                        st.markdown(f"- {raw[:200]}")
                    if len(group) > 10:
                        st.markdown(f"_... and {len(group) - 10} more_")

# --- Model-Suggested Polemic Vocabulary ---

if vocab_df is not None:
    doc_vocab = vocab_df[vocab_df["doc_id"] == doc_id]
    if len(doc_vocab) > 0:
        st.subheader("Suggested Polemic Markers")
        vrow = doc_vocab.iloc[0]
        markers = vrow["polemic_markers"]
        explanations = vrow["marker_explanations"]

        if markers:
            # Initialize approvals for this doc if needed
            if doc_id not in st.session_state["vocab_approvals"]:
                st.session_state["vocab_approvals"][doc_id] = {}

            marker_cols = st.columns(min(len(markers), 5))
            for j, (marker, expl) in enumerate(zip(markers, explanations)):
                with marker_cols[j % len(marker_cols)]:
                    status = st.session_state["vocab_approvals"].get(doc_id, {}).get(marker, "pending")
                    status_icon = {"approved": "✅", "rejected": "❌", "pending": "⬜"}[status]
                    st.markdown(
                        f'<div dir="rtl" style="text-align:center; font-size:18px; '
                        f'border:1px solid #ddd; border-radius:8px; padding:8px; margin:4px;">'
                        f'{status_icon} <b>{restore_final_forms(marker)}</b></div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(expl)
                    bcol1, bcol2 = st.columns(2)
                    with bcol1:
                        if st.button("✅", key=f"approve_{doc_id}_{j}"):
                            st.session_state["vocab_approvals"].setdefault(doc_id, {})[marker] = "approved"
                            st.rerun()
                    with bcol2:
                        if st.button("❌", key=f"reject_{doc_id}_{j}"):
                            st.session_state["vocab_approvals"].setdefault(doc_id, {})[marker] = "rejected"
                            st.rerun()

# --- Annotation panel ---

st.subheader("Annotation")

LABELS = ["explicit polemic", "implicit polemic", "meta-polemic (descriptive)", "non-polemic"]
current_label = st.session_state["annotations"].get(doc_id, None)
default_idx = LABELS.index(current_label) if current_label in LABELS else 0

label = st.radio(
    "Label this text:",
    LABELS,
    index=default_idx,
    key=f"label_{doc_id}",
    horizontal=True,
)

# Reviewer comment
current_comment = st.session_state["comments"].get(doc_id, "")
comment = st.text_area(
    "Reviewer comment (reasoning, notes, disagreements with models):",
    value=current_comment,
    key=f"comment_{doc_id}",
    height=100,
)

if st.button("Save annotation", key=f"save_{doc_id}"):
    st.session_state["annotations"][doc_id] = label
    st.session_state["comments"][doc_id] = comment
    st.success(f"Saved: {doc_id} → {label}")

# --- Keyword suggestion ---

st.subheader("Keyword Suggestions")

new_kw = st.text_input("Suggest a Hebrew polemic keyword:", key=f"kw_{doc_id}")
if st.button("Add keyword", key=f"addkw_{doc_id}") and new_kw:
    st.session_state["keyword_suggestions"].append({
        "doc_id": doc_id, "keyword": new_kw, "source": "human"
    })
    st.success(f'Keyword "{new_kw}" added!')

if st.session_state["keyword_suggestions"]:
    kw_df = pd.DataFrame(st.session_state["keyword_suggestions"])
    doc_kws = kw_df[kw_df["doc_id"] == doc_id]
    if len(doc_kws) > 0:
        st.write("Keywords for this text:", ", ".join(doc_kws["keyword"].tolist()))

# --- Export ---

st.sidebar.header("Export")

if st.sidebar.button("Download annotations CSV"):
    if st.session_state["annotations"]:
        ann_df = pd.DataFrame([
            {"doc_id": k, "label": v, "comment": st.session_state["comments"].get(k, "")}
            for k, v in st.session_state["annotations"].items()
        ])
        csv = ann_df.to_csv(index=False)
        st.sidebar.download_button(
            "Download", csv, file_name="annotations.csv", mime="text/csv"
        )
    else:
        st.sidebar.warning("No annotations yet.")

if st.sidebar.button("Download keywords CSV"):
    # Combine human suggestions + approved model vocabulary
    all_keywords = list(st.session_state["keyword_suggestions"])
    for did, approvals in st.session_state["vocab_approvals"].items():
        for marker, status in approvals.items():
            all_keywords.append({
                "doc_id": did, "keyword": marker,
                "source": f"model_{status}",
            })
    if all_keywords:
        kw_df = pd.DataFrame(all_keywords)
        csv = kw_df.to_csv(index=False)
        st.sidebar.download_button(
            "Download", csv, file_name="keyword_suggestions.csv", mime="text/csv"
        )
    else:
        st.sidebar.warning("No keyword suggestions yet.")
