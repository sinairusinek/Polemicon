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


def load_classifications():
    path = os.path.join(DATA_DIR, "pilot_classifications.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    st.warning(f"Classifications not found at {path}")
    return None


def load_disagreements():
    path = os.path.join(DATA_DIR, "pilot_disagreements.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    st.warning(f"Disagreements not found at {path}")
    return None


def load_references():
    path = os.path.join(DATA_DIR, "pilot_references.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    st.warning(f"References not found at {path}")
    return None


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
    st.warning(f"Vocab not found at {path}")
    return None


# --- App config ---

st.set_page_config(page_title="Polemicon Annotation", layout="wide")
st.title("Polemicon Annotation & Keyword Discovery")

df = load_pilot_sample()
clf_df = load_classifications()
disagree_df = load_disagreements()
refs_df = load_references()
vocab_df = load_vocab()
cluster_labels_df = load_cluster_labels()

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

# --- Navigation ---

st.sidebar.header("Navigation")

if len(filtered) == 0:
    st.info("No texts match the current filters.")
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
row1[2].metric("Keyword Score", f"{row['polemic_score']:.3f}")
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
