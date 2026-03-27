"""
streamlit_app.py - Polemicon Annotation & Keyword Discovery

Real corpus-backed annotation tool for Phase B.2 pilot.
- Loads 200-text stratified pilot sample
- Multi-class polemic labeling
- Metadata display (source, year, keyword score, cluster)
- Human keyword suggestion
- CSV export of annotations
"""
import streamlit as st
import pandas as pd
import os

# --- Data loading ---

@st.cache_data
def load_pilot_sample():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "pilot_sample.parquet")
    if not os.path.exists(path):
        st.error("Pilot sample not found. Run src/sample_pilot.py first.")
        st.stop()
    return pd.read_parquet(path)


# --- App config ---

st.set_page_config(page_title="Polemicon Annotation", layout="wide")
st.title("Polemicon Annotation & Keyword Discovery")

df = load_pilot_sample()

# --- Session state ---

if "annotations" not in st.session_state:
    st.session_state["annotations"] = {}
if "keyword_suggestions" not in st.session_state:
    st.session_state["keyword_suggestions"] = []
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

filtered = filtered.reset_index(drop=True)

st.sidebar.markdown(f"**{len(filtered)}** texts match filters")
st.sidebar.markdown(f"**{len(st.session_state['annotations'])}** / {len(df)} annotated")

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

# Metadata bar
meta_cols = st.columns(5)
meta_cols[0].metric("Source", row["source"])
meta_cols[1].metric("Year", int(row["year"]) if pd.notna(row.get("year")) else "N/A")
meta_cols[2].metric("Keyword Score", f"{row['polemic_score']:.3f}")
if "cluster_id" in row and pd.notna(row.get("cluster_id")):
    meta_cols[3].metric("Cluster", int(row["cluster_id"]))
else:
    meta_cols[3].metric("Cluster", "N/A")
meta_cols[4].metric("Doc ID", doc_id)

# Text display
st.subheader("Text")
st.markdown(
    f'<div dir="rtl" style="text-align:right; font-size:16px; line-height:1.8; '
    f'max-height:400px; overflow-y:auto; border:1px solid #ddd; padding:12px; '
    f'border-radius:4px;">{row["text"][:5000]}{"..." if len(str(row["text"])) > 5000 else ""}</div>',
    unsafe_allow_html=True,
)

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

if st.button("Save annotation", key=f"save_{doc_id}"):
    st.session_state["annotations"][doc_id] = label
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
            {"doc_id": k, "label": v}
            for k, v in st.session_state["annotations"].items()
        ])
        csv = ann_df.to_csv(index=False)
        st.sidebar.download_button(
            "Download", csv, file_name="annotations.csv", mime="text/csv"
        )
    else:
        st.sidebar.warning("No annotations yet.")

if st.sidebar.button("Download keywords CSV"):
    if st.session_state["keyword_suggestions"]:
        kw_df = pd.DataFrame(st.session_state["keyword_suggestions"])
        csv = kw_df.to_csv(index=False)
        st.sidebar.download_button(
            "Download", csv, file_name="keyword_suggestions.csv", mime="text/csv"
        )
    else:
        st.sidebar.warning("No keyword suggestions yet.")
