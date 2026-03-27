"""
streamlit_app.py - Annotation and keyword suggestion app for Polemicon Phase B

Features:
- Multi-class labeling: explicit, implicit, meta, non-polemic
- Human keyword suggestion UI
- Automated keyword suggestion via Gemini (for polemic labels)
- Logging of all suggestions
"""
import streamlit as st
import pandas as pd
from typing import List

# Placeholder for Gemini integration
def gemini_suggest_keywords(text: str) -> List[str]:
    # TODO: Replace with real Gemini API call
    return ["דוגמה", "פולמוס", "מחלוקת"]

# Load sample data (replace with real corpus slice)
def load_samples():
    # For demo: 5 sample texts
    return pd.DataFrame({
        'doc_id': [f'doc_{i}' for i in range(5)],
        'text': [f'טקסט לדוגמה {i} עם פולמוס' for i in range(5)]
    })

st.title("Polemicon Annotation & Keyword Discovery")
samples = load_samples()

if 'annotations' not in st.session_state:
    st.session_state['annotations'] = {}
if 'keyword_suggestions' not in st.session_state:
    st.session_state['keyword_suggestions'] = []

for idx, row in samples.iterrows():
    st.header(f"Text {idx+1} (ID: {row['doc_id']})")
    st.write(row['text'])
    label = st.radio(
        "Label:",
        ["explicit polemic", "implicit polemic", "meta-polemic (descriptive)", "non-polemic"],
        key=f'label_{row["doc_id"]}'
    )
    st.session_state['annotations'][row['doc_id']] = label
    # Human keyword suggestion
    new_kw = st.text_input("Suggest a Hebrew polemic keyword (optional):", key=f'kw_{row["doc_id"]}')
    if st.button("Add keyword", key=f'addkw_{row["doc_id"]}') and new_kw:
        st.session_state['keyword_suggestions'].append({'doc_id': row['doc_id'], 'keyword': new_kw, 'source': 'human'})
        st.success(f'Keyword "{new_kw}" added!')
    # Automated keyword suggestion (Gemini)
    if label != "non-polemic":
        auto_kws = gemini_suggest_keywords(row['text'])
        if st.button("Accept Gemini suggestions", key=f'gemini_{row["doc_id"]}'):
            for kw in auto_kws:
                st.session_state['keyword_suggestions'].append({'doc_id': row['doc_id'], 'keyword': kw, 'source': 'gemini'})
            st.success(f'Gemini keywords added: {", ".join(auto_kws)}')
        st.info(f'Gemini suggests: {", ".join(auto_kws)}')

# Export results
if st.button("Export annotations & keywords"):
    ann_df = pd.DataFrame([{'doc_id': k, 'label': v} for k, v in st.session_state['annotations'].items()])
    kw_df = pd.DataFrame(st.session_state['keyword_suggestions'])
    ann_df.to_csv('annotations.csv', index=False)
    kw_df.to_csv('keyword_suggestions.csv', index=False)
    st.success('Exported annotations.csv and keyword_suggestions.csv')
