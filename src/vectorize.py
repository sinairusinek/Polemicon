"""
vectorize.py - Vectorization for the Polemicon project

- Dense: multilingual-e5-large embeddings (512-token window, 256 stride, mean pool)
- Sparse: char n-gram TF-IDF (3-5), word n-gram TF-IDF (1-2)
- Only chunk texts >512 tokens
"""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import os

def chunk_text(text, window=512, stride=256):
    tokens = text.split()  # crude tokenization; replace with model tokenizer for production
    if len(tokens) <= window:
        return [' '.join(tokens)]
    chunks = []
    for i in range(0, len(tokens), stride):
        chunk = tokens[i:i+window]
        if len(chunk) < window and i != 0:
            break
        chunks.append(' '.join(chunk))
    return chunks

def mean_pool(vectors):
    return np.mean(vectors, axis=0)

def main():
    corpus = pd.read_parquet('corpus.parquet')
    model = SentenceTransformer('intfloat/multilingual-e5-large')
    dense_vectors = []
    doc_ids = []
    for idx, row in corpus.iterrows():
        text = row['text']
        chunks = chunk_text(text)
        chunk_vecs = model.encode(chunks, batch_size=32, show_progress_bar=False)
        if len(chunk_vecs) > 1:
            vec = mean_pool(chunk_vecs)
        else:
            vec = chunk_vecs[0]
        dense_vectors.append(vec)
        doc_ids.append(row['doc_id'])
    dense_vectors = np.stack(dense_vectors)
    np.save('dense_vectors.npy', dense_vectors)
    with open('doc_ids.txt', 'w') as f:
        for doc_id in doc_ids:
            f.write(f"{doc_id}\n")
    # Build FAISS index
    index = faiss.IndexFlatIP(dense_vectors.shape[1])
    index.add(dense_vectors.astype('float32'))
    faiss.write_index(index, 'dense_faiss.index')
    # Sparse vectorization
    char_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), max_features=50000)
    word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=30000)
    char_tfidf = char_vectorizer.fit_transform(corpus['text'])
    word_tfidf = word_vectorizer.fit_transform(corpus['text'])
    from scipy import sparse
    sparse.save_npz('char_tfidf.npz', char_tfidf)
    sparse.save_npz('word_tfidf.npz', word_tfidf)
    print('Vectorization complete.')

if __name__ == '__main__':
    main()
