# Embeddings + FAISS store

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def build_embeddings(texts, model_name: str):
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return np.array(emb, dtype="float32")

def build_faiss_index(vectors: np.ndarray):
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via normalized IP
    index.add(vectors)
    return index

def save_faiss(index, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)

def load_faiss(path: str):
    return faiss.read_index(path)

def load_docstore(jsonl_path: str):
    docs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs
