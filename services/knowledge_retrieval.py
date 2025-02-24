import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Paths
FAISS_INDEX_PATH = "knowledge_base/faiss_index/"
INDEX_FILE = "C:/Users/PC/Desktop/MentalHealthAI/knowledge_base/faiss_index/faiss_index.bin"
TEXTS_FILE = "C:/Users/PC/Desktop/MentalHealthAI/knowledge_base/faiss_index/texts.pkl"

# Check if FAISS index file exists
if not os.path.exists(INDEX_FILE):
    raise FileNotFoundError(f"FAISS index file not found at {INDEX_FILE}. Run prepare_kb.py first.")

# Load FAISS index
faiss_index = faiss.read_index(INDEX_FILE)

# Load texts associated with embeddings
if os.path.exists(TEXTS_FILE):
    with open(TEXTS_FILE, "rb") as f:
        texts = pickle.load(f)
else:
    raise FileNotFoundError(f"Text data file {TEXTS_FILE} not found. Run prepare_kb.py again.")

# Load SentenceTransformer model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def retrieve_response(query, top_k=3):
    query_embedding = np.array([embedding_model.encode(query)], dtype=np.float32)
    _, indices = faiss_index.search(query_embedding, top_k)
    
    # Get matching responses
    results = [texts[i] for i in indices[0] if i < len(texts)]
    return results

# Example Query
query = "I had a really good day today."
responses = retrieve_response(query)
print("🔍 Retrieved Responses:", responses)
