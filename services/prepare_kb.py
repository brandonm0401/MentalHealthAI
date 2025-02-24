import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Define paths
DOCUMENTS_PATH = "C:/Users/PC/Desktop/MentalHealthAI/knowledge_base/documents/"
FAISS_INDEX_PATH = "C:/Users/PC/Desktop/MentalHealthAI/knowledge_base/faiss_index/"

# Ensure the FAISS index directory exists
os.makedirs(FAISS_INDEX_PATH, exist_ok=True)

# Load SentenceTransformer Model (FREE)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Function to generate embeddings
def get_embedding(text):
    return embedding_model.encode(text)

# Load text documents
documents = []
for file in os.listdir(DOCUMENTS_PATH):
    if file.endswith(".txt"):
        file_path = os.path.join(DOCUMENTS_PATH, file)
        loader = TextLoader(file_path, encoding="utf-8")
        documents.extend(loader.load())

# Split texts into smaller chunks for better retrieval
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# Convert text chunks into embeddings
texts = [chunk.page_content for chunk in chunks]
embeddings = np.array([get_embedding(text) for text in texts], dtype=np.float32)  # Convert to NumPy array

# Create FAISS index
dimension = embeddings.shape[1]  # Get embedding size
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embeddings)

# Save FAISS index
faiss.write_index(faiss_index, os.path.join(FAISS_INDEX_PATH, "faiss_index.bin"))

# Save text chunks for retrieval
with open(os.path.join(FAISS_INDEX_PATH, "texts.pkl"), "wb") as f:
    pickle.dump(texts, f)

print("✅ FAISS Knowledge Base Created & Saved with SentenceTransformers!")
