import os
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

# Define paths
DOCUMENTS_PATH = "C:/Users/PC/Desktop/MentalHealth/knowledge_base/documents/"
FAISS_INDEX_PATH = "C:/Users/PC/Desktop/MentalHealth/knowledge_base/faiss_index/"

# Ensure FAISS index directory exists
os.makedirs(FAISS_INDEX_PATH, exist_ok=True)

# Load SentenceTransformer Model (FREE)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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

# Print the first few documents for verification
print("\n🔎 Verifying document structure before storing in FAISS...")
for i, doc in enumerate(chunks[:5]):  # Print only first 5 docs
    print(f"📄 Document {i+1}: {doc.page_content[:200]}...")  # Print first 200 chars

# Ensure all documents are properly formatted
if any(not isinstance(doc, Document) for doc in chunks):
    raise ValueError("❌ Some documents are not in the correct `Document` format!")

# Ensure documents are not empty
if any(not doc.page_content.strip() for doc in chunks):
    raise ValueError("❌ Some documents are empty! Check data loading.")

print(f"✅ All {len(chunks)} documents are correctly formatted and ready for FAISS storage.")

# Create FAISS index using LangChain
vector_store = FAISS.from_documents(chunks, embedding_model)

# Save FAISS index properly
vector_store.save_local(FAISS_INDEX_PATH)

print(f"✅ FAISS Knowledge Base Created! Total Chunks: {len(chunks)}")
