import os
import faiss
import pickle
import cohere
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

FAISS_INDEX_PATH = "C:/Users/PC/Desktop/MentalHealth/knowledge_base/faiss_index/"
FAISS_INDEX_FILE = os.path.join(FAISS_INDEX_PATH, "index.faiss")
TEXTS_FILE = os.path.join(FAISS_INDEX_PATH, "index.pkl")

if not (os.path.exists(FAISS_INDEX_FILE) and os.path.exists(TEXTS_FILE)):
    raise FileNotFoundError("❌ FAISS index or metadata not found. Run `prepare_kb.py` first.")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
try:
    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH,
        embedding_model,
        index_name="index",
        allow_dangerous_deserialization=True
    )
except Exception as e:
    raise ValueError(f"❌ FAISS loading failed: {e}")

print("🔍 Checking FAISS index structure...")

# 🔹 Ensure FAISS index is correctly structured
if not hasattr(vector_store, "docstore"):
    raise ValueError("⚠️ FAISS index loaded, but `docstore` attribute is missing!")

if isinstance(vector_store.docstore, list):
    raise ValueError("⚠️ FAISS `docstore` is incorrectly loaded as a list instead of a document store!")

if not vector_store.docstore:
    raise ValueError("⚠️ FAISS index is empty! Try rebuilding it with `prepare_kb.py`.")

print(f"✅ FAISS Index Loaded Successfully! Total documents: {len(vector_store.index_to_docstore_id)}")

with open(TEXTS_FILE, "rb") as f:
    texts, doc_ids = pickle.load(f)

index_id_map = faiss.read_index(FAISS_INDEX_FILE)
faiss_index = index_id_map

retriever = vector_store.as_retriever()
COHERE_API_KEY = "YOUR_COHERE_API_KEY"
co = cohere.Client(COHERE_API_KEY)

EMOTION_TEMPLATES = {
    "Happy": "That's wonderful! What made your day so great?",
    "Sad": "I'm really sorry you're feeling this way. Want to talk about it?",
    "Angry": "I see that you're upset. What happened?",
    "Neutral": "Got it. How has your day been so far?",
}

GREETINGS = {"hi", "hello", "hey"}

def generate_text(prompt):
    #Use Cohere API for text generation.
    response = co.generate(
        model="command",
        prompt=prompt,
        max_tokens=100,
        temperature=0.7
    )
    return response.generations[0].text.strip()

def generate_response(user_input, emotion):
    #Retrieve relevant knowledge and generate an emotion-aware response.
    user_input_lower = user_input.lower().strip()
    if user_input_lower in GREETINGS:
        return "Hi! I am a virtual mental health evaluator. How are you feeling today?"

    retrieved_docs = retriever.invoke(user_input)
    filtered_context = [doc.page_content.split("Response:")[-1].strip() for doc in retrieved_docs]
    context = "\n".join(filtered_context) if filtered_context else "I'm here to listen. How can I support you?"
    prompt = (
        "You are a Mental Health Evaluator AI designed to assess users' emotions and respond appropriately."
        "Your job is to carefully analyze both the user's words and detected emotion to generate a thoughtful and empathetic response.\n\n"
        "Guidelines:\n"
        "- Strictly consider the detected emotion while responding.\n"
        "- Use empathetic, engaging, and supportive language.\n"
        "- If the user is feeling negative (Sad, Angry, Stressed, etc.), respond with concern and ask follow-up questions.\n"
        "- If the user is neutral or positive, engage in a friendly and supportive manner.\n\n"
        "User's Input: {user_input}\n"
        "Detected Emotion: {emotion}\n"
        "Context from Knowledge Base: {context}\n"
        "Your Response:"
    ).format(user_input=user_input, emotion=emotion, context=context)

    response_text = generate_text(prompt)
    return response_text

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit","Bye","Thank you","Thanks"]:
            print("AI: Take care! Feel free to talk anytime. 😊")
            break
        emotion = "Sad"
        response = generate_response(user_input, emotion)
        print(f"AI: {response}")
