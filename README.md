🧠 MentalHealthAI
MentalHealthAI is a conversational AI designed to evaluate mental well-being based on speech and text analysis. It integrates Speech Emotion Recognition (SER), Sentiment Analysis, and a FAISS-based Retrieval-Augmented Generation (RAG) system to generate responses dynamically.

🌟 Features
✅ Speech Emotion Recognition using MFCCs, Chroma Features, Pitch, and Energy
✅ Text Sentiment Analysis to determine emotional intent
✅ FAISS-based RAG for knowledge retrieval and response generation
✅ Real-time Speech-to-Text (STT) & Text-to-Speech (TTS)
✅ Dynamic Response Generation based on detected emotions
✅ Behavioral Trend Analysis for detecting long-term mood changes

📌 Project Structure
MentalHealthAI/
│── knowledge_base/          # Stores documents for FAISS knowledge retrieval
│── models/                  # Contains models for speech emotion recognition & STT
│   │── emotion_detection.py  # Extracts MFCCs, Chroma, Pitch, and Energy features
│   │── speech-to-text.py     # Converts speech to text using STT
│── services/                # Handles RAG, knowledge retrieval, and embeddings
│   │── prepare_kb.py         # Prepares FAISS knowledge base
│   │── knowledge_retrieval.py # Retrieves relevant responses from FAISS
│   │── rag_pipeline.py       # Manages response generation
│── static/                  # Stores images, audio, and reference files
│── requirements.txt         # List of required dependencies
│── main.py                  # Entry point for the application
│── README.md                # Project documentation


🔧 Installation
1️⃣ Clone the repository
git clone https://github.com/brandonm0401/MentalHealthAI.git
cd MentalHealthAI
2️⃣ Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3️⃣ Install dependencies
pip install -r requirements.txt


🚀 Usage
1️⃣ Prepare the FAISS knowledge base
python services/prepare_kb.py
2️⃣ Run the AI model
python main.py

🏗️ Technologies Used
FastAPI - Backend API
FAISS - Vector search for knowledge retrieval
Scikit-learn - Machine Learning (SVM for emotion classification)
SpeechRecognition - Converts speech to text
Librosa - Extracts MFCC, Chroma, Pitch, and Energy features
LangChain - Handles RAG-based retrieval and response generation

💡 Future Enhancements
🔹 Improve the SVM model for speech emotion recognition
🔹 Implement Transformer-based NLP for better sentiment analysis
🔹 Optimize TTS responses for more natural conversations
🤝 Contributing
Feel free to fork this repository, create a new branch, and submit a pull request!

📜 License
This project is open-source under the MIT License.

