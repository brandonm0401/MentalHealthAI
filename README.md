### MentalHealthAI  

**MentalHealthAI** is a conversational AI designed to evaluate mental well-being based on speech and text analysis. It integrates **Speech Emotion Recognition (SER), Sentiment Analysis, and a FAISS-based Retrieval-Augmented Generation (RAG) system** to generate responses dynamically.  

---

### 🌟 Features  

- ✅ **Speech Emotion Recognition** using **MFCCs, Chroma Features, Pitch, and Energy**  
- ✅ **Text Sentiment Analysis** to determine emotional intent  
- ✅ **FAISS-based RAG** for knowledge retrieval and response generation  
- ✅ **Real-time Speech-to-Text (STT) & Text-to-Speech (TTS)**  
- ✅ **Dynamic Response Generation** based on detected emotions  
- ✅ **Behavioral Trend Analysis** for detecting long-term mood changes  

---

### 📌 Project Structure  

MentalHealthAI/
│── knowledge_base/          # Stores documents for FAISS knowledge retrieval
│── models/                  # Contains models for speech emotion recognition & STT
│   │── emotion_detection.py  # Extracts MFCCs, Chroma, Pitch, and Energy features
│   │── speech-to-text.py     # Converts speech to text using STT
│   │── text-to-speech.py     # Converts text to speech using TTS
│── services/                # Handles RAG, knowledge retrieval, and embeddings
│   │── prepare_kb.py         # Prepares FAISS knowledge base
│   │── knowledge_retrieval.py # Retrieves relevant responses from FAISS
│   │── rag_pipeline.py       # Manages response generation
│   │── trend_analysis.py     # Analyzes behavioral trends
│── static/                  # Stores images, audio, and reference files
│── requirements.txt         # List of required dependencies
│── main.py                  # Entry point for the application
│── README.md                # Project documentation
│── LICENSE                  # License agreement

---

### 🔧 Installation  

#### 1️⃣ Clone the Repository  

```bash
# Windows
git clone https://github.com/brandonm0401/MentalHealthAI.git
cd MentalHealthAI

# Mac / Linux
python3 -m venv venv
source venv/bin/activate

```
#### 2️⃣ Install Dependencies

```bash
python -m venv venv
venv\Scripts\activate
```
#### 3️⃣ Install Required Packages

```bash
pip install -r requirements.txt
```

---

### 🚀 Usage

#### 1️⃣ Prepare the FAISS Knowledge Base
```bash
python services/prepare_kb.py
```
#### 2️⃣ Run the Application
```bash
python main.py
```

---

### 🏗️ Technologies Used
- 🖥️ FastAPI - Backend API
- 🔍 FAISS - Vector search for knowledge retrieval
- 🗣️ SpeechRecognition - Converts speech to text
- 🎤 Librosa - Audio feature extraction
- 🤖 LangChain - RAG & knowledge retrieval
- 📝 Transformers - NLP & sentiment analysis

---

### 📜 License
- This project is licensed under the MIT License.