**🩺 Medical RAG AI Chatbot**

An AI-powered Medical Question Answering System built using Retrieval-Augmented Generation (RAG), FAISS, Sentence Transformers, and Ollama LLMs.

The chatbot retrieves relevant medical information from a knowledge base and generates context-aware responses, reducing hallucinations and improving answer reliability.

**🚀 Features**
Retrieval-Augmented Generation (RAG)
Semantic Search using FAISS
Sentence Embeddings with Sentence Transformers
Local LLM inference using Ollama
Context-aware Medical Question Answering
Streamlit Web Interface
Fast and Lightweight Deployment

**🏗️ System Architecture**

User Query
     │
     ▼
Embedding Model
     │
     ▼
FAISS Vector Search
     │
Retrieved Context
     │
     ▼
Ollama LLM
     │
     ▼
Generated Response

**🛠️ Technologies Used**
Python
Streamlit
FAISS
Ollama
Sentence Transformers
Retrieval-Augmented Generation (RAG)
NLP

**📂 Dataset**

The chatbot uses medical documents that are:

Embedded using Sentence Transformers
Indexed in FAISS
Retrieved based on semantic similarity


**⚙️ Installation**

git clone https://github.com/AyehsaFarooq/medical-rag-chatbot

cd medical-rag-chatbot

pip install -r requirements.txt

**▶️ Run Application**

streamlit run app.py

**💡 Example Questions**
What are the symptoms of diabetes?
How does hypertension affect the body?
What are common causes of anemia?
Explain asthma in simple terms.

**📸 Screenshots**

<img width="601" height="677" alt="image" src="https://github.com/user-attachments/assets/b06ab41d-9d8d-45dc-96be-4630ec87c06f" />

**🔮 Future Improvements**
Multi-document retrieval
Medical PDF ingestion
Chat history memory
Source citation support
Hybrid Search (BM25 + Vector Search)
Evaluation Metrics Dashboard

**⚠️ Disclaimer**

This chatbot is intended for educational and research purposes only and should not be considered professional medical advice.

**👩‍💻 Author**

**Ayesha Farooq**

**AI | NLP | Machine Learning | LLM Applications**
