import os
import pickle
import streamlit as st
import faiss
import requests
import json
from sentence_transformers import SentenceTransformer

# Ollama endpoint
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"  
OLLAMA_MODEL = "llama3.2:latest"  # change to gemma:2b or llama3 if you prefer

# -------------------- Load System --------------------
@st.cache_resource
def load_system():
    try:
        required_files = ['knowledge_base.pkl', 'medical_index.faiss', 'model_name.txt']
        for file in required_files:
            if not os.path.exists(file):
                st.error(f"Missing: {file}")
                return None, None, None

        # Load model
        with open('model_name.txt', 'r') as f:
            model_name = f.read().strip()
        model = SentenceTransformer(model_name)

        # Load knowledge base
        with open('knowledge_base.pkl', 'rb') as f:
            knowledge_base = pickle.load(f)

        # Load FAISS index
        index = faiss.read_index('medical_index.faiss')

        return model, knowledge_base, index
    except Exception as e:
        st.error(f"Error while loading system: {str(e)}")
        return None, None, None

# -------------------- Ollama Query (Streaming) --------------------
def query_ollama_stream(prompt, model=OLLAMA_MODEL, temperature=0.7):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,  # Enable streaming
        "options": {
            "temperature": temperature,
            "top_p": 0.9
        }
    }
    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=300) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        if "response" in data:
                            yield data["response"]
                        if data.get("done", False):
                            break
                    except Exception:
                        continue
    except Exception as e:
        yield f"⚠️ Error querying local LLM: {e}"

def rewrite_query(query, chat_history):
    rewrite_prompt = f"""
You are a query rewriting assistant.

Given the conversation history and the latest user question,
rewrite the question into a complete standalone question.

Conversation History:
{chat_history}

Latest User Question:
{query}

Standalone Question:
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": rewrite_prompt,
                "stream": False
            },
            timeout=60
        )

        response.raise_for_status()

        rewritten = response.json().get("response", "").strip()

        return rewritten if rewritten else query

    except Exception:
        return query
    
def get_chat_history(messages, max_messages=6):
    history = []

    for msg in messages[-max_messages:]:
        history.append(
            f"{msg['role'].capitalize()}: {msg['content']}"
        )

    return "\n".join(history)
# -------------------- Context Retrieval --------------------
def retrieve_context(query, model, knowledge_base, index, top_k=3):
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)
    scores, indices = index.search(query_embedding.astype('float32'), top_k)

    context_chunks = []
    for idx in indices[0]:
        if idx < len(knowledge_base):
            kb_item = knowledge_base[idx]
            context_chunks.append(f"Q: {kb_item['question']}\nA: {kb_item['answer']}")
    return "\n\n".join(context_chunks)

# -------------------- RAG Pipeline --------------------
def get_rag_prompt(
    query,
    chat_history,
    model,
    knowledge_base,
    index,
    top_k=3
):
    context = retrieve_context(
        query,
        model,
        knowledge_base,
        index,
        top_k
    )

    prompt = f"""
You are a helpful and empathetic medical assistant.

Use the retrieved medical context whenever relevant.

If the answer is not available in the context,
use your general medical knowledge.

Always recommend consulting a healthcare professional
for diagnosis, treatment, or emergencies.

Keep answers concise (maximum 5 sentences).

Conversation History:
{chat_history}

Retrieved Context:
{context}

Current User Question:
{query}

Answer:
"""

    return prompt

# -------------------- Streamlit UI --------------------
def main():
    st.set_page_config(
        page_title="AI Medical Chatbot (RAG + LLM)",
        layout="centered"
    )

    st.title("🩺 AI Medical Chatbot (Local LLM)")
    st.caption(
        "⚠️ This chatbot provides general medical information only. "
        "Always consult a professional for medical advice."
    )

    model, knowledge_base, index = load_system()

    if not all([model, knowledge_base, index]):
        st.stop()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a medical question..."):

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Build history BEFORE current question
        chat_history = get_chat_history(
            st.session_state.messages
        )

        # Rewrite follow-up questions
        rewritten_query = rewrite_query(
            prompt,
            chat_history
        )

        # Debug panel
        st.sidebar.markdown("### Query Rewriting")
        st.sidebar.write("Original:", prompt)
        st.sidebar.write("Rewritten:", rewritten_query)

        # Build final prompt
        system_prompt = get_rag_prompt(
            rewritten_query,
            chat_history,
            model,
            knowledge_base,
            index
        )

        # Save user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        # Generate assistant response
        with st.chat_message("assistant"):

            placeholder = st.empty()
            full_response = ""

            for chunk in query_ollama_stream(system_prompt):
                full_response += chunk
                placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)

        # Save assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })

    # Clear chat button
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()
