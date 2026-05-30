from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import pickle


def create_medical_knowledge_base():
    print("📥 Loading medical dataset...")
    dataset = load_dataset("ruslanmv/ai-medical-chatbot")
    print(f"✅ Dataset loaded: {len(dataset['train'])} examples")

    knowledge_base = []
    for example in dataset['train']:
        question = example['Patient']
        answer = example['Doctor']
        intent = example['Description']
        category = "medical_consultation"

        # --- Cleaning step ---
        answer = answer.replace("\n", " ").replace("\t", " ")
        answer = answer.replace("...", ".").strip()
        question = question.replace("\n", " ").strip()

        for char in ["{{", "}}", "[", "]"]:
            question = question.replace(char, "")
            answer = answer.replace(char, "")

        # Shortened answer for context injection (first 40 words only)
        short_answer = " ".join(answer.split()[:40])

        knowledge_base.append({
            "question": question,
            "answer": short_answer,
            "intent": intent,
            "category": category
        })

    print(f"✅ Knowledge base created with {len(knowledge_base)} entries")
    return knowledge_base


def create_embeddings(knowledge_base, model_name):
    print("🔄 Loading sentence transformer model...")
    model = SentenceTransformer(model_name)

    print("🔄 Creating embeddings...")
    questions = [item['question'] for item in knowledge_base]
    embeddings = model.encode(questions, show_progress_bar=True, normalize_embeddings=True)
    return embeddings, model


def create_faiss_index(embeddings):
    print("📌 Creating FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # inner product similarity
    index.add(embeddings.astype('float32'))
    return index


def save_files(knowledge_base, index, model_name):
    print("💾 Saving knowledge base and index...")
    with open('knowledge_base.pkl', 'wb') as f:
        pickle.dump(knowledge_base, f)
    faiss.write_index(index, 'medical_index.faiss')
    with open('model_name.txt', 'w') as f:
        f.write(model_name)

    print("✅ Knowledge base created successfully!")
    print("Files created:")
    print("- knowledge_base.pkl")
    print("- medical_index.faiss")
    print("- model_name.txt")


def main():
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    knowledge_base = create_medical_knowledge_base()
    embeddings, model = create_embeddings(knowledge_base, model_name)
    index = create_faiss_index(embeddings)
    save_files(knowledge_base, index, model_name)


if __name__ == "__main__":
    main()
