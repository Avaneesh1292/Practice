import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# ===============================================
# CONFIG - LOCAL
# ===============================================
MODEL_NAME = "google/flan-t5-base"  # ← FREE, smaller & faster

# ===============================================
# STEP 1: Load Knowledge Base
# ===============================================
def load_knowledge_base(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        return lines
    except FileNotFoundError:
        print(f"Error: '{filename}' not found!")
        return []

print("1. Loading knowledge base...")
knowledge_base = load_knowledge_base("knowledge_base.txt")
print(f"   Loaded {len(knowledge_base)} lines.")

# ===============================================
# STEP 2: Retrieval (FREE SentenceTransformer)
# ===============================================
print("2. Loading retrieval model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # free

corpus_embeddings = embedder.encode(knowledge_base, convert_to_tensor=True)

def retrieve_info(query):
    query_emb = embedder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, corpus_embeddings, top_k=3)
    results = [knowledge_base[h["corpus_id"]] for h in hits[0]]
    return " ".join(results)

# ===============================================
# STEP 3: GENERATION (LOCAL FLAN-T5)
# ===============================================
print(f"3. Loading local generator: {MODEL_NAME}")
generator = pipeline(
    "text2text-generation",
    model=MODEL_NAME,
    device_map="auto"   # uses GPU if available
)

def generate_answer(question, context):
    prompt = f"""
    Answer using ONLY the context below.
    If answer is not found, say: "I don't have enough information."

    Context: {context}

    Question: {question}

    Answer:
    """

    output = generator(prompt, max_new_tokens=150)
    return output[0]["generated_text"].strip()

# ===============================================
# STEP 4: Chat Loop
# ===============================================
print("\n==========================================")
print("RAG Chat Ready!")
print("Type 'exit' to quit.")
print("==========================================")

while True:
    user_query = input("\nAsk a question: ")
    
    if user_query.lower() in ["exit", "quit"]:
        print("Exitting chat.")
        break
    
    context = retrieve_info(user_query)
    answer = generate_answer(user_query, context)
    
    print("\nAI Answer:", answer)
