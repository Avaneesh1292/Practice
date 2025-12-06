import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# ==================================================
# STEP 1: The Knowledge Base
# ==================================================
knowledge_base = [
    "The jaguar is a large cat species and the only living member of the genus Panthera native to the Americas.",
    "Jaguars are excellent swimmers and often hunt in water.",
    "The cheetah is the fastest land animal, capable of reaching speeds up to 75 mph.",
    "Lions are social cats and live in groups called prides."
]

# ==================================================
# STEP 2: The Retrieval System (The "Librarian")
# ==================================================
print("1. Loading Retrieval Model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
corpus_embeddings = embedder.encode(knowledge_base, convert_to_tensor=True)

def retrieve_info(query):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=1)
    best_hit = hits[0][0]
    return knowledge_base[best_hit['corpus_id']]

# ==================================================
# STEP 3: The Generator (Smarter & Stricter)
# ==================================================
print("2. Loading Generation Model (google/flan-t5-base)...")

# We switch from 'small' to 'base' (250MB). It is much smarter.
generator = pipeline(
    "text2text-generation", 
    model="google/flan-t5-base" 
)

def generate_answer(question, context):
    # STRICTER PROMPT: We explicitly tell it what to do if it doesn't know.
    prompt = f"""
    Use the following context to answer the question. 
    Context: {context}
    Question: {question}
    
    If the answer is not in the context, say "I don't know based on this context".
    Answer:
    """
    
    # We generated slightly more text (max_new_tokens) to let it explain itself
    result = generator(prompt, max_new_tokens=100)
    return result[0]['generated_text']

# ==================================================
# STEP 4: Running the Pipeline
# ==================================================
user_query = "Do jaguars hunt in prides?"

print(f"\nUser asks: '{user_query}'")

# A. Retrieve
context = retrieve_info(user_query)
print(f"--> Retrieved Fact: '{context}'")

# B. Generate
print("--> AI is thinking...")
final_answer = generate_answer(user_query, context)

print(f"\nAI Answer: {final_answer}")