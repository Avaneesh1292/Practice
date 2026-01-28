import config 

from load_data import load_and_chunk_pdf
from embeddings_store import build_vector_store
from retriever import HybridRetriever
from decision_model import decide_next_step
from rag_pipeline import run_rag

chunks = load_and_chunk_pdf()
store, embeddings = build_vector_store(chunks)
retriever = HybridRetriever(store, embeddings, chunks)

print("\n---Model is ready ---\n")

while True:
    user_q = input("Ask a question (or exit): ")
    if user_q.lower() == "exit":
        break

    decision = decide_next_step(user_q)

    if decision.tool_calls:
        tool_call = decision.tool_calls[0]
        reason = tool_call["args"]["reason"]

        print("\nClarification needed:")
        print(reason)

        extra = input("Additional context: ")
        user_q = user_q + " " + extra

    answer = run_rag(user_q, retriever)
    print("\nANSWER:\n", answer, "\n")