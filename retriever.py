import numpy as np
from rank_bm25 import BM25Okapi
from config import SEMANTIC_K, BM25_K, RERANK_K

class HybridRetriever:
    def __init__(self, vector_store, embeddings, chunks):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.chunks = chunks
        self.bm25 = BM25Okapi(
            [c.page_content.lower().split() for c in chunks]
        )

    def retrieve(self, query):
        semantic = self.vector_store.similarity_search(query, k=SEMANTIC_K)

        scores = self.bm25.get_scores(query.lower().split())
        top = np.argsort(scores)[::-1][:BM25_K]
        bm_docs = [self.chunks[i] for i in top]

        merged = {d.page_content: d for d in semantic + bm_docs}
        return list(merged.values())

    def rerank(self, query, docs):
        q_emb = self.embeddings.embed_query(query)
        d_embs = self.embeddings.embed_documents(
            [d.page_content for d in docs]
        )
        scores = np.dot(d_embs, q_emb)
        idx = np.argsort(scores)[::-1][:RERANK_K]
        return [docs[i] for i in idx]