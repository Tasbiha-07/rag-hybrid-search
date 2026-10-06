from typing import List, Tuple
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS

def reciprocal_rank_fusion(
    bm25_results: List[Tuple[Document, float]],
    vector_results: List[Tuple[Document, float]],
    k_constant: int = 60,
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
    top_k: int = 4
) -> List[Tuple[Document, float]]:
    scores = {}
    doc_map = {}

    for rank, (doc, _) in enumerate(bm25_results):
        doc_id = doc.page_content
        doc_map[doc_id] = doc
        scores[doc_id] = scores.get(doc_id, 0.0) + bm25_weight * (1.0 / (rank + 1 + k_constant))

    for rank, (doc, _) in enumerate(vector_results):
        doc_id = doc.page_content
        doc_map[doc_id] = doc
        scores[doc_id] = scores.get(doc_id, 0.0) + vector_weight * (1.0 / (rank + 1 + k_constant))

    sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [(doc_map[doc_id], score) for doc_id, score in sorted_docs]

def hybrid_search(
    query: str,
    vector_store: FAISS,
    bm25: BM25Okapi,
    chunks: List[Document],
    top_k: int = 4,
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5
) -> List[Tuple[Document, float]]:
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k * 2]
    bm25_docs = [(chunks[i], bm25_scores[i]) for i in bm25_top_indices]

    vector_docs_scores = vector_store.similarity_search_with_score(query, k=top_k * 2)

    return reciprocal_rank_fusion(
        bm25_results=bm25_docs,
        vector_results=vector_docs_scores,
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
        top_k=top_k
    )