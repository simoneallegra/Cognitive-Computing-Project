"""Vector store locale (FAISS + sentence-transformers, entrambi gratuiti)
per la parte di document retrieval del K-RAG.

L'indice e' costruito "on the fly" ad ogni sessione di retrieval (i
documenti recuperati da internet cambiano ogni volta), non e' persistito
tra esecuzioni diverse: questo e' coerente con lo scenario, dove le fonti
sono pagine web aggiornate di volta in volta dal Search tool.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings

import config


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)


def build_index(documents: list[Document]) -> FAISS | None:
    if not documents:
        return None
    return FAISS.from_documents(documents, get_embeddings())


def get_retriever(index: FAISS, k: int = 4) -> VectorStoreRetriever:
    """Espone l'indice come retriever LangChain standard (`as_retriever`),
    cosi' il resto della pipeline lo invoca con `retriever.invoke(query)`
    invece di chiamare direttamente `index.similarity_search(...)`."""
    return index.as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    # Debug: `python rag/vectorstore.py` costruisce un indice FAISS su
    # documenti finti e verifica che la similarity search torni quello
    # semanticamente piu' vicino, senza toccare la rete.
    docs = [
        Document(page_content="The Dolomites are famous for hiking and skiing.", metadata={"source_url": "a"}),
        Document(page_content="Bangkok street food is best explored at night markets.", metadata={"source_url": "b"}),
        Document(page_content="Bali has many vegan-friendly beachfront resorts.", metadata={"source_url": "c"}),
    ]
    index = build_index(docs)
    results = get_retriever(index, k=1).invoke("mountains for hiking")
    print("Query: 'mountains for hiking' ->", results[0].page_content, f"(source={results[0].metadata['source_url']})")
