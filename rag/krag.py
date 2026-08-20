"""K-RAG: la query viene espansa coi topic correlati nel KG, poi passa
per search -> fetch -> chunk -> embedding, e i passaggi tornano con la
fonte allegata per poterli citare."""
from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from rag.chunking import chunk_text
from rag.fetch import domain_of, fetch_and_clean
from rag.search import search_web
from rag.vectorstore import build_index, get_retriever


def expand_query(query: str, topic: str | None, kg: Any | None) -> str:
    """Aggiunge alla query i topic correlati trovati nel KG"""
    if not topic or kg is None:
        return query
    related = kg.expand_query_topics(topic)
    if not related:
        return query
    return f"{query} ({', '.join(related)})"


def gather_documents(query: str, max_sources: int = 4) -> list[Document]:
    """Search -> fetch -> chunk. Ogni Document porta la fonte nei metadata"""
    results = search_web(query, max_results=max_sources)
    documents: list[Document] = []
    for result in results:
        url = result.get("href", "")
        title = result.get("title", "")
        if not url:
            continue
        page_text = fetch_and_clean(url)
        if not page_text:
            # fallback: usa almeno lo snippet restituito dal motore di ricerca
            page_text = result.get("body", "")
        for chunk in chunk_text(page_text):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={"source_url": url, "source_title": title, "domain": domain_of(url)},
                )
            )
    return documents


def retrieve_with_citations(
    query: str,
    topic: str | None = None,
    kg: Any | None = None,
    max_sources: int = 4,
    k: int = 5,
) -> list[dict[str, str]]:
    """Pipeline K-RAG completa: espande la query, recupera e ritorna i
    passaggi con la relativa fonte"""
    expanded = expand_query(query, topic, kg)
    documents = gather_documents(expanded, max_sources=max_sources)
    index = build_index(documents)
    if index is None:
        return []

    retrieved = get_retriever(index, k=k).invoke(query)
    return [
        {
            "text": doc.page_content,
            "source_url": doc.metadata.get("source_url", ""),
            "source_title": doc.metadata.get("source_title", ""),
        }
        for doc in retrieved
    ]


if __name__ == "__main__":
    # Debug: `python rag/krag.py` esegue la pipeline K-RAG completa
    # (senza espansione KG, topic=None) su una query reale
    passages = retrieve_with_citations("best hiking trails in the Dolomites")
    print(f"{len(passages)} passaggi recuperati")
    for p in passages[:2]:
        print(f"- ({p['source_url']}) {p['text'][:150]}")
