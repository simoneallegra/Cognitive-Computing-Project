"""Tool di retrieval K-RAG: espande la query via Knowledge Graph, poi
recupera passaggi dal web con la fonte allegata."""
from __future__ import annotations

from langchain_core.tools import tool

from rag.krag import retrieve_with_citations
from tools._shared import get_kg


@tool
def rag_retrieval_tool(query: str, topic: str = "") -> str:
    """Retrieve passages of text from the web that are relevant to `query`,
    grounded with explicit source citations. If `topic` is given, the query
    is first expanded using the Knowledge Graph's related topics (K-RAG),
    so the retrieval also covers content connected to what the blog has
    already published. Use the returned passages (and their [n] source URL)
    as the factual basis for the post, and cite them explicitly."""
    kg = get_kg()
    passages = retrieve_with_citations(query, topic=topic or None, kg=kg)
    if not passages:
        return "No relevant passages retrieved for this query."

    lines = []
    for i, passage in enumerate(passages, start=1):
        lines.append(f"[{i}] (source: {passage['source_url']}) {passage['text']}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Debug: `python tools/rag_tool.py` esegue la pipeline K-RAG completa
    # (search -> fetch -> chunk -> embed -> retrieval) su una query reale.
    print(rag_retrieval_tool.invoke({"query": "best hiking trails in the Dolomites"}))
