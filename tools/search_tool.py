"""Tool di ricerca web, primo passo per raccogliere fonti candidate."""
from __future__ import annotations

from langchain_core.tools import tool

from rag.search import search_web


@tool
def search_web_tool(query: str, max_results: int = 5) -> str:
    """Search the public web for up-to-date information about a travel
    destination, activity, or event. Use this FIRST to discover candidate
    sources (titles + URLs + short snippets) before deeper retrieval with
    rag_retrieval_tool. Good for checking facts, events, and news."""
    results = search_web(query, max_results=max_results)
    if not results:
        return "No results found for this query."

    lines = []
    for r in results:
        title = r.get("title", "")
        href = r.get("href", "")
        body = (r.get("body", "") or "")[:200]
        lines.append(f"- {title} | {href}\n  {body}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Debug: `python tools/search_tool.py` esegue una ricerca reale e ne
    # stampa l'output cosi' come lo vedrebbe il LLM nel ciclo ReAct.
    print(search_web_tool.invoke({"query": "best hiking trails in the Dolomites", "max_results": 3}))
