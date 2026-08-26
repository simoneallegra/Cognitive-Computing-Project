"""Wrapper su DuckDuckGo Search: nessuna API key richiesta."""
from __future__ import annotations

from ddgs import DDGS


def search_web(query: str, max_results: int = 5):
    """Ritorna una lista di risultati {title, href, body} per la query data."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as exc:  # boundary di rete: la ricerca esterna puo' fallire per molte ragioni
        return [{"title": "ERRORE_RICERCA", "href": "", "body": str(exc)}]
    return results


if __name__ == "__main__":
    # Debug: `python rag/search.py` esegue una ricerca reale e stampa i
    # risultati grezzi, senza passare dal tool LangChain
    for r in search_web("best hiking trails in the Dolomites", max_results=3):
        print(f"- {r.get('title')} | {r.get('href')}")
