"""Recupero e pulizia del contenuto testuale di una pagina web (sorgenti
esterne per il RAG)."""
from __future__ import annotations

from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; BloggerSupportAgent/1.0; "
        "+educational-prototype)"
    )
}
_TIMEOUT_SECONDS = 8
_MAX_CHARS = 8000


def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc
    except ValueError:
        return ""


def fetch_and_clean(url: str) -> str:
    """Scarica una pagina e ne estrae il testo. In caso di errore/timeout
    ritorna stringa vuota, cosi' il chiamante puo' andare avanti."""
    try:
        response = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException:
        return ""

    soup = BeautifulSoup(response.text, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    text = " ".join(soup.get_text(separator=" ").split())
    return text[:_MAX_CHARS]


if __name__ == "__main__":
    # Debug: `python rag/fetch.py` scarica e pulisce una pagina reale
    url = "https://en.wikipedia.org/wiki/Dolomites"
    text = fetch_and_clean(url)
    print(f"domain_of(url) = {domain_of(url)!r}")
    print(f"{len(text)} caratteri estratti")
    print(text[:300])
