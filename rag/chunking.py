"""Chunking del testo recuperato per il vector store RAG.

Strategia: recursive chunking, che rispetta paragrafi/frasi quando
possibile - un buon compromesso tra semplicita' e qualita' semantica
rispetto al fixed-size chunking.
"""
from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_text(text: str) -> list[str]:
    if not text:
        return []
    return _splitter.split_text(text)


if __name__ == "__main__":
    # Debug: `python rag/chunking.py` mostra come un testo di prova viene
    # spezzato in chunk, per verificare a colpo d'occhio la dimensione e
    # l'overlap configurati sopra.
    sample = (
        "Paragraph one talks about hiking trails in the mountains.\n\n"
        "Paragraph two is a bit longer and talks about something else "
        "entirely, spanning multiple sentences to test the splitter "
        "behaviour across sentence boundaries. " * 5
    )
    chunks = chunk_text(sample)
    print(f"{len(chunks)} chunk generati da un testo di {len(sample)} caratteri")
    for i, c in enumerate(chunks[:3]):
        print(f"--- chunk {i} ({len(c)} caratteri) ---")
        print(c)
