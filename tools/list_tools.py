"""Debug: `python tools/list_tools.py` elenca i tool di REACT_TOOLS con
una breve descrizione in italiano (per il LLM il docstring resta in
inglese, vedi i singoli file in tools/)"""
from __future__ import annotations

from tools import REACT_TOOLS

DESCRIZIONI = {
    "search_web_tool": "cerca sul web fonti candidate",
    "rag_retrieval_tool": "recupera passaggi dal web con citazioni (K-RAG)",
    "kg_query_tool": "interroga il Knowledge Graph",
    "resource_quality_tool": "valuta qualita'/interesse di una fonte",
    "draft_with_finetuned_model": "scrive il post col modello fine-tuned",
}

for t in REACT_TOOLS:
    print(f"- {t.name}")
    print(f"    descrizione: {DESCRIZIONI.get(t.name, '(nessuna descrizione)')}")
