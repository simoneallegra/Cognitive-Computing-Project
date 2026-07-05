"""Logging delle run dell'agente.

Un logger JSONL locale scrive sempre, senza bisogno di account: e' quello
che poi legge evaluate_agent.py per l'analisi qualitativa. In piu', se in
.env c'e' LANGCHAIN_API_KEY, si attiva anche il tracing su LangSmith (free
tier) impostando le variabili d'ambiente standard di LangChain.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import config

_RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
_RUN_LOG_PATH = config.RUN_LOGS_DIR / f"run_{_RUN_ID}.jsonl"


def setup_langsmith() -> bool:
    if not config.LANGCHAIN_API_KEY:
        return False
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = config.LANGCHAIN_API_KEY
    os.environ.setdefault("LANGCHAIN_PROJECT", "blogger-support-agent")
    return True


def current_run_log_path() -> Path:
    return _RUN_LOG_PATH


def log_event(event_type: str, payload: dict) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        "payload": payload,
    }
    config.RUN_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with _RUN_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # Debug: `python observability/tracing.py` scrive un evento di prova e
    # mostra dove finisce, senza dover lanciare l'agente per generarne uno.
    log_event("debug_test", {"hello": "world"})
    print("Evento scritto in:", current_run_log_path())
    print("LangSmith attivato:", setup_langsmith())
