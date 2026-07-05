"""Configurazione centrale del progetto Blogger Support Agent."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent

load_dotenv(PROJECT_DIR / ".env")

# credenziali / endpoint
HF_TOKEN = os.getenv("HF_TOKEN")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")  # LangSmith, opzionale

# modelli usati dall'agente, tutti gratuiti/locali eccetto il reasoning LLM
REASONING_LLM_REPO_ID = "Qwen/Qwen2.5-3B-Instruct"  # gira in locale su GPU
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FINETUNE_BASE_MODEL = "distilgpt2"

# dominio del blog
BLOG_DOMAIN = "travel blogging"
POST_TYPES = ["how_to", "review", "ranking"]
TOPIC_CATEGORIES = [
    "culture", "adventure", "nature", "beaches", "nightlife",
    "cuisine", "wellness", "urban", "seclusion",
]
POSTING_INTERVAL_DAYS = 3
MAX_POST_WORDS = 280

# percorsi
FINETUNING_DIR = PROJECT_DIR / "finetuning"
FT_DATA_DIR = FINETUNING_DIR / "data"
FT_SAVED_MODELS_DIR = FINETUNING_DIR / "saved_models"
FT_RESULTS_DIR = FINETUNING_DIR / "results"
FT_FINAL_MODEL_DIR = FT_SAVED_MODELS_DIR / "distilgpt2_context_grounded"

# vecchio modello fine-tuned (senza contesto), usato solo come riferimento
# facoltativo in finetuning/evaluate.py se presente
LEGACY_FT_MODEL_DIR = PROJECT_DIR / "saved_models" / "travel_posts_distilgpt2"
LEGACY_BASE_MODEL_DIR = PROJECT_DIR / "saved_models" / "distilgpt2_base"

RAG_INDEX_DIR = PROJECT_DIR / "rag" / "faiss_index"
RUN_LOGS_DIR = PROJECT_DIR / "results" / "run_logs"
EVAL_REPORT_PATH = PROJECT_DIR / "results" / "evaluation_report.md"

RAG_INDEX_DIR.mkdir(parents=True, exist_ok=True)
RUN_LOGS_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Debug: `python config.py` stampa la configurazione effettiva caricata
    # da .env, senza dover avviare l'intero agente. Utile per verificare
    # rapidamente che le credenziali siano lette correttamente.
    print(f"PROJECT_DIR          = {PROJECT_DIR}")
    print(f".env trovato         = {(PROJECT_DIR / '.env').exists()}")
    print(f"HF_TOKEN impostato   = {bool(HF_TOKEN)}")
    print(f"NEO4J_URI            = {NEO4J_URI}")
    print(f"NEO4J_USER           = {NEO4J_USER}")
    print(f"NEO4J_PASSWORD impostato = {bool(NEO4J_PASSWORD)}")
    print(f"LANGCHAIN_API_KEY impostato = {bool(LANGCHAIN_API_KEY)}")
    print(f"REASONING_LLM_REPO_ID = {REASONING_LLM_REPO_ID}")
    print(f"EMBEDDING_MODEL_NAME  = {EMBEDDING_MODEL_NAME}")
    print(f"FINETUNE_BASE_MODEL   = {FINETUNE_BASE_MODEL}")
    print(f"POST_TYPES            = {POST_TYPES}")
    print(f"TOPIC_CATEGORIES      = {TOPIC_CATEGORIES}")
    print(f"FT_FINAL_MODEL_DIR esiste = {FT_FINAL_MODEL_DIR.exists()} ({FT_FINAL_MODEL_DIR})")
    print(f"RAG_INDEX_DIR esiste      = {RAG_INDEX_DIR.exists()} ({RAG_INDEX_DIR})")
    print(f"RUN_LOGS_DIR esiste       = {RUN_LOGS_DIR.exists()} ({RUN_LOGS_DIR})")
