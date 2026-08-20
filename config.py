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

# Costanti Globali per il blog
BLOG_DOMAIN = "travel blogging"
POST_TYPES = ["how_to", "review", "ranking"]
TOPIC_CATEGORIES = [
    "culture", "adventure", "nature", "beaches", "nightlife",
    "cuisine", "wellness", "urban", "seclusion",
]
POSTING_INTERVAL_DAYS = 3
MAX_POST_WORDS = 280

# Paths
FINETUNING_DIR = PROJECT_DIR / "finetuning"
FT_DATA_DIR = FINETUNING_DIR / "data"
FT_SAVED_MODELS_DIR = FINETUNING_DIR / "saved_models"
FT_RESULTS_DIR = FINETUNING_DIR / "results"
FT_FINAL_MODEL_DIR = FT_SAVED_MODELS_DIR / "distilgpt2_context_grounded"

LEGACY_FT_MODEL_DIR = PROJECT_DIR / "saved_models" / "travel_posts_distilgpt2"
LEGACY_BASE_MODEL_DIR = PROJECT_DIR / "saved_models" / "distilgpt2_base"

RAG_INDEX_DIR = PROJECT_DIR / "rag" / "faiss_index"
EVAL_REPORT_PATH = PROJECT_DIR / "results" / "evaluation_report.md"

RAG_INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Langchain settings
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ.setdefault("LANGCHAIN_PROJECT", "blogger-support-agent")


