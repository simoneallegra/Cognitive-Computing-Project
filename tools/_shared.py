"""Risorse condivise tra i tool dell'agente (connessione KG, modello
fine-tuned), inizializzate pigramente e riusate tra le chiamate."""
from __future__ import annotations

from functools import lru_cache

import torch
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import config
from kg.client import KnowledgeGraph

_kg_instance: KnowledgeGraph | None = None


def get_kg() -> KnowledgeGraph:
    global _kg_instance
    if _kg_instance is None:
        _kg_instance = KnowledgeGraph()
    return _kg_instance


@lru_cache(maxsize=1)
def get_finetuned_generator():
    model_dir = config.FT_FINAL_MODEL_DIR
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Modello fine-tuned non trovato in {model_dir}. "
            "Esegui prima 'python -m finetuning.train'."
        )
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(model_dir))
    model.config.pad_token_id = tokenizer.pad_token_id

    device = 0 if torch.cuda.is_available() else -1
    hf_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
    # Wrappato nell'astrazione LangChain `HuggingFacePipeline` invece di
    # usare la pipeline transformers grezza: stesso modello/generazione,
    # ma invocabile con la stessa interfaccia .invoke() del resto del progetto.
    generator = HuggingFacePipeline(pipeline=hf_pipe)
    return generator, tokenizer


if __name__ == "__main__":
    # Debug: `python tools/_shared.py` verifica che entrambe le risorse
    # condivise (connessione KG e modello fine-tuned) si inizializzino
    # correttamente, senza dover passare da tutto l'agente.
    print("Test get_kg()...")
    try:
        print("KG OK:", get_kg().summary())
    except Exception as exc:
        print(f"KG non disponibile: {exc}")

    print("\nTest get_finetuned_generator() (carica il modello su GPU/CPU)...")
    try:
        generator, tokenizer = get_finetuned_generator()
        print("Generatore caricato correttamente, pad_token_id =", tokenizer.pad_token_id)
    except FileNotFoundError as exc:
        print(f"Modello non trovato: {exc}")
