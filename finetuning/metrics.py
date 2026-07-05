"""Metriche per valutare il drafting fine-tuned.

- perplexity: la solita metrica di qualita' del language model.
- grounding score: quanti dei "fatti" passati nel prompt si ritrovano
  davvero nel testo generato. E' quella che conta di piu' qui, perche'
  misura se il draft e' effettivamente basato su quello che RAG/KG hanno
  recuperato, invece che inventato.
- ROUGE-L: sovrapposizione col riferimento sintetico, come controprova.
"""
from __future__ import annotations

import math

from rouge_score import rouge_scorer

_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "for", "and", "or", "to", "with",
    "is", "are", "this", "that", "above", "using", "only", "facts", "blog",
    "post", "write", "task", "topic",
}
_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def perplexity_from_loss(eval_loss: float) -> float:
    return math.exp(eval_loss)


def _significant_tokens(line: str) -> set[str]:
    tokens: list[str] = []
    current = ""
    for ch in line.lower():
        if ("a" <= ch <= "z") or ("0" <= ch <= "9"):
            current += ch
        elif current:
            tokens.append(current)
            current = ""
    if current:
        tokens.append(current)
    return {t for t in tokens if len(t) >= 3 and t not in _STOPWORDS}


def grounding_score(generated_text: str, facts: list[str]) -> float:
    """Per ogni fatto, controlla quanti dei suoi token significativi
    compaiono nel testo generato; ritorna la media sui fatti (0..1)."""
    if not facts:
        return 0.0
    generated_lower = generated_text.lower()
    covered = 0.0
    for fact in facts:
        tokens = _significant_tokens(fact)
        if not tokens:
            continue
        hits = sum(1 for t in tokens if t in generated_lower)
        covered += hits / len(tokens)
    return covered / len(facts)


def rouge_l(generated_text: str, reference_text: str) -> float:
    scores = _rouge.score(reference_text, generated_text)
    return scores["rougeL"].fmeasure


if __name__ == "__main__":
    # Debug: `python finetuning/metrics.py` verifica le tre metriche su
    # un esempio semplice, senza caricare nessun modello.
    facts = ["The museum opens at 9am.", "Entrance costs 15 euros."]
    generated = "Visit the museum, which opens at 9am and costs 15 euros to enter."
    reference = "The museum opens early in the morning and has an entrance fee."

    print("grounding_score:", grounding_score(generated, facts))
    print("rouge_l:", rouge_l(generated, reference))
    print("perplexity_from_loss(0.5):", perplexity_from_loss(0.5))
    print("_significant_tokens('The Museum Opens at 9am!'):", _significant_tokens("The Museum Opens at 9am!"))
