"""Genera il testo del draft dai fatti raccolti nel ReAct, usando il
distilgpt2 fine-tuned in finetuning/train.py."""
from __future__ import annotations

from langchain_core.tools import tool

from tools._shared import get_finetuned_generator

_TASK_INSTRUCTION = {
    "review": "Write a travel review blog post using ONLY the facts above.",
    "how_to": "Write a 'how to spend your time' travel guide post using ONLY the facts above.",
    "ranking": "Write a 'Top cities' ranking blog post using ONLY the facts above.",
}


@tool
def draft_with_finetuned_model(
    topic: str, post_type: str, facts: list[str], max_new_tokens: int = 380
) -> str:
    """Draft the body text of a blog post using the team's fine-tuned
    distilgpt2 model. Inputs: `topic`, `post_type` (how_to/review/ranking),
    and `facts` — verified statements gathered via the other tools. Call
    this ONLY AFTER gathering and quality-checking the facts."""
    generator, tokenizer = get_finetuned_generator()
    instruction = _TASK_INSTRUCTION.get(post_type, _TASK_INSTRUCTION["review"])
    facts_block = "\n".join(facts) if facts else "(no facts provided)"
    prompt = f"Facts:\n{facts_block}\n\nTopic: {topic}\nTask: {instruction}"
    full_prompt = f"<PROMPT>\n{prompt}\n<RESP>\n"

    output = generator.invoke(
        full_prompt,
        pipeline_kwargs={
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min(180, max_new_tokens),
            "do_sample": True,
            "top_k": 50,
            "temperature": 0.8,
            "repetition_penalty": 1.3,
            "no_repeat_ngram_size": 3,
            "pad_token_id": tokenizer.pad_token_id,
        },
    )

    return output[len(full_prompt):].strip()


if __name__ == "__main__":
    # Debug: `python tools/finetuned_draft_tool.py` genera un draft di
    # prova per ciascun post_type, per ispezionare rapidamente la qualita'
    # del modello fine-tuned senza passare dal ciclo ReAct completo
    debug_facts = [
        "The old town has cobblestone streets dating back to the 14th century.",
        "Local markets open at 7am and close by 2pm (source: example.com).",
    ]
    for post_type in _TASK_INSTRUCTION:
        print(f"--- post_type={post_type} ---")
        print(draft_with_finetuned_model.invoke(
            {"topic": "A weekend in Verona", "post_type": post_type, "facts": debug_facts}
        ))
        print()
