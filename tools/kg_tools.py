"""Interrogazione e aggiornamento del Knowledge Graph.

`update_knowledge_graph` non e' esposta come @tool al LLM di proposito:
la scrittura sul grafo deve avvenire solo dopo l'approvazione umana del
post, e lasciare quella decisione al modello dentro il ciclo ReAct
sarebbe troppo rischioso (potrebbe scrivere prima del tempo, o mai). Per
questo la chiama direttamente il nodo `update_kg` del grafo, in modo
deterministico. La query invece e' innocua e serve all'agente per
ragionare (evitare ripetizioni, restare coerente con quanto gia' pubblicato).
"""
from __future__ import annotations

import json

from langchain_core.tools import tool

import config
from kg.client import KnowledgeGraph
from tools._shared import get_kg


@tool
def kg_query_tool(action: str, topic: str = "") -> str:
    """Query the editorial Knowledge Graph. `action` must be one of:
    - 'recent_topics': topics covered most recently (use to AVOID repetition)
    - 'coverage_gaps': domain topics never covered yet (use to find gaps)
    - 'related_context': previous posts related to `topic` (use to ensure
      consistency with previously published content when drafting)
    - 'claims_for_topic': claims already published about `topic` (use to
      verify accuracy / avoid contradicting earlier posts)
    `topic` is required for 'related_context' and 'claims_for_topic'."""
    kg = get_kg()
    if action == "recent_topics":
        return json.dumps(kg.get_recent_topics())
    if action == "coverage_gaps":
        return json.dumps(kg.get_coverage_gaps(config.TOPIC_CATEGORIES))
    if action == "related_context":
        return json.dumps(kg.get_related_context(topic))
    if action == "claims_for_topic":
        return json.dumps(kg.get_claims_for_topic(topic))
    return f"Unknown action '{action}'. Valid actions: recent_topics, coverage_gaps, related_context, claims_for_topic."


def update_knowledge_graph(
    kg: KnowledgeGraph,
    post_id: str,
    title: str,
    post_type: str,
    topic: str,
    summary: str = "",
    related_topics: list[str] | None = None,
    sources: list[str] | None = None,
    claims: list[str] | None = None,
) -> None:
    """Scrittura incrementale del KG, chiamata SOLO dal nodo `update_kg`
    del grafo dell'agente, dopo approvazione umana del post."""
    kg.add_post(post_id, title, post_type, summary)
    kg.link_post_topic(post_id, topic)
    for related in related_topics or []:
        kg.link_topics_related(topic, related)
    for url in sources or []:
        kg.link_post_source(post_id, url)
    for i, claim_text in enumerate(claims or []):
        source_url = sources[i] if sources and i < len(sources) else None
        kg.add_claim(post_id, f"{post_id}_claim_{i}", claim_text, source_url=source_url)


if __name__ == "__main__":
    # Debug: `python tools/kg_tools.py` esercita kg_query_tool su ogni
    # action supportata, cosi' come farebbe il LLM nel ciclo ReAct.
    for action in ("recent_topics", "coverage_gaps"):
        print(f"--- {action} ---")
        print(kg_query_tool.invoke({"action": action}))
    print("--- unknown_action (verifica messaggio di errore) ---")
    print(kg_query_tool.invoke({"action": "unknown_action"}))
