"""Stato condiviso del grafo LangGraph.

Ogni nodo legge/scrive un sottoinsieme di questi campi; LangGraph si
occupa di propagarlo da uno step al successivo.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class PlannedPost(TypedDict):
    topic: str
    post_type: str  # uno tra config.POST_TYPES
    justification: str
    status: Literal["planned", "drafting", "approved", "rejected"]


class ReasoningStep(TypedDict):
    step: int
    thought: str
    tool: str | None
    tool_args: dict[str, Any] | None
    observation: str | None
    justification: str


class RetrievedFact(TypedDict):
    text: str
    source_url: str
    source_title: str
    quality_score: float


class Draft(TypedDict):
    topic: str
    post_type: str
    title: str
    body: str
    citations: list[dict[str, str]]
    facts_used: list[str]


class BlogAgentState(TypedDict):
    # --- user input ---
    user_input: str
    n_posts: int

    # --- messaggi ReAct (per il sotto-ciclo di tool-calling) ---
    messages: Annotated[list[AnyMessage], add_messages]

    # --- reasoning trace esplicito (Thought/Action/Observation) ---
    reasoning_trace: list[ReasoningStep]

    # --- output dei tool, nell'ordine in cui sono stati invocati ---
    tool_outputs: list[dict[str, Any]]

    # --- riassunto del Knowledge Graph (query in fase di planning/drafting) ---
    kg_summary: dict[str, Any]

    # --- pianificazione editoriale ---
    plan: list[PlannedPost]
    current_post_idx: int

    # --- fatti raccolti per il post corrente (K-RAG) ---
    gathered_facts: list[RetrievedFact]

    # --- draft corrente in revisione ---
    draft: Draft | None

    # --- human-in-the-loop ---
    human_decision: Literal["approve", "modify", "reject", None]
    human_feedback: str | None

    # --- storico dei post approvati nella sessione ---
    approved_posts: list[Draft]


if __name__ == "__main__":
    # Debug: `python state.py` costruisce uno stato iniziale vuoto (identico
    # a quello che main.py passa a app.invoke) e ne stampa le chiavi, utile
    # per verificare a colpo d'occhio che nessun campo di BlogAgentState sia
    # stato dimenticato quando main.py costruisce initial_state.
    example: BlogAgentState = {
        "user_input": "debug", "n_posts": 1, "messages": [],
        "reasoning_trace": [], "tool_outputs": [], "kg_summary": {},
        "plan": [], "current_post_idx": 0, "gathered_facts": [],
        "draft": None, "human_decision": None, "human_feedback": None,
        "approved_posts": [],
    }
    print(f"BlogAgentState ha {len(example)} campi:")
    for key, value in example.items():
        print(f"  {key!r:20} -> {value!r}")
