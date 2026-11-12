"""Il grafo LangGraph dell'agente.

Flusso: load_kg_context -> plan_posts -> [ gather_and_draft -> human_review
-> (reject -> gather_and_draft) | (approve/modify -> update_kg) ] per ogni
post del piano.

`gather_and_draft` fa internamente il ciclo ReAct (Thought/Action/
Observation) per un post: raccoglie e verifica i fatti con i tool, poi
genera il testo con il modello fine-tuned.
"""
from __future__ import annotations

import json
import uuid

from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

import config
from agent.llm import get_chat_model
from agent.prompts import build_planning_messages, build_react_system_prompt
from state import BlogAgentState
from tools import REACT_TOOLS, draft_with_finetuned_model, update_knowledge_graph
from tools._shared import get_kg

TOOLS_BY_NAME = {}
for t in REACT_TOOLS:
    TOOLS_BY_NAME[t.name] = t

MAX_REACT_STEPS = 10
_CITATION_MARKER = "(source: "


def _normalize_topic(topic: str) -> str:
    return topic.strip().casefold()


def _dedupe_plan(plan, all_topics, coverage_gaps):
    """Controllo hard anti-duplicati: il prompt di planning istruisce il LLM
    a evitare i topic gia' pubblicati (via recent_topics/coverage_gaps), ma
    e' solo un suggerimento -- questo filtro garantisce che nessun topic
    identico (case-insensitive) a uno gia' nel KG, ne' ripetuto due volte
    nello stesso piano, arrivi fino al drafting."""
    existing = set()
    for t in all_topics:
        existing.add(_normalize_topic(t))

    replacement_candidates = []
    for t in coverage_gaps:
        if _normalize_topic(t) not in existing:
            replacement_candidates.append(t)
    replacements = iter(replacement_candidates)

    deduped = []
    seen = set()
    for item in plan:
        key = _normalize_topic(item["topic"])
        if key in existing or key in seen:
            replacement = next(replacements, None)
            if replacement is None:
                print(f"[plan_posts] '{item['topic']}' e' gia' nel KG, scartato (nessuna categoria di ricambio disponibile).")
                continue
            print(f"[plan_posts] '{item['topic']}' e' gia' nel KG, sostituito con '{replacement}'.")
            item = {
                **item,
                "topic": replacement,
                "justification": f"{item['justification']} (topic originale duplicato di un post gia' pubblicato, sostituito)",
            }
            key = _normalize_topic(item["topic"])
            if key in seen:
                continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _parse_plan_json(text: str):
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):]
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):]
    cleaned = cleaned.strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[: -len("```")]
    cleaned = cleaned.strip()
    return json.loads(cleaned)


def _extract_citations(tool_outputs):
    """Estrae gli URL dalle righe di rag_retrieval_tool, formato
    "[i] (source: URL) testo"."""
    citations = []
    seen = set()
    for record in tool_outputs:
        if record["tool"] != "rag_retrieval_tool":
            continue
        for line in str(record["observation"]).splitlines():
            start = line.find(_CITATION_MARKER)
            if start == -1:
                continue
            start += len(_CITATION_MARKER)
            end = line.find(")", start)
            if end == -1:
                continue
            url = line[start:end].strip()
            if url and url not in seen:
                seen.add(url)
                citations.append({"url": url})
    return citations


def _split_fact_source(fact_text: str):
    """Estrae il marcatore inline "(source: URL)" da un singolo fatto (il
    ReAct prompt richiede esplicitamente di includerlo in ogni fatto, si
    veda build_react_system_prompt). Ritorna (testo_pulito, url) -- url
    vuota se il fatto non porta un marcatore (es. i fatti di fallback)."""
    start = fact_text.find(_CITATION_MARKER)
    if start == -1:
        return fact_text, ""
    end = fact_text.find(")", start)
    if end == -1:
        return fact_text, ""
    url = fact_text[start + len(_CITATION_MARKER):end].strip()
    clean_text = (fact_text[:start] + fact_text[end + 1:]).strip()
    return (clean_text or fact_text), url


def _facts_with_sources(raw_facts):
    """Converte i fatti grezzi (stringhe con eventuale "(source: URL)"
    inline) in RetrievedFact, con la fonte di ciascun fatto estratta dal
    fatto stesso -- non piu' accoppiata per posizione a una lista di
    citazioni separata (si veda update_knowledge_graph)."""
    facts = []
    for raw in raw_facts:
        text, source_url = _split_fact_source(raw)
        facts.append({"text": text, "source_url": source_url, "source_title": "", "quality_score": 0.0})
    return facts


# nodi del grafo
def load_kg_context(state: BlogAgentState):
    """Interroga il KG per sapere cosa e' gia' stato pubblicato."""
    kg = get_kg()
    kg_summary = {
        "recent_topics": kg.get_recent_topics(),
        "coverage_gaps": kg.get_coverage_gaps(config.TOPIC_CATEGORIES),
        "all_topics": kg.get_all_topics(),
        **kg.summary(),
    }
    print(
        f"[load_kg_context] KG: {kg_summary['posts']} post, {kg_summary['topics']} topic, "
        f"{kg_summary['sources']} fonti, {kg_summary['claims']} claim "
        f"({len(kg_summary['coverage_gaps'])} categorie ancora scoperte)"
    )
    return {"kg_summary": kg_summary}


def plan_posts(state: BlogAgentState):
    """Chiede al LLM il piano dei post, basandosi sul riassunto del KG."""
    n_posts = state.get("n_posts", 3)
    llm = get_chat_model()
    messages = build_planning_messages(state["kg_summary"], n_posts)
    response = llm.invoke(messages)
    print("[plan_posts] LLM ha generato un piano, parsing JSON...")

    try:
        raw_plan = _parse_plan_json(response.content)
    except (json.JSONDecodeError, TypeError, ValueError):
        fallback_topics = state["kg_summary"].get("coverage_gaps") or config.TOPIC_CATEGORIES
        raw_plan = []
        for topic in fallback_topics[:n_posts]:
            raw_plan.append(
                {
                    "topic": topic,
                    "post_type": "review",
                    "justification": "Fallback: planner output was not valid JSON; using a coverage gap.",
                }
            )

    plan = []
    for i, item in enumerate(raw_plan[:n_posts]):
        post_type = item.get("post_type") if item.get("post_type") in config.POST_TYPES else "review"
        plan.append(
            {
                "topic": item.get("topic", f"Untitled topic {i}"),
                "post_type": post_type,
                "justification": item.get("justification", ""),
                "status": "planned",
            }
        )

    plan = _dedupe_plan(
        plan,
        all_topics=state["kg_summary"].get("all_topics", []),
        coverage_gaps=state["kg_summary"].get("coverage_gaps") or list(config.TOPIC_CATEGORIES),
    )

    print(f"[plan_posts] piano di {len(plan)} post:")
    for p in plan:
        print(f"  - {p['topic']} ({p['post_type']})")
    return {"plan": plan, "current_post_idx": 0}


def gather_and_draft(state: BlogAgentState):
    """Ciclo ReAct per il post corrente: raccoglie i fatti coi tool, poi
    genera il draft col modello fine-tuned."""
    plan = state["plan"]
    idx = state["current_post_idx"]
    post = plan[idx]

    llm = get_chat_model().bind_tools(REACT_TOOLS)
    messages = [SystemMessage(content=build_react_system_prompt(
        post["topic"], post["post_type"], post["justification"]
    ))]

    reasoning_trace = []
    tool_outputs = []
    gathered_facts = []
    draft_body = None
    seen_calls = set()  # (tool, args) gia' invocati in questa sessione

    print(f"[gather_and_draft] '{post['topic']}': inizio ciclo ReAct...")
    for _ in range(MAX_REACT_STEPS):
        ai_message = llm.invoke(messages)
        messages.append(ai_message)

        if not ai_message.tool_calls:
            break

        for tool_call in ai_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool = TOOLS_BY_NAME.get(tool_name)
            if tool is None and "facts" in tool_args:
                # il LLM a volte storpia il nome del tool di drafting (es.
                # "draft_with_facts" invece di "draft_with_finetuned_model"):
                # la firma con `facts` e' univoca tra i 5 tool, quindi la
                # chiamata e' recuperabile invece di scartarla come errore
                tool_name = "draft_with_finetuned_model"
                tool = TOOLS_BY_NAME.get(tool_name)
            call_signature = (tool_name, json.dumps(tool_args, sort_keys=True, default=str))
            tool_failed = False
            if tool is None:
                observation = f"Tool error: unknown tool '{tool_name}'."
                tool_failed = True
            elif call_signature in seen_calls:
                observation = (
                    "You already called this exact tool with these exact arguments earlier "
                    "in this session -- repeating it gives no new information. Call "
                    "draft_with_finetuned_model now with the facts you have gathered so far."
                )
                tool_failed = True
            else:
                seen_calls.add(call_signature)
                try:
                    observation = tool.invoke(tool_args)
                except Exception as exc:
                    observation = (
                        f"Tool error calling {tool_name} with args {tool_args}: {exc}. "
                        f"Check the expected argument names and try again."
                    )
                    tool_failed = True

            print(f"  step {len(reasoning_trace) + 1}: {tool_name}" + (" (fallito)" if tool_failed else ""))

            thought = (ai_message.content or "").strip()
            justification = thought or (
                f"Calling {tool_name} with {tool_args} to progress towards a grounded, "
                f"verified draft for '{post['topic']}'."
            )

            reasoning_trace.append(
                {
                    "step": len(reasoning_trace) + 1,
                    "thought": thought,
                    "tool": tool_name,
                    "tool_args": tool_args,
                    "observation": str(observation)[:1000],
                    "justification": justification,
                }
            )
            tool_outputs.append({"tool": tool_name, "args": tool_args, "observation": str(observation)})
            messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))

            if tool_name == "draft_with_finetuned_model" and not tool_failed:
                draft_body = observation
                gathered_facts = tool_args.get("facts", [])

        if draft_body is not None:
            break

    facts_used_for_draft = gathered_facts
    if draft_body is None:
        # safety net: garantisce che il drafting passi SEMPRE dal tool fine-tuned
        # (topic/post_type sono metadati del post, non fatti estratti da una fonte:
        # servono solo a far partire il drafting, non devono diventare Claim nel KG,
        # quindi non toccano `gathered_facts`)
        facts_fallback = gathered_facts or [f"Topic: {post['topic']}", f"Post type: {post['post_type']}"]
        draft_body = draft_with_finetuned_model.invoke(
            {"topic": post["topic"], "post_type": post["post_type"], "facts": facts_fallback}
        )
        facts_used_for_draft = facts_fallback
        reasoning_trace.append(
            {
                "step": len(reasoning_trace) + 1,
                "thought": "Reasoning budget exhausted before the model called the drafting tool.",
                "tool": "draft_with_finetuned_model",
                "tool_args": {"topic": post["topic"], "post_type": post["post_type"], "facts": facts_fallback},
                "observation": str(draft_body)[:1000],
                "justification": "Fallback to guarantee a fine-tuned, grounded draft within the step budget.",
            }
        )

    citations = _extract_citations(tool_outputs)
    draft = {
        "topic": post["topic"],
        "post_type": post["post_type"],
        "title": f"{post['topic']} — {post['post_type'].replace('_', ' ').title()}",
        "body": draft_body,
        "citations": citations,
        "facts_used": facts_used_for_draft,
        "reasoning_trace": reasoning_trace,
    }

    tools_used = []
    for step in reasoning_trace:
        tools_used.append(step["tool"])
    print(
        f"[gather_and_draft] '{post['topic']}': {len(reasoning_trace)} step "
        f"({', '.join(tools_used)}), {len(citations)} citazioni — draft pronto per la revisione"
    )

    return {
        "messages": messages,
        "reasoning_trace": state.get("reasoning_trace", []) + reasoning_trace,
        "tool_outputs": state.get("tool_outputs", []) + tool_outputs,
        "gathered_facts": _facts_with_sources(facts_used_for_draft),
        "draft": draft,
    }


def human_review(state: BlogAgentState):
    """Sospende il grafo e aspetta la decisione dell'utente sul draft."""
    decision = interrupt(
        {
            "draft": state["draft"],
            "instructions": "Respond with decision in {'approve','modify','reject'}; "
            "if 'modify', also provide 'feedback' with the edited post text.",
        }
    )
    return {
        "human_decision": decision.get("decision"),
        "human_feedback": decision.get("feedback"),
    }


def update_kg(state: BlogAgentState):
    """Scrive il post nel KG, solo dopo l'approvazione umana."""
    draft = dict(state["draft"])
    if state.get("human_decision") == "modify" and state.get("human_feedback"):
        draft["body"] = state["human_feedback"]

    kg = get_kg()
    post_id = str(uuid.uuid4())
    plan = state["plan"]
    idx = state["current_post_idx"]

    related_topics = []
    for i, p in enumerate(plan):
        if i != idx and p["status"] == "approved":
            related_topics.append(p["topic"])

    source_urls = []
    for c in draft.get("citations", []):
        if c.get("url"):
            source_urls.append(c["url"])

    update_knowledge_graph(
        kg,
        post_id=post_id,
        title=draft["title"],
        post_type=draft["post_type"],
        topic=draft["topic"],
        summary=draft["body"][:300],
        related_topics=related_topics,
        sources=source_urls,
        facts=state.get("gathered_facts", []),
    )

    new_plan = list(plan)
    new_plan[idx] = {**new_plan[idx], "status": "approved"}

    print(f"[update_kg] post_id={post_id} title={draft['title']!r}")

    return {
        "plan": new_plan,
        "current_post_idx": idx + 1,
        "approved_posts": state.get("approved_posts", []) + [draft],
        "draft": None,
        "human_decision": None,
        "human_feedback": None,
    }


# routing tra i nodi
def route_after_review(state: BlogAgentState) -> str:
    if state.get("human_decision") == "reject":
        return "gather_and_draft"
    return "update_kg"


def route_after_update(state: BlogAgentState) -> str:
    if state["current_post_idx"] >= len(state["plan"]):
        return END
    return "gather_and_draft"


def build_graph():
    graph = StateGraph(BlogAgentState)
    graph.add_node("load_kg_context", load_kg_context)
    graph.add_node("plan_posts", plan_posts)
    graph.add_node("gather_and_draft", gather_and_draft)
    graph.add_node("human_review", human_review)
    graph.add_node("update_kg", update_kg)

    graph.add_edge(START, "load_kg_context")
    graph.add_edge("load_kg_context", "plan_posts")
    graph.add_edge("plan_posts", "gather_and_draft")
    graph.add_edge("gather_and_draft", "human_review")
    graph.add_conditional_edges(
        "human_review", route_after_review, {"gather_and_draft": "gather_and_draft", "update_kg": "update_kg"}
    )
    graph.add_conditional_edges("update_kg", route_after_update, {END: END, "gather_and_draft": "gather_and_draft"})

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    # Debug: `python agent/graph.py` costruisce il grafo (senza eseguirlo)
    # e ne stampa la struttura in formato mermaid, per verificare rapidamente
    # nodi/archi dopo una modifica, senza dover lanciare l'intero agente
    app = build_graph()
    print("Grafo costruito correttamente.\n")
    print(app.get_graph().draw_mermaid())
