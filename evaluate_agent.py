"""Valuta l'agente: fa girare il grafo in-process (approvando i draft
in automatico) e produce un report di analisi qualitativa.

Uso: python evaluate_agent.py [--posts N]
"""
from __future__ import annotations

import argparse
import uuid

from langgraph.types import Command

import config
from agent.graph import MAX_REACT_STEPS, build_graph
from finetuning.metrics import grounding_score

LOW_GROUNDING_THRESHOLD = 0.2


def _run_agent_and_collect_drafts(n_posts: int) -> list[dict]:
    """Fa girare l'agente end-to-end approvando ogni draft senza chiedere
    conferma, e ritorna i post approvati con la loro reasoning_trace."""
    app = build_graph()
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial_state = {
        "user_input": f"Plan and draft {n_posts} travel blog posts.",
        "n_posts": n_posts,
        "messages": [],
        "reasoning_trace": [],
        "tool_outputs": [],
        "kg_summary": {},
        "plan": [],
        "current_post_idx": 0,
        "gathered_facts": [],
        "draft": None,
        "human_decision": None,
        "human_feedback": None,
        "approved_posts": [],
    }

    result = app.invoke(initial_state, config=thread_config)
    while "__interrupt__" in result:
        result = app.invoke(Command(resume={"decision": "approve"}), config=thread_config)
    return result["approved_posts"]


def _analyze_draft(draft: dict) -> dict:
    reasoning_trace = draft["reasoning_trace"]
    facts_used = draft.get("facts_used", [])
    g_score = grounding_score(draft["body"], facts_used) if facts_used else 0.0
    quality_checks = [s for s in reasoning_trace if s["tool"] == "resource_quality_tool"]
    return {
        "topic": draft["topic"],
        "post_type": draft["post_type"],
        "body_excerpt": draft["body"][:300],
        "n_citations": len(draft.get("citations", [])),
        "n_facts_used": len(facts_used),
        "n_quality_checks": len(quality_checks),
        "n_reasoning_steps": len(reasoning_trace),
        "grounding_score": g_score,
        "reasoning_trace": reasoning_trace,
        "draft": draft,
    }


def _looks_like_rating_out_of_five(text: str) -> bool:
    """True se `text` contiene un voto tipo "4/5": segno che il LLM ha
    copiato un rating dal dataset invece di scrivere testo vero."""
    n = len(text)
    for i, ch in enumerate(text):
        if ch not in "12345":
            continue
        if i > 0 and text[i - 1].isalnum():
            continue
        j = i + 1
        while j < n and text[j].isspace():
            j += 1
        if j >= n or text[j] != "/":
            continue
        j += 1
        while j < n and text[j].isspace():
            j += 1
        if j < n and text[j] == "5" and (j + 1 >= n or not text[j + 1].isalnum()):
            return True
    return False


def _identify_failure_cases(analyses: list[dict]) -> list[dict]:
    failures = []
    for a in analyses:
        facts_used = a["draft"].get("facts_used", [])
        suspicious_facts = [f for f in facts_used if _looks_like_rating_out_of_five(f)]
        if suspicious_facts:
            failures.append(
                {
                    "topic": a["topic"],
                    "reason": "Likely fact fabrication: the agent passed numeric 'x/5' style ratings to "
                    "the drafting tool, a pattern that does not occur in real web search results and "
                    "was explicitly forbidden in the system prompt — the LLM hallucinated structured "
                    "data instead of using only retrieved evidence.",
                    "evidence": f"suspicious_facts={suspicious_facts}",
                }
            )
        if a["n_citations"] == 0:
            failures.append(
                {
                    "topic": a["topic"],
                    "reason": "No citations attached to the draft: RAG retrieval returned no usable "
                    "passages, or the agent drafted without citing any retrieved source.",
                    "evidence": f"n_facts_used={a['n_facts_used']}, n_reasoning_steps={a['n_reasoning_steps']}",
                }
            )
        if a["grounding_score"] < LOW_GROUNDING_THRESHOLD:
            failures.append(
                {
                    "topic": a["topic"],
                    "reason": f"Low grounding score ({a['grounding_score']:.2f}): the drafted text does "
                    "not clearly reflect the facts gathered during retrieval.",
                    "evidence": f"facts_used={a['draft'].get('facts_used', [])}",
                }
            )
        last_step = a["reasoning_trace"][-1] if a["reasoning_trace"] else None
        if last_step and "fallback" in last_step.get("justification", "").lower():
            failures.append(
                {
                    "topic": a["topic"],
                    "reason": f"Reasoning loop exhausted its step budget ({MAX_REACT_STEPS} steps) before "
                    "the model voluntarily called the drafting tool; a forced fallback draft was used.",
                    "evidence": str(last_step),
                }
            )
        for step in a["reasoning_trace"]:
            obs = str(step.get("observation", ""))
            if obs.startswith("No results found") or obs.startswith("No relevant passages"):
                failures.append(
                    {
                        "topic": a["topic"],
                        "reason": f"Tool '{step['tool']}' returned an empty result for this query, "
                        "reducing the factual basis available for the post.",
                        "evidence": f"args={step['tool_args']}",
                    }
                )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Blogger Support Agent evaluation")
    parser.add_argument("--posts", type=int, default=3, help="Number of posts to plan, draft and evaluate.")
    args = parser.parse_args()

    drafts = _run_agent_and_collect_drafts(args.posts)
    analyses = [_analyze_draft(d) for d in drafts]
    failures = _identify_failure_cases(analyses)

    lines = ["# Agent evaluation report\n"]
    lines.append(f"Analyzed {len(analyses)} drafted post(s) from a fresh end-to-end run (--posts {args.posts}).\n")

    lines.append("## Qualitative analysis of generated posts\n")
    for a in analyses:
        lines.append(f"### {a['topic']} ({a['post_type']})")
        lines.append(f"- Reasoning steps: {a['n_reasoning_steps']} | Quality checks performed: {a['n_quality_checks']}")
        lines.append(f"- Citations: {a['n_citations']} | Facts used: {a['n_facts_used']} | Grounding score: {a['grounding_score']:.2f}")
        lines.append(f"- Excerpt: {a['body_excerpt']!r}\n")

    lines.append("## Source quality and grounding assessment\n")
    if analyses:
        avg_grounding = sum(a["grounding_score"] for a in analyses) / len(analyses)
        avg_citations = sum(a["n_citations"] for a in analyses) / len(analyses)
        lines.append(f"- Average grounding score: {avg_grounding:.2f}")
        lines.append(f"- Average citations per post: {avg_citations:.1f}")
    lines.append("")

    lines.append("## Documented failure cases\n")
    if failures:
        for i, f in enumerate(failures, start=1):
            lines.append(f"### Failure case #{i} — {f['topic']}")
            lines.append(f"- Reason: {f['reason']}")
            lines.append(f"- Evidence: {f['evidence']}\n")
    else:
        lines.append("No failure cases were detected by the automated heuristics in this run.\n")

    lines.append("## Observability\n")
    if config.LANGCHAIN_API_KEY:
        lines.append(
            "- LangSmith tracing was active for this run (project: blogger-support-agent). "
            "Per-step latency, token usage and the full Thought/Action/Observation trace are "
            "available at https://smith.langchain.com."
        )
    else:
        lines.append(
            "- LangSmith tracing was NOT active for this run (no LANGCHAIN_API_KEY in .env). "
            "Set it to get per-step latency/token observability parameters."
        )

    config.EVAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.EVAL_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Evaluation report written to {config.EVAL_REPORT_PATH}")


if __name__ == "__main__":
    main()
