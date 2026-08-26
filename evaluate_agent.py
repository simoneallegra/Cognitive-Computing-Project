"""Valuta l'agente: fa girare il grafo in-process (approvando i draft
in automatico) e produce un report di sole metriche, senza prosa.

Uso: python evaluate_agent.py [--posts N]
"""
from __future__ import annotations

import argparse
import re
import uuid

from langgraph.types import Command

import config
import state as state_module
from agent.graph import MAX_REACT_STEPS, build_graph
from finetuning.metrics import grounding_score

LOW_GROUNDING_THRESHOLD = 0.2
_RATING_OUT_OF_FIVE_RE = re.compile(r"(?<!\w)[1-5]\s*/\s*5(?!\w)")


def _run_agent_and_collect_drafts(n_posts: int):
    """Fa girare l'agente end-to-end approvando ogni draft senza chiedere
    conferma, e ritorna i post approvati con la loro reasoning_trace."""

    app = build_graph()
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    result = app.invoke(state_module.initial_state(n_posts), config=thread_config)
    while "__interrupt__" in result:
        result = app.invoke(Command(resume={"decision": "approve"}), config=thread_config)
    return result["approved_posts"]


def _looks_like_rating_out_of_five(text: str) -> bool:
    """True se `text` contiene un voto tipo "4/5": segno che il LLM ha
    copiato un rating dal dataset invece di scrivere testo vero."""
    return bool(_RATING_OUT_OF_FIVE_RE.search(text))


def _analyze_draft(draft: dict) -> dict:
    reasoning_trace = draft["reasoning_trace"]
    facts_used = draft.get("facts_used", [])

    n_quality_checks = 0
    for s in reasoning_trace:
        if s["tool"] == "resource_quality_tool":
            n_quality_checks += 1

    return {
        "topic": draft["topic"],
        "post_type": draft["post_type"],
        "n_reasoning_steps": len(reasoning_trace),
        "n_quality_checks": n_quality_checks,
        "n_citations": len(draft.get("citations", [])),
        "n_facts_used": len(facts_used),
        "grounding_score": grounding_score(draft["body"], facts_used) if facts_used else 0.0,
        "reasoning_trace": reasoning_trace,
        "facts_used": facts_used,
    }


def _identify_failure_cases(analyses):
    """Quattro euristiche automatiche per post: fatti fabbricati, nessuna
    citazione, grounding basso, budget ReAct esaurito -- piu' un controllo
    per ogni step con un tool andato a vuoto. `evidence` e' sempre un dato
    grezzo (numero, lista, dict), mai una frase."""
    failures = []
    for a in analyses:
        topic = a["topic"]

        suspicious = []
        for f in a["facts_used"]:
            if _looks_like_rating_out_of_five(f):
                suspicious.append(f)
        if suspicious:
            failures.append({"topic": topic, "code": "fabricated_facts", "evidence": suspicious})
        if a["n_citations"] == 0:
            failures.append({"topic": topic, "code": "no_citations", "evidence": a["n_facts_used"]})
        if a["grounding_score"] < LOW_GROUNDING_THRESHOLD:
            failures.append({"topic": topic, "code": "low_grounding", "evidence": a["grounding_score"]})

        last_step = a["reasoning_trace"][-1] if a["reasoning_trace"] else None
        if last_step and "fallback" in last_step.get("justification", "").lower():
            failures.append({"topic": topic, "code": "budget_exhausted", "evidence": MAX_REACT_STEPS})

        for step in a["reasoning_trace"]:
            obs = str(step.get("observation", ""))
            if obs.startswith(("No results found", "No relevant passages")):
                failures.append({"topic": topic, "code": "empty_tool_result", "evidence": step["tool"]})
    return failures


def _report(analyses, failures, n_posts: int) -> str:
    posts_lines = []
    for a in analyses:
        posts_lines.append(
            f"| {a['topic']} | {a['post_type']} | {a['n_reasoning_steps']} | {a['n_quality_checks']} | "
            f"{a['n_citations']} | {a['n_facts_used']} | {a['grounding_score']:.2f} |"
        )
    posts_table = "\n".join(posts_lines)

    grounding_sum = 0.0
    citations_sum = 0
    for a in analyses:
        grounding_sum += a["grounding_score"]
        citations_sum += a["n_citations"]
    avg_grounding = grounding_sum / len(analyses) if analyses else 0.0
    avg_citations = citations_sum / len(analyses) if analyses else 0.0

    failures_lines = []
    for f in failures:
        failures_lines.append(f"| {f['topic']} | {f['code']} | {f['evidence']} |")
    failures_table = "\n".join(failures_lines) or "| — | none | — |"

    return (
        f"# Agent evaluation report ({n_posts} posts)\n\n"
        "## posts\n"
        "| topic | type | steps | quality_checks | citations | facts | grounding |\n"
        "|---|---|---|---|---|---|---|\n"
        f"{posts_table}\n\n"
        f"avg_grounding: {avg_grounding:.2f}\n"
        f"avg_citations: {avg_citations:.1f}\n\n"
        "## failure_cases\n"
        "| topic | code | evidence |\n"
        "|---|---|---|\n"
        f"{failures_table}\n\n"
        "## observability\n"
        f"langsmith_active: {bool(config.LANGCHAIN_API_KEY)}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Blogger Support Agent evaluation")
    parser.add_argument("--posts", type=int, default=3, help="Number of posts to plan, draft and evaluate.")
    args = parser.parse_args()

    drafts = _run_agent_and_collect_drafts(args.posts)
    analyses = []
    for d in drafts:
        analyses.append(_analyze_draft(d))
    failures = _identify_failure_cases(analyses)
    report = _report(analyses, failures, args.posts)

    config.EVAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.EVAL_REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Evaluation report written to {config.EVAL_REPORT_PATH}")


if __name__ == "__main__":
    main()
