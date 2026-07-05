"""Valutazione end-to-end dell'agente: legge i run log prodotti da
observability/tracing.py durante le esecuzioni di `python main.py`
e ne ricava un'analisi qualitativa (grounding, qualita' delle fonti,
principali failure case), salvata in results/evaluation_report.md.

Uso (da dentro questa cartella):
    python evaluate_agent.py
"""
from __future__ import annotations

import json

import config
from agent.graph import MAX_REACT_STEPS
from finetuning.metrics import grounding_score

LOW_GROUNDING_THRESHOLD = 0.2


def _load_all_events() -> list[dict]:
    events = []
    for path in sorted(config.RUN_LOGS_DIR.glob("run_*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    return events


def _draft_payloads(events: list[dict]) -> list[dict]:
    return [e["payload"] for e in events if e["event"] == "gather_and_draft"]


def _analyze_draft(payload: dict) -> dict:
    draft = payload["draft"]
    reasoning_trace = payload["reasoning_trace"]
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
    """Vero se `text` contiene un pattern tipo "4/5" o "4 / 5" - i punteggi
    strutturati del dataset che non dovrebbero mai finire, tali e quali,
    nel testo generato (segno che il LLM li ha copiati invece di scrivere
    prosa a partire dai fatti)."""
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
    events = _load_all_events()
    if not events:
        print(
            f"No run logs found in {config.RUN_LOGS_DIR}. "
            "Run 'python main.py' at least once first."
        )
        return

    draft_payloads = _draft_payloads(events)
    analyses = [_analyze_draft(p) for p in draft_payloads]
    failures = _identify_failure_cases(analyses)

    lines = ["# Agent evaluation report\n"]
    lines.append(f"Analyzed {len(analyses)} drafted post(s) across {len(list(config.RUN_LOGS_DIR.glob('run_*.jsonl')))} run log(s).\n")

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
        lines.append("No failure cases were detected by the automated heuristics in the available run logs.\n")

    lines.append("## Observability\n")
    lines.append(
        "- Local structured run logs (Thought/Action/Observation, tool outputs, KG summaries, "
        f"plan): {config.RUN_LOGS_DIR}/run_*.jsonl (always on, no account required).\n"
        "- LangSmith tracing: enabled automatically if LANGCHAIN_API_KEY is set in .env "
        "(see observability/tracing.py)."
    )

    config.EVAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.EVAL_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Evaluation report written to {config.EVAL_REPORT_PATH}")


if __name__ == "__main__":
    main()
