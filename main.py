"""Entrypoint CLI: fa girare tutto il ciclo, dalla pianificazione fino
all'aggiornamento del KG.

Uso: python main.py [--posts N]
"""
from __future__ import annotations

import argparse
import textwrap
import uuid

from langgraph.types import Command

import config
from agent.graph import build_graph


def _print_draft(draft: dict) -> None:
    print("\n" + "=" * 78)
    print(f"DRAFT — {draft['title']}  [{draft['post_type']}]")
    print("=" * 78)
    print(textwrap.fill(draft["body"], width=78))
    if draft.get("citations"):
        print("\nSources:")
        for c in draft["citations"]:
            print(f"  - {c['url']}")
    print("=" * 78)


def _ask_human_decision() -> dict:
    while True:
        choice = input("\nApprove / Modify / Reject this post? [a/m/r]: ").strip().lower()
        if choice in ("a", "approve"):
            return {"decision": "approve"}
        if choice in ("r", "reject"):
            return {"decision": "reject"}
        if choice in ("m", "modify"):
            print("Enter the edited post text (end with an empty line):")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            return {"decision": "modify", "feedback": "\n".join(lines)}
        print("Please answer with a, m, or r.")


def run(n_posts: int) -> None:
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
        interrupt_payload = result["__interrupt__"][0].value
        _print_draft(interrupt_payload["draft"])
        decision = _ask_human_decision()
        result = app.invoke(Command(resume=decision), config=thread_config)

    print(f"\nSession complete. {len(result['approved_posts'])} post(s) approved and written to the Knowledge Graph.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Blogger Support Agent")
    parser.add_argument("--posts", type=int, default=3, help="Number of posts to plan and draft.")
    args = parser.parse_args()
    run(args.posts)


if __name__ == "__main__":
    main()
