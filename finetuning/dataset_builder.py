"""Costruisce il dataset di fine-tuning per il draft generator: dato un
elenco di fatti, il modello deve imparare a generare il post

Riusa i generatori euristici di dataset_generation/ come completion
supervisionata, associata a un prompt che elenca esplicitamente i fatti
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import kagglehub  # noqa: E402
import pandas as pd  # noqa: E402
from datasets import Dataset, DatasetDict  # noqa: E402

from dataset_generation.bestof_generator import generate_ranking_post  # noqa: E402
from dataset_generation.how_to_post_generator import generate_howto  # noqa: E402
from dataset_generation.review_generator import generate_review  # noqa: E402

import config  # noqa: E402

SEED = 42
KAGGLE_DATASET = "furkanima/worldwide-travel-cities-ratings-and-climate"
TASK_INSTRUCTION = {
    "review": "Write a travel review blog post using ONLY the facts above.",
    "how_to": "Write a 'how to spend your time' travel guide post using ONLY the facts above.",
    "ranking": "Write a 'Top cities' ranking blog post using ONLY the facts above.",
}


def _load_destinations() -> list[dict[str, Any]]:
    download_path = kagglehub.dataset_download(KAGGLE_DATASET)
    csv_file = sorted(Path(download_path).rglob("*.csv"))[0]
    df = pd.read_csv(csv_file)
    return [df.iloc[i].to_dict() for i in range(len(df))]


def _fact_lines_for_destination(data: dict[str, Any]) -> list[str]:
    lines = [f"City: {data['city']}, {data.get('region', '')}, {data['country']}".replace(", ,", ",")]
    if data.get("short_description"):
        lines.append(f"Description: {data['short_description']}")
    lines.append(f"Ideal durations: {data.get('ideal_durations')}")
    lines.append(f"Budget level: {data.get('budget_level')}")
    for category in config.TOPIC_CATEGORIES:
        lines.append(f"{category}: {data.get(category)}/5")
    return lines


def _fact_lines_for_ranking(top_destinations: list[dict[str, Any]], category: str) -> list[str]:
    lines = [f"Category: {category}"]
    for i, dest in enumerate(top_destinations, start=1):
        lines.append(f"{i}. {dest['city']}, {dest['country']} - {category}: {dest.get(category)}/5")
    return lines


def _build_example(post_type: str, fact_lines: list[str], topic: str, completion: str) -> dict[str, Any]:
    facts_block = "\n".join(fact_lines)
    prompt = f"Facts:\n{facts_block}\n\nTopic: {topic}\nTask: {TASK_INSTRUCTION[post_type]}"
    return {
        "prompt": prompt,
        "completion": completion,
        "post_type": post_type,
        "facts": fact_lines,
        "topic": topic,
    }


def build_examples(destinations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []

    for dest in destinations:
        review_facts = _fact_lines_for_destination(dest)
        examples.append(
            _build_example("review", review_facts, dest["city"], generate_review(dest))
        )

        howto_facts = _fact_lines_for_destination(dest)
        examples.append(
            _build_example("how_to", howto_facts, dest["city"], generate_howto(dest))
        )

    for category in config.TOPIC_CATEGORIES:
        ranked_post = generate_ranking_post(destinations, category=category, top_n=5)
        top_destinations = sorted(
            destinations, key=lambda d: d.get(category, 0), reverse=True
        )[:5]
        ranking_facts = _fact_lines_for_ranking(top_destinations, category)
        examples.append(
            _build_example("ranking", ranking_facts, f"Top cities for {category}", ranked_post)
        )

    random.Random(SEED).shuffle(examples)
    return examples


def split_dataset(examples: list[dict[str, Any]]) -> DatasetDict:
    """Divide il dataset 80/10/10 in train/validation/test."""
    dataset = Dataset.from_list(examples)
    train_test = dataset.train_test_split(test_size=0.2, seed=SEED)
    val_test = train_test["test"].train_test_split(test_size=0.5, seed=SEED)
    return DatasetDict(
        {
            "train": train_test["train"],
            "validation": val_test["train"],
            "test": val_test["test"],
        }
    )


def main() -> None:
    print("Downloading/loading Kaggle dataset...")
    destinations = _load_destinations()
    print(f"Loaded {len(destinations)} destinations.")

    examples = build_examples(destinations)
    print(f"Built {len(examples)} fact-grounded examples "
          f"(review/how_to per city + ranking per category).")

    dataset_dict = split_dataset(examples)
    for split_name, split in dataset_dict.items():
        print(f"  {split_name}: {len(split)} examples")

    config.FT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(config.FT_DATA_DIR / "context_grounded_dataset"))

    # copia leggibile in jsonl, utile per ispezione manuale / report
    for split_name, split in dataset_dict.items():
        out_path = config.FT_DATA_DIR / f"{split_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in split:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Dataset saved to {config.FT_DATA_DIR}")


if __name__ == "__main__":
    main()
