"""Confronto finale sul test set tra baseline, vecchio modello fine-tuned
e nuovo modello, con analisi degli errori (almeno 3 failure case)
"""
from __future__ import annotations

import json

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from finetuning.metrics import grounding_score, perplexity_from_loss, rouge_l

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_QUALITATIVE_SAMPLES = 30


def _load_model(model_dir_or_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir_or_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir_or_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


def _generate(model, tokenizer, prompt_text: str, max_new_tokens: int = 150) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=400).to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id
        )
    return tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def _perplexity_on_rows(model, tokenizer, rows, prompt_builder) -> float:
    total_loss, total_tokens = 0.0, 0
    for row in rows:
        full_text = prompt_builder(row, tokenizer) + row["completion"] + tokenizer.eos_token
        enc = tokenizer(full_text, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
        n_tokens = enc["input_ids"].shape[1]
        total_loss += out.loss.item() * n_tokens
        total_tokens += n_tokens
    return perplexity_from_loss(total_loss / total_tokens)


def _context_grounded_prompt(row, _tokenizer) -> str:
    return f"<PROMPT>\n{row['prompt']}\n<RESP>\n"


_LEGACY_INSTRUCTION = {
    "review": "Write a travel review for {topic}.",
    "how_to": "Write a 'how to' travel post for {topic}.",
    "ranking": "Write a ranking travel post of the top cities for {topic}",
}


def _legacy_prompt(row, _tokenizer) -> str:
    instruction = _LEGACY_INSTRUCTION.get(row["post_type"], _LEGACY_INSTRUCTION["review"]).format(topic=row["topic"])
    return f"<PROMPT> {instruction} \n <RESP> "


def evaluate_model(label: str, model_dir_or_name: str, rows, prompt_builder) -> dict:
    print(f"\n=== Evaluating: {label} ===")
    model, tokenizer = _load_model(model_dir_or_name)

    perplexity = _perplexity_on_rows(model, tokenizer, rows, prompt_builder)

    sample = rows.shuffle(seed=42).select(range(min(N_QUALITATIVE_SAMPLES, len(rows))))
    records = []
    for row in sample:
        prompt_text = prompt_builder(row, tokenizer)
        generated = _generate(model, tokenizer, prompt_text)
        records.append(
            {
                "topic": row["topic"],
                "post_type": row["post_type"],
                "facts": row["facts"],
                "generated": generated,
                "reference": row["completion"],
                "grounding_score": grounding_score(generated, row["facts"]),
                "rouge_l": rouge_l(generated, row["completion"]),
            }
        )

    grounding_sum = 0.0
    rouge_sum = 0.0
    for r in records:
        grounding_sum += r["grounding_score"]
        rouge_sum += r["rouge_l"]
    avg_grounding = grounding_sum / len(records)
    avg_rouge = rouge_sum / len(records)

    print(f"{label}: perplexity={perplexity:.2f} grounding={avg_grounding:.3f} rougeL={avg_rouge:.3f}")

    del model
    torch.cuda.empty_cache()

    return {
        "label": label,
        "perplexity": perplexity,
        "avg_grounding_score": avg_grounding,
        "avg_rouge_l": avg_rouge,
        "records": records,
    }


def _format_failure_case(record: dict, reason: str) -> str:
    return (
        f"### Failure case ({reason})\n"
        f"- Topic: {record['topic']} ({record['post_type']})\n"
        f"- Facts given: {record['facts']}\n"
        f"- Grounding score: {record['grounding_score']:.2f} | ROUGE-L: {record['rouge_l']:.2f}\n"
        f"- Generated: {record['generated'][:400]!r}\n"
        f"- Reference: {record['reference'][:400]!r}\n"
    )


def main() -> None:
    dataset_dict = load_from_disk(str(config.FT_DATA_DIR / "context_grounded_dataset"))
    test_rows = dataset_dict["test"]

    results = []
    results.append(
        evaluate_model("baseline_zero_shot_distilgpt2", config.FINETUNE_BASE_MODEL, test_rows, _context_grounded_prompt)
    )
    if config.LEGACY_FT_MODEL_DIR.exists():
        results.append(
            evaluate_model(
                "legacy_finetuned_no_context", str(config.LEGACY_FT_MODEL_DIR), test_rows, _legacy_prompt
            )
        )
    results.append(
        evaluate_model(
            "finetuned_context_grounded (final tool model)",
            str(config.FT_FINAL_MODEL_DIR),
            test_rows,
            _context_grounded_prompt,
        )
    )

    # error analysis sul modello finale (quello effettivamente usato dal tool dell'agente)
    final_records = results[-1]["records"]
    sorted_by_grounding = sorted(final_records, key=lambda r: r["grounding_score"])
    sorted_by_rouge = sorted(final_records, key=lambda r: r["rouge_l"])
    best_example = max(final_records, key=lambda r: r["grounding_score"])

    failure_cases = sorted_by_grounding[:2] + sorted_by_rouge[:1]

    comparison = []
    for r in results:
        r_without_records = {}
        for k, v in r.items():
            if k != "records":
                r_without_records[k] = v
        comparison.append(r_without_records)

    config.FT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (config.FT_RESULTS_DIR / "final_evaluation.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "comparison": comparison,
                "failure_cases": failure_cases,
                "best_example": best_example,
            },
            f,
            indent=2,
        )

    report_lines = ["# Fine-tuning: final comparison (test set)\n"]
    report_lines.append("| Model | Perplexity | Grounding score | ROUGE-L |")
    report_lines.append("|---|---|---|---|")
    for r in results:
        report_lines.append(f"| {r['label']} | {r['perplexity']:.2f} | {r['avg_grounding_score']:.3f} | {r['avg_rouge_l']:.3f} |")

    report_lines.append("\n## Qualitative success example\n")
    report_lines.append(_format_failure_case(best_example, "best grounding score"))

    report_lines.append("\n## Documented failure cases\n")
    for i, case in enumerate(failure_cases, start=1):
        report_lines.append(_format_failure_case(case, f"failure case #{i}"))

    report_path = config.FT_RESULTS_DIR / "final_evaluation.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\nFinal comparison + error analysis written to {report_path}")


if __name__ == "__main__":
    main()
