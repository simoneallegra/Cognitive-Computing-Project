"""Fine-tuning del draft generator grounded sui fatti.

1. baseline = distilgpt2 non fine-tuned, stesso prompt fact-grounded (zero-shot)
2. hyperparameter search (lr x epochs) sul validation set
3. training finale con la configurazione migliore
4. confronto finale con baseline + vecchio modello (senza contesto) ed
   error analysis: in evaluate.py, sul test set
"""
from __future__ import annotations

import gc
import json
import math
import random
from dataclasses import dataclass, asdict

import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

import config
from finetuning.metrics import grounding_score, perplexity_from_loss

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HYPERPARAM_GRID = [
    {"learning_rate": 5e-5, "num_train_epochs": 3},
    {"learning_rate": 5e-5, "num_train_epochs": 5},
    {"learning_rate": 1e-4, "num_train_epochs": 3},
    {"learning_rate": 1e-4, "num_train_epochs": 5},
]
N_GROUNDING_SAMPLES = 20


@dataclass
class RunResult:
    config: dict
    eval_loss: float
    perplexity: float
    grounding_score: float


def _tokenize(example: dict, tokenizer) -> dict:
    """Tokenizza prompt+completamento e maschera la loss sul prompt (label
    -100): il modello deve imparare a SCRIVERE il post, non a prevedere il
    blocco "Facts:" che gli viene dato in input. Senza questo masking,
    DataCollatorForLanguageModeling calcolerebbe la loss sull'intera
    sequenza, insegnando al modello a rigurgitare i fatti (inclusi i
    punteggi "x/5") invece di limitarsi a scriverci sopra un post."""
    prompt_section = f"<PROMPT>\n{example['prompt']}\n<RESP>\n"
    full_text = prompt_section + example["completion"] + tokenizer.eos_token

    prompt_ids = tokenizer(prompt_section, truncation=True, max_length=512)["input_ids"]
    encoded = tokenizer(full_text, truncation=True, max_length=512)

    prompt_len = min(len(prompt_ids), len(encoded["input_ids"]))
    labels = list(encoded["input_ids"])
    for i in range(prompt_len):
        labels[i] = -100
    encoded["labels"] = labels
    return encoded


def load_tokenized_dataset(tokenizer):
    dataset_dict = load_from_disk(str(config.FT_DATA_DIR / "context_grounded_dataset"))
    text_columns = dataset_dict["train"].column_names

    tokenized = dataset_dict.map(
        _tokenize,
        fn_kwargs={"tokenizer": tokenizer},
        batched=False,
        remove_columns=[c for c in text_columns if c not in ("prompt", "facts")],
    )
    return dataset_dict, tokenized


def fresh_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(config.FINETUNE_BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config.FINETUNE_BASE_MODEL)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(DEVICE)
    return model, tokenizer


def evaluate_grounding(model, tokenizer, raw_validation, n_samples: int = N_GROUNDING_SAMPLES) -> float:
    """Genera il continuamento per un campione di prompt di validation e
    misura quanti dei fatti elencati nel prompt compaiono nel testo
    generato (vedi finetuning/metrics.py)."""
    model.eval()
    sample = raw_validation.shuffle(seed=SEED).select(range(min(n_samples, len(raw_validation))))
    scores = []
    for row in sample:
        prompt_text = f"<PROMPT>\n{row['prompt']}\n<RESP>\n"
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=400).to(DEVICE)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        scores.append(grounding_score(generated, row["facts"]))
    return float(np.mean(scores)) if scores else 0.0


def run_baseline(tokenized, raw_validation) -> RunResult:
    print("\n=== BASELINE (distilgpt2 zero-shot) ===")
    model, tokenizer = fresh_model_and_tokenizer()
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(config.FT_RESULTS_DIR / "tmp_baseline"),
            per_device_eval_batch_size=4,
            report_to="none",
            seed=SEED,
        ),
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    eval_metrics = trainer.evaluate()
    grounding = evaluate_grounding(model, tokenizer, raw_validation)
    result = RunResult(
        config={"name": "baseline_zero_shot"},
        eval_loss=eval_metrics["eval_loss"],
        perplexity=perplexity_from_loss(eval_metrics["eval_loss"]),
        grounding_score=grounding,
    )
    print(result)
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_config(hp: dict, tokenized, raw_validation, save_dir=None):
    print(f"\n=== TRAINING config: {hp} ===")
    model, tokenizer = fresh_model_and_tokenizer()
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)

    training_args = TrainingArguments(
        output_dir=str(config.FT_RESULTS_DIR / "tmp_sweep"),
        learning_rate=hp["learning_rate"],
        num_train_epochs=hp["num_train_epochs"],
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    grounding = evaluate_grounding(model, tokenizer, raw_validation)

    result = RunResult(
        config=hp,
        eval_loss=eval_metrics["eval_loss"],
        perplexity=perplexity_from_loss(eval_metrics["eval_loss"]),
        grounding_score=grounding,
    )
    print(result)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(save_dir))
        tokenizer.save_pretrained(str(save_dir))

    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    return result, model, tokenizer


def main() -> None:
    config.FT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config.FINETUNE_BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_dataset, tokenized = load_tokenized_dataset(tokenizer)

    baseline_result = run_baseline(tokenized, raw_dataset["validation"])

    sweep_results = []
    best_result = None
    best_hp = None
    for hp in HYPERPARAM_GRID:
        result, model, tok = run_config(hp, tokenized, raw_dataset["validation"])
        sweep_results.append(result)
        if best_result is None or result.perplexity < best_result.perplexity:
            best_result = result
            best_hp = hp
        del model, tok
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n=== BEST CONFIG (by validation perplexity): {best_hp} ===")
    print(best_result)

    print("\n=== Re-training best config to save the final model ===")
    final_result, final_model, final_tokenizer = run_config(
        best_hp, tokenized, raw_dataset["validation"], save_dir=config.FT_FINAL_MODEL_DIR
    )

    log = {
        "baseline": asdict(baseline_result),
        "sweep": [asdict(r) for r in sweep_results],
        "best_config": best_hp,
        "final_model_validation": asdict(final_result),
        "final_model_dir": str(config.FT_FINAL_MODEL_DIR),
    }
    with (config.FT_RESULTS_DIR / "training_log.json").open("w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    print(f"\nFinal model saved to {config.FT_FINAL_MODEL_DIR}")
    print(f"Training log saved to {config.FT_RESULTS_DIR / 'training_log.json'}")


if __name__ == "__main__":
    main()
