# Blogger Support Agent

Sistema agentico LangGraph per supportare un blog di viaggi: pianifica una
sequenza di post, raccoglie e verifica informazioni da internet (Search +
K-RAG), le confronta con un Knowledge Graph editoriale (Neo4j), genera il
draft con un modello fine-tuned, e richiede approvazione umana prima di
aggiornare il Knowledge Graph.

Tutte le componenti girano in locale o su free tier: Neo4j Community
(locale), Qwen2.5-3B-Instruct per il reasoning (locale, su GPU), embeddings
e fine-tuning locali (sentence-transformers / distilgpt2), DuckDuckGo
Search, FAISS.

Tutti i comandi sotto vanno lanciati da dentro questa cartella (`progetto/`),
con l'interprete del virtualenv: `.venv/bin/python` (oppure attiva il
virtualenv con `source .venv/bin/activate` e usa semplicemente `python`).

## Setup

1. Le credenziali (`HF_TOKEN`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`)
   sono nel file `.env` in questa cartella.
2. Avvia Neo4j (se non e' gia' in esecuzione):
   ```bash
   ./runtime/start_neo4j.sh
   ```
   Console disponibile su http://localhost:7474 (bolt su :7687).
3. Installa le dipendenze:
   ```bash
   .venv/bin/pip install -r requirements.txt
   ```
4. (Opzionale) Osservabilita' via LangSmith: imposta `LANGCHAIN_API_KEY` in
   `.env` con una chiave gratuita da https://smith.langchain.com — se
   presente, `main.py` ed `evaluate_agent.py` attivano automaticamente il
   tracing (progetto `blogger-support-agent`), con traccia completa
   Thought/Action/Observation, latenza e token per ogni step, consultabile
   sulla dashboard di LangSmith. Se assente, l'esecuzione continua
   normalmente, solo senza quella traccia esterna.
   Se il tuo account LangSmith e' sulla regione **EU** (non US, il
   default), imposta anche `LANGCHAIN_ENDPOINT=https://eu.api.smith.langchain.com`
   in `.env` — altrimenti l'invio delle trace fallisce con `403 Forbidden`
   (il client punta all'endpoint sbagliato, non e' un problema di chiave).

## Pipeline di fine-tuning (tool basato sul modello fine-tuned)

```bash
.venv/bin/python -m finetuning.dataset_builder   # costruisce il dataset fact-grounded (train/val/test 80/10/10)
.venv/bin/python -m finetuning.train             # baseline + hyperparameter sweep + training finale
.venv/bin/python -m finetuning.evaluate          # confronto finale su test set + error analysis
```

Output in `finetuning/results/` (`training_log.json`, `final_evaluation.md`/
`.json`) e modello finale in
`finetuning/saved_models/distilgpt2_context_grounded/`.

## Esecuzione dell'agente

```bash
.venv/bin/python main.py --posts 3
```

Per ogni post pianificato, l'agente mostra il draft generato e chiede
`[a]pprove / [m]odify / [r]eject`. Il Knowledge Graph viene aggiornato solo
dopo l'approvazione (eventualmente con il testo modificato).

## Valutazione dell'agente

```bash
.venv/bin/python evaluate_agent.py --posts 3
```

Esegue l'agente end-to-end approvando automaticamente ogni draft (nessuna
revisione umana reale) e produce `results/evaluation_report.md` (analisi
qualitativa, grounding/qualita' delle fonti, failure case, stato del
tracing LangSmith). Il dettaglio di ogni fase (piano, tool chiamati, draft)
viene comunque stampato a console con dei semplici `print()` di debug.

## Struttura

```
config.py, state.py        # configurazione e stato condiviso (MCP-style)
kg/client.py                # Knowledge Graph (Neo4j)
rag/                         # search, fetch, chunking, FAISS, K-RAG
finetuning/                  # dataset, training, valutazione del modello fine-tuned
tools/                        # i 5 tool dell'agente (3 core + 2 custom)
agent/                         # LLM, prompt, grafo LangGraph
main.py, evaluate_agent.py       # entrypoint CLI
```
