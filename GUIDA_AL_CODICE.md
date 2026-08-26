# Guida allo studio del codice — Blogger Support Agent (versione estesa)

Percorso di lettura pensato per capire il progetto dal più semplice al più
complesso, senza saltare da un file all'altro senza contesto. Per ogni file:
perché leggerlo ora, il codice per intero (o quasi), cosa notare riga per
riga, come si collega al resto, e qualche domanda per verificare di aver
capito prima di andare avanti.

**Percorso:**
0. Concetti di base (da leggere una volta sola, prima di tutto)
1. `state.py` — il vocabolario condiviso
2. `config.py` — configurazione e percorsi
3. `kg/client.py` — il Knowledge Graph (Neo4j)
4. `tools/` — i 5 strumenti dell'agente
5. `rag/` — ricerca e retrieval dal web
6. `agent/llm.py` + `agent/prompts.py` — il modello e i prompt
7. `agent/graph.py` — il cuore: il grafo LangGraph e il ciclo ReAct
8. `main.py` — l'entrypoint CLI
9. `evaluate_agent.py` — valutazione end-to-end dell'agente
10. `finetuning/` — pipeline di training del modello di drafting

Appendice: glossario dei termini usati.

---

## Step 0 — Concetti di base

Prima di leggere codice, vale la pena fissare cinque idee. Se già le
conosci, salta al Step 1.

### LangGraph in breve

LangGraph modella un agente come una **macchina a stati**: un grafo dove i
nodi sono funzioni Python (`def nodo(state: dict) -> dict`, ognuna riceve
lo stato e ritorna un dizionario con i campi da aggiornare) e gli archi
dicono quale nodo eseguire dopo. Due tipi di arco:
- **fisso** (`add_edge("a", "b")`): dopo `a` si va sempre a `b`.
- **condizionale** (`add_conditional_edges("a", funzione_di_routing, {...})`):
  dopo `a` viene chiamata `funzione_di_routing(state)`, che ritorna una
  stringa che decide il prossimo nodo.

Il grafo viene costruito con `StateGraph(TypeDelloStato)` e poi
`.compile()`. Con un **checkpointer** (qui `MemorySaver`), lo stato viene
salvato ad ogni step: questo è ciò che permette a `interrupt()` di
sospendere l'esecuzione a metà e riprenderla più tardi esattamente da dove
si era fermata, anche se nel frattempo il processo ha aspettato input da
tastiera.

### Il pattern ReAct

"ReAct" = **Reason + Act**. Il LLM, invece di rispondere subito, alterna:
*Thought* (un pensiero testuale su cosa fare), *Action* (chiama uno
strumento/tool), *Observation* (il risultato del tool, che rientra nella
conversazione). Si ripete finché il LLM non decide di fermarsi (di solito
perché ha raccolto abbastanza informazioni per rispondere). Nel nostro
progetto questo è implementato "a mano" nel nodo `gather_and_draft` (Step 7).

### Tool-calling (function calling)

I LLM moderni possono, invece di scrivere solo testo, emettere una
richiesta strutturata tipo "chiama la funzione `search_web_tool` con
argomento `{"query": "Bangkok street food"}`". LangChain rappresenta questa
richiesta come `AIMessage.tool_calls`, una lista di dizionari
`{"name", "args", "id"}`. Chi ha scritto il codice deve poi davvero eseguire
quella funzione e restituire il risultato al modello come `ToolMessage`,
prima di lasciarlo continuare a ragionare. Il modo in cui questo viene
comunicato al modello (che formato di prompt usare) dipende dal modello:
Qwen2.5 usa blocchi XML `<tool_call>{...}</tool_call>` — è il motivo per
cui esiste `agent/llm.py` (Step 6).

### RAG e K-RAG

**RAG** (Retrieval-Augmented Generation): invece di fidarsi solo di quello
che il LLM "sa" già, si recuperano documenti freschi (qui: pagine web), si
spezzano in passaggi (chunk), si trasformano in vettori numerici
(embedding) e si cerca per similarità semantica quali passaggi sono
rilevanti per la domanda corrente. **K-RAG** in questo progetto è
un'estensione: prima di cercare, la query viene arricchita con topic
correlati presi dal Knowledge Graph, per recuperare anche contenuti
collegati a ciò che il blog ha già pubblicato.

### Knowledge Graph e Cypher

Un Knowledge Graph è un database a grafo: nodi con etichette e proprietà,
collegati da relazioni tipizzate. Qui gira su **Neo4j**, interrogato con il
linguaggio **Cypher**. Tre parole chiave bastano per leggere le query di
questo progetto: `MATCH` (cerca un pattern esistente), `MERGE` (crea il
pattern se non esiste, altrimenti lo riusa — evita duplicati), `RETURN`
(restituisce i risultati).

### Fine-tuning: vocabolario minimo

- **Baseline zero-shot**: il modello *non* allenato, valutato con lo stesso
  prompt che userà il modello finale — il punto di partenza da battere.
- **Hyperparameter sweep**: provare più combinazioni di parametri di
  training (qui: learning rate × numero di epoche) e tenere la migliore.
- **Perplexity**: quanto il modello è "sorpreso" dal testo corretto — più
  bassa è, meglio il modello prevede il linguaggio atteso.
- **ROUGE-L**: sovrapposizione (sequenza comune più lunga) tra testo
  generato e un riferimento — usato qui come controprova, non come metrica
  principale.
- **Grounding score**: metrica *ad hoc* di questo progetto — quanti dei
  fatti forniti nel prompt si ritrovano davvero nel testo generato. È la
  metrica più importante perché misura la cosa che conta di più per un
  blog editoriale: non inventare.

---

## Step 1 — `state.py`

Si parte da qui perché **non contiene nessuna logica**, solo definizioni di
dati: è il modo più veloce per imparare il vocabolario che si userà in tutti
gli altri file.

```python
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


def initial_state(n_posts: int) -> BlogAgentState:
    return {
        "user_input": f"Plan and draft {n_posts} travel blog posts.",
        "n_posts": n_posts, "messages": [], "reasoning_trace": [], "tool_outputs": [],
        "kg_summary": {}, "plan": [], "current_post_idx": 0, "gathered_facts": [],
        "draft": None, "human_decision": None, "human_feedback": None, "approved_posts": [],
    }
```
`initial_state(n_posts)` è la fabbrica dello stato di partenza: un valore
per ognuna delle 13 chiavi di `BlogAgentState`, nessuna omessa (LangGraph
non assume default impliciti). Vive qui, non duplicata in `main.py` e
`evaluate_agent.py` (Step 8-9): entrambi la chiamano invece di ricopiare
lo stesso dizionario due volte.

**Cosa notare, campo per campo:**
- `TypedDict` è solo un dizionario Python normale a runtime — le
  annotazioni di tipo servono all'editor/type-checker, non cambiano il
  comportamento. Quindi `state["draft"]` è un dict vero, accedibile con
  `["chiave"]` come sempre.
- `BlogAgentState` è l'unico "stato" che LangGraph passa da un nodo
  all'altro del grafo: ogni funzione che vedremo dopo (`plan_posts`,
  `gather_and_draft`, ecc.) riceve l'intero stato e ritorna un dizionario
  parziale con solo i campi che vuole aggiornare — LangGraph si occupa di
  fare il merge.
- `messages: Annotated[list[AnyMessage], add_messages]` è l'unico campo
  "speciale": quell'`Annotated` dice a LangGraph di usare la funzione
  `add_messages` come *reducer* — invece di sovrascrivere la lista,
  accoda i nuovi messaggi a quelli già presenti. Tutti gli altri campi
  (come `plan` o `draft`) vengono invece sovrascritti per intero ogni volta
  che un nodo li restituisce.
- `current_post_idx` traccia quale post del piano si sta processando in
  questo momento: l'agente lavora su un post alla volta, in un ciclo che
  avanza questo indice (lo vedremo in `update_kg`, Step 7).
- `RetrievedFact` esiste come tipo ma nella pratica del codice attuale
  viene popolato con `source_url`/`source_title` vuoti e `quality_score`
  a 0.0 (vedi `gather_and_draft` in Step 7) — è più uno scaffolding per
  estensioni future che qualcosa di pienamente sfruttato oggi.

**Verifica la comprensione:**
- Se un nodo ritorna `{"plan": nuovo_piano}`, cosa succede agli altri campi
  dello stato, tipo `kg_summary`? (Risposta: restano invariati — LangGraph
  aggiorna solo le chiavi presenti nel dizionario ritornato.)
- Perché `messages` ha bisogno di un reducer (`add_messages`) e `plan` no?

---

## Step 2 — `config.py`

Niente logica applicativa, solo valori globali che tutti gli altri moduli
importano con `import config` e usano come `config.TOPIC_CATEGORIES`,
`config.NEO4J_URI`, ecc.

```python
PROJECT_DIR = Path(__file__).resolve().parent

load_dotenv(PROJECT_DIR / ".env")

# credenziali / endpoint
HF_TOKEN = os.getenv("HF_TOKEN")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")  # LangSmith, opzionale

# modelli usati dall'agente, tutti gratuiti/locali eccetto il reasoning LLM
REASONING_LLM_REPO_ID = "Qwen/Qwen2.5-3B-Instruct"  # gira in locale su GPU
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FINETUNE_BASE_MODEL = "distilgpt2"

# dominio del blog
BLOG_DOMAIN = "travel blogging"
POST_TYPES = ["how_to", "review", "ranking"]
TOPIC_CATEGORIES = [
    "culture", "adventure", "nature", "beaches", "nightlife",
    "cuisine", "wellness", "urban", "seclusion",
]
POSTING_INTERVAL_DAYS = 3
MAX_POST_WORDS = 280

# percorsi
FINETUNING_DIR = PROJECT_DIR / "finetuning"
FT_DATA_DIR = FINETUNING_DIR / "data"
FT_SAVED_MODELS_DIR = FINETUNING_DIR / "saved_models"
FT_RESULTS_DIR = FINETUNING_DIR / "results"
FT_FINAL_MODEL_DIR = FT_SAVED_MODELS_DIR / "distilgpt2_context_grounded"

LEGACY_FT_MODEL_DIR = PROJECT_DIR / "saved_models" / "travel_posts_distilgpt2"
LEGACY_BASE_MODEL_DIR = PROJECT_DIR / "saved_models" / "distilgpt2_base"

RAG_INDEX_DIR = PROJECT_DIR / "rag" / "faiss_index"
EVAL_REPORT_PATH = PROJECT_DIR / "results" / "evaluation_report.md"

RAG_INDEX_DIR.mkdir(parents=True, exist_ok=True)


def setup_langsmith() -> bool:
    if not LANGCHAIN_API_KEY:
        return False
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ.setdefault("LANGCHAIN_PROJECT", "blogger-support-agent")
    return True
```

**Cosa notare:**
- `PROJECT_DIR = Path(__file__).resolve().parent` è la cartella di questo
  stesso file (`progetto/`), calcolata sempre allo stesso modo
  indipendentemente da dove viene lanciato il comando. `load_dotenv`
  carica `.env` da lì: è la ragione per cui le credenziali si trovano
  sempre, a prescindere dalla working directory da cui parte lo script.
- `os.getenv("NOME", default)`: il secondo argomento è un valore di
  ripiego. `NEO4J_URI` ha un default sensato (`bolt://localhost:7687`),
  mentre `HF_TOKEN` no (`None` se assente) perché non esiste un default
  ragionevole per un token.
- **Tre modelli, tre ruoli diversi** — è il punto di riferimento mentale
  più utile di tutto il file:
  - `REASONING_LLM_REPO_ID` → il "cervello": ragiona e chiama i tool
    (Step 6-7).
  - `EMBEDDING_MODEL_NAME` → trasforma testo in vettori per il retrieval
    (Step 5).
  - `FINETUNE_BASE_MODEL` → il modello piccolo allenato per scrivere il
    corpo del post (Step 4, Step 10).
- `POST_TYPES` e `TOPIC_CATEGORIES` sono il "dominio" del blog: li
  ritroveremo nel prompt di planning (Step 6), nelle query del Knowledge
  Graph (Step 3) e nella costruzione del dataset di training (Step 10).
- `LEGACY_FT_MODEL_DIR`/`LEGACY_BASE_MODEL_DIR` puntano a un vecchio
  modello di riferimento usato solo (facoltativamente) in
  `finetuning/evaluate.py` per un confronto storico — se quella cartella
  non esiste, il codice se ne accorge e salta quel confronto (vedi Step 10).
- `setup_langsmith()` attiva il tracing LangSmith (servizio esterno di
  osservabilità per applicazioni LangChain, gratuito per uso individuale)
  impostando le variabili d'ambiente standard che LangChain legge da solo
  — solo se `LANGCHAIN_API_KEY` è presente in `.env`, altrimenti ritorna
  `False` senza fare nulla. Chiamata da `main.py` (Step 8) ed
  `evaluate_agent.py` (Step 9) prima di costruire il grafo.
- Nota pratica su LangSmith: se l'account è sulla regione **EU** (LangSmith
  di default punta all'endpoint US), va impostata anche
  `LANGCHAIN_ENDPOINT=https://eu.api.smith.langchain.com` in `.env` — non
  è una variabile che `config.py` legge esplicitamente, ma `load_dotenv()`
  la mette comunque in `os.environ`, dove il client LangSmith la trova da
  solo. Senza, l'invio delle trace fallisce con `403 Forbidden` pur con
  una chiave valida (è un problema di endpoint, non di credenziali).
- L'ultima riga prima della funzione (`.mkdir(...)`) è l'unico effetto
  collaterale a livello di import: ad ogni `import config`, il progetto si
  assicura che quella cartella esista già, per evitare errori "file not
  found" più avanti.

**Verifica la comprensione:**
- Se cancellassi il file `.env`, quale variabile smetterebbe subito di
  avere un valore utilizzabile e quali invece avrebbero comunque un
  default? Prova a elencarle.
- Perché ha senso calcolare i percorsi (`FT_DATA_DIR`, ecc.) partendo da
  `PROJECT_DIR` invece di scriverli come stringhe assolute?

---

## Step 3 — `kg/client.py`

Qui inizia vera logica applicativa, ma isolata e autocontenuta: è
**l'unico posto nel progetto che parla con Neo4j**. Tutto il resto del
codice chiama metodi di questa classe (`KnowledgeGraph`) senza mai scrivere
Cypher direttamente altrove.

### Lo schema del grafo

```
(:Post {id, title, post_type, created_at, summary})
(:Topic {name})
(:Source {url, domain})
(:Claim {id, text})

(Post)-[:TALKS_ABOUT]->(Topic)
(Post)-[:USES_SOURCE]->(Source)
(Post)-[:CONTAINS]->(Claim)
(Claim)-[:SUPPORTED_BY]->(Source)
(Topic)-[:RELATED_TO]->(Topic)
```

Quattro tipi di nodi, cinque tipi di relazioni. Ogni post parla di uno o
più topic, usa fonti, contiene affermazioni ("claim"), e ogni affermazione
è supportata da una fonte specifica (che può essere diversa dalle fonti
generali del post).

**Cosa rappresenta ciascun nodo:**
- **`Post`** `{id, title, post_type, created_at, summary}` — un articolo
  effettivamente **approvato e pubblicato**. Creato solo da `add_post()`,
  chiamata dal nodo `update_kg` del grafo dell'agente (Step 7) — cioè solo
  dopo che l'utente ha risposto "approve"/"modify" all'human review, mai
  per un draft rifiutato. `id` è un UUID, `post_type` è uno tra
  `how_to`/`review`/`ranking`.
- **`Topic`** `{name}` — l'argomento di cui parla un post (es. "Hiking the
  Alta Via 1 in the Dolomites"). Non ha un elenco chiuso: viene creato al
  volo con `MERGE` la prima volta che un post lo tocca (`link_post_topic`).
  Due topic possono essere collegati tra loro con `RELATED_TO`
  (`link_topics_related`) — è il pezzo che rende possibile il K-RAG: se si
  scrive di un topic nuovo ma vicino a uno già trattato, `expand_query_topics`
  (Step 5) lo trova.
- **`Source`** `{url, domain}` — una fonte web citata, o da un post
  (`link_post_source`, relazione `USES_SOURCE`) o da un singolo claim
  (`add_claim`, relazione `SUPPORTED_BY`). Creata anch'essa al volo con
  `MERGE`: se due post citano lo stesso URL è lo **stesso nodo**, non
  duplicato.
- **`Claim`** `{id, text}` — una singola **affermazione fattuale**
  estratta durante il ciclo ReAct (uno dei "fatti" raccolti e passati al
  tool di drafting, Step 7), collegata al post che la contiene
  (`CONTAINS`) e, se nota, alla fonte specifica che la supporta
  (`SUPPORTED_BY`) — è il livello più granulare: mentre `Source` dice
  "questo post ha usato questo URL", `Claim`+`SUPPORTED_BY` dice "questa
  frase specifica viene da quell'URL specifico".

Sono esattamente i 5 elementi richiesti dalle specifiche del progetto per
il Knowledge Graph ("post generati in precedenza, argomenti trattati,
fonti utilizzate, affermazioni chiave estratte dal contenuto, relazioni
tra gli argomenti") — mappano 1:1 su `Post`/`Topic`/`Source`/`Claim` +
`RELATED_TO`.

### Inizializzazione e connessione

```python
class KnowledgeGraph:
    def __init__(self, uri=None, user=None, password=None):
        self.driver = GraphDatabase.driver(
            uri or config.NEO4J_URI,
            auth=(user or config.NEO4J_USER, password or config.NEO4J_PASSWORD),
        )
        self._ensure_constraints()

    def close(self) -> None:
        self.driver.close()

    def __enter__(self) -> "KnowledgeGraph":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
```
- Gli argomenti `uri`/`user`/`password` sono opzionali: se non passati,
  usa i default di `config.py` — comodo sia per l'uso reale sia per i test.
- `__enter__`/`__exit__` rendono la classe usabile con `with KnowledgeGraph() as kg: ...`,
  garantendo che la connessione venga chiusa anche in caso di eccezione.
- `_ensure_constraints()` crea (se non esistono già) dei vincoli di unicità
  su `Post.id`, `Topic.name`, `Source.url`, `Claim.id` — così Neo4j stesso
  impedisce duplicati, anche se il codice sbagliasse qualcosa.

### Scritture (solo dopo approvazione umana)

```python
def add_post(self, post_id, title, post_type, summary=""):
    query = """
    MERGE (p:Post {id: $id})
    SET p.title = $title, p.post_type = $post_type,
        p.summary = $summary, p.created_at = $created_at
    """
    ...
```
`MERGE` + `SET` è un pattern comune: cerca il nodo per chiave univoca
(`id`), se non esiste lo crea, poi (ri)scrive le sue proprietà — questo
rende il metodo **idempotente**: chiamarlo due volte con lo stesso `id`
aggiorna, non duplica. Lo stesso pattern si ripete in `link_post_topic`,
`link_topics_related`, `link_post_source`, `add_claim`: tutte usano `MERGE`
sia per i nodi sia per le relazioni.

Nota `link_topics_related`: `if topic_a == topic_b: return` — piccola
guardia per non creare un auto-collegamento inutile di un topic con se
stesso.

`add_claim` fa due `session.run` distinti: prima crea/collega il claim al
post, poi — solo se è stata passata una `source_url` — crea/collega anche
il claim alla fonte che lo supporta.

### Interrogazioni (usate durante il ragionamento)

- **`get_recent_topics(limit=10)`**: ordina i topic per l'ultima volta che
  sono stati usati (`max(p.created_at)`) e prende i più recenti. Serve a
  **evitare ripetizioni** nel planning.
- **`get_coverage_gaps(candidate_topics)`**: il trucco Cypher da imparare
  a memoria — `OPTIONAL MATCH` (che non fallisce se non trova nulla) +
  `count(t) AS uses` + `WHERE uses = 0` è il modo standard per trovare
  "cosa manca" in un grafo. Riceve una lista di topic candidati (di solito
  `config.TOPIC_CATEGORIES`) e ritorna solo quelli mai collegati a nessun post.
- **`get_related_context(topic, limit=5)`**: trova post precedenti sullo
  stesso topic *o* su topic collegati da `RELATED_TO`, per garantire
  coerenza in fase di drafting.
- **`get_all_topics()`**: tutti i topic mai associati a un post pubblicato,
  senza limite di recenza — a differenza di `get_recent_topics` (che guarda
  solo agli ultimi `limit`), è la base del controllo *hard* anti-duplicati
  in `plan_posts` (Step 7): `get_recent_topics`/`coverage_gaps` sono solo un
  suggerimento nel prompt di planning, che il LLM potrebbe ignorare;
  `get_all_topics()` alimenta invece un filtro che scarta o sostituisce
  qualunque topic pianificato identico a uno già pubblicato, a prescindere
  da cosa fa il LLM.
- **`expand_query_topics(topic, limit=5)`**: topic collegati a uno dato —
  è il pezzo "K" di "K-RAG" (Step 5): il retrieval userà questi topic per
  allargare la query di ricerca.
- **`get_claims_for_topic(topic, limit=10)`**: affermazioni già pubblicate
  su un topic — usato per verificare di non contraddire post precedenti.
- **`summary()`**: quattro conteggi aggregati (post, topic, fonti, claim)
  in un'unica query, con `WITH` per incatenare i conteggi parziali. Usato
  per riempire `state["kg_summary"]` (collegamento diretto con `state.py`,
  Step 1, e con `load_kg_context`, Step 7).

**Verifica la comprensione:**
- Perché `get_coverage_gaps` usa `OPTIONAL MATCH` invece di un `MATCH`
  normale? Cosa succederebbe con un `MATCH` normale se un topic candidato
  non fosse mai stato collegato a nessun post?
- Quale metodo chiameresti per rispondere alla domanda "di cosa ha parlato
  il blog nell'ultimo mese"? E per "quali argomenti del nostro dominio non
  abbiamo ancora mai trattato"?

---

## Step 4 — `tools/`

Cinque strumenti che il modello di reasoning può chiamare durante il ciclo
ReAct, più uno **non** esposto al modello (`update_knowledge_graph`).

### `tools/__init__.py` — il punto di assemblaggio

```python
from tools.finetuned_draft_tool import draft_with_finetuned_model
from tools.kg_tools import kg_query_tool, update_knowledge_graph
from tools.quality_tool import resource_quality_tool
from tools.rag_tool import rag_retrieval_tool
from tools.search_tool import search_web_tool

REACT_TOOLS = [
    search_web_tool,
    rag_retrieval_tool,
    kg_query_tool,
    resource_quality_tool,
    draft_with_finetuned_model,
]
```
`REACT_TOOLS` è esattamente la lista passata a `.bind_tools(...)` nel grafo
(Step 7) — l'unico posto dove serve importare i tool tutti insieme.

### `tools/_shared.py` — risorse condivise, inizializzate pigramente

```python
_kg_instance = None

def get_kg() -> KnowledgeGraph:
    global _kg_instance
    if _kg_instance is None:
        _kg_instance = KnowledgeGraph()
    return _kg_instance

@lru_cache(maxsize=1)
def get_finetuned_generator():
    model_dir = config.FT_FINAL_MODEL_DIR
    if not model_dir.exists():
        raise FileNotFoundError(...)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    ...
    hf_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
    generator = HuggingFacePipeline(pipeline=hf_pipe)  # wrapper LangChain
    return generator, tokenizer
```
Due tecniche diverse per lo stesso obiettivo (non ricreare risorse costose
ad ogni chiamata):
- `get_kg()` usa una variabile globale col pattern manuale
  "crea se `None`, poi riusa".
- `get_finetuned_generator()` usa il decoratore `@lru_cache(maxsize=1)` di
  Python: la prima chiamata esegue la funzione e ne mette il risultato in
  cache, le chiamate successive (con gli stessi argomenti — qui nessuno)
  ritornano il valore cachato senza rieseguire nulla. Utile perché caricare
  un modello su GPU richiede secondi/minuti.

La pipeline `transformers` grezza viene incapsulata in
`langchain_huggingface.HuggingFacePipeline`: stesso modello, stessa
generazione, ma esposta con l'interfaccia standard `.invoke(prompt,
pipeline_kwargs={...})` invece di essere chiamata come funzione — è la
stessa astrazione "LLM" di LangChain usata per wrappare modelli locali
nel notebook di riferimento sul RAG (dove il caso è invece un chat model,
quindi serve anche `ChatHuggingFace`; qui il modello fine-tuned è un
semplice completatore di testo, quindi basta `HuggingFacePipeline`).

### I cinque tool, in ordine di utilizzo tipico nel ciclo ReAct

**1. `search_tool.py`** — il più semplice: wrapper su
`rag.search.search_web`, formatta i risultati come elenco puntato
`- titolo | url\n  estratto`. Primo passo: trova URL candidati.

**2. `rag_tool.py`**:
```python
@tool
def rag_retrieval_tool(query: str, topic: str = "") -> str:
    kg = get_kg()
    passages = retrieve_with_citations(query, topic=topic or None, kg=kg)
    ...
    for i, passage in enumerate(passages, start=1):
        lines.append(f"[{i}] (source: {passage['source_url']}) {passage['text']}")
    return "\n".join(lines)
```
Se gli viene passato `topic`, la query viene prima espansa con il
Knowledge Graph (pattern K-RAG, Step 5). Il formato di output
`[n] (source: URL) testo` **non è casuale**: `agent/graph.py` (Step 7) lo
riconosce per estrarre automaticamente le citazioni del post finale.

**3. `kg_tools.py`** — contiene due cose concettualmente diverse:
```python
@tool
def kg_query_tool(action: str, topic: str = "") -> str:
    kg = get_kg()
    if action == "recent_topics":
        return json.dumps(kg.get_recent_topics())
    if action == "coverage_gaps":
        return json.dumps(kg.get_coverage_gaps(config.TOPIC_CATEGORIES))
    if action == "related_context":
        return json.dumps(kg.get_related_context(topic))
    if action == "claims_for_topic":
        return json.dumps(kg.get_claims_for_topic(topic))
    return f"Unknown action '{action}'. ..."


def update_knowledge_graph(kg, post_id, title, post_type, topic, summary="",
                            related_topics=None, sources=None, facts=None):
    kg.add_post(post_id, title, post_type, summary)
    kg.link_post_topic(post_id, topic)
    for related in related_topics or []:
        kg.link_topics_related(topic, related)
    for url in sources or []:
        kg.link_post_source(post_id, url)
    for i, fact in enumerate(facts or []):
        claim_text = fact.get("text", "")
        if not claim_text:
            continue
        source_url = fact.get("source_url") or None
        kg.add_claim(post_id, f"{post_id}_claim_{i}", claim_text, source_url=source_url)
        if source_url:
            kg.link_post_source(post_id, source_url)
```
- `kg_query_tool` è decorato con `@tool`: un semplice dispatcher su
  `action`, che chiama il metodo giusto di `KnowledgeGraph` (Step 3) e
  serializza il risultato in JSON (perché il tool deve restituire una
  stringa, non un oggetto Python).
- `update_knowledge_graph` **non** ha `@tool`: è una funzione Python
  normale, chiamata direttamente dal nodo `update_kg` del grafo (Step 7),
  mai dal LLM. Design intenzionale, spiegato nel docstring del file: la
  scrittura sul grafo deve avvenire *solo* dopo l'approvazione umana, e
  lasciare quella decisione al giudizio del LLM dentro un ciclo ReAct
  sarebbe rischioso (potrebbe scrivere troppo presto, o dimenticarsene).
  Meglio renderla deterministica e fuori dal controllo del modello.
- `facts` è la lista di `RetrievedFact` raccolta in `gather_and_draft`
  (Step 7): ogni fatto porta già la propria `source_url`, estratta dal
  marcatore `"(source: URL)"` inline nel testo del fatto stesso
  (`agent/graph.py:_split_fact_source`) — niente più accoppiamento per
  indice con una lista di source separata, che poteva lasciare un claim
  senza fonte quando le due liste avevano lunghezze diverse.
- Nota come costruisce gli id dei claim: `f"{post_id}_claim_{i}"` — un id
  leggibile derivato dall'indice nella lista, non un UUID separato.

**4. `quality_tool.py`** — punteggio euristico, **nessun modello**:
```python
def score_resource(text, url=""):
    score = 0.5
    ...
    if word_count >= 150: score += 0.15       # contenuto sostanzioso
    elif word_count < 40: score -= 0.2        # troppo corto

    has_recency_signal = False
    for k in _RECENCY_KEYWORDS:
        if k in text_lower:
            has_recency_signal = True
            break
    if has_recency_signal: score += 0.15
    ...                                        # stesso schema (loop + flag) per dominio e "interestingness"
    return max(0.0, min(1.0, score)), reasons
```
Parte da 0.5 (neutro) e aggiusta in base a: lunghezza del testo, presenza
di parole legate all'attualità ("2025", "latest"...), dominio
affidabile/inaffidabile (una lista nera di domini noti per contenuti di
bassa qualità), presenza di marcatori di "interestingness" ("hidden gem",
"award"...). Il punteggio finale è sempre clampato tra 0 e 1
(`max(0.0, min(1.0, score))`). Serve all'agente per scartare fonti deboli
prima del drafting, e — separatamente — come baseline euristica di
paragone nella valutazione del modello fine-tuned.

**5. `finetuned_draft_tool.py`** — l'ultimo anello della catena:
```python
_TASK_INSTRUCTION = {
    "review": "Write a travel review blog post using ONLY the facts above.",
    "how_to": "Write a 'how to spend your time' travel guide post using ONLY the facts above.",
    "ranking": "Write a 'Top cities' ranking blog post using ONLY the facts above.",
}

@tool
def draft_with_finetuned_model(topic, post_type, facts, max_new_tokens=380):
    generator, tokenizer = get_finetuned_generator()
    instruction = _TASK_INSTRUCTION.get(post_type, _TASK_INSTRUCTION["review"])
    facts_block = "\n".join(facts) if facts else "(no facts provided)"
    prompt = f"Facts:\n{facts_block}\n\nTopic: {topic}\nTask: {instruction}"
    full_prompt = f"<PROMPT>\n{prompt}\n<RESP>\n"

    output = generator.invoke(
        full_prompt,
        pipeline_kwargs={
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min(180, max_new_tokens),
            "do_sample": True,
            "top_k": 50,
            "temperature": 0.8,
            "repetition_penalty": 1.3,
            "no_repeat_ngram_size": 3,
            "pad_token_id": tokenizer.pad_token_id,
        },
    )
    return output[len(full_prompt):].strip()
```
Costruisce un prompt **nello stesso formato esatto** usato in fase di
training (`<PROMPT>\n...\n<RESP>\n`, Step 10) — coerenza tra come il
modello è stato allenato e come viene usato in produzione è cruciale per
un modello piccolo come distilgpt2. `output[len(full_prompt):]` scarta il
prompt dall'output generato (il modello, essendo un semplice
"continuatore" di testo, restituisce prompt+completamento insieme, e
`HuggingFacePipeline` di default non toglie il prompt automaticamente).
I parametri di generazione (temperatura, `repetition_penalty`,
`no_repeat_ngram_size`...) passano ora come `pipeline_kwargs=`, l'unico
punto di ingresso che `HuggingFacePipeline._generate` inoltra alla
pipeline `transformers` sottostante.

**Verifica la comprensione:**
- Perché `update_knowledge_graph` non ha il decoratore `@tool` mentre gli
  altri quattro sì? Cosa cambierebbe se lo avesse?
- Se un'osservazione di `rag_retrieval_tool` non contenesse il pattern
  `(source: ...)`, cosa succederebbe alla fase di estrazione citazioni in
  `agent/graph.py`?

---

## Step 5 — `rag/`

La pipeline di retrieval dal web, usata da `rag_tool.py`. Ordine di
chiamata: `search` → `fetch` → `chunk` → `vectorstore`, orchestrati da
`krag`.

### `search.py`
```python
def search_web(query, max_results=5):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as exc:
        return [{"title": "ERRORE_RICERCA", "href": "", "body": str(exc)}]
    return results
```
Wrapper su DuckDuckGo (`ddgs`), nessuna API key richiesta. Se la ricerca
fallisce (rete assente, rate limit...), ritorna un risultato-segnaposto
invece di sollevare un'eccezione: chi chiama può continuare a funzionare
(magari mostrando all'utente/LLM che la ricerca non ha funzionato), invece
di far crollare tutto il ciclo ReAct.

### `fetch.py`
```python
def fetch_and_clean(url):
    try:
        response = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException:
        return ""
    soup = BeautifulSoup(response.text, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    return text[:_MAX_CHARS]
```
Scarica una pagina, rimuove i tag che quasi certamente non contengono
contenuto utile (`script`, `nav`, `footer`...), estrae solo il testo e lo
tronca a 8000 caratteri (`_MAX_CHARS`). Stesso principio di `search.py`:
qualunque errore di rete diventa stringa vuota, non eccezione — il
confine con siti web arbitrari di terzi non è controllabile, quindi va
gestito con grazia.

### `chunking.py`
```python
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)

def chunk_text(text):
    if not text:
        return []
    return _splitter.split_text(text)
```
"Recursive" perché prova a tagliare prima sui separatori più "naturali"
(doppio a capo → paragrafi), e solo se un pezzo è ancora troppo lungo passa
al separatore successivo nella lista (singolo a capo, poi punto, poi
spazio, infine carattere per carattere). Risultato: chunk che rispettano
il più possibile confini di paragrafo/frase, invece di tagliare a metà
frase come farebbe un chunking a lunghezza fissa ingenuo.
`chunk_overlap=100` fa sì che chunk consecutivi condividano un po' di
testo, per non perdere contesto ai bordi.

### `vectorstore.py`
```python
@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

def build_index(documents):
    if not documents:
        return None
    return FAISS.from_documents(documents, get_embeddings())

def get_retriever(index, k=4) -> VectorStoreRetriever:
    return index.as_retriever(search_kwargs={"k": k})
```
Il modello di embedding (`sentence-transformers/all-MiniLM-L6-v2`) viene
caricato una sola volta (`lru_cache`). L'indice FAISS invece viene
ricostruito **da zero ad ogni retrieval**, senza persisterlo su disco: non
avrebbe senso, perché le fonti sono pagine web recuperate di volta in
volta dal Search tool — persistere un indice basato su ricerche passate lo
renderebbe presto obsoleto.
Il retrieval avviene tramite `index.as_retriever(...)`, l'astrazione
standard `VectorStoreRetriever` di LangChain (invocabile con
`.invoke(query)`), invece di chiamare direttamente
`index.similarity_search(...)`: stesso risultato, ma nella forma
"retriever" usata anche nelle catene LCEL.

### `krag.py` — l'orchestratore
```python
def expand_query(query, topic, kg):
    if not topic or kg is None:
        return query
    related = kg.expand_query_topics(topic)
    if not related:
        return query
    return f"{query} ({', '.join(related)})"

def gather_documents(query, max_sources=4):
    results = search_web(query, max_results=max_sources)
    documents = []
    for result in results:
        url = result.get("href", "")
        if not url:
            continue
        page_text = fetch_and_clean(url)
        if not page_text:
            page_text = result.get("body", "")   # fallback: usa lo snippet del motore di ricerca
        for chunk in chunk_text(page_text):
            documents.append(Document(page_content=chunk,
                metadata={"source_url": url, "source_title": result.get("title", ""), "domain": domain_of(url)}))
    return documents

def retrieve_with_citations(query, topic=None, kg=None, max_sources=4, k=5):
    expanded = expand_query(query, topic, kg)
    documents = gather_documents(expanded, max_sources=max_sources)
    index = build_index(documents)
    if index is None:
        return []
    retrieved = get_retriever(index, k=k).invoke(query)
    passages = []
    for doc in retrieved:
        passages.append({"text": doc.page_content, "source_url": doc.metadata.get("source_url", ""),
                          "source_title": doc.metadata.get("source_title", "")})
    return passages
```
- `expand_query`: se c'è un `topic` e un'istanza KG, aggiunge alla query
  testuale i nomi dei topic correlati tra parentesi — un modo semplice ma
  efficace di "spingere" la ricerca verso contenuti già collegati nel
  grafo (il pattern K-RAG).
- `gather_documents`: nota il fallback `page_text = result.get("body", "")`
  — se il download/parsing della pagina fallisce (torna stringa vuota da
  `fetch_and_clean`), si usa almeno il breve estratto ("snippet") che il
  motore di ricerca aveva già fornito, invece di perdere del tutto quel
  risultato.
- `retrieve_with_citations`: nota che la *query di ricerca web* usa la
  versione espansa (`expanded`), ma la *similarity search sui vettori*
  usa la query **originale** (`query`, non `expanded`) — l'espansione
  serve a trovare più documenti candidati, non a spostare cosa viene
  considerato "rilevante" una volta che i documenti sono già stati
  recuperati e indicizzati.

**Verifica la comprensione:**
- Perché la query espansa via KG viene usata per la ricerca web ma non per
  la similarity search finale? Che problema eviterebbe di causare farla
  usare in entrambi i posti?
- Cosa succede, passo per passo, se `fetch_and_clean(url)` ritorna
  stringa vuota per tutti i risultati di ricerca?

---

## Step 6 — `agent/llm.py` e `agent/prompts.py`

### `agent/llm.py` — il problema e la soluzione

Il problema tecnico da risolvere: `ChatHuggingFace` di
`langchain_huggingface`, quando usa un modello locale (backend
"pipeline"), **non inoltra lo schema dei tool** al chat template — lo fa
solo per l'API a pagamento (backend "endpoint"). Qwen2.5 però gestisce
il tool-calling tramite il proprio chat template, che se riceve `tools=`
istruisce il modello a emettere blocchi XML
`<tool_call>{"name": ..., "arguments": ...}</tool_call>`. Serve quindi un
wrapper che lo faccia manualmente.

```python
_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"
_MISSING_COMMA_RE = re.compile(r'"\s+"')

def _parse_tool_call_json(payload_text):
    for candidate in (payload_text, _MISSING_COMMA_RE.sub('", "', payload_text)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None

def _extract_tool_calls(text):
    tool_calls = []
    remaining = text
    cursor = 0
    while True:
        start = remaining.find(_TOOL_CALL_OPEN, cursor)
        if start == -1:
            break
        end = remaining.find(_TOOL_CALL_CLOSE, start)
        if end != -1:
            payload_end = end
        else:
            next_open = remaining.find(_TOOL_CALL_OPEN, start + len(_TOOL_CALL_OPEN))
            payload_end = next_open if next_open != -1 else len(remaining)
        payload_text = remaining[start + len(_TOOL_CALL_OPEN):payload_end].strip()
        payload = _parse_tool_call_json(payload_text)
        if payload is None:
            cursor = (end + len(_TOOL_CALL_CLOSE)) if end != -1 else payload_end
            continue
        tool_calls.append({
            "name": payload.get("name"),
            "args": payload.get("arguments", {}),
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "tool_call",
        })
        close_at = (end + len(_TOOL_CALL_CLOSE)) if end != -1 else payload_end
        remaining = remaining[:start] + remaining[close_at:]
    return tool_calls, remaining.strip()
```
Scansiona il testo generato cercando ogni coppia `<tool_call>`/`</tool_call>`
(con `str.find`, senza regex), fa il parse del JSON dentro, e per ognuna
costruisce il dizionario nel formato standard che LangChain/LangGraph si
aspettano in `AIMessage.tool_calls` (`name`, `args`, `id`, `type`). Ogni
blocco riconosciuto viene anche rimosso dal testo, così quello che resta
(`remaining`) è il "pensiero" (`Thought`) da mostrare come contenuto del
messaggio.

Questa versione è più robusta di un semplice "trova apertura, trova
chiusura, fai il parse" perché il modello locale (3B, senza decoding
vincolato da una grammatica) a volte non si comporta come ci si
aspetterebbe: **non emette `</tool_call>`** e si ferma subito dopo il JSON
(gestito considerando il payload chiuso al prossimo `<tool_call>` — utile
se il modello ne apre un secondo "auto-correggendosi" a metà risposta — o
a fine testo), oppure **dimentica le virgole** tra le stringhe di un array
come `facts` (`_parse_tool_call_json` ritenta inserendole prima di
arrendersi). Senza questi due margini di tolleranza, una chiamata
altrimenti valida (magari proprio quella a `draft_with_finetuned_model`,
con dentro i fatti veri raccolti) verrebbe scartata in silenzio, il ciclo
ReAct penserebbe che il modello abbia smesso di voler agire, e scatterebbe
prematuramente il fallback con pochissimi fatti — vedi anche i due
controlli difensivi analoghi in `gather_and_draft` (Step 7): recupero del
nome del tool storpiato e guardia anti-ripetizione.

```python
class LocalQwenToolCallingChat(BaseChatModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    hf_model: Any
    hf_tokenizer: Any
    max_new_tokens: int = 1024
    temperature: float = 0.4

    def bind_tools(self, tools, **kwargs):
        formatted_tools = []
        for t in tools:
            formatted_tools.append(convert_to_openai_tool(t))
        return super().bind(tools=formatted_tools, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        tools = kwargs.get("tools")
        message_dicts = []
        for m in messages:
            message_dicts.append(_message_to_dict(m))
        prompt = self.hf_tokenizer.apply_chat_template(
            message_dicts, tools=tools, tokenize=False, add_generation_prompt=True)
        inputs = self.hf_tokenizer(prompt, return_tensors="pt").to(self.hf_model.device)
        with torch.no_grad():
            output_ids = self.hf_model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                do_sample=True, temperature=self.temperature, pad_token_id=self.hf_tokenizer.eos_token_id)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        text = self.hf_tokenizer.decode(new_tokens, skip_special_tokens=True)
        tool_calls, content = _extract_tool_calls(text)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content, tool_calls=tool_calls))])
```
- `LocalQwenToolCallingChat` eredita da `BaseChatModel` di LangChain: per
  essere un "chat model" a tutti gli effetti (utilizzabile in LangGraph
  esattamente come qualunque altro), basta implementare `_generate` (e
  opzionalmente `bind_tools`).
- `bind_tools` converte i tool LangChain (oggetti Python) nel formato
  standard "OpenAI tool schema" con `convert_to_openai_tool`, e li
  attacca al modello con `.bind(tools=...)` — un meccanismo di LangChain
  che ritorna una nuova versione "legata" del modello, che passerà sempre
  quei `tools` come kwarg extra ad ogni `_generate`.
- `_generate` è il cuore: costruisce il prompt col chat template (passando
  `tools=`), tokenizza, genera con `model.generate(...)` su GPU (`torch.no_grad()`
  perché siamo in inferenza, non training, quindi non serve calcolare i
  gradienti), decodifica solo i **token nuovi** (`output_ids[0][lunghezza_input:]`,
  altrimenti si riavrebbe indietro anche il prompt), e infine passa il
  testo a `_extract_tool_calls`.
- `get_chat_model()`, con `@lru_cache(maxsize=1)`, carica modello e
  tokenizer una sola volta per processo.
- `max_new_tokens=1024` (alzato da 512): quando il modello ragiona a
  lungo prima di chiamare `draft_with_finetuned_model` (ripete più
  "Thought:", riassume i fatti in prosa), la chiamata finale — quella con
  dentro l'intera lista `facts` — rischia di essere tagliata a metà JSON
  prima ancora di arrivare a `</tool_call>`, con conseguente perdita
  silenziosa della chiamata piu' importante del ciclo.

### `agent/prompts.py` — i due prompt di sistema

```python
PLANNING_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _PLANNING_SYSTEM_TEXT), ("user", "Plan the posts now.")]
)
REACT_PROMPT = ChatPromptTemplate.from_messages([("system", _REACT_SYSTEM_TEXT)])

def build_planning_messages(kg_summary, n_posts):
    return PLANNING_PROMPT.format_messages(
        domain=config.BLOG_DOMAIN, n_posts=n_posts, interval=config.POSTING_INTERVAL_DAYS,
        post_types=", ".join(config.POST_TYPES), categories=", ".join(config.TOPIC_CATEGORIES),
        recent_topics=json.dumps(kg_summary.get("recent_topics", [])),
        coverage_gaps=json.dumps(kg_summary.get("coverage_gaps", [])))

def build_react_system_prompt(topic, post_type, justification):
    return REACT_PROMPT.format_messages(domain=config.BLOG_DOMAIN, topic=topic,
                                          post_type=post_type, justification=justification)[0].content
```
I testi dei due prompt (`_PLANNING_SYSTEM_TEXT`, `_REACT_SYSTEM_TEXT`) sono
stringhe Python con placeholder (`{domain}`, `{n_posts}`...), ma invece di
riempirli con `.format(...)` a mano vengono incapsulati in
`ChatPromptTemplate` (astrazione LangChain per i prompt, § notebook RAG
introduttivo sezione 6.3): `format_messages(...)` restituisce direttamente
`BaseMessage` già pronti per `llm.invoke(...)`, invece di dizionari
`{"role": ..., "content": ...}` da convertire implicitamente. Per il
prompt ReAct, che al call site (`agent/graph.py`) serve come stringa dentro
un `SystemMessage`, si prende semplicemente `[0].content` del messaggio
formattato. Cose da notare leggendo per intero `_PLANNING_SYSTEM_TEXT` e
`_REACT_SYSTEM_TEXT` nel file:

- Il prompt di planning **obbliga** una risposta JSON (`Respond ONLY with
  a JSON array...`), e specifica regole esplicite: i topic devono essere
  concreti ("Hiking the Alta Via 1 in the Dolomites"), mai una categoria
  generica ("adventure") usata come topic — le categorie sono "lenti"
  tematiche per scegliere un gap, non il topic stesso.
- Il prompt ReAct descrive un piano d'azione passo-passo (cerca UNA volta →
  RAG una o due volte, **fonte primaria dei fatti** → al massimo 1-2
  controlli di qualità → opzionale verifica coerenza col KG → chiama il
  tool di drafting), con un budget dichiarato di 6-8 tool call e il
  divieto esplicito di richiamare lo stesso tool ripetutamente su input
  simili — aggiunto dopo aver osservato run reali in cui il modello locale
  si bloccava a ri-valutare la qualità di ogni singolo snippet di ricerca
  senza mai passare al RAG, esaurendo il budget senza raccogliere fatti
  veri. Contiene anche un blocco anti-allucinazione molto esplicito: ogni fatto
  passato a `draft_with_finetuned_model` **deve** provenire letteralmente
  da un'Observation di un tool vista in questa conversazione, mai
  inventato — con tanto di esempio negativo specifico ("non produrre un
  blocco City/Description/.../culture: x/5..."), perché è esattamente il
  formato dei dati strutturati del dataset di training (Step 10), che il
  modello potrebbe "ricordare" e riprodurre senza che sia mai stato
  davvero osservato in questa sessione. `evaluate_agent.py` (Step 9)
  controlla proprio questo pattern a posteriori per rilevare allucinazioni.

**Verifica la comprensione:**
- Perché `_generate` decodifica solo `output_ids[0][inputs["input_ids"].shape[1]:]`
  invece di tutto `output_ids[0]`?
- Se il modello generasse un blocco `<tool_call>` con JSON non valido
  (es. virgola in eccesso), cosa succederebbe in `_extract_tool_calls`?
  (Suggerimento: guarda il blocco `except json.JSONDecodeError`.)

---

## Step 7 — `agent/graph.py` (il cuore del sistema)

Il file più importante: mette insieme tutto quello visto finora in un
grafo LangGraph. Flusso generale (dal docstring del file):

```
load_kg_context -> plan_posts -> [ gather_and_draft -> human_review
                                    -> (reject -> gather_and_draft)
                                    | (approve/modify -> update_kg) ] *
                                  -> (altri post pianificati?
                                      -> gather_and_draft | END)
```

### Le due utility di parsing (senza regex)

```python
def _parse_plan_json(text):
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):]
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):]
    cleaned = cleaned.strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-len("```")]
    cleaned = cleaned.strip()
    return json.loads(cleaned)
```
Ripulisce eventuali fence markdown (` ```json ... ``` `) che il LLM a
volte aggiunge nonostante il prompt dica di non farlo, con semplici
controlli di prefisso/suffisso su stringhe — nessuna libreria di regex
necessaria per un caso così regolare.

```python
_CITATION_MARKER = "(source: "

def _extract_citations(tool_outputs):
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
```
Sfrutta il formato fisso prodotto da `rag_retrieval_tool` (Step 4):
`[n] (source: URL) testo`. Cerca il marcatore `(source: `, prende tutto
fino alla parentesi chiusa successiva: è l'URL. `seen` evita doppioni se
la stessa fonte compare in più passaggi.

### I cinque nodi

**1. `load_kg_context`** — interroga il KG e popola `state["kg_summary"]`:
```python
def load_kg_context(state):
    kg = get_kg()
    kg_summary = {
        "recent_topics": kg.get_recent_topics(),
        "coverage_gaps": kg.get_coverage_gaps(config.TOPIC_CATEGORIES),
        "all_topics": kg.get_all_topics(),
        **kg.summary(),
    }
    print(f"[load_kg_context] KG: {kg_summary['posts']} post, ... "
          f"({len(kg_summary['coverage_gaps'])} categorie ancora scoperte)")
    return {"kg_summary": kg_summary}
```
`**kg.summary()` fonde nel dizionario anche `posts`, `topics`, `sources`,
`claims`. Primo nodo, eseguito una volta sola all'inizio della sessione.
`all_topics` (Step 3) alimenta il controllo hard anti-duplicati di
`plan_posts`, sotto.

**2. `plan_posts`** — chiama il LLM, produce il piano e lo filtra contro i
topic già pubblicati:
```python
def _dedupe_plan(plan, all_topics, coverage_gaps):
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
                continue  # nessuna categoria di ricambio: il post viene scartato
            item = {**item, "topic": replacement,
                    "justification": f"{item['justification']} (topic duplicato, sostituito)"}
            key = _normalize_topic(item["topic"])
            if key in seen:
                continue
        seen.add(key)
        deduped.append(item)
    return deduped


def plan_posts(state):
    n_posts = state.get("n_posts", 3)
    llm = get_chat_model()
    messages = build_planning_messages(state["kg_summary"], n_posts)
    response = llm.invoke(messages)
    try:
        raw_plan = _parse_plan_json(response.content)
    except (json.JSONDecodeError, TypeError, ValueError):
        fallback_topics = state["kg_summary"].get("coverage_gaps") or config.TOPIC_CATEGORIES
        raw_plan = []
        for topic in fallback_topics[:n_posts]:
            raw_plan.append({"topic": topic, "post_type": "review",
                "justification": "Fallback: planner output was not valid JSON; using a coverage gap."})

    plan = []
    for i, item in enumerate(raw_plan[:n_posts]):
        post_type = item.get("post_type") if item.get("post_type") in config.POST_TYPES else "review"
        plan.append({"topic": item.get("topic", f"Untitled topic {i}"), "post_type": post_type,
                      "justification": item.get("justification", ""), "status": "planned"})

    plan = _dedupe_plan(plan, all_topics=state["kg_summary"].get("all_topics", []),
                         coverage_gaps=state["kg_summary"].get("coverage_gaps") or list(config.TOPIC_CATEGORIES))

    print(f"[plan_posts] piano di {len(plan)} post:")
    for p in plan:
        print(f"  - {p['topic']} ({p['post_type']})")
    return {"plan": plan, "current_post_idx": 0}
```
Nota il **fallback deterministico**: se il parsing JSON fallisce per
qualunque motivo, invece di far crollare l'intero programma, si costruisce
comunque un piano valido usando direttamente i `coverage_gaps` del KG. È
lo stesso principio di resilienza visto in `rag/search.py` e
`rag/fetch.py`: un componente esterno (qui, l'output non garantito di un
LLM) può fallire, e il codice deve avere un percorso alternativo pronto.
Nota anche la validazione riga per riga di `post_type`: se il LLM
inventasse un tipo non previsto, viene silenziosamente sostituito con
`"review"`.

`_dedupe_plan` è il **controllo hard anti-duplicati**: il prompt di
planning (`agent/prompts.py`) istruisce già il LLM a evitare topic recenti
e a preferire i `coverage_gaps`, ma è solo un suggerimento — il LLM può
ignorarlo o proporre lo stesso topic due volte nello stesso piano.
`_dedupe_plan` confronta ogni topic pianificato (normalizzato con
`_normalize_topic`: `strip().casefold()`, per non farsi ingannare da
maiuscole/spazi) contro `all_topics` (tutto lo storico del KG, non solo i
`recent_topics` mostrati al LLM) *e* contro i topic già visti nello stesso
piano. Un duplicato viene sostituito con la prossima categoria di
`coverage_gaps` ancora libera; se le categorie di ricambio finiscono, il
post duplicato viene semplicemente scartato — per questo il piano finale
può avere meno di `n_posts` elementi (gestito da `route_after_update`, più
sotto, che itera su `len(state["plan"])` e non su `n_posts`).

**3. `gather_and_draft`** — il nodo più complesso, il ciclo ReAct:
```python
def gather_and_draft(state):
    plan = state["plan"]
    idx = state["current_post_idx"]
    post = plan[idx]

    llm = get_chat_model().bind_tools(REACT_TOOLS)
    messages = [SystemMessage(content=build_react_system_prompt(
        post["topic"], post["post_type"], post["justification"]))]

    reasoning_trace, tool_outputs, gathered_facts = [], [], []
    draft_body = None
    seen_calls = set()  # (tool, args) gia' invocati in questa sessione

    print(f"[gather_and_draft] '{post['topic']}': inizio ciclo ReAct...")
    for _ in range(MAX_REACT_STEPS):          # MAX_REACT_STEPS = 10
        ai_message = llm.invoke(messages)
        messages.append(ai_message)
        if not ai_message.tool_calls:
            break

        for tool_call in ai_message.tool_calls:
            tool_name, tool_args = tool_call["name"], tool_call["args"]
            tool = TOOLS_BY_NAME.get(tool_name)
            if tool is None and "facts" in tool_args:
                tool_name = "draft_with_finetuned_model"       # nome storpiato dal LLM, recuperato
                tool = TOOLS_BY_NAME.get(tool_name)
            call_signature = (tool_name, json.dumps(tool_args, sort_keys=True, default=str))
            tool_failed = False
            if tool is None:
                observation = f"Tool error: unknown tool '{tool_name}'."
                tool_failed = True
            elif call_signature in seen_calls:
                observation = ("You already called this exact tool with these exact arguments "
                    "... Call draft_with_finetuned_model now with the facts you have gathered so far.")
                tool_failed = True
            else:
                seen_calls.add(call_signature)
                try:
                    observation = tool.invoke(tool_args)
                except Exception as exc:
                    observation = f"Tool error calling {tool_name} with args {tool_args}: {exc}. ..."
                    tool_failed = True

            print(f"  step {len(reasoning_trace) + 1}: {tool_name}" + (" (fallito)" if tool_failed else ""))

            thought = (ai_message.content or "").strip()
            justification = thought or f"Calling {tool_name} with {tool_args} to progress ..."
            reasoning_trace.append({"step": len(reasoning_trace) + 1, "thought": thought,
                "tool": tool_name, "tool_args": tool_args, "observation": str(observation)[:1000],
                "justification": justification})
            tool_outputs.append({"tool": tool_name, "args": tool_args, "observation": str(observation)})
            messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))

            if tool_name == "draft_with_finetuned_model" and not tool_failed:
                draft_body = observation
                gathered_facts = tool_args.get("facts", [])

        if draft_body is not None:
            break

    facts_used_for_draft = gathered_facts
    if draft_body is None:
        facts_fallback = gathered_facts or [f"Topic: {post['topic']}", f"Post type: {post['post_type']}"]
        draft_body = draft_with_finetuned_model.invoke(
            {"topic": post["topic"], "post_type": post["post_type"], "facts": facts_fallback})
        facts_used_for_draft = facts_fallback
        reasoning_trace.append({...})  # traccia esplicita del fallback

    citations = _extract_citations(tool_outputs)
    draft = {"topic": post["topic"], "post_type": post["post_type"],
             "title": f"{post['topic']} — {post['post_type'].replace('_', ' ').title()}",
             "body": draft_body, "citations": citations, "facts_used": facts_used_for_draft,
             "reasoning_trace": reasoning_trace}

    tools_used = []
    for step in reasoning_trace:
        tools_used.append(step["tool"])
    print(f"[gather_and_draft] '{post['topic']}': {len(reasoning_trace)} step "
          f"({', '.join(tools_used)}), {len(citations)} citazioni — draft pronto per la revisione")
    return {"messages": messages,
            "reasoning_trace": state.get("reasoning_trace", []) + reasoning_trace,
            "tool_outputs": state.get("tool_outputs", []) + tool_outputs,
            "gathered_facts": _facts_with_sources(facts_used_for_draft),
            "draft": draft}
```
`_facts_with_sources` (definita accanto a `_extract_citations`) converte
ogni fatto grezzo in un `RetrievedFact`, estraendo la `source_url` dal
marcatore `"(source: URL)"` inline nel testo stesso del fatto
(`_split_fact_source`) — non più accoppiata per posizione a una lista di
citazioni separata, come succedeva in una versione precedente: quello
schema poteva lasciare un claim senza fonte se le due liste (fatti e
citazioni) avevano lunghezze diverse.

Da notare `facts_used_for_draft` come variabile separata da
`gathered_facts`: nel ramo di fallback, `topic`/`post_type` sono metadati
del post (già scritti come proprietà del nodo `Post` da `add_post`), non
fatti estratti da una fonte — servono solo a dare a
`draft_with_finetuned_model` qualcosa da scrivere quando il budget ReAct si
esaurisce senza aver raccolto nulla di vero. Tenerli fuori da
`gathered_facts` evita che finiscano nel KG come `Claim` fasulli quando
`update_kg` (più sotto) li scrive: `facts_used_for_draft` resta comunque
visibile in `draft["facts_used"]`, per la valutazione (`evaluate_agent.py`,
Step 9) — che così vede correttamente "zero fatti reali raccolti" invece di
un dato inquinato.
Punti chiave, in ordine di importanza:
- **Due margini di tolleranza in più**, oltre a quelli di `_extract_tool_calls`
  (Step 6), scoperti osservando run reali del modello locale: (1) se il
  nome del tool non esiste ma gli argomenti contengono `facts`, si assume
  che il LLM abbia storpiato `draft_with_finetuned_model` (es.
  `draft_with_facts`) e si recupera comunque la chiamata — è l'unico tool
  con quella firma di argomenti; (2) `seen_calls` tiene traccia delle
  coppie (tool, argomenti) già invocate in questa sessione: il modello
  locale a volte si blocca ripetendo la stessa identica chiamata (es.
  `rag_retrieval_tool` con la stessa query) invece di procedere al
  drafting — anziché rieseguirla (rete/tempo sprecati), si risponde con
  un'Observation che lo spinge esplicitamente verso
  `draft_with_finetuned_model`.
- I `print(...)` sono un compromesso deliberato tra due estremi: **una
  riga per step, in tempo reale** (`step N: tool_name`), cosi' l'utente
  vede che il ciclo sta lavorando (ogni step comporta una vera chiamata al
  LLM/tool, puo' richiedere anche decine di secondi — senza nessun output
  intermedio l'esecuzione sembrerebbe bloccata, specialmente per il
  secondo post di un piano, quando non c'e' nessun altro output tra
  `update_kg` del post precedente e questo), **piu' un riassunto finale di
  una riga**, non un dump del `draft` intero: quest'ultimo contiene anche
  `reasoning_trace` (le observation di ogni tool, fino a 1000 caratteri
  l'una) — stamparlo per intero seppellirebbe il blocco pulito
  `=== DRAFT ===` che `main.py` (Step 8) mostra subito dopo per la
  revisione umana, rendendo illeggibile proprio il testo che l'utente
  deve valutare.
- **Il ciclo `for _ in range(MAX_REACT_STEPS)`** è l'implementazione
  esplicita del pattern ReAct (Step 0): ad ogni iterazione il modello
  produce un `AIMessage` (che può contenere `tool_calls`), e se non ne
  contiene nessuno il ciclo si ferma (il modello ha "deciso" di non agire
  oltre). Ogni tool chiamato produce un `ToolMessage` che rientra nella
  lista `messages` — è così che la "memoria" della conversazione cresce
  turno dopo turno, esattamente come una chat normale, solo che alcuni
  messaggi sono generati da tool invece che da un umano.
- **Tolleranza ai fallimenti dei tool**: se `tool.invoke(tool_args)`
  solleva un'eccezione (per esempio il LLM ha passato un argomento
  sbagliato), l'errore **non** interrompe il ciclo — diventa semplicemente
  il contenuto della prossima `Observation`, così il modello, leggendo
  quel messaggio di errore al turno successivo, ha la possibilità di
  correggersi da solo.
- **La "safety net" dopo il ciclo** (`if draft_body is None: ...`) è forse
  il dettaglio di design più importante del file: garantisce che **ogni**
  post, senza eccezioni, passi dal modello fine-tuned per la generazione
  del testo — mai un testo scritto "a mano" dal LLM di reasoning. Se il
  budget di `MAX_REACT_STEPS` si esaurisce senza che il modello abbia mai
  chiamato volontariamente `draft_with_finetuned_model`, il codice lo
  invoca comunque, direttamente, con qualunque fatto sia stato raccolto
  fino a quel momento (o un fallback minimale col solo topic/post_type).
  Questo evento viene anche registrato esplicitamente nella
  `reasoning_trace`, ed è uno dei quattro segnali che `evaluate_agent.py`
  (Step 9) usa per identificare failure case.
- `_extract_citations(tool_outputs)` chiude il cerchio con quanto visto
  sopra: raccoglie tutte le fonti citate nelle Observation di
  `rag_retrieval_tool` durante l'intero ciclo.
- Il `draft` include anche la propria `reasoning_trace` (non solo lo stato
  cumulativo `state["reasoning_trace"]`): così, quando il post viene
  approvato e finisce in `approved_posts` (vedi `update_kg` più sotto),
  porta con sé tutto il necessario per l'analisi post-hoc, senza dover
  rileggere nulla da un log esterno — è ciò che `evaluate_agent.py`
  (Step 9) usa direttamente.
- Da notare: `state["messages"]` accumula l'intera traccia per
  tracciabilità, ma **ogni chiamata** di `gather_and_draft` costruisce un
  prompt fresco (`messages` locale, inizializzato da zero con solo il
  `SystemMessage` per il post corrente) — non rilegge i messaggi dei post
  precedenti nel prompt che invia al LLM, per non degradare la qualità del
  reasoning mescolando contesti di post diversi. Ciò che invece
  attraversa davvero l'intera sessione, da un post al successivo, sono
  `kg_summary`, `plan`, `reasoning_trace` e `gathered_facts`.

**4. `human_review`** — il punto di sospensione:
```python
def human_review(state):
    decision = interrupt({
        "draft": state["draft"],
        "instructions": "Respond with decision in {'approve','modify','reject'}; ..."
    })
    return {"human_decision": decision.get("decision"), "human_feedback": decision.get("feedback")}
```
`interrupt(payload)` è una primitiva di LangGraph: **sospende
letteralmente l'esecuzione del grafo a questo punto**, e restituisce il
`payload` al chiamante esterno (`main.py`, Step 8) tramite una chiave
speciale nel risultato di `app.invoke(...)`. Grazie al checkpointer
(`MemorySaver`, visto in `build_graph`), lo stato viene conservato: quando
più tardi arriva un `Command(resume={"decision": ..., "feedback": ...})`,
l'esecuzione riprende esattamente da qui, e quel dizionario diventa il
valore di ritorno di `interrupt(...)` — cioè `decision` in questa funzione.
Nessun polling, nessun salvataggio manuale su file: è il meccanismo nativo
di LangGraph per l'human-in-the-loop.

**5. `update_kg`** — scrittura finale, solo dopo approvazione:
```python
def update_kg(state):
    draft = dict(state["draft"])
    if state.get("human_decision") == "modify" and state.get("human_feedback"):
        draft["body"] = state["human_feedback"]

    kg = get_kg()
    post_id = str(uuid.uuid4())
    plan, idx = state["plan"], state["current_post_idx"]

    related_topics = []
    for i, p in enumerate(plan):
        if i != idx and p["status"] == "approved":
            related_topics.append(p["topic"])

    source_urls = []
    for c in draft.get("citations", []):
        if c.get("url"):
            source_urls.append(c["url"])

    update_knowledge_graph(kg, post_id=post_id, title=draft["title"], post_type=draft["post_type"],
                            topic=draft["topic"], summary=draft["body"][:300],
                            related_topics=related_topics, sources=source_urls,
                            facts=state.get("gathered_facts", []))

    new_plan = list(plan)
    new_plan[idx] = {**new_plan[idx], "status": "approved"}
    print(f"[update_kg] post_id={post_id} title={draft['title']!r}")
    return {"plan": new_plan, "current_post_idx": idx + 1,
            "approved_posts": state.get("approved_posts", []) + [draft],
            "draft": None, "human_decision": None, "human_feedback": None}
```
Se la decisione era `"modify"`, il corpo del post viene sostituito col
testo corretto a mano dall'utente **prima** di scrivere nel KG — quindi il
grafo conserva sempre la versione finale davvero pubblicata, non il draft
originale. `related_topics` collega il nuovo post agli altri topic già
approvati **in questa stessa sessione** (non a tutto lo storico — quello
lo gestisce già `link_topics_related` chiamato per ciascuno). Da notare
`idx + 1`: è qui che l'indice del post corrente avanza, permettendo al
prossimo giro del ciclo (via `route_after_update`) di passare al post
successivo del piano.

### Le due funzioni di routing

```python
def route_after_review(state):
    if state.get("human_decision") == "reject":
        return "gather_and_draft"
    return "update_kg"

def route_after_update(state):
    if state["current_post_idx"] >= len(state["plan"]):
        return END
    return "gather_and_draft"
```
Semplici funzioni pure: leggono lo stato e ritornano il nome del prossimo
nodo (o la costante speciale `END`). `route_after_review` implementa la
regola "solo un rifiuto esplicito rimanda al drafting, approve *e* modify
procedono entrambi verso la scrittura nel KG" (con la differenza che
`modify` avrà già sostituito il testo, come visto sopra). `route_after_update`
implementa il ciclo sui post del piano: continua finché ce ne sono ancora
da processare, altrimenti termina.

### `build_graph()` — il cablaggio

```python
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
    graph.add_conditional_edges("human_review", route_after_review,
        {"gather_and_draft": "gather_and_draft", "update_kg": "update_kg"})
    graph.add_conditional_edges("update_kg", route_after_update,
        {END: END, "gather_and_draft": "gather_and_draft"})

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
```
`StateGraph(BlogAgentState)` dice a LangGraph quale forma ha lo stato
condiviso (Step 1). I quattro `add_edge` sono i passaggi fissi del flusso;
i due `add_conditional_edges` sono i due bivi del sistema (dopo la revisione
umana, e dopo la scrittura nel KG). Il terzo argomento di
`add_conditional_edges` è una mappa "valore ritornato dalla funzione di
routing" → "nome del nodo": serve a LangGraph per validare in anticipo che
tutti i possibili output della funzione di routing corrispondano a nodi
realmente esistenti nel grafo. Infine, `MemorySaver` come `checkpointer` è
ciò che rende possibile l'`interrupt()`/`resume` visto in `human_review`.

**Verifica la comprensione:**
- Se il LLM chiamasse `draft_with_finetuned_model` al primo tentativo,
  senza prima cercare nulla, quanti giri del `for` farebbe il ciclo in
  `gather_and_draft`? Il draft prodotto sarebbe comunque valido?
- Traccia a mano cosa succede allo stato se l'utente risponde `"reject"`
  a `human_review`: quale nodo viene eseguito subito dopo, e cosa cambia
  (o non cambia) in `state["plan"]`?
- Perché `route_after_update` controlla `current_post_idx >= len(plan)`
  invece di, per esempio, un flag booleano separato tipo "piano_finito"?

---

## Step 8 — `main.py`

L'entrypoint CLI, il più semplice da leggere dopo aver capito il grafo:

```python
def run(n_posts):
    if config.setup_langsmith():
        print("LangSmith tracing attivo (progetto: blogger-support-agent)")
    app = build_graph()
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = app.invoke(state.initial_state(n_posts), config=thread_config)

    while "__interrupt__" in result:
        interrupt_payload = result["__interrupt__"][0].value
        _print_draft(interrupt_payload["draft"])
        decision = _ask_human_decision()
        result = app.invoke(Command(resume=decision), config=thread_config)

    print(f"\nSession complete. {len(result['approved_posts'])} post(s) approved and written to the Knowledge Graph.")
```
- `config.setup_langsmith()`: attiva il tracing esterno se configurato
  (Step 2) — l'unica riga aggiunta rispetto a un `run()` "nudo"; il resto
  della funzione non sa nulla di LangSmith, il tracing è trasparente per
  il grafo grazie alle variabili d'ambiente lette automaticamente da
  LangChain.
- `thread_config` con un `thread_id` univoco: è la chiave con cui il
  `MemorySaver` (Step 7) tiene traccia dello stato di **questa specifica**
  esecuzione — necessario perché `interrupt`/`resume` funzionino
  correttamente anche se, in teoria, più sessioni girassero in parallelo.
- `state.initial_state(n_posts)` (in `state.py`, Step 1) costruisce lo
  stato iniziale, con un valore per **ogni** chiave di `BlogAgentState` —
  LangGraph non assume default impliciti. È una funzione condivisa: sia
  `main.py` che `evaluate_agent.py` (Step 9) partono da qui, invece di
  duplicare lo stesso dizionario di 13 campi in entrambi i file.
- Il ciclo `while "__interrupt__" in result` è l'altro lato di
  `interrupt()` visto in `human_review`: ogni volta che il grafo si ferma,
  `result` contiene quella chiave speciale con il payload passato a
  `interrupt(...)`. Si stampa il draft (`_print_draft`), si chiede la
  decisione da tastiera (`_ask_human_decision`), e si rilancia il grafo
  con `Command(resume=decision)` — che lo fa ripartire esattamente da
  dove si era fermato (dentro `human_review`), con `decision` come valore
  di ritorno di quella chiamata a `interrupt(...)`.

```python
def _ask_human_decision():
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
```
Un semplice loop di validazione input da tastiera: continua a chiedere
finché non riceve `a`/`m`/`r` (anche per esteso). Per `modify`, legge righe
multiple finché non trova una riga vuota — un modo semplice per permettere
un testo corretto lungo più righe da terminale.

**Verifica la comprensione:**
- Cosa succederebbe se `initial_state` non includesse la chiave
  `"approved_posts"`? (Prova a pensare a dove viene letta per la prima
  volta in `agent/graph.py`.)
- Perché serve un `thread_id` diverso per ogni esecuzione di `run()`,
  invece di usarne sempre uno fisso?

---

## Step 9 — `evaluate_agent.py`

Due livelli di osservabilità, indipendenti tra loro:
- Il debug "in diretta" dei nodi del grafo (Step 7: `load_kg_context`,
  `plan_posts`, `gather_and_draft`, `update_kg`) è affidato a semplici
  `print(...)` piazzati direttamente nei nodi — utile per seguire
  l'esecuzione da terminale mentre il programma gira, non c'è un modulo
  dedicato.
- **LangSmith** (Step 2, `config.setup_langsmith()`), se una
  `LANGCHAIN_API_KEY` è configurata: traccia esternamente ogni chiamata
  al LLM e ai tool con latenza, token usati e l'albero completo
  Thought/Action/Observation, consultabile sulla dashboard di LangSmith —
  è il "parametro di osservabilità" più ricco a disposizione, e viene
  attivato sia da `main.py` sia da `evaluate_agent.py` prima di costruire
  il grafo.

`evaluate_agent.py` non dipende dai `print` né da nessun file su disco per
la propria analisi: **esegue l'intero grafo end-to-end in-process**, un
modo alternativo di lanciare l'agente pensato per la valutazione
automatica.

```python
def _run_agent_and_collect_drafts(n_posts):
    config.setup_langsmith()
    app = build_graph()
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = app.invoke(state.initial_state(n_posts), config=thread_config)
    while "__interrupt__" in result:
        result = app.invoke(Command(resume={"decision": "approve"}), config=thread_config)
    return result["approved_posts"]
```
A differenza di `main.py`, che ad ogni `interrupt` chiede una decisione a
tastiera, qui si **approva sempre automaticamente** (`{"decision":
"approve"}`): per valutare l'agente non serve un revisore umano reale, e
approvare senza modifiche garantisce che l'analisi lavori sul testo
esattamente come generato dal modello. Ogni elemento di `approved_posts`
è un `draft` che, come visto nello Step 7, porta già con sé la propria
`reasoning_trace` — è tutto ciò che serve per l'analisi, senza dover
rileggere nulla da un log esterno.

La logica interessante resta tutta in `_identify_failure_cases`, che
applica quattro euristiche automatiche a ciascun post analizzato:

```python
_RATING_OUT_OF_FIVE_RE = re.compile(r"(?<!\w)[1-5]\s*/\s*5(?!\w)")

def _looks_like_rating_out_of_five(text):
    """True se `text` contiene un voto tipo "4/5": segno che il LLM ha
    copiato un rating dal dataset invece di scrivere testo vero."""
    return bool(_RATING_OUT_OF_FIVE_RE.search(text))
```
Un pattern "cifra 1-5" + "/" (spazi opzionali) + "5", con controlli di
confine di parola (`(?<!\w)`/`(?!\w)`, per non far scattare falsi positivi
su "14/5" o "4/50") — se il testo del draft contiene qualcosa come "4/5",
è quasi certamente un rating copiato dal formato strutturato del dataset
di training (Step 10), non un fatto realmente osservato in una fonte web —
esattamente ciò che il prompt ReAct (Step 6) proibiva esplicitamente.

```python
def _identify_failure_cases(analyses):
    failures = []
    for a in analyses:
        # 1. rating "x/5" nei fatti usati -> probabile allucinazione
        # 2. nessuna citazione -> RAG non ha trovato nulla o il draft non cita fonti
        # 3. grounding_score sotto soglia -> il testo non riflette i fatti raccolti
        # 4. ultimo step di reasoning_trace menziona "fallback" -> budget ReAct esaurito
        # + per ogni step: se un tool ha restituito un risultato vuoto, lo segnala come evidenza
        ...
        failures.append({"topic": a["topic"], "code": "...", "evidence": ...})
    return failures
```
Le quattro euristiche (qui riassunte, il codice completo è nel file) danno
una diagnostica automatica senza bisogno di rileggere manualmente ogni
run log: allucinazione di dati strutturati, assenza di fonti, scarsa
aderenza ai fatti, o esaurimento del budget di ragionamento prima del
drafting volontario. Ogni failure case è un record di **sole tre chiavi**:
`topic`, `code` (una stringa breve da un vocabolario fisso — `fabricated_facts`,
`no_citations`, `low_grounding`, `budget_exhausted`, `empty_tool_result`) ed
`evidence` (un **dato grezzo**: un numero, una lista, un nome di tool — mai
una frase in inglese che lo spiega). Scelta deliberata: niente prosa da
generare o mantenere, il report resta puramente dati.

Il report finale (`results/evaluation_report.md`) è costruito da
`_report(analyses, failures, n_posts)` come **due tabelle Markdown** (post
analizzati con le loro metriche; failure case con `topic`/`code`/`evidence`)
più due righe di media (`avg_grounding`, `avg_citations`) e una riga finale
`langsmith_active: True/False` per l'osservabilità — nessun paragrafo
discorsivo, solo numeri e stringhe corte pronte per essere lette a colpo
d'occhio o incollate in una tabella della relazione.

**Verifica la comprensione:**
- Se un post avesse `grounding_score = 0.15` (sotto `LOW_GROUNDING_THRESHOLD = 0.2`)
  ma anche 3 citazioni valide, quanti failure case verrebbero registrati
  per quel post secondo la logica descritta sopra?
- Perché `_run_agent_and_collect_drafts` approva sempre automaticamente
  ogni draft invece di, per esempio, chiedere anche qui una decisione da
  tastiera come fa `main.py`?

---

## Step 10 — `finetuning/` (pipeline di training, a parte)

Indipendente dall'esecuzione dell'agente: costruisce e allena il modello
usato da `draft_with_finetuned_model` (Step 4). Da leggere nell'ordine in
cui andrebbero eseguiti.

### `dataset_builder.py`

```python
def _fact_lines_for_destination(data):
    lines = [f"City: {data['city']}, {data.get('region', '')}, {data['country']}".replace(", ,", ",")]
    if data.get("short_description"):
        lines.append(f"Description: {data['short_description']}")
    lines.append(f"Ideal durations: {data.get('ideal_durations')}")
    lines.append(f"Budget level: {data.get('budget_level')}")
    for category in config.TOPIC_CATEGORIES:
        lines.append(f"{category}: {data.get(category)}/5")
    return lines

def _build_example(post_type, fact_lines, topic, completion):
    facts_block = "\n".join(fact_lines)
    prompt = f"Facts:\n{facts_block}\n\nTopic: {topic}\nTask: {TASK_INSTRUCTION[post_type]}"
    return {"prompt": prompt, "completion": completion, "post_type": post_type,
            "facts": fact_lines, "topic": topic}
```
Scarica un dataset Kaggle di destinazioni di viaggio (`kagglehub`), e per
ognuna costruisce righe di "fatti" strutturati (città, descrizione,
durata ideale, budget, punteggio 1-5 per ogni categoria di
`config.TOPIC_CATEGORIES`) — **sono proprio questi** i pattern "x/5" che
il prompt ReAct (Step 6) e `evaluate_agent.py` (Step 9) vogliono impedire
che finiscano, tali e quali, nel testo generato dall'agente in produzione:
qui sono legittimi (sono dati strutturati reali del dataset), ma in
produzione il modello deve trasformarli in prosa, non copiarli. Ogni
esempio di training diventa un dizionario `{prompt, completion, post_type,
facts, topic}`; il testo di riferimento (`completion`) viene preso da
generatori euristici esterni al progetto (`dataset_generation/`, fuori da
questa cartella). Split riproducibile 80/10/10 con seed fisso
(`split_dataset`, `SEED = 42`) in train/validation/test.

### `metrics.py`

Le tre metriche condivise, già introdotte nello Step 0 e riviste in
`evaluate_agent.py` (Step 9): `perplexity_from_loss` (`math.exp(eval_loss)`),
`grounding_score` (interseca i token significativi di ogni fatto — parole
di almeno 3 caratteri, non in una lista di stopword — con il testo
generato), `rouge_l` (via libreria `rouge_score`).

### `train.py` — tre fasi in sequenza

```python
def main():
    tokenizer = AutoTokenizer.from_pretrained(config.FINETUNE_BASE_MODEL)
    raw_dataset, tokenized = load_tokenized_dataset(tokenizer)

    baseline_result = run_baseline(tokenized, raw_dataset["validation"])   # 1. baseline zero-shot

    sweep_results, best_result, best_hp = [], None, None
    for hp in HYPERPARAM_GRID:                                            # 2. hyperparameter sweep
        result, model, tok = run_config(hp, tokenized, raw_dataset["validation"])
        sweep_results.append(result)
        if best_result is None or result.perplexity < best_result.perplexity:
            best_result, best_hp = result, hp
        ...

    final_result, final_model, final_tokenizer = run_config(              # 3. retraining finale
        best_hp, tokenized, raw_dataset["validation"], save_dir=config.FT_FINAL_MODEL_DIR)

    sweep_log = []
    for r in sweep_results:
        sweep_log.append(asdict(r))

    log = {"baseline": asdict(baseline_result), "sweep": sweep_log,
           "best_config": best_hp, "final_model_validation": asdict(final_result),
           "final_model_dir": str(config.FT_FINAL_MODEL_DIR)}
    # salvato in finetuning/results/training_log.json
```
- **Fase 1 — `run_baseline`**: valuta distilgpt2 **non** fine-tuned, con
  lo stesso prompt fact-grounded che userà il modello finale (zero-shot):
  è il punto di partenza da battere, per poter dire con dati alla mano
  "il fine-tuning ha migliorato le cose di *tanto*".
- **Fase 2 — sweep**: prova ogni combinazione in `HYPERPARAM_GRID`
  (learning rate `5e-5`/`1e-4` × epoche `3`/`5`, quattro combinazioni in
  totale), valutando ciascuna sul validation set, e tiene quella con la
  perplexity più bassa (`best_hp`).
- **Fase 3 — retraining finale**: riallena da zero con la configurazione
  vincente, e stavolta salva il modello su disco (`save_dir=config.FT_FINAL_MODEL_DIR`)
  — è il modello che l'agente userà davvero, tramite
  `tools/_shared.get_finetuned_generator()` (Step 4).
- `evaluate_grounding` (chiamata dentro `run_baseline`/`run_config`) fa
  generare al modello un piccolo campione di completions e calcola il
  `grounding_score` medio — la stessa identica metrica di
  `evaluate_agent.py` (Step 9), qui usata per scegliere il modello invece
  che per diagnosticare l'agente in produzione.
- Da notare la gestione esplicita della memoria GPU: `del model, trainer`
  + `gc.collect()` + `torch.cuda.empty_cache()` dopo ogni configurazione
  provata — necessario perché lo sweep alloca e libera più modelli in
  sequenza sulla stessa GPU, senza questo la memoria si esaurirebbe dopo
  poche iterazioni.

### `evaluate.py` — confronto finale sul test set

```python
def main():
    dataset_dict = load_from_disk(str(config.FT_DATA_DIR / "context_grounded_dataset"))
    test_rows = dataset_dict["test"]     # MAI usato per scegliere iperparametri

    results = [evaluate_model("baseline_zero_shot_distilgpt2", config.FINETUNE_BASE_MODEL,
                                test_rows, _context_grounded_prompt)]
    if config.LEGACY_FT_MODEL_DIR.exists():
        results.append(evaluate_model("legacy_finetuned_no_context", str(config.LEGACY_FT_MODEL_DIR),
                                        test_rows, _legacy_prompt))
    results.append(evaluate_model("finetuned_context_grounded (final tool model)",
                                    str(config.FT_FINAL_MODEL_DIR), test_rows, _context_grounded_prompt))

    final_records = results[-1]["records"]
    failure_cases = sorted(final_records, key=lambda r: r["grounding_score"])[:2] \
                  + sorted(final_records, key=lambda r: r["rouge_l"])[:1]
    # scrive finetuning/results/final_evaluation.json e .md
```
Confronto **sul test set**, esplicitamente mai toccato durante lo sweep
(per non "sovra-adattare" la scelta degli iperparametri anche al test).
Tre modelli a confronto: baseline zero-shot, un eventuale vecchio modello
"legacy" (se la cartella esiste — altrimenti quel confronto viene
saltato), e il modello finale. Da notare `_legacy_prompt` vs
`_context_grounded_prompt`: il vecchio modello legacy era stato allenato
con un formato di prompt diverso (senza il blocco `Facts:`), quindi
l'errore analysis lo valuta con il *suo* formato originale, non con quello
nuovo — un confronto equo deve rispettare il formato con cui ciascun
modello è stato effettivamente allenato. `failure_cases` prende
deterministicamente i 2 esempi con grounding più basso e 1 con ROUGE-L più
basso dal modello finale, per un error analysis riproducibile e
documentato in `finetuning/results/final_evaluation.md`.

**Verifica la comprensione:**
- Perché il confronto finale usa il test set e non il validation set già
  usato nello sweep? Cosa si rischierebbe a riutilizzare lo stesso set?
- Se `config.LEGACY_FT_MODEL_DIR` non esistesse (come nell'ambiente
  attuale), quanti modelli verrebbero effettivamente confrontati in
  `finetuning/evaluate.py`?

---

## Come si collega tutto (riepilogo)

```
main.py               ─┐  entrambi chiamano config.setup_langsmith()
evaluate_agent.py     ─┴─ agent/graph.py (build_graph)
       ├─ kg/client.py            (via tools/_shared.get_kg)
       ├─ agent/llm.py + prompts.py
       ├─ tools/ (i 5 tool)
       │    ├─ rag/ (search → fetch → chunk → vectorstore → krag)
       │    └─ finetuning/saved_models/.../  (modello di drafting)
       ├─ print(...) di debug ad ogni nodo (nessun modulo dedicato)
       └─ LangSmith (tracing esterno, opzionale — Step 2)

evaluate_agent.py  esegue il grafo end-to-end, approva ogni draft
                    automaticamente, e analizza i post ottenuti in memoria
finetuning/train.py + evaluate.py  producono il modello che finetuned_draft_tool.py usa
```

## Comandi per eseguire tutto

```bash
cd /home/sallegra/progetto
./runtime/start_neo4j.sh          # se Neo4j non è già avviato
.venv/bin/python main.py --posts 3
.venv/bin/python evaluate_agent.py --posts 3
```

---

## Appendice — Glossario

- **StateGraph**: la classe LangGraph che rappresenta il grafo prima della
  compilazione (`.compile()`).
- **Nodo**: una funzione `state -> dict parziale`, registrata con
  `graph.add_node(nome, funzione)`.
- **Edge (arco)**: collegamento fisso (`add_edge`) o condizionale
  (`add_conditional_edges`) tra due nodi.
- **Checkpointer**: componente che salva lo stato del grafo ad ogni step
  (qui `MemorySaver`, in memoria); necessario per `interrupt`/`resume`.
- **`interrupt()` / `Command(resume=...)`**: primitiva LangGraph per
  sospendere un grafo a metà e riprenderlo più tardi con un valore
  fornito dall'esterno (qui: la decisione umana).
- **ReAct**: pattern Reason+Act, alternanza di Thought/Action/Observation.
- **Tool-calling / function calling**: capacità di un LLM di richiedere
  l'esecuzione di una funzione con argomenti strutturati, invece di
  scrivere solo testo libero.
- **RAG**: Retrieval-Augmented Generation — recuperare documenti esterni
  per fondare la generazione del LLM su informazioni reali e verificabili.
- **K-RAG**: qui, RAG la cui query viene arricchita con topic correlati
  presi da un Knowledge Graph prima della ricerca.
- **Embedding**: rappresentazione numerica (vettore) di un testo, tale
  che testi semanticamente simili abbiano vettori vicini.
- **FAISS**: libreria per la ricerca per similarità tra vettori (usata qui
  come indice in-memory, ricostruito ad ogni sessione).
- **Cypher**: linguaggio di interrogazione di Neo4j (`MATCH`/`MERGE`/`RETURN`).
- **Fine-tuning**: continuare l'addestramento di un modello pre-esistente
  su un dataset specifico per un task particolare.
- **Baseline zero-shot**: valutazione del modello *non* fine-tuned, con lo
  stesso prompt che userà la versione finale — il punto di partenza.
- **Hyperparameter sweep**: provare più combinazioni di iperparametri di
  training e scegliere la migliore in base a una metrica sul validation set.
- **Perplexity**: metrica di qualità di un language model — più bassa è,
  meglio il modello prevede il testo corretto.
- **ROUGE-L**: metrica di sovrapposizione testuale con un riferimento,
  basata sulla sequenza comune più lunga.
- **Grounding score**: metrica *ad hoc* di questo progetto — percentuale
  di token significativi dei fatti forniti che compaiono nel testo
  generato.
- **lru_cache**: decoratore Python (`functools.lru_cache`) che mette in
  cache il risultato di una funzione in base ai suoi argomenti — con
  `maxsize=1` e nessun argomento, equivale a "esegui solo la prima volta".
