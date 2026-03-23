# Knowledge Worker — RAG Q&A Assistant

A Retrieval-Augmented Generation (RAG) system that turns your internal markdown documents into a conversational Q&A assistant. Ask natural-language questions and get grounded answers sourced directly from your knowledge base.

---

## What Problem It Solves

Internal documents — recipes, contracts, product specs, employee handbooks — are hard to search. You either remember where something is or you don't. This system ingests your `.md` files, converts them into searchable vector embeddings, and lets anyone on the team ask plain-English questions and get precise answers instantly.

**Example use case (this repo):** Dragon Palace kitchen staff can ask "Does the lamb flatbread contain fish?" and get a safety-aware answer drawn from the confidential recipe and allergen database — no need to scan the whole document manually.

---

## Framework & Architecture

The system is built on three operations:

```
INGEST    →  Documents → Chunks → Vectors → ChromaDB
RETRIEVE  →  Question → Vector Search → Top-K Chunks
GENERATE  →  Chunks + Question + History → LLM → Answer
```

### Packages Used

| Package | Role |
|---|---|
| `langchain` | Orchestration framework — connects retriever, prompt, and LLM |
| `langchain-openai` | OpenAI LLM and embedding wrappers |
| `langchain-chroma` | ChromaDB vector store integration |
| `langchain-community` | Document loaders (`TextLoader`, `DirectoryLoader`) |
| `langchain-text-splitters` | `CharacterTextSplitter` for chunking documents |
| `chromadb` | Local vector database — stores and searches embeddings |
| `openai` | Underlying OpenAI API client |
| `python-dotenv` | Loads `OPENAI_API_KEY` from `.env` file |
| `gradio` | Local chat web UI — zero frontend code required |

---

## Project Structure

```
Knowledge-Worker/
├── knowledge-base/          # Your source documents (.md files)
│   └── products/
│       └── marketllm.md    # Example: Dragon Palace recipe & allergen database
├── src/
│   ├── ingest.py           # Step 1: Load → chunk → embed → store to ChromaDB
│   ├── retriever.py        # Step 2: Embed query → cosine similarity search
│   ├── chain.py            # Step 3: Retriever + LLM + chat history → answer
│   └── app.py              # Gradio chat UI
├── vector_db/              # ChromaDB persistent storage (auto-created, gitignored)
├── config.py               # All settings in one place (model, chunk size, etc.)
├── requirements.txt
├── .env                    # Your API key (gitignored)
└── .gitignore
```

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up your API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxx
```

Rules:
- No quotes around the key
- No spaces around the `=`
- No trailing whitespace or blank lines

### 3. Ingest your knowledge base

```bash
python -m src.ingest
```

This reads all `.md` files in `knowledge-base/`, splits them into chunks, embeds each chunk, and stores them in `vector_db/`.

### 4. Launch the chat UI

```bash
python -m src.app
```

Opens the chat interface at `http://localhost:7860`.

---

## Usage for Users in Mainland China

OpenAI's API is not directly accessible from mainland China. You need a US or Japan VPN active, then set your terminal proxy before running any command.

**Step 1:** Connect your VPN (select a US or Japan server)

**Step 2:** Export proxy settings in your terminal:

```bash
export https_proxy=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
```

> Port `7890` is the default for Clash. If you use a different proxy client, replace the port accordingly.

**Step 3:** Run the app in the same terminal session:

```bash
python -m src.ingest    # first time only
python -m src.app
```

The proxy settings only apply to the current terminal session. You need to re-export them each time you open a new terminal.

---

## How to Change the OpenAI Model

Open `config.py` and change `LLM_MODEL`:

```python
# Current default
LLM_MODEL = "gpt-4o-mini"

# Other options
LLM_MODEL = "gpt-4o"          # More powerful, higher cost
LLM_MODEL = "gpt-4-turbo"     # Previous generation GPT-4
LLM_MODEL = "gpt-3.5-turbo"   # Fastest, lowest cost
```

You can also adjust temperature for different use cases:

```python
LLM_TEMPERATURE = 0.1   # Very factual — best for safety-critical info (allergens, contracts)
LLM_TEMPERATURE = 0.5   # Balanced
LLM_TEMPERATURE = 0.9   # More creative — better for brainstorming
```

No re-ingestion needed after changing the model or temperature — only restart the app.

---

## How to Swap the Knowledge Base for a Different Company

### Step 1 — Replace the documents

Delete or archive the existing files in `knowledge-base/` and add your new `.md` files:

```
knowledge-base/
└── your-company/
    ├── products.md
    ├── contracts.md
    └── employees.md
```

The ingest script automatically finds all `.md` files recursively — no code changes needed.

### Step 2 — Update the UI labels (optional)

In `config.py`, change the app title:

```python
APP_TITLE = "Your Company — Internal Assistant"
```

In `src/chain.py`, update the system prompt to match your domain:

```python
SYSTEM_PROMPT = """You are an internal assistant for [Your Company].
Answer questions using ONLY the context provided below.
...
```

### Step 3 — Re-ingest

```bash
python -m src.ingest
```

This clears the old vector store and rebuilds from your new documents. Then relaunch the app:

```bash
python -m src.app
```

---

## Tuning Parameters

All tunable settings live in `config.py`:

| Parameter | Default | When to change |
|---|---|---|
| `CHUNK_SIZE` | 1000 | Increase if your docs have long sections; decrease for short bullet-point docs |
| `CHUNK_OVERLAP` | 200 | Increase if answers are getting cut off at chunk boundaries |
| `TOP_K` | 5 | Increase if the assistant misses relevant context; decrease to reduce noise |
| `LLM_TEMPERATURE` | 0.3 | Lower for factual/safety content; higher for creative use cases |

After changing chunk settings, always re-run `python -m src.ingest` to rebuild the vector store.
