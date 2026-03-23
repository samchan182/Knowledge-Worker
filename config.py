import os
from dotenv import load_dotenv

load_dotenv(override=True)

# API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-key-if-not-using-env")

# Models
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.3          # Low temperature — allergen accuracy is critical

# Embeddings
EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI default via LangChain

# Chunking
CHUNK_SIZE = 1000          # characters per chunk
CHUNK_OVERLAP = 200        # overlap between adjacent chunks

# Retrieval
TOP_K = 5                 # 5 chunks is sufficient for a single-document knowledge base

# Storage
DB_NAME = "vector_db"
KNOWLEDGE_BASE_DIR = "knowledge-base"

# UI
APP_TITLE = "Dragon Palace — Recipe & Allergen Assistant"
APP_PORT = 7860
