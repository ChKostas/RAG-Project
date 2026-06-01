import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

KNOWLEDGE_BASE_PATH = BASE_DIR / "Knowledge-Base-Incurance"
DB_PATH = str(BASE_DIR / "preprocessed_db")
COLLECTION_NAME = "docs"

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 5

LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080/completion")