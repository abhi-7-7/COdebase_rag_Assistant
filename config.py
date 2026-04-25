"""
config.py — Central Configuration
All constants and settings for the Codebase RAG Assistant.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Directory paths ──────────────────────────────────────────────────────────
CLONE_DIR = "cloned_repo"
VECTOR_STORE_DIR = "vector_store"
HASH_FILE = "vector_store/.repo_hash"
CONVERSATION_LOG = "conversation_log.jsonl"

# ── Chunking parameters ─────────────────────────────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ── Embedding model ─────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_K = 5

# ── LLM via Groq ────────────────────────────────────────────────────────────
GROQ_MODEL = "llama-3.1-8b-instant"
try:
    import streamlit as st
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
except Exception:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── File filtering ───────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".cpp", ".c", ".h", ".cs",
    ".go", ".rs", ".rb", ".php", ".swift",
    ".md", ".txt", ".json", ".yaml", ".yml",
    ".html", ".css", ".scss", ".sh", ".env.example",
}

SKIP_DIRS = {
    "node_modules", ".git", "__pycache__",
    ".venv", "venv", "env",
    "dist", "build", ".next", ".vercel",
    "coverage", ".mypy_cache",
}

SKIP_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "poetry.lock", "composer.lock", "Cargo.lock",
}
