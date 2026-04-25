"""
ingest.py — Full Ingestion Pipeline
Clones a GitHub repo, collects source files, chunks them,
embeds with HuggingFace, and persists a FAISS vector store.
Also provides helpers for Q&A retention (append to store + log).
"""

import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from functools import lru_cache
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import (
    CLONE_DIR,
    VECTOR_STORE_DIR,
    HASH_FILE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    SUPPORTED_EXTENSIONS,
    SKIP_DIRS,
    SKIP_FILES,
    CONVERSATION_LOG,
)


# ── Embedding Model Cache ───────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_embeddings():
    """Return a cached instance of HuggingFaceEmbeddings."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


# ── Hashing helpers ─────────────────────────────────────────────────────────

def _repo_hash(url: str) -> str:
    """Return MD5 hex-digest of the repository URL string."""
    return hashlib.md5(url.encode("utf-8")).hexdigest()


def _cached(url: str) -> bool:
    """Check if the repo URL was already ingested (hash file matches)."""
    hash_path = Path(HASH_FILE)
    if not hash_path.exists():
        return False
    return hash_path.read_text().strip() == _repo_hash(url)


def _save_hash(url: str) -> None:
    """Persist the MD5 hash of the URL to disk."""
    Path(VECTOR_STORE_DIR).mkdir(parents=True, exist_ok=True)
    Path(HASH_FILE).write_text(_repo_hash(url))


# ── Clone ────────────────────────────────────────────────────────────────────

def _clone(url: str, progress_cb=None) -> None:
    """Shallow-clone the repository into CLONE_DIR."""
    if progress_cb:
        progress_cb("🔄 Cloning repository …")

    clone_path = Path(CLONE_DIR)
    if clone_path.exists():
        shutil.rmtree(clone_path)

    subprocess.run(
        ["git", "clone", "--depth=1", url, CLONE_DIR],
        check=True,
        capture_output=True,
    )


# ── Collect documents ────────────────────────────────────────────────────────

def _collect_documents(progress_cb=None) -> list[Document]:
    """Walk CLONE_DIR, read supported files, return LangChain Documents."""
    docs: list[Document] = []
    root = Path(CLONE_DIR)

    for filepath in root.rglob("*"):
        # Skip directories themselves
        if filepath.is_dir():
            continue

        # Skip files inside excluded directories
        parts = filepath.relative_to(root).parts
        if any(d in SKIP_DIRS for d in parts):
            continue

        # Skip lockfiles or other noise
        if filepath.name in SKIP_FILES:
            continue

        # Filter by extension
        if filepath.suffix not in SUPPORTED_EXTENSIONS:
            continue

        text = filepath.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            continue

        relative_path = str(filepath.relative_to(root))
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": relative_path,
                    "extension": filepath.suffix,
                    "filename": filepath.name,
                },
            )
        )

        if progress_cb and len(docs) % 20 == 0:
            progress_cb(f"📂 Collected {len(docs)} files …")

    return docs


# ── Split ────────────────────────────────────────────────────────────────────

def _split(docs: list[Document], progress_cb=None) -> list[Document]:
    """Split documents into chunks using RecursiveCharacterTextSplitter."""
    if progress_cb:
        progress_cb("✂️ Splitting into chunks …")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


# ── Embed & index ────────────────────────────────────────────────────────────

def _embed_and_index(chunks: list[Document], progress_cb=None):
    """Create HuggingFace embeddings and persist a FAISS index."""
    if progress_cb:
        progress_cb("🧠 Generating embeddings (this may take a moment) …")

    embeddings = get_embeddings()
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(VECTOR_STORE_DIR)
    return store


# ── Public API ───────────────────────────────────────────────────────────────

def ingest(url: str, progress_cb=None, force: bool = False):
    """
    Full ingestion pipeline.

    Returns
    -------
    tuple : (FAISS vector store, stats dict, was_cached bool)
    """
    if not force and _cached(url):
        if progress_cb:
            progress_cb("⚡ Loading cached vector store …")
        store = load_existing()
        return store, {}, True

    # Step 1 — Clone
    _clone(url, progress_cb)

    # Step 2 — Collect
    docs = _collect_documents(progress_cb)

    # Step 3 — Split
    chunks = _split(docs, progress_cb)

    # Step 4 — Embed & Index
    store = _embed_and_index(chunks, progress_cb)

    # Step 5 — Save hash
    _save_hash(url)

    stats = {"files": len(docs), "chunks": len(chunks)}
    return store, stats, False


def load_existing():
    """Load a previously saved FAISS index from disk."""
    embeddings = get_embeddings()
    return FAISS.load_local(
        VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True
    )


# ── Q&A Retention ────────────────────────────────────────────────────────────

def append_qa_to_store(store, question: str, answer: str, sources: list) -> None:
    """
    Add a Q&A pair as a document to the live FAISS store.
    This enriches future retrievals with past conversation context.
    """
    source_files = ", ".join(s.get("file", "") for s in sources) if sources else "general"
    doc = Document(
        page_content=f"Question: {question}\n\nAnswer: {answer}",
        metadata={
            "source": f"qa_history/{source_files}",
            "extension": ".qa",
            "filename": "conversation_history",
            "type": "qa_pair",
        },
    )
    store.add_documents([doc])


def save_store(store) -> None:
    """Persist the current FAISS store to disk (call after adding Q&A docs)."""
    store.save_local(VECTOR_STORE_DIR)


def log_conversation(question: str, answer: str, sources: list) -> None:
    """Append a Q&A entry to the persistent JSONL conversation log."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "sources": [s.get("file", "") for s in sources],
    }
    with open(CONVERSATION_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
