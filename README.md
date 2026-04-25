# 🧠 CodeBase RAG Assistant

A production-ready **Retrieval-Augmented Generation** application that lets you chat with any public GitHub repository. Paste a repo URL, ingest the entire codebase into a local FAISS vector store, and ask questions — the LLM answers with source-file citations and full conversation memory.

---

## 🏗️ Tech Stack

| Layer | Tool |
|-------|------|
| Ingestion | GitPython + `subprocess` |
| Parsing | LangChain Document Loaders |
| Splitting | `RecursiveCharacterTextSplitter` |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace, free, local) |
| Vector Store | FAISS (local, no server needed) |
| LLM | LLaMA 3 via Groq API (free tier) |
| RAG Chain | LangChain `ConversationalRetrievalChain` |
| Frontend | Streamlit |

---

## 🚀 Setup (Conda)

```bash
git clone <your-repo-url>
cd codebase-rag-assistant

# Create and activate the Conda environment
conda create -n cod python=3.11 -y
conda activate cod

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Open .env and paste your GROQ_API_KEY

# Launch
streamlit run app.py
```

### 🔑 Get a Free Groq API Key

1. Visit **[https://console.groq.com](https://console.groq.com)**
2. Sign up for a free account
3. Navigate to **API Keys → Create key**
4. Copy the key and paste it into your `.env` file (or the sidebar input at runtime)

---

## ⚙️ How It Works

1. **Clone** — User pastes a GitHub URL → the app runs `git clone --depth=1` to fetch the repo.
2. **Collect** — All source files are walked, filtered by supported extensions (`.py`, `.js`, `.ts`, `.java`, `.go`, `.rs`, etc.), and directories like `node_modules`, `.git`, `__pycache__` are skipped.
3. **Split** — Files are split into **500-token chunks** with **50-token overlap** using `RecursiveCharacterTextSplitter`.
4. **Embed** — Chunks are embedded with the `all-MiniLM-L6-v2` model (runs locally, no API needed).
5. **Index** — Embeddings are saved to a local **FAISS** vector store on disk.
6. **Query** — User question → **MMR retrieval** of top-5 chunks → **LLaMA 3** (via Groq) generates an answer with file citations.

---

## 🔥 God-Mode Features

- ⚡ **Smart caching** — MD5 hash of the repo URL; skips re-ingestion if already indexed
- 🧠 **Conversation memory** — sliding window of last 6 turns via `ConversationBufferWindowMemory`
- 📄 **Source attribution** — every answer shows the exact file paths it was derived from
- 🔍 **MMR retrieval** — Max Marginal Relevance for diverse, non-redundant chunks
- 🔄 **Force re-ingest toggle** — override cache when the repo has been updated
- 🎨 **Dark UI** — production-grade Streamlit interface with custom CSS, Inter & JetBrains Mono fonts

---

## 📁 Project Structure

```
codebase-rag-assistant/
├── app.py              # Streamlit UI — chat interface & controls
├── ingest.py           # Clone → collect → split → embed → FAISS
├── rag_chain.py        # ConversationalRetrievalChain + memory
├── config.py           # All constants (no hardcoded values elsewhere)
├── requirements.txt    # Python dependencies
├── .env.example        # Template for Groq API key
└── README.md           # This file
```

---
