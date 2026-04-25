"""
rag_chain.py — RAG Chain with Conversation Memory
Builds a ConversationalRetrievalChain backed by Groq LLaMA 3.1,
with sliding-window memory and MMR retrieval.
"""

import os
import re

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from config import GROQ_MODEL, TOP_K, GROQ_API_KEY


# ── System Prompt ────────────────────────────────────────────────────────────
# NOTE: The combine_docs chain in ConversationalRetrievalChain only receives
# {context} and {question} — NOT {chat_history}. Chat history is handled
# separately by the condense-question step.

_TEMPLATE = """\
You are a senior software engineer answering questions about a codebase.
Use ONLY the provided context to answer. Be technically precise.

Rules:
1. ALWAYS try to answer using the context provided. Most questions ARE about
   the codebase — even broad ones like "describe the files" or "project layout."
2. Include code snippets when the context contains relevant code.
3. Mention which file(s) your answer comes from.
4. If the context is insufficient, say briefly what you found and what is missing.
5. ONLY if the question is completely unrelated to any codebase or software
   (e.g. "what is the weather?", "tell me a joke"), reply with a single line:
   "⚠️ Not related to the codebase — please ask about the project's code."
6. Keep answers clear, structured, and concise.
7. At the very end of your answer, include a single line: "Confidence Score: [X]%" where X is your confidence (0-100) that the answer is accurate based ONLY on the provided context. If you are unsure or the context is partial, reduce the score accordingly.

Context:
{context}

Question: {question}

Answer:"""

_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=_TEMPLATE,
)


# ── Chain builder ────────────────────────────────────────────────────────────

def build_chain(vector_store):
    """
    Build and return a ConversationalRetrievalChain.

    Reads GROQ_API_KEY from the environment dynamically so the .env
    file is respected even if config was imported before dotenv ran.
    """
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0.2,
        max_tokens=1024,
        streaming=True,
    )

    memory = ConversationBufferWindowMemory(
        k=6,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": TOP_K * 3},
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": _PROMPT},
        output_key="answer",
    )

    return chain


# ── Query helper ─────────────────────────────────────────────────────────────

def ask(chain, question: str):
    """
    Query the chain and return the answer text and source metadata.

    Returns
    -------
    tuple : (answer_text, list of source dicts, confidence_score)
        Each source dict: {"file", "filename", "extension", "snippet"}
    """
    result = chain.invoke({"question": question})

    answer = result["answer"]
    
    # Extract confidence score if present
    score_match = re.search(r"Confidence Score:\s*(\d+)%", answer)
    score = int(score_match.group(1)) if score_match else None

    # Deduplicate source documents by file path
    seen = set()
    sources = []
    for doc in result.get("source_documents", []):
        src = doc.metadata.get("source", "unknown")
        if src not in seen:
            seen.add(src)
            sources.append(
                {
                    "file": src,
                    "filename": doc.metadata.get("filename", ""),
                    "extension": doc.metadata.get("extension", ""),
                    "snippet": doc.page_content[:300],
                }
            )

    return answer, sources, score
