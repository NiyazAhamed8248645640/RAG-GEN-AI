import json
import os
import logging
from typing import List, Dict, Any

from app.vectorstore.vector_store import VectorStore
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.conversation_service import ConversationService
from app.prompts.templates import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.30"))
TOP_K = int(os.getenv("TOP_K", "3"))
CHUNK_SIZE_CHARS = int(os.getenv("CHUNK_SIZE_CHARS", "1600"))  # ≈ 400 tokens
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))
FALLBACK_RESPONSE = (
    "I could not find enough information in the knowledge base to answer this question. "
    "Please rephrase your query or contact support for further assistance."
)
# ──────────────────────────────────────────────────────────────────────────────


class RAGService:
    """
    Orchestrates the full RAG pipeline:

    Indexing  : load docs → chunk → embed → store in VectorStore
    Querying  : embed query → similarity search → build prompt → LLM → respond
    """

    def __init__(self):
        self.vector_store = VectorStore()
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService()
        self.conversation_service = ConversationService()

    # ── Chunking ──────────────────────────────────────────────────────────────

    @staticmethod
    def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Sliding-window character-based chunker with overlap.

        Splits on word boundaries to avoid cutting mid-word.
        Recommended: chunk_size ≈ 1600 chars (~400 tokens), overlap ≈ 200 chars.
        """
        words = text.split()
        chunks: List[str] = []
        current_words: List[str] = []
        current_len = 0

        for word in words:
            current_words.append(word)
            current_len += len(word) + 1  # +1 for space

            if current_len >= chunk_size:
                chunks.append(" ".join(current_words))
                # Roll back overlap worth of words
                rollback_len = 0
                rollback_words: List[str] = []
                for w in reversed(current_words):
                    rollback_len += len(w) + 1
                    rollback_words.insert(0, w)
                    if rollback_len >= overlap:
                        break
                current_words = rollback_words
                current_len = sum(len(w) + 1 for w in current_words)

        if current_words:
            chunks.append(" ".join(current_words))

        return chunks if chunks else [text]

    # ── Indexing ──────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Load docs.json, chunk all documents, generate embeddings in batch,
        and populate the vector store.
        """
        docs_path = os.getenv("DOCS_PATH", "docs.json")

        if not os.path.exists(docs_path):
            raise FileNotFoundError(
                f"Knowledge base not found at '{docs_path}'. "
                "Ensure docs.json exists in the project root."
            )

        with open(docs_path, "r", encoding="utf-8") as fh:
            documents: List[Dict[str, Any]] = json.load(fh)

        logger.info(f"[RAGService] Loaded {len(documents)} documents from '{docs_path}'")

        all_texts: List[str] = []
        all_metadata: List[Dict[str, Any]] = []

        for doc in documents:
            title: str = doc.get("title", "Untitled")
            content: str = doc.get("content", "")

            if not content.strip():
                logger.warning(f"[RAGService] Document '{title}' has empty content, skipping.")
                continue

            chunks = self._chunk_text(content, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS)
            logger.info(f"[RAGService] '{title}' → {len(chunks)} chunk(s)")

            for i, chunk in enumerate(chunks):
                all_texts.append(chunk)
                all_metadata.append(
                    {
                        "title": title,
                        "chunk_id": f"{title}_chunk_{i}",
                        "source": title,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    }
                )

        logger.info(
            f"[RAGService] Total chunks to embed: {len(all_texts)} — calling Embeddings API…"
        )

        # Single batched API call — much cheaper than N individual calls
        embeddings = await self.embedding_service.get_embeddings_batch(all_texts)

        for embedding, text, metadata in zip(embeddings, all_texts, all_metadata):
            self.vector_store.add(embedding, text, metadata)

        stats = self.vector_store.stats()
        logger.info(
            f"[RAGService] Vector store ready — "
            f"chunks={stats['total_chunks']}, "
            f"dim={stats['embedding_dim']}, "
            f"sources={stats['unique_sources']}"
        )

    # ── Querying ──────────────────────────────────────────────────────────────

    async def query(self, session_id: str, question: str) -> Dict[str, Any]:
        """
        Full RAG query pipeline:
          1. Embed the user question
          2. Retrieve top-k similar chunks (with threshold gating)
          3. Build context + history prompt
          4. Call LLM
          5. Update conversation history
          6. Return structured response
        """
        logger.info(f"[RAGService] Query — session='{session_id}' question='{question[:80]}'")

        # ── Step 1: Embed the query ────────────────────────────────────────────
        query_embedding = await self.embedding_service.get_embedding(question)

        # ── Step 2: Retrieve relevant chunks ──────────────────────────────────
        results = self.vector_store.search(
            query_embedding,
            top_k=TOP_K,
            threshold=SIMILARITY_THRESHOLD,
        )

        # ── Step 3: Fallback if nothing above threshold ────────────────────────
        if not results:
            logger.info(
                f"[RAGService] No chunks above threshold {SIMILARITY_THRESHOLD} — "
                "returning fallback response."
            )
            self.conversation_service.add_message(session_id, "user", question)
            self.conversation_service.add_message(session_id, "assistant", FALLBACK_RESPONSE)
            return {
                "reply": FALLBACK_RESPONSE,
                "tokensUsed": 0,
                "retrievedChunks": 0,
            }

        # ── Step 4: Build context string ──────────────────────────────────────
        context_parts = []
        for rank, result in enumerate(results, start=1):
            source = result["metadata"]["source"]
            score = result["score"]
            context_parts.append(
                f"[{rank}] Source: {source} (similarity: {score:.3f})\n{result['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # ── Step 5: Gather conversation history ───────────────────────────────
        history_str = self.conversation_service.format_history_for_prompt(session_id)

        # ── Step 6: Construct LLM message ─────────────────────────────────────
        user_message = build_user_prompt(context, history_str, question)

        # ── Step 7: Call the LLM ──────────────────────────────────────────────
        reply, tokens_used = await self.llm_service.generate(
            system_prompt=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        # ── Step 8: Persist to conversation history ───────────────────────────
        self.conversation_service.add_message(session_id, "user", question)
        self.conversation_service.add_message(session_id, "assistant", reply)

        logger.info(
            f"[RAGService] Response generated — "
            f"chunks_used={len(results)} tokens={tokens_used}"
        )

        return {
            "reply": reply,
            "tokensUsed": tokens_used,
            "retrievedChunks": len(results),
        }
