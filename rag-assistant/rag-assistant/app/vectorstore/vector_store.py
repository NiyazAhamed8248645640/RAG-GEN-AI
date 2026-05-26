import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    """
    In-memory vector store supporting cosine similarity search.

    Design:
    - Embeddings stored as a 2D numpy matrix for fast batch operations.
    - Metadata and raw texts stored in parallel lists indexed by position.
    - Cosine similarity computed via normalized dot product.
    """

    def __init__(self):
        self._embeddings: List[np.ndarray] = []
        self._texts: List[str] = []
        self._metadata: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add(self, embedding: List[float], text: str, metadata: Dict[str, Any]) -> None:
        """Add a single chunk and its embedding to the store."""
        vec = np.array(embedding, dtype=np.float32)
        # Pre-normalise for cheaper similarity computation later
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        self._embeddings.append(vec)
        self._texts.append(text)
        self._metadata.append(metadata)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine similarity between two L2-normalised vectors.
        Since both vectors are already normalised on ingestion/query,
        this reduces to a plain dot product: O(d).
        """
        return float(np.dot(a, b))

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k chunks whose cosine similarity >= threshold.

        Returns:
            List of dicts: {text, metadata, score}
        """
        if not self._embeddings:
            logger.warning("Vector store is empty — no chunks to search.")
            return []

        # Normalise the query vector
        query_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec = query_vec / q_norm

        # Stack all stored vectors into a matrix and batch dot-product
        matrix = np.stack(self._embeddings)          # shape: (N, D)
        scores = matrix @ query_vec                  # shape: (N,)

        # Sort descending, take top_k
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            chunk_title = self._metadata[idx].get("title", "unknown")
            logger.info(
                f"[VectorStore] chunk='{chunk_title}' "
                f"chunk_id='{self._metadata[idx].get('chunk_id')}' "
                f"similarity={score:.4f}"
            )
            if score >= threshold:
                results.append(
                    {
                        "text": self._texts[idx],
                        "metadata": self._metadata[idx],
                        "score": score,
                    }
                )

        logger.info(
            f"[VectorStore] {len(results)}/{top_k} chunks above threshold {threshold}"
        )
        return results

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._embeddings)

    def stats(self) -> Dict[str, Any]:
        titles = [m.get("title", "?") for m in self._metadata]
        unique_sources = list(set(titles))
        return {
            "total_chunks": len(self._embeddings),
            "unique_sources": unique_sources,
            "embedding_dim": self._embeddings[0].shape[0] if self._embeddings else 0,
        }
