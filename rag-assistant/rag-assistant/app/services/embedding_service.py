import os
import logging
from typing import List

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

# Retry on transient errors only
_RETRYABLE = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
)


class EmbeddingService:
    """
    Wraps the OpenAI Embeddings API.

    Strategy
    --------
    - Single texts  → `get_embedding()`
    - Batch texts   → `get_embeddings_batch()` (one API call, cheaper)
    - Both methods apply a 3-attempt exponential-backoff retry on transient errors.
    """

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        logger.info(f"[EmbeddingService] Using model: {self._model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type(_RETRYABLE),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def get_embedding(self, text: str) -> List[float]:
        """Embed a single string."""
        try:
            response = await self._client.embeddings.create(
                input=text.replace("\n", " "),
                model=self._model,
            )
            return response.data[0].embedding
        except openai.AuthenticationError as exc:
            logger.error("[EmbeddingService] Authentication failed — check OPENAI_API_KEY.")
            raise ValueError("Invalid OpenAI API key.") from exc
        except _RETRYABLE as exc:
            logger.warning(f"[EmbeddingService] Transient error, will retry: {exc}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type(_RETRYABLE),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of strings in a single API call.
        OpenAI supports batches up to 2048 inputs per request.
        """
        cleaned = [t.replace("\n", " ") for t in texts]
        try:
            response = await self._client.embeddings.create(
                input=cleaned,
                model=self._model,
            )
            # API returns items ordered by their index
            items = sorted(response.data, key=lambda x: x.index)
            logger.info(
                f"[EmbeddingService] Embedded {len(items)} chunks via batch call."
            )
            return [item.embedding for item in items]
        except openai.AuthenticationError as exc:
            logger.error("[EmbeddingService] Authentication failed — check OPENAI_API_KEY.")
            raise ValueError("Invalid OpenAI API key.") from exc
        except _RETRYABLE as exc:
            logger.warning(f"[EmbeddingService] Transient error, will retry: {exc}")
            raise
