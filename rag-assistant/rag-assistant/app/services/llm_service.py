import os
import logging
from typing import List, Dict, Tuple, Optional

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

_RETRYABLE = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
)


class LLMService:
    """
    Wraps the OpenAI Chat Completions API.

    Configuration (via env vars):
    - LLM_MODEL        default: gpt-3.5-turbo
    - LLM_TEMPERATURE  default: 0.2  (spec: 0–0.3)
    - LLM_MAX_TOKENS   default: 512
    """

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        self._temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        self._max_tokens = int(os.getenv("LLM_MAX_TOKENS", "512"))
        logger.info(
            f"[LLMService] model={self._model} "
            f"temperature={self._temperature} "
            f"max_tokens={self._max_tokens}"
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type(_RETRYABLE),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def generate(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
    ) -> Tuple[str, Optional[int]]:
        """
        Call the LLM and return (reply_text, tokens_used).

        Parameters
        ----------
        system_prompt : str
            The system-role instruction passed to the model.
        messages : list of {role, content} dicts
            Conversation turns to include (should include the final user message).
        """
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                timeout=30,
                messages=[{"role": "system", "content": system_prompt}] + messages,
            )

            reply: str = response.choices[0].message.content.strip()
            tokens_used: Optional[int] = (
                response.usage.total_tokens if response.usage else None
            )

            logger.info(
                f"[LLMService] model={self._model} "
                f"prompt_tokens={getattr(response.usage, 'prompt_tokens', '?')} "
                f"completion_tokens={getattr(response.usage, 'completion_tokens', '?')} "
                f"total_tokens={tokens_used}"
            )
            return reply, tokens_used

        except openai.AuthenticationError as exc:
            logger.error("[LLMService] Authentication failed — check OPENAI_API_KEY.")
            raise ValueError("Invalid API key. Please check your server configuration.") from exc

        except openai.RateLimitError as exc:
            logger.warning("[LLMService] Rate limit hit — will retry.")
            raise  # Let tenacity retry

        except openai.APITimeoutError as exc:
            logger.error("[LLMService] Request timed out.")
            raise  # Let tenacity retry

        except openai.BadRequestError as exc:
            logger.error(f"[LLMService] Bad request: {exc}")
            raise ValueError(f"LLM request was malformed: {exc}") from exc

        except Exception as exc:
            logger.error(f"[LLMService] Unexpected error: {exc}")
            raise
