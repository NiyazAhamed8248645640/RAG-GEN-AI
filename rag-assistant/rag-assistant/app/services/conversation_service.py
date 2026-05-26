import logging
from collections import defaultdict
from typing import List, Dict

logger = logging.getLogger(__name__)

# Store the last N user+assistant pairs per session
MAX_HISTORY_PAIRS = int(__import__("os").getenv("MAX_HISTORY_PAIRS", "5"))


class ConversationService:
    """
    Manages per-session conversation history in memory.

    Each session stores up to MAX_HISTORY_PAIRS message pairs
    (user + assistant), giving the LLM short-term context continuity.
    """

    def __init__(self):
        # session_id → list of {role, content} dicts
        self._sessions: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message and prune to the rolling window."""
        self._sessions[session_id].append({"role": role, "content": content})
        max_messages = MAX_HISTORY_PAIRS * 2
        if len(self._sessions[session_id]) > max_messages:
            self._sessions[session_id] = self._sessions[session_id][-max_messages:]

    def clear_session(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"[ConversationService] Cleared session '{session_id}'")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        return list(self._sessions[session_id])

    def format_history_for_prompt(self, session_id: str) -> str:
        """
        Return a human-readable conversation transcript for injection
        into the LLM prompt, e.g.:

            User: How do I reset my password?
            Assistant: Navigate to Settings > Security …
        """
        history = self.get_history(session_id)
        if not history:
            return ""
        lines = []
        for msg in history:
            label = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{label}: {msg['content']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def session_count(self) -> int:
        return len(self._sessions)
