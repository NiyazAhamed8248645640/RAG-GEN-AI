"""
Prompt templates for the RAG assistant.

Design principles:
- System prompt: tells the model to stay strictly within the provided context.
- User message: injects retrieved context, conversation history, and the question.
- Temperature kept low (0–0.3) to reduce hallucination.
"""

SYSTEM_PROMPT = """You are a precise and helpful customer support assistant.

Rules you MUST follow:
1. Answer ONLY using the information in the provided Context section.
2. Do NOT use any prior training knowledge that is not reflected in the context.
3. If the context does not contain enough information to answer the question, respond with:
   "I could not find enough information in the knowledge base to answer this question."
4. Be concise, accurate, and professional.
5. If you quote specific steps or values, make sure they come directly from the context."""


def build_user_prompt(context: str, history: str, question: str) -> str:
    """
    Construct the full user-turn message injected into the LLM.

    Parameters
    ----------
    context  : Retrieved document chunks joined as a string.
    history  : Formatted past conversation turns (may be empty).
    question : The current user question.
    """
    history_section = (
        f"Conversation History:\n{history}\n\n"
        if history
        else "Conversation History:\n(No prior conversation)\n\n"
    )

    return (
        f"Context:\n{context}\n\n"
        f"{history_section}"
        f"Question: {question}\n\n"
        "Answer based ONLY on the context above:"
    )
