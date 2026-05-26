import logging
from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import JSONResponse

from app.models.schemas import ChatRequest, ChatResponse, HealthResponse, ErrorResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        503: {"model": ErrorResponse, "description": "LLM or embedding service unavailable"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Send a message to the RAG assistant",
)
async def chat(request: Request, body: ChatRequest):
    """
    RAG chat endpoint.

    Validates payload → retrieves relevant chunks via cosine similarity →
    injects context + history into LLM prompt → returns grounded reply.
    """
    # Manual validation on top of Pydantic (belt-and-suspenders)
    if not body.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Message field is required and must not be empty."},
        )
    if not body.sessionId.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "sessionId field is required and must not be empty."},
        )

    rag_service = request.app.state.rag_service

    try:
        result = await rag_service.query(
            session_id=body.sessionId.strip(),
            question=body.message.strip(),
        )
        return ChatResponse(**result)

    except ValueError as exc:
        # Raised for bad API keys, rate limits, etc.
        logger.warning(f"[/api/chat] Service error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": str(exc)},
        )
    except FileNotFoundError as exc:
        logger.error(f"[/api/chat] Missing resource: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Knowledge base not initialised. Contact administrator."},
        )
    except Exception as exc:
        logger.exception(f"[/api/chat] Unexpected error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "An unexpected error occurred. Please try again."},
        )


@router.delete(
    "/sessions/{session_id}",
    summary="Clear conversation history for a session",
)
async def clear_session(request: Request, session_id: str):
    """Reset the conversation history for the given session (used by New Chat)."""
    rag_service = request.app.state.rag_service
    rag_service.conversation_service.clear_session(session_id)
    return {"status": "cleared", "sessionId": session_id}
