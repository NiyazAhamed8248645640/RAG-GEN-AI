from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    sessionId: str = Field(..., min_length=1, description="Unique session identifier")
    message: str = Field(..., min_length=1, description="User message")

    model_config = {"json_schema_extra": {"example": {"sessionId": "abc123", "message": "How do I reset my password?"}}}


class ChatResponse(BaseModel):
    reply: str
    tokensUsed: Optional[int] = None
    retrievedChunks: int


class HealthResponse(BaseModel):
    status: str


class ErrorResponse(BaseModel):
    error: str
