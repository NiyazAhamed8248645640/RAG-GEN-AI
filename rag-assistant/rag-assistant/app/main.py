import os
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.utils.logger import setup_logging
from app.routes.chat import router as chat_router
from app.services.rag_service import RAGService

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)


# ── Lifespan: index documents on startup ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("Starting GenAI RAG Assistant")
    logger.info("=" * 60)

    rag_service = RAGService()
    try:
        await rag_service.initialize()
        app.state.rag_service = rag_service
        logger.info("RAG service initialised — server is ready.")
    except Exception as exc:
        logger.error(f"Failed to initialise RAG service: {exc}")
        raise

    yield  # ── application running ──

    logger.info("Shutting down GenAI RAG Assistant.")


# ── Application factory ───────────────────────────────────────────────────────
app = FastAPI(
    title="GenAI RAG Assistant",
    description=(
        "A production-grade Retrieval-Augmented Generation chat assistant. "
        "Answers questions grounded strictly in the knowledge base."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Health check (before static mount so it takes priority) ───────────────────
@app.get("/health", tags=["Infrastructure"], summary="Health check")
async def health():
    return {"status": "healthy"}

# ── API routes ────────────────────────────────────────────────────────────────
app.include_router(chat_router, prefix="/api", tags=["Chat"])

# ── Serve frontend static files ───────────────────────────────────────────────
_frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
    logger.info(f"Serving frontend from: {_frontend_dir}")
else:
    logger.warning("'frontend/' directory not found — UI will not be served.")
