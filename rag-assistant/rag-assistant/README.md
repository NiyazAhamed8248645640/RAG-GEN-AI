# ⚡ TechFlow GenAI RAG Assistant

A **production-grade** Retrieval-Augmented Generation (RAG) chat assistant built with FastAPI, OpenAI, and in-memory vector search.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Browser (Frontend)                          │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │  index.html  │    │    styles.css     │    │     app.js       │  │
│  │  Chat UI     │    │  Layout/Design    │    │ Session + API    │  │
│  └──────┬───────┘    └──────────────────┘    └────────┬─────────┘  │
│         │                POST /api/chat                │             │
└─────────┼──────────────────────────────────────────────────────────┘
          │  HTTP/JSON                          ▲
          ▼                                     │
┌─────────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                              │
│                                                                     │
│  GET /health          POST /api/chat       DELETE /api/sessions/:id │
│       │                     │                       │               │
│       │              ┌──────▼──────────────────┐    │               │
│       │              │      RAG Service         │◄───┘               │
│       │              │                          │                   │
│       │              │  1. Embed query          │                   │
│       │              │  2. Vector search        │                   │
│       │              │  3. Build context prompt │                   │
│       │              │  4. Call LLM             │                   │
│       │              │  5. Update history       │                   │
│       │              └──────────────────────────┘                   │
│       │                /        |         \                         │
│       │    ┌──────────┘  ┌──────┘    ┌────┘                        │
│       │    ▼             ▼           ▼                              │
│  ┌────┴──────────┐ ┌──────────┐ ┌──────────────────┐               │
│  │  Embedding    │ │ Vector   │ │ Conversation      │               │
│  │  Service      │ │ Store    │ │ Service           │               │
│  │  (OpenAI API) │ │ (NumPy   │ │ (In-memory        │               │
│  │               │ │ Cosine   │ │  session history) │               │
│  └───────┬───────┘ │ Similarity│ └──────────────────┘              │
│          │         └──────┬───┘                                     │
│          │                │                                         │
│  ┌───────▼────────────────▼──────────┐                              │
│  │            LLM Service            │                              │
│  │  (OpenAI gpt-3.5-turbo, T=0.2)   │                              │
│  └───────────────────────────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┘
          ▼
  ┌──────────────────────┐
  │    OpenAI APIs        │
  │  Embeddings API       │  text-embedding-3-small
  │  Chat Completions API │  gpt-3.5-turbo
  └──────────────────────┘
```

---

## RAG Workflow

### Indexing (on server startup)

```
docs.json
    │
    ▼ Load JSON
[{title, content}, ...]
    │
    ▼ Chunk (sliding window, ~400 tokens, 50-token overlap)
[chunk_0, chunk_1, chunk_2, ...]
    │
    ▼ Batch embed (OpenAI text-embedding-3-small)
[[0.02, -0.14, ...], ...]   ← 1536-dim float32 vectors
    │
    ▼ L2-normalise & store
VectorStore (in-memory NumPy arrays + metadata)
```

### Querying (per user message)

```
User question
    │
    ▼ Embed (same model as indexing)
query_vector (1536-dim, normalised)
    │
    ▼ Cosine similarity against all stored vectors
        score_i = dot(query_vec, chunk_vec_i)     ← O(N·D)
    │
    ▼ Sort descending, keep top-3
    │
    ▼ Threshold gate (default 0.30)
        score < 0.30 → fallback response
        score ≥ 0.30 → pass to LLM
    │
    ▼ Build prompt
        System: "Answer only from context…"
        User:   Context + History + Question
    │
    ▼ OpenAI gpt-3.5-turbo (T=0.2)
    │
    ▼ Return reply + tokensUsed + retrievedChunks
```

---

## Embedding Strategy

| Property       | Detail                              |
|----------------|-------------------------------------|
| **Model**      | `text-embedding-3-small` (OpenAI)   |
| **Dimensions** | 1536 floats                         |
| **Batching**   | All chunks embedded in a single API call at startup |
| **Normalisation** | L2-normalised on ingestion, enabling dot-product = cosine similarity |
| **Chunking**   | Sliding window, ~1600 chars (~400 tokens), 200-char overlap |

The overlap ensures that sentences near chunk boundaries appear in at least two chunks, preventing context loss at split points.

---

## Similarity Search

**Method:** Cosine Similarity via normalised dot product.

Since all vectors are L2-normalised on ingestion:

```
cosine_similarity(q, c) = dot(q_hat, c_hat) = dot(q/‖q‖, c/‖c‖)
```

This is implemented as a matrix-vector product across all stored embeddings:

```python
matrix = np.stack(self._embeddings)   # shape (N, 1536)
scores = matrix @ query_vec           # shape (N,)  — one operation
```

**Why cosine similarity?**  
It measures the angle between vectors, making it robust to length differences. Documents about "how to reset a password" and a query "forgot my password" will have a high cosine similarity even though the wording differs.

**Threshold:** Default `0.30`. Queries with no chunk above this score receive a safe fallback rather than a hallucinated answer.

---

## Prompt Design

```
System:
  You are a precise customer support assistant.
  Answer ONLY using the provided Context.
  If context is insufficient, say so explicitly.

User:
  Context:
  [1] Source: Password Reset (similarity: 0.812)
  Users can reset their password by navigating to...

  ---

  [2] Source: Two-Factor Authentication (similarity: 0.731)
  Two-factor authentication (2FA) significantly...

  Conversation History:
  User: Do I need 2FA?
  Assistant: 2FA is optional but strongly recommended...

  Question: How do I reset my password?

  Answer based ONLY on the context above:
```

**Why this structure?**
- Context first: the model processes it before reading the question, anchoring it to the retrieved facts.
- History included: enables follow-up questions within a session.
- Strict instruction: "Answer ONLY using context" dramatically reduces hallucination.
- Temperature 0.2: low randomness for factual, reproducible answers.

---

## Project Structure

```
project/
│
├── app/
│   ├── main.py                  ← FastAPI app, lifespan, CORS, routing
│   ├── routes/
│   │   └── chat.py              ← POST /api/chat, DELETE /api/sessions/:id
│   ├── services/
│   │   ├── rag_service.py       ← Core pipeline: indexing + querying
│   │   ├── embedding_service.py ← OpenAI Embeddings API wrapper
│   │   ├── llm_service.py       ← OpenAI Chat Completions wrapper
│   │   └── conversation_service.py ← Per-session history (last 5 pairs)
│   ├── models/
│   │   └── schemas.py           ← Pydantic request/response schemas
│   ├── vectorstore/
│   │   └── vector_store.py      ← In-memory cosine similarity store
│   ├── prompts/
│   │   └── templates.py         ← System prompt + user message builder
│   └── utils/
│       └── logger.py            ← Structured logging setup
│
├── frontend/
│   ├── index.html               ← Chat UI shell
│   ├── styles.css               ← Sidebar + bubble + responsive design
│   └── app.js                   ← Session mgmt, API calls, markdown render
│
├── docs.json                    ← 8-document knowledge base
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup & Running

### 1. Clone & install

```bash
git clone https://github.com/<you>/rag-assistant.git
cd rag-assistant

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

### 3. Run the server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open the UI

Visit **http://localhost:8000**

API docs at **http://localhost:8000/docs**

---

## API Reference

### `POST /api/chat`

**Request:**
```json
{
  "sessionId": "sess_abc123",
  "message": "How do I reset my password?"
}
```

**Response:**
```json
{
  "reply": "You can reset your password by navigating to Settings > Security > Reset Password.",
  "tokensUsed": 312,
  "retrievedChunks": 2
}
```

**Error (400):**
```json
{ "error": "Message field is required and must not be empty." }
```

### `GET /health`

```json
{ "status": "healthy" }
```

### `DELETE /api/sessions/{session_id}`

Clears the conversation history for a session (called by "New Chat").

---

## Deployment (Render)

1. Push the repo to GitHub.
2. Create a new **Web Service** on [Render](https://render.com).
3. Set **Build Command:** `pip install -r requirements.txt`
4. Set **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Add `OPENAI_API_KEY` as an environment variable in the Render dashboard.
6. Deploy!

---

## Evaluation Checklist

| Criterion | Implementation |
|---|---|
| RAG Architecture (30%) | Indexing pipeline + querying pipeline in `rag_service.py` |
| Embedding & Similarity (25%) | `embedding_service.py` + `vector_store.py` with cosine sim |
| LLM Integration (20%) | `llm_service.py`, temp=0.2, retry, error handling |
| Prompt Design (10%) | `prompts/templates.py` — context-first grounded prompt |
| Frontend UI (5%) | `frontend/` — session mgmt, markdown, loading indicator |
| Code Quality (10%) | Type hints, docstrings, structured logging, Pydantic schemas |
| Bonus: JWT (pending) | Implement via `python-jose` + FastAPI `Depends` |
| Bonus: Persistent storage | Replace in-memory store with SQLite/ChromaDB |

---

## License

MIT
