import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient
from pydantic import BaseModel
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HUGGINGFACE_EMBEDDING_MODEL = os.getenv(
    "HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
EXPECTED_EMBEDDING_DIMENSION = int(os.getenv("EXPECTED_EMBEDDING_DIMENSION", "384"))
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "pdf_documents")
VECTOR_FUNCTION = os.getenv("SUPABASE_VECTOR_FUNCTION", "match_pdf_documents")

app = FastAPI(title="Aksaraku Python API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
hf_client = InferenceClient(token=HUGGINGFACE_API_KEY) if HUGGINGFACE_API_KEY else None


def validate_embedding(embedding: Any) -> bool:
    return (
        isinstance(embedding, list)
        and len(embedding) == EXPECTED_EMBEDDING_DIMENSION
        and all(isinstance(value, (int, float)) for value in embedding)
    )


def embed_text(text: str) -> List[float]:
    if not hf_client:
        raise HTTPException(
            status_code=500,
            detail="No Hugging Face API key configured. Set HUGGINGFACE_API_KEY.",
        )

    try:
        result = hf_client.feature_extraction(text)
    except AttributeError:
        # newer `InferenceClient` may provide `embeddings(model=..., inputs=...)`
        result = hf_client.embeddings(model=HUGGINGFACE_EMBEDDING_MODEL, inputs=text)

    # Normalize possible response shapes into a single embedding list
    embedding = None
    if isinstance(result, dict):
        if "embedding" in result and isinstance(result["embedding"], list):
            embedding = result["embedding"]
        else:
            vals = [v for v in result.values() if isinstance(v, list)]
            embedding = vals[0] if vals else None
    elif isinstance(result, list):
        embedding = result[0] if result and isinstance(result[0], list) else result
    else:
        embedding = None

    if not isinstance(embedding, list):
        raise HTTPException(
            status_code=500,
            detail=f"Hugging Face response invalid: {result}",
        )
    if not validate_embedding(embedding):
        raise HTTPException(
            status_code=500,
            detail=f"Invalid embedding shape from Hugging Face: {result}",
        )

    return embedding


def normalize_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    content = str(chunk.get("text", ""))
    embedding = chunk.get("embedding")

    if embedding is not None and not validate_embedding(embedding):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid embedding vector. Expected length {EXPECTED_EMBEDDING_DIMENSION}.",
        )

    row: Dict[str, Any] = {
        "pdf_name": chunk.get("pdf_name") or None,
        "content": content,
    }

    if embedding is not None:
        row["embedding"] = embedding

    if "metadata" in chunk:
        row["metadata"] = chunk.get("metadata")

    return row


class ChunkModel(BaseModel):
    text: str
    pdf_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = None


class EmbedUpsertRequest(BaseModel):
    chunks: List[ChunkModel]


class QueryRequest(BaseModel):
    question: str


@app.post("/embed-upsert")
async def embed_upsert(body: EmbedUpsertRequest):
    if not body.chunks:
        raise HTTPException(status_code=400, detail="invalid chunks")

    rows: List[Dict[str, Any]] = []
    for chunk in body.chunks:
        row = normalize_chunk(chunk.dict())
        if "embedding" not in row:
            row["embedding"] = embed_text(row["content"])
        rows.append(row)

    response = supabase.table(SUPABASE_TABLE).insert(rows).select("*").execute()
    if response.error:
        raise HTTPException(status_code=500, detail=str(response.error))

    return {"data": response.data}


@app.post("/embed-upsert-raw")
async def embed_upsert_raw(body: EmbedUpsertRequest):
    if not body.chunks:
        raise HTTPException(status_code=400, detail="invalid chunks")

    rows: List[Dict[str, Any]] = []
    for chunk in body.chunks:
        row = normalize_chunk(chunk.dict())
        if "embedding" not in row:
            raise HTTPException(
                status_code=400,
                detail=f"Missing embedding for chunk with text: {row.get('content')[:50]}",
            )
        rows.append(row)

    response = supabase.table(SUPABASE_TABLE).insert(rows).select("*").execute()
    if response.error:
        raise HTTPException(status_code=500, detail=str(response.error))

    return {"data": response.data}


@app.post("/query-docs")
async def query_docs(body: QueryRequest):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    embedding = embed_text(question)
    response = supabase.rpc(
        VECTOR_FUNCTION,
        {
            "query_embedding": embedding,
            "match_threshold": 0.3,
            "match_count": 4,
        },
    ).execute()

    if response.error:
        raise HTTPException(status_code=500, detail=str(response.error))

    return {"data": response.data or []}
