import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient
import json
import math
import inspect
from openai import OpenAI
import logging
from numbers import Number
from typing import Sequence
import numpy as np
from pydantic import BaseModel
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HUGGINGFACE_EMBEDDING_MODEL = os.getenv(
    "HUGGINGFACE_EMBEDDING_MODEL", "BAAI/bge-m3"
)
EXPECTED_EMBEDDING_DIMENSION = os.getenv("EXPECTED_EMBEDDING_DIMENSION")
EXPECTED_EMBEDDING_DIMENSION = int(EXPECTED_EMBEDDING_DIMENSION) if EXPECTED_EMBEDDING_DIMENSION else None
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "pdf_documents")
VECTOR_FUNCTION = os.getenv("SUPABASE_VECTOR_FUNCTION", "match_pdf_documents")
SUPABASE_METADATA_COLUMN = os.getenv("SUPABASE_METADATA_COLUMN", "metadata")
SUPABASE_USE_METADATA = os.getenv("SUPABASE_USE_METADATA", "false").lower() in ("1", "true", "yes")
# LLM providers: Groq first, OpenRouter second.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "llama-3.1-8b-instant")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_CHAT_MODEL = os.getenv("OPENROUTER_CHAT_MODEL", "openrouter/free")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Aksaraku Python API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
hf_client = InferenceClient(token=HUGGINGFACE_API_KEY) if HUGGINGFACE_API_KEY else None

groq_client = (
    OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )
    if GROQ_API_KEY
    else None
)

openrouter_client = (
    OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:3000"),
            "X-Title": os.getenv("OPENROUTER_SITE_NAME", "Aksaraku"),
        },
    )
    if OPENROUTER_API_KEY
    else None
)


def _is_numeric(value: Any) -> bool:
    return isinstance(value, Number)


def _is_flat_numeric_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(_is_numeric(item) for item in value)


def _is_list_of_numeric_lists(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, list) and bool(item) and all(_is_numeric(v) for v in item) for item in value)
    )


def _infer_shape(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return str(value.shape)
    if isinstance(value, list):
        shape = []
        current = value
        while isinstance(current, list):
            shape.append(len(current))
            if not current:
                break
            current = current[0]
        return str(tuple(shape))
    return type(value).__name__


def _mean_pool_embedding(value: Sequence[Sequence[Number]]) -> List[float]:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Cannot mean-pool embedding with ndim={arr.ndim}")
    return arr.mean(axis=0).tolist()


def _normalize_embedding_result(result: Any) -> List[float]:
    raw_type = type(result).__name__
    if hasattr(result, "tolist"):
        try:
            result = result.tolist()
        except Exception:
            pass

    if isinstance(result, dict):
        if "embedding" in result:
            result = result["embedding"]
        elif "embeddings" in result:
            result = result["embeddings"]
        else:
            values = [v for v in result.values() if isinstance(v, list)]
            result = values[0] if values else result

    if _is_flat_numeric_list(result):
        return [float(x) for x in result]

    if _is_list_of_numeric_lists(result):
        if len(result) == 1:
            return [float(x) for x in result[0]]
        inner_lengths = {len(item) for item in result}
        if len(inner_lengths) == 1:
            return [float(x) for x in _mean_pool_embedding(result)]

    raise ValueError(
        f"Unable to normalize embedding result. raw_type={raw_type}, raw_shape={_infer_shape(result)}"
    )


def validate_embedding(embedding: Any) -> bool:
    if not isinstance(embedding, list) or not embedding:
        return False
    if not all(_is_numeric(value) for value in embedding):
        return False
    if EXPECTED_EMBEDDING_DIMENSION is not None and len(embedding) != EXPECTED_EMBEDDING_DIMENSION:
        return False
    return True


def embed_text(text: str) -> List[float]:
    if not hf_client:
        raise HTTPException(
            status_code=500,
            detail="No Hugging Face API key configured. Set HUGGINGFACE_API_KEY.",
        )

    result = None
    try:
        result = hf_client.feature_extraction(text, model=HUGGINGFACE_EMBEDDING_MODEL)
    except Exception as e:
        detail = (
            f"Hugging Face request failed. Error: {e}. "
            f"Ensure `HUGGINGFACE_EMBEDDING_MODEL` is an embedding-capable model, "
            f"e.g. 'sentence-transformers/all-MiniLM-L6-v2'."
        )
        raise HTTPException(status_code=500, detail=detail)

    logger.info(
        "Embedding model=%s expected_dim=%s raw_type=%s raw_shape=%s",
        HUGGINGFACE_EMBEDDING_MODEL,
        EXPECTED_EMBEDDING_DIMENSION or "unset",
        type(result).__name__,
        _infer_shape(result),
    )

    try:
        embedding = _normalize_embedding_result(result)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Hugging Face response invalid: {result}. "
                f"Error normalizing embedding: {exc}. "
                f"Ensure `HUGGINGFACE_EMBEDDING_MODEL` is an embedding-capable model."
            ),
        )

    logger.info(
        "Normalized embedding model=%s dim=%s sample=%s",
        HUGGINGFACE_EMBEDDING_MODEL,
        len(embedding),
        embedding[:5],
    )

    if not validate_embedding(embedding):
        if EXPECTED_EMBEDDING_DIMENSION is None:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Invalid embedding format from Hugging Face: {result}. "
                    f"Expected a flat list of numeric values, got shape {_infer_shape(embedding)}."
                ),
            )
        raise HTTPException(
            status_code=500,
            detail=(
                f"Invalid embedding shape from Hugging Face: {result}. "
                f"Expected length {EXPECTED_EMBEDDING_DIMENSION}, got {len(embedding)}."
            ),
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

    if SUPABASE_USE_METADATA and "metadata" in chunk:
        row[SUPABASE_METADATA_COLUMN] = chunk.get("metadata")

    return row


def _extract_response_parts(response: Any) -> Dict[str, Any]:
    """Safely extract `error` and `data` from various Supabase client response shapes."""
    # dict-like responses
    if isinstance(response, dict):
        return {"error": response.get("error"), "data": response.get("data")}

    error = None
    data = None
    try:
        error = getattr(response, "error", None)
    except Exception:
        error = None

    try:
        data = getattr(response, "data", None)
    except Exception:
        data = None

    # fallback keys sometimes used
    if error is None:
        try:
            error = getattr(response, "errors", None)
        except Exception:
            error = None

    if data is None:
        # some clients return `body` or `output`
        try:
            data = getattr(response, "body", None)
        except Exception:
            data = None
        if data is None:
            try:
                data = getattr(response, "output", None)
            except Exception:
                data = None

    return {"error": error, "data": data}


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0 or norm_b == 0:
        return -1.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _parse_embedding(raw: Any) -> Optional[List[float]]:
    if raw is None:
        return None
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return None
    return None


def _extract_openai_chat_text(response: Any) -> str:
    """Extract assistant text from an OpenAI-compatible chat response."""
    try:
        content = response.choices[0].message.content
        return content.strip() if isinstance(content, str) else ""
    except Exception:
        return ""


def _call_openai_compatible_chat(client: OpenAI, model: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Kamu adalah asisten RAG untuk Aksaraku. Jawab dalam Bahasa Indonesia. "
                    "Gunakan hanya informasi dari konteks dokumen yang diberikan. "
                    "Jika informasi tidak ada di konteks, katakan bahwa informasi tidak ditemukan. "
                    "Jangan mengarang sumber atau fakta."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    answer = _extract_openai_chat_text(response)
    if not answer:
        raise RuntimeError(f"{model} returned an empty response")
    return answer


def groq_generate(prompt: str) -> str:
    if not groq_client:
        raise RuntimeError("GROQ_API_KEY is not configured")
    return _call_openai_compatible_chat(groq_client, GROQ_CHAT_MODEL, prompt)


def openrouter_generate(prompt: str) -> str:
    if not openrouter_client:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")
    return _call_openai_compatible_chat(openrouter_client, OPENROUTER_CHAT_MODEL, prompt)


def generate_with_fallback(prompt: str) -> tuple[str, str]:
    """Try Groq first, then OpenRouter. Returns (answer, provider)."""
    errors: List[str] = []

    providers = [
        ("groq", groq_generate),
        ("openrouter", openrouter_generate),
    ]

    for name, generator in providers:
        try:
            answer = generator(prompt)
            logger.info("LLM response provider=%s model=%s", name,
                        GROQ_CHAT_MODEL if name == "groq" else OPENROUTER_CHAT_MODEL)
            return answer, name
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            logger.warning("LLM provider %s failed: %s", name, exc)

    raise RuntimeError("All LLM providers failed. " + " | ".join(errors))


def get_similar_documents(query_embedding: List[float], top_k: int = 4) -> List[Dict[str, Any]]:
    # Try RPC/vector function first (fast when available)
    try:
        response = supabase.rpc(
            VECTOR_FUNCTION,
            {"query_embedding": query_embedding, "match_threshold": 0.0, "match_count": top_k},
        ).execute()
        parts = _extract_response_parts(response)
        if not parts.get("error") and parts.get("data"):
            return parts.get("data")
    except Exception:
        # fallback to client-side similarity
        pass

    # Fallback: fetch rows and compute cosine similarity in Python
    response = supabase.table(SUPABASE_TABLE).select("*").limit(1000).execute()
    parts = _extract_response_parts(response)
    if parts.get("error"):
        raise HTTPException(status_code=500, detail=str(parts.get("error")))

    rows = parts.get("data") or []
    scored: List[Dict[str, Any]] = []
    for r in rows:
        emb = _parse_embedding(r.get("embedding") or r.get("embbbed") or r.get("embeddings"))
        if emb is None:
            continue
        score = _cosine_similarity(query_embedding, emb)
        scored.append({"row": r, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return [s["row"] for s in scored[:top_k]]


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
    parts = _extract_response_parts(response)
    if parts.get("error"):
        raise HTTPException(status_code=500, detail=str(parts.get("error")))

    return {"data": parts.get("data")}


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
    parts = _extract_response_parts(response)
    if parts.get("error"):
        raise HTTPException(status_code=500, detail=str(parts.get("error")))

    return {"data": parts.get("data")}


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
    parts = _extract_response_parts(response)
    if parts.get("error"):
        raise HTTPException(status_code=500, detail=str(parts.get("error")))

    return {"data": parts.get("data") or []}


@app.post("/chat")
async def chat(body: QueryRequest):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    # embed the question
    query_embedding = embed_text(question)

    # fetch similar documents (tries RPC first, falls back to client-side)
    docs = get_similar_documents(query_embedding, top_k=4)

    # Build context including akurasi field if present
    context_parts: List[str] = []
    for d in docs:
        pdf_name = d.get("pdf_name") or d.get("pdf_name")
        content = d.get("content") or d.get("text") or ""
        akurasi = d.get("akurasi") or d.get("accuracy") or d.get("akurasi_data") or None
        header = f"File: {pdf_name} | Akurasi: {akurasi}\n" if pdf_name else (f"Akurasi: {akurasi}\n" if akurasi is not None else "")
        context_parts.append(header + content[:2000])

    context_text = "\n\n---\n\n".join(context_parts) if context_parts else ""

    system_instruction = (
        "Kamu asisten yang menjawab dalam Bahasa Indonesia. "
        "Gunakan hanya informasi dari dokumen yang diberikan di bawah ini. "
        "Jika menjawab, cantumkan sumber dari nama file PDF dan sertakan nilai 'akurasi' bila tersedia. "
        "Jika informasi tidak ada di dokumen, jelaskan bahwa tidak ditemukan di dokumen."
    )

    prompt = f"{system_instruction}\n\nDOKUMEN:\n{context_text}\n\nPERTANYAAN: {question}\n\nJAWAB:"

    try:
        answer_text, provider = generate_with_fallback(prompt)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"All LLM providers failed: {e}")

    # Return answer with sources and provider used.
    return {"answer": answer_text, "provider": provider, "sources": docs}
