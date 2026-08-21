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
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_CHAT_MODEL = os.getenv("OPENROUTER_CHAT_MODEL", "openrouter/free")
GROQ_MODEL_CANDIDATES = (
    GROQ_CHAT_MODEL,
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
)
OPENROUTER_MODEL_CANDIDATES = (
    OPENROUTER_CHAT_MODEL,
    "openrouter/free",
)

# TOKEN / CONTEXT OPTIMIZATION
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "3500"))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "2600"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "500"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
COMPRESS_CONTEXT = os.getenv("COMPRESS_CONTEXT", "true").lower() in ("1", "true", "yes")

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


def _find_available_model(client: OpenAI, candidates: tuple[str, ...]) -> str:
    """Return the first configured model that the provider currently exposes."""
    try:
        available = {model.id for model in client.models.list().data}
        for candidate in candidates:
            if candidate in available:
                return candidate
        raise RuntimeError("No configured model is available")
    except Exception as exc:
        logger.warning("Could not verify provider models: %s", exc)
        return candidates[0]


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
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            ]
            return "".join(parts).strip()
        return ""
    except Exception:
        return ""


def _public_sources(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return source metadata without exposing large embedding vectors."""
    return [
        {key: value for key, value in document.items() if key not in {"embedding", "embbbed", "embeddings"}}
        for document in documents
    ]


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
        max_tokens=MAX_OUTPUT_TOKENS,
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


# =========================
# CONTEXT COMPRESSION
# =========================

def estimate_tokens(text: str) -> int:
    """Fast token estimate for budgeting; provider usage remains authoritative."""
    return max(1, (len(text) + 3) // 4) if text else 0


def _words(text: str) -> set[str]:
    import re
    return set(re.findall(r"[a-zA-Z0-9À-ÿ]+", text.lower()))


def _sentences(text: str) -> List[str]:
    import re
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def compress_context(question: str, docs: List[Dict[str, Any]], max_tokens: int) -> tuple[str, Dict[str, Any]]:
    """Free deterministic compression: rank sentences by query overlap + vector score."""
    if not docs:
        return "", {"original_tokens": 0, "compressed_tokens": 0, "compression_ratio": 0.0, "sentences_kept": 0}

    qwords = _words(question)
    candidates=[]
    original=[]
    for di,d in enumerate(docs):
        pdf=d.get("pdf_name") or "Sumber tidak diketahui"
        content=str(d.get("content") or d.get("text") or "").strip()
        ak=d.get("akurasi")
        if ak is None: ak=d.get("accuracy")
        if ak is None: ak=d.get("akurasi_data")
        score=float(d.get("similarity") or d.get("score") or d.get("similarity_score") or 0.0)
        header=f"File: {pdf}" + (f" | Akurasi: {ak}" if ak is not None else "")
        original.append(header+"\n"+content)
        for si,s in enumerate(_sentences(content)):
            sw=_words(s)
            overlap=len(sw & qwords)/max(1,len(qwords))
            relevance=overlap*0.7 + max(0.0,min(score,1.0))*0.3 + (0.03 if si==0 else 0)
            candidates.append({"di":di,"si":si,"pdf":pdf,"ak":ak,"s":s,"score":relevance})

    original_text="\n\n---\n\n".join(original)
    original_tokens=estimate_tokens(original_text)
    if not COMPRESS_CONTEXT or original_tokens <= max_tokens:
        return original_text,{"original_tokens":original_tokens,"compressed_tokens":original_tokens,"compression_ratio":1.0,"sentences_kept":len(candidates),"compressed":False}

    by_doc={}
    for c in candidates: by_doc.setdefault(c["di"],[]).append(c)
    selected=[]
    selected_ids=set()
    for di in range(len(docs)):
        arr=sorted(by_doc.get(di,[]),key=lambda x:x["score"],reverse=True)
        if arr:
            selected.append(arr[0]); selected_ids.add(id(arr[0]))

    ranked=sorted(candidates,key=lambda x:x["score"],reverse=True)
    def render(items):
        grouped={}
        for c in items: grouped.setdefault(c["di"],[]).append(c)
        blocks=[]
        for di in sorted(grouped):
            arr=sorted(grouped[di],key=lambda x:x["si"])
            header=f"File: {arr[0]['pdf']}" + (f" | Akurasi: {arr[0]['ak']}" if arr[0]["ak"] is not None else "")
            blocks.append(header+"\n"+" ".join(x["s"] for x in arr))
        return "\n\n---\n\n".join(blocks)

    for c in ranked:
        if id(c) in selected_ids: continue
        trial=selected+[c]
        if estimate_tokens(render(trial)) <= max_tokens:
            selected.append(c); selected_ids.add(id(c))

    compressed=render(selected)
    max_chars=max_tokens*4
    if len(compressed)>max_chars:
        compressed=compressed[:max_chars].rsplit(" ",1)[0]+"…"
    ct=estimate_tokens(compressed)
    return compressed,{"original_tokens":original_tokens,"compressed_tokens":ct,"compression_ratio":round(ct/max(1,original_tokens),4),"sentences_kept":len(selected),"compressed":True}


def _usage(response: Any) -> Dict[str, Optional[int]]:
    u=getattr(response,"usage",None)
    if u is None: return {"prompt_tokens":None,"completion_tokens":None,"total_tokens":None}
    def g(n):
        try:
            v=getattr(u,n,None); return int(v) if v is not None else None
        except Exception: return None
    return {"prompt_tokens":g("prompt_tokens"),"completion_tokens":g("completion_tokens"),"total_tokens":g("total_tokens")}


def generate_with_fallback_with_usage(prompt: str) -> tuple[str,str,Dict[str,Optional[int]]]:
    errors=[]
    providers = [
        ("groq", groq_client, GROQ_MODEL_CANDIDATES),
        ("openrouter", openrouter_client, OPENROUTER_MODEL_CANDIDATES),
    ]
    for name, client, candidates in providers:
        if client is None:
            errors.append(f"{name}: API key/client not configured"); continue
        model = _find_available_model(client, candidates)
        try:
            response=client.chat.completions.create(
                model=model,
                messages=[
                    {"role":"system","content":"Kamu adalah asisten RAG untuk Aksaraku. Jawab dalam Bahasa Indonesia. Gunakan hanya informasi dari konteks dokumen. Jika informasi tidak ditemukan, katakan tidak ditemukan. Jangan mengarang fakta atau sumber."},
                    {"role":"user","content":prompt},
                ],
                temperature=0.2,
                max_tokens=MAX_OUTPUT_TOKENS,
            )
            answer=_extract_openai_chat_text(response)
            if not answer: raise RuntimeError(f"{model} returned an empty response")
            return answer,name,_usage(response)
        except Exception as exc:
            errors.append(f"{name}: {exc}"); logger.warning("LLM provider %s failed: %s",name,exc)
    raise RuntimeError("All LLM providers failed. " + " | ".join(errors))


@app.post("/chat")
async def chat(body: QueryRequest):
    question=body.question.strip()
    if not question: raise HTTPException(status_code=400,detail="question is required")
    query_embedding=embed_text(question)
    docs=get_similar_documents(query_embedding,top_k=RAG_TOP_K)
    context_text,compression=compress_context(question,docs,MAX_CONTEXT_TOKENS)
    system_instruction=(
        "Kamu asisten yang menjawab dalam Bahasa Indonesia. "
        "Gunakan hanya informasi dari dokumen yang diberikan. "
        "Cantumkan sumber dari nama file PDF dan nilai akurasi bila tersedia. "
        "Jika informasi tidak ada di dokumen, jelaskan bahwa tidak ditemukan di dokumen."
    )
    prompt=f"{system_instruction}\n\nDOKUMEN RELEVAN:\n{context_text}\n\nPERTANYAAN: {question}\n\nJAWAB:"
    estimated_input=estimate_tokens(prompt)
    try:
        answer,provider,usage=generate_with_fallback_with_usage(prompt)
    except Exception as e:
        raise HTTPException(status_code=502,detail=f"All LLM providers failed: {e}")
    return {
        "answer":answer,
        "provider":provider,
        "sources":_public_sources(docs),
        "token_optimization":{
            "enabled":COMPRESS_CONTEXT,
            "estimated_input_tokens":estimated_input,
            "context_original_tokens":compression["original_tokens"],
            "context_compressed_tokens":compression["compressed_tokens"],
            "compression_ratio":compression["compression_ratio"],
            "sentences_kept":compression["sentences_kept"],
            "max_context_tokens":MAX_CONTEXT_TOKENS,
            "max_output_tokens":MAX_OUTPUT_TOKENS,
            "provider_usage":usage,
        },
    }
