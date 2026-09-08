# Aksaraku - FastAPI RAG backend
import os, json, math, logging, time, re
from numbers import Number
from typing import Any, Dict, List, Optional, Sequence, Literal

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient
from openai import OpenAI
from pydantic import BaseModel
from supabase import create_client

load_dotenv()

# ========================= CONFIG =========================
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "pdf_documents")
VECTOR_FUNCTION = os.getenv("SUPABASE_VECTOR_FUNCTION", "match_pdf_documents")
SUPABASE_METADATA_COLUMN = os.getenv("SUPABASE_METADATA_COLUMN", "metadata")
SUPABASE_USE_METADATA = os.getenv("SUPABASE_USE_METADATA", "false").lower() in {"1", "true", "yes"}

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HUGGINGFACE_EMBEDDING_MODEL = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EXPECTED_EMBEDDING_DIMENSION = int(os.getenv("EXPECTED_EMBEDDING_DIMENSION", os.getenv("HUGGINGFACE_EXPECTED_EMBEDDING_DIMENSION", "384")))
HUGGINGFACE_EMBEDDING_RETRIES = max(1, int(os.getenv("HUGGINGFACE_EMBEDDING_RETRIES", "3")))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_CHAT_MODEL = os.getenv("OPENROUTER_CHAT_MODEL", "openrouter/free")

MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "3500"))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "2000"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "600"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "8"))
RAG_FINAL_K = int(os.getenv("RAG_FINAL_K", "5"))
RAG_MIN_SIMILARITY = float(os.getenv("RAG_MIN_SIMILARITY", "0.30"))
COMPRESS_CONTEXT = os.getenv("COMPRESS_CONTEXT", "true").lower() in {"1", "true", "yes"}

SESSION_TABLE = os.getenv("SESSION_TABLE", "chat_sessions")
MESSAGE_TABLE = os.getenv("MESSAGE_TABLE", "chat_messages")
SESSION_HISTORY_LIMIT = int(os.getenv("SESSION_HISTORY_LIMIT", "6"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aksaraku")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")

app = FastAPI(title="Aksaraku Python API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
hf_client = InferenceClient(token=HUGGINGFACE_API_KEY) if HUGGINGFACE_API_KEY else None
groq_client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1") if GROQ_API_KEY else None
openrouter_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:3000"),
        "X-Title": os.getenv("OPENROUTER_SITE_NAME", "Aksaraku"),
    },
) if OPENROUTER_API_KEY else None


# ========================= MODELS =========================
class ChunkModel(BaseModel):
    text: str
    pdf_name: Optional[str] = None
    category: Literal["private", "public"] = "public"
    metadata: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = None

class EmbedUpsertRequest(BaseModel):
    chunks: List[ChunkModel]

class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class DocumentRenameRequest(BaseModel):
    name: Optional[str] = None
    pdf_name: Optional[str] = None
    category: Optional[Literal["private", "public"]] = None


# ========================= HELPERS =========================
def _extract_response_parts(response: Any) -> Dict[str, Any]:
    if isinstance(response, dict):
        return {"error": response.get("error"), "data": response.get("data")}
    error = getattr(response, "error", None)
    data = getattr(response, "data", None)
    if data is None:
        data = getattr(response, "body", None)
    if data is None:
        data = getattr(response, "output", None)
    return {"error": error, "data": data}


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4) if text else 0


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9À-ÿ]+", text.lower()))


def _sentences(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", text) if p.strip()]


def _is_numeric(value: Any) -> bool:
    return isinstance(value, Number)


def validate_embedding(embedding: Any) -> bool:
    return isinstance(embedding, list) and bool(embedding) and all(_is_numeric(v) for v in embedding) and len(embedding) == EXPECTED_EMBEDDING_DIMENSION


def _normalize_embedding_result(result: Any) -> List[float]:
    if hasattr(result, "tolist"):
        result = result.tolist()
    if isinstance(result, dict):
        result = result.get("embedding", result.get("embeddings", result))
    if isinstance(result, list) and result and all(_is_numeric(v) for v in result):
        return [float(v) for v in result]
    if isinstance(result, list) and result and all(isinstance(v, list) for v in result):
        lengths = {len(v) for v in result}
        if len(lengths) == 1:
            return np.asarray(result, dtype=float).mean(axis=0).tolist()
    raise ValueError(f"Unable to normalize embedding result; shape={_infer_shape(result)}")


def _infer_shape(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return str(value.shape)
    if isinstance(value, list):
        shape, cur = [], value
        while isinstance(cur, list):
            shape.append(len(cur))
            if not cur: break
            cur = cur[0]
        return str(tuple(shape))
    return type(value).__name__


def _is_retryable_hf_error(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    text = str(error).lower()
    return status in {502, 503, 504} or any(x in text for x in ("502", "503", "504", "timeout", "timed out"))


def embed_text(text: str) -> List[float]:
    if not hf_client:
        raise HTTPException(500, "No Hugging Face API key configured. Set HUGGINGFACE_API_KEY.")
    for attempt in range(HUGGINGFACE_EMBEDDING_RETRIES):
        try:
            result = hf_client.feature_extraction(text, model=HUGGINGFACE_EMBEDDING_MODEL)
            embedding = _normalize_embedding_result(result)
            if not validate_embedding(embedding):
                raise ValueError(f"Expected embedding dimension {EXPECTED_EMBEDDING_DIMENSION}, got {len(embedding)}")
            return embedding
        except Exception as exc:
            if attempt == HUGGINGFACE_EMBEDDING_RETRIES - 1 or not _is_retryable_hf_error(exc):
                raise HTTPException(500, f"Hugging Face embedding failed: {exc}")
            delay = 2 ** attempt
            logger.warning("HF retry %s/%s in %ss: %s", attempt + 1, HUGGINGFACE_EMBEDDING_RETRIES, delay, exc)
            time.sleep(delay)
    raise HTTPException(500, "Embedding failed")


def _parse_embedding(raw: Any) -> Optional[List[float]]:
    if isinstance(raw, list): return raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
            return value if isinstance(value, list) else None
        except Exception:
            return None
    return None


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b): return -1.0
    aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(aa), np.linalg.norm(bb)
    return -1.0 if na == 0 or nb == 0 else float(np.dot(aa, bb) / (na * nb))


# ========================= AUTH =========================
def _extract_profile_role(response: Any) -> Optional[str]:
    parts = _extract_response_parts(response)
    if parts["error"]: return None
    data = parts["data"]
    if isinstance(data, list): data = data[0] if data else None
    if not isinstance(data, dict): return None
    return str(data.get("role")).strip().lower() if data.get("role") is not None else None


def _authenticated_user(authorization: Optional[str]) -> tuple[str, str]:
    if not isinstance(authorization, str) or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Bearer token is required")
    token = authorization[7:].strip()
    if not token: raise HTTPException(401, "Bearer token is required")
    try:
        auth = supabase.auth.get_user(token)
        user = getattr(auth, "user", None) or (auth.get("user") if isinstance(auth, dict) else None)
        user_id = getattr(user, "id", None) or (user.get("id") if isinstance(user, dict) else None)
        if not user_id: raise ValueError("user missing")
    except Exception as exc:
        logger.warning("Token validation failed: %s", exc)
        raise HTTPException(401, "Invalid or expired token")
    try:
        profile = supabase.table("profiles").select("role").eq("id", user_id).limit(1).execute()
    except Exception as exc:
        logger.warning("Profile lookup failed: %s", exc)
        raise HTTPException(403, "User profile is not available")
    role = _extract_profile_role(profile)
    if role not in {"user", "teacher", "admin"}: raise HTTPException(403, "User role is not valid")
    return str(user_id), role


def _authenticated_role(authorization: Optional[str]) -> str:
    return _authenticated_user(authorization)[1]


def _chat_role(authorization: Optional[str]) -> str:
    """Return a public role for anonymous chat requests; validate supplied tokens."""
    if not isinstance(authorization, str) or not authorization.strip():
        return "anonymous"
    return _authenticated_role(authorization)


def allowed_categories(role: str) -> List[str]:
    return ["public", "private"] if role in {"teacher", "admin"} else ["public"]


def filter_documents_for_role(documents: List[Dict[str, Any]], role: str) -> List[Dict[str, Any]]:
    categories = set(allowed_categories(role))
    return [
        document for document in documents
        if str(document.get("category") or "public").strip().lower() in categories
    ]


# ========================= SESSION =========================
def ensure_session(session_id: Optional[str]) -> None:
    if not session_id: return
    try:
        found = supabase.table(SESSION_TABLE).select("id").eq("id", session_id).limit(1).execute()
        if not (_extract_response_parts(found)["data"] or []):
            supabase.table(SESSION_TABLE).insert({"id": session_id}).execute()
    except Exception as exc:
        logger.warning("Session ensure failed: %s", exc)


def get_session_history(session_id: Optional[str]) -> List[Dict[str, str]]:
    if not session_id: return []
    try:
        response = supabase.table(MESSAGE_TABLE).select("role,content").eq("session_id", session_id).order("created_at", desc=True).limit(SESSION_HISTORY_LIMIT).execute()
        rows = _extract_response_parts(response)["data"] or []
        rows.reverse()
        return [{"role": r["role"], "content": r["content"]} for r in rows if r.get("role") in {"user", "assistant"} and r.get("content")]
    except Exception as exc:
        logger.warning("History load failed: %s", exc)
        return []


def save_message(session_id: Optional[str], role: str, content: str) -> None:
    if not session_id: return
    try:
        supabase.table(MESSAGE_TABLE).insert({"session_id": session_id, "role": role, "content": content}).execute()
    except Exception as exc:
        logger.warning("Message save failed: %s", exc)


def build_search_query(question: str, history: List[Dict[str, str]]) -> str:
    if not history: return question
    markers = ("yang tadi", "yang sebelumnya", "yang itu", "bagaimana dengan", "kalau yang", "kalau untuk", "berapa yang", "dan yang", "terus", "lalu", "sedangkan", "dibandingkan", "tahun sebelumnya")
    if not any(m in question.lower() for m in markers): return question
    previous = [x["content"] for x in history if x["role"] == "user"][-2:]
    return " ".join(previous + [question]) if previous else question


# ========================= RETRIEVAL =========================
def get_similar_documents(query_embedding: List[float], role: str, top_k: int = RAG_TOP_K) -> List[Dict[str, Any]]:
    categories = allowed_categories(role)
    # Preferred RPC: add category_filter to SQL function for DB-level access control.
    try:
        response = supabase.rpc(VECTOR_FUNCTION, {"query_embedding": query_embedding, "match_threshold": RAG_MIN_SIMILARITY, "match_count": top_k, "category_filter": categories}).execute()
        parts = _extract_response_parts(response)
        if not parts["error"] and parts["data"]:
            return parts["data"]
    except Exception as exc:
        logger.warning("Filtered vector RPC failed; trying compatibility RPC: %s", exc)
    # Compatibility path: still use vector RPC, then filter returned candidates.
    try:
        response = supabase.rpc(VECTOR_FUNCTION, {"query_embedding": query_embedding, "match_threshold": RAG_MIN_SIMILARITY, "match_count": max(top_k * 3, 15)}).execute()
        parts = _extract_response_parts(response)
        if not parts["error"]:
            return [r for r in (parts["data"] or []) if str(r.get("category") or "public").lower() in categories][:top_k]
    except Exception as exc:
        logger.warning("Vector RPC failed: %s", exc)
    return []


def rerank_documents(question: str, docs: List[Dict[str, Any]], final_k: int = RAG_FINAL_K) -> List[Dict[str, Any]]:
    qwords = _words(question)
    scored = []
    for i, doc in enumerate(docs):
        content = str(doc.get("content") or doc.get("text") or "")
        vector = float(doc.get("similarity") or doc.get("score") or doc.get("similarity_score") or 0)
        overlap = len(qwords & _words(content)) / max(1, len(qwords))
        score = vector * 0.75 + overlap * 0.25
        scored.append((score, i, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[2] for x in scored[:final_k]]


# ========================= CONTEXT =========================
def compress_context(question: str, docs: List[Dict[str, Any]], max_tokens: int) -> tuple[str, Dict[str, Any]]:
    if not docs:
        return "", {"original_tokens": 0, "compressed_tokens": 0, "compression_ratio": 0.0, "sentences_kept": 0, "compressed": False}
    qwords = _words(question); candidates=[]; original=[]
    for di, doc in enumerate(docs):
        pdf = str(doc.get("pdf_name") or "Sumber tidak diketahui")
        content = str(doc.get("content") or doc.get("text") or "").strip()
        score = float(doc.get("similarity") or doc.get("score") or doc.get("similarity_score") or 0)
        original.append(f"[SOURCE {di+1}]\nFile: {pdf}\n{content}")
        for si, sentence in enumerate(_sentences(content)):
            overlap = len(_words(sentence) & qwords) / max(1, len(qwords))
            candidates.append({"di":di,"si":si,"pdf":pdf,"s":sentence,"score":overlap*0.7 + max(0,min(score,1))*0.3 + (0.03 if si==0 else 0)})
    original_text="\n\n---\n\n".join(original); original_tokens=estimate_tokens(original_text)
    if not COMPRESS_CONTEXT or original_tokens <= max_tokens:
        return original_text,{"original_tokens":original_tokens,"compressed_tokens":original_tokens,"compression_ratio":1.0,"sentences_kept":len(candidates),"compressed":False}
    grouped={}
    for c in candidates: grouped.setdefault(c["di"],[]).append(c)
    selected=[]; selected_keys=set()
    for di in range(len(docs)):
        arr=sorted(grouped.get(di,[]),key=lambda x:x["score"],reverse=True)
        if arr: selected.append(arr[0]); selected_keys.add((arr[0]["di"],arr[0]["si"]))
    ranked=sorted(candidates,key=lambda x:x["score"],reverse=True)
    def render(items):
        groups={}
        for c in items: groups.setdefault(c["di"],[]).append(c)
        blocks=[]
        for di in sorted(groups):
            arr=sorted(groups[di],key=lambda x:x["si"])
            blocks.append(f"[SOURCE {di+1}]\nFile: {arr[0]['pdf']}\n" + " ".join(x["s"] for x in arr))
        return "\n\n---\n\n".join(blocks)
    for c in ranked:
        key=(c["di"],c["si"])
        if key in selected_keys: continue
        trial=selected+[c]
        if estimate_tokens(render(trial)) <= max_tokens:
            selected.append(c); selected_keys.add(key)
    compressed=render(selected)
    if estimate_tokens(compressed) > max_tokens:
        compressed=compressed[:max_tokens*4].rsplit(" ",1)[0] + "…"
    ct=estimate_tokens(compressed)
    return compressed,{"original_tokens":original_tokens,"compressed_tokens":ct,"compression_ratio":round(ct/max(1,original_tokens),4),"sentences_kept":len(selected),"compressed":True}


# ========================= LLM =========================
SYSTEM_PROMPT = """Kamu adalah Aksaraku, asisten AI berbasis RAG untuk informasi sekolah.
Jawab dalam Bahasa Indonesia. Gunakan hanya informasi dari SOURCE yang diberikan.
Jangan mengarang fakta, angka, nama, tanggal, atau sumber.
Jika informasi tidak ditemukan, katakan bahwa informasi tidak ditemukan pada dokumen yang tersedia.
Jangan menampilkan similarity score, confidence score, atau nilai akurasi internal.
Jika SOURCE bertentangan, jelaskan bahwa terdapat perbedaan data dan jangan memilih secara sembarangan.
Mulai jawaban langsung dengan isi jawaban. Jawab secara langsung, ringkas, dan tidak bertele-tele.
Jangan pernah menampilkan proses berpikir, langkah pemecahan masalah, analisis internal, ringkasan instruksi,
atau komentar tentang format jawaban. Larang keras menulis kata-kata seperti "Analisis Masukan", "Berikut proses
berpikir", "Langkah untuk menyelesaikan", dan sejenisnya.
Gunakan format yang mudah dibaca:
- Gunakan heading Markdown seperlunya untuk memisahkan topik.
- Gunakan daftar bernomor untuk urutan atau daftar nama.
- Gunakan bullet list untuk rincian singkat.
- Gunakan tabel Markdown hanya jika SOURCE memiliki data berbentuk kolom dan headernya jelas.
- Jangan membuat tabel jika data kolomnya tidak lengkap atau ambigu.
- Untuk daftar nama, tampilkan satu orang per baris dan sertakan jabatan atau keterangan yang tersedia.
Jika terdapat beberapa SOURCE dengan data berbeda, pisahkan berdasarkan SOURCE atau jelaskan perbedaannya.
Jangan mengulang pertanyaan pengguna dan jangan menambahkan pembuka seperti "Berikut analisis saya".
"""


def _extract_openai_chat_text(response: Any) -> str:
    try:
        content=response.choices[0].message.content
        if isinstance(content,str): return content.strip()
        if isinstance(content,list): return "".join(item.get("text","") for item in content if isinstance(item,dict) and isinstance(item.get("text"),str)).strip()
    except Exception: pass
    return ""


def _usage(response: Any) -> Dict[str, Optional[int]]:
    u=getattr(response,"usage",None)
    if u is None: return {"prompt_tokens":None,"completion_tokens":None,"total_tokens":None}
    def g(n):
        try:
            v=getattr(u,n,None); return int(v) if v is not None else None
        except Exception: return None
    return {"prompt_tokens":g("prompt_tokens"),"completion_tokens":g("completion_tokens"),"total_tokens":g("total_tokens")}


def generate_with_fallback_with_usage(question: str, context: str, history: List[Dict[str,str]], force_direct: bool = False) -> tuple[str,str,Dict[str,Optional[int]]]:
    messages=[{"role":"system","content":SYSTEM_PROMPT.strip()}]
    messages.extend(history[-SESSION_HISTORY_LIMIT:])
    user_content=f"SOURCE DOKUMEN:\n\n{context or '[Tidak ada SOURCE yang relevan]'}\n\nPERTANYAAN:\n{question}\n\nJAWAB:"
    if force_direct:
        user_content += "\n\nPENTING: Jawab langsung dan sangat ringkas tanpa kalimat pengantar, tanpa proses berpikir, tanpa analisis."
    messages.append({"role":"user","content":user_content})
    providers=[("groq",groq_client,GROQ_CHAT_MODEL),("openrouter",openrouter_client,OPENROUTER_CHAT_MODEL)]
    errors=[]
    for name,client,model in providers:
        if client is None:
            errors.append(f"{name}: not configured"); continue
        try:
            response=client.chat.completions.create(model=model,messages=messages,temperature=0.2,max_tokens=MAX_OUTPUT_TOKENS)
            answer=_extract_openai_chat_text(response)
            if not answer: raise RuntimeError(f"{model} returned empty response")
            return answer,name,_usage(response)
        except Exception as exc:
            errors.append(f"{name}: {exc}"); logger.warning("LLM provider %s failed: %s",name,exc)
    raise RuntimeError("All LLM providers failed. " + " | ".join(errors))


def _strip_chain_of_thought(answer: str) -> str:
    if not answer:
        return answer
    result = answer
    for tag in ("reasoning", "chain_of_thought", "cot", "thinking", "scratchpad", "thought"):
        result = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", " ", result, flags=re.S | re.I)
    lowered = result.lower()
    for marker in ("jawaban akhir:", "jawaban:", "final answer:", "kesimpulan:", "output final:", "hasil jadinya:", "hasil:"):
        offset = lowered.rfind(marker)
        if offset >= 0:
            return result[offset + len(marker):].strip()
    return result


def _clean_answer(answer: str) -> str:
    answer = _strip_chain_of_thought(answer)
    patterns=[r"\s*\*{0,2}Tidak ada nilai akurasi yang tersedia dalam dokumen\.?\*{0,2}\s*", r"\s*nilai akurasi tidak tersedia\.?\s*"]
    for p in patterns: answer=re.sub(p," ",answer,flags=re.I)
    answer = re.sub(r"[ \t]{2,}", " ", answer)
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    return answer.strip()


def calculate_evidence_confidence(question: str, docs: List[Dict[str,Any]], answer: str) -> Dict[str,Any]:
    if not docs: return {"score":0.0,"level":"low","supported":False}
    awords=_words(answer); scores=[]
    for doc in docs:
        content=str(doc.get("content") or doc.get("text") or "")
        lexical=len(awords & _words(content))/max(1,len(awords))
        vector=float(doc.get("similarity") or doc.get("score") or doc.get("similarity_score") or 0)
        scores.append(vector*0.65 + lexical*0.35)
    score=max(0,min(max(scores),1))
    level="high" if score>=0.78 else "medium" if score>=0.55 else "low"
    return {"score":round(score,3),"level":level,"supported":score>=0.55}


def _public_sources(documents: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    hidden={"embedding","embbbed","embeddings","similarity","score","similarity_score","confidence","confidence_score","akurasi","accuracy"}
    return [{k:v for k,v in d.items() if k not in hidden} for d in documents]


# ========================= DOCUMENTS =========================
def _fetch_document_rows() -> List[Dict[str,Any]]:
    response=supabase.table(SUPABASE_TABLE).select("*").limit(10000).execute()
    parts=_extract_response_parts(response)
    if parts["error"]: raise HTTPException(500,str(parts["error"]))
    return parts["data"] or []


def _document_summary(rows: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    docs={}
    for row in rows:
        name=str(row.get("pdf_name") or "").strip()
        if not name: continue
        d=docs.setdefault(name,{"id":name,"name":name,"pdf_name":name,"chunk_count":0,"created_at":row.get("created_at"),"status":row.get("status") or "completed","category":row.get("category") or "public"})
        d["chunk_count"]+=1
        if row.get("created_at") and (not d.get("created_at") or str(row["created_at"])<str(d["created_at"])): d["created_at"]=row["created_at"]
        if row.get("status"): d["status"]=row["status"]
        if row.get("category"): d["category"]=row["category"]
    return list(docs.values())


def _fetch_table_rows(table_name: str, limit: int = 1000) -> List[Dict[str, Any]]:
    response = supabase.table(table_name).select("*").limit(limit).execute()
    parts = _extract_response_parts(response)
    if parts["error"]:
        raise HTTPException(500, str(parts["error"]))
    return parts["data"] or []


def _dashboard_activity(rows: List[Dict[str, Any]], label: str, name_key: str) -> List[Dict[str, Any]]:
    activities = []
    for row in rows:
        name = row.get(name_key) or row.get("title") or row.get("content") or "Aktivitas baru"
        timestamp = row.get("updated_at") or row.get("created_at") or row.get("inserted_at")
        activities.append({"label": label, "name": str(name)[:120], "timestamp": timestamp})
    return activities


# ========================= ROUTES =========================
@app.get("/health")
async def health():
    return {"status":"ok","service":"Aksaraku","embedding_model":HUGGINGFACE_EMBEDDING_MODEL,"embedding_dimension":EXPECTED_EMBEDDING_DIMENSION,"rag_top_k":RAG_TOP_K,"rag_final_k":RAG_FINAL_K,"context_tokens":MAX_CONTEXT_TOKENS}


@app.get("/admin/dashboard")
async def admin_dashboard(authorization: Optional[str] = Header(default=None)):
    _, role = _authenticated_user(authorization)
    if role != "admin":
        raise HTTPException(403, "Admin role is required")

    document_rows = _fetch_table_rows(SUPABASE_TABLE, 10000)
    profiles = _fetch_table_rows("profiles", 1000)
    sessions = _fetch_table_rows(SESSION_TABLE, 1000)
    messages = _fetch_table_rows(MESSAGE_TABLE, 1000)
    documents = _document_summary(document_rows)

    category_counts = {"public": 0, "private": 0}
    for document in documents:
        category = str(document.get("category") or "public").lower()
        category_counts[category if category in category_counts else "public"] += 1

    role_counts = {"admin": 0, "teacher": 0, "user": 0, "other": 0}
    for profile in profiles:
        profile_role = str(profile.get("role") or "user").strip().lower()
        role_counts[profile_role if profile_role in role_counts else "other"] += 1

    recent_users = []
    for profile in profiles:
        recent_users.append({
            "id": profile.get("id"),
            "email": profile.get("email") or "",
            "full_name": profile.get("full_name") or profile.get("email") or "Pengguna",
            "role": str(profile.get("role") or "user").lower(),
            "provider": profile.get("provider") or "email",
            "created_at": profile.get("created_at"),
        })
    recent_users.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)

    messages_role_counts = {"user": 0, "assistant": 0}
    for message in messages:
        message_role = str(message.get("role") or "").lower()
        if message_role == "user":
            messages_role_counts["user"] += 1
        elif message_role == "assistant":
            messages_role_counts["assistant"] += 1

    activities = (
        _dashboard_activity(documents, "Dokumen", "pdf_name")
        + _dashboard_activity(sessions, "Sesi chat", "title")
        + _dashboard_activity(messages, "Pesan AI", "content")
        + _dashboard_activity(recent_users, "Pengguna", "full_name")
    )
    activities.sort(key=lambda item: str(item.get("timestamp") or ""), reverse=True)

    return {
        "stats": {
            "documents": len(documents),
            "chunks": len(document_rows),
            "users": len(profiles),
            "sessions": len(sessions),
            "messages": len(messages),
            "public_documents": category_counts["public"],
            "private_documents": category_counts["private"],
            "users_by_role": role_counts,
            "messages_by_role": messages_role_counts,
        },
        "documents": documents[:8],
        "activity": activities[:10],
        "recent_users": recent_users[:8],
    }

@app.get("/documents")
async def list_documents():
    return {"data":_document_summary(_fetch_document_rows())}

@app.put("/documents/{document_id}")
async def rename_document(document_id:str,body:DocumentRenameRequest):
    new_name=(body.name or body.pdf_name or document_id).strip()
    if not new_name: raise HTTPException(400,"name is required")
    rows=_fetch_document_rows(); names={str(r.get("pdf_name") or "").strip() for r in rows}
    if document_id not in names: raise HTTPException(404,"document not found")
    if new_name!=document_id and new_name in names: raise HTTPException(409,"document name already exists")
    payload={"pdf_name":new_name}
    if body.category is not None: payload["category"]=body.category
    response=supabase.table(SUPABASE_TABLE).update(payload).eq("pdf_name",document_id).execute()
    parts=_extract_response_parts(response)
    if parts["error"]: raise HTTPException(500,str(parts["error"]))
    return {"data":{"id":new_name,"name":new_name,"pdf_name":new_name}}

@app.delete("/documents/{document_id}")
async def delete_document(document_id:str):
    names={str(r.get("pdf_name") or "").strip() for r in _fetch_document_rows()}
    if document_id not in names: raise HTTPException(404,"document not found")
    response=supabase.table(SUPABASE_TABLE).delete().eq("pdf_name",document_id).execute()
    parts=_extract_response_parts(response)
    if parts["error"]: raise HTTPException(500,str(parts["error"]))
    return {"data":{"id":document_id,"deleted":True}}

@app.post("/embed-upsert")
async def embed_upsert(body:EmbedUpsertRequest):
    if not body.chunks: raise HTTPException(400,"invalid chunks")
    rows=[]
    for chunk in body.chunks:
        content=chunk.text.strip()
        if not content: raise HTTPException(400,"content tidak boleh kosong")
        row={"content":content,"pdf_name":chunk.pdf_name or None,"category":chunk.category}
        if chunk.metadata is not None and SUPABASE_USE_METADATA: row[SUPABASE_METADATA_COLUMN]=chunk.metadata
        row["embedding"]=chunk.embedding if chunk.embedding is not None else embed_text(content)
        if not validate_embedding(row["embedding"]): raise HTTPException(400,f"Invalid embedding vector; expected {EXPECTED_EMBEDDING_DIMENSION} dimensions")
        rows.append(row)
    response=supabase.table(SUPABASE_TABLE).insert(rows).select("*").execute()
    parts=_extract_response_parts(response)
    if parts["error"]: raise HTTPException(500,str(parts["error"]))
    return {"success":True,"data":parts["data"] or []}

@app.post("/embed-upsert-raw")
async def embed_upsert_raw(body:EmbedUpsertRequest):
    if not body.chunks: raise HTTPException(400,"invalid chunks")
    rows=[]
    for chunk in body.chunks:
        if chunk.embedding is None: raise HTTPException(400,"Missing embedding for chunk")
        if not validate_embedding(chunk.embedding): raise HTTPException(400,f"Invalid embedding; expected {EXPECTED_EMBEDDING_DIMENSION} dimensions")
        row={"pdf_name":chunk.pdf_name or None,"content":chunk.text.strip(),"category":chunk.category,"embedding":chunk.embedding}
        if SUPABASE_USE_METADATA and chunk.metadata is not None: row[SUPABASE_METADATA_COLUMN]=chunk.metadata
        rows.append(row)
    response=supabase.table(SUPABASE_TABLE).insert(rows).select("*").execute()
    parts=_extract_response_parts(response)
    if parts["error"]: raise HTTPException(500,str(parts["error"]))
    return {"data":parts["data"] or []}

@app.post("/query-docs")
async def query_docs(body:QueryRequest,authorization:Optional[str]=Header(default=None)):
    role=_authenticated_role(authorization); question=body.question.strip()
    if not question: raise HTTPException(400,"question is required")
    retrieved = get_similar_documents(embed_text(question), role, RAG_TOP_K)
    docs=rerank_documents(question,filter_documents_for_role(retrieved, role),RAG_FINAL_K)
    return {"data":_public_sources(docs)}

@app.post("/chat")
async def chat(body:QueryRequest,authorization:Optional[str]=Header(default=None)):
    role=_chat_role(authorization)
    question=body.question.strip()
    if not question: raise HTTPException(400,"question is required")
    persist_session = role != "anonymous"
    if persist_session:
        ensure_session(body.session_id)
    history=get_session_history(body.session_id) if persist_session else []
    search_query=build_search_query(question,history)
    if persist_session:
        save_message(body.session_id,"user",question)

    retrieved = get_similar_documents(embed_text(search_query), role, RAG_TOP_K)
    docs=rerank_documents(search_query,filter_documents_for_role(retrieved, role),RAG_FINAL_K)
    context,compression=compress_context(search_query,docs,MAX_CONTEXT_TOKENS)

    try:
        answer,provider,usage=generate_with_fallback_with_usage(question,context,history)
        answer=_clean_answer(answer)
        if not answer:
            logger.warning("LLM produced only chain-of-thought; retrying with direct-answer instruction")
            answer,provider,usage=generate_with_fallback_with_usage(question,context,history,force_direct=True)
            answer=_clean_answer(answer)
    except Exception as exc:
        raise HTTPException(502,f"All LLM providers failed: {exc}")

    if persist_session:
        save_message(body.session_id,"assistant",answer)
    prompt_estimate=estimate_tokens(SYSTEM_PROMPT+"\n"+context+"\n"+question)
    evidence=calculate_evidence_confidence(question,docs,answer)

    return {
        "answer":answer,
        "session_id":body.session_id,
        "provider":provider,
        "sources":_public_sources(docs),
        "evidence":evidence,
        "retrieval":{"retrieved_count":len(docs),"final_count":len(docs),"min_similarity":RAG_MIN_SIMILARITY},
        "token_optimization":{
            "enabled":COMPRESS_CONTEXT,
            "estimated_input_tokens":prompt_estimate,
            "context_original_tokens":compression["original_tokens"],
            "context_compressed_tokens":compression["compressed_tokens"],
            "compression_ratio":compression["compression_ratio"],
            "sentences_kept":compression["sentences_kept"],
            "max_context_tokens":MAX_CONTEXT_TOKENS,
            "max_input_tokens":MAX_INPUT_TOKENS,
            "max_output_tokens":MAX_OUTPUT_TOKENS,
            "provider_usage":usage,
        },
    }
