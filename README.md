# Aksaraku Python API

API server for embedding text and querying Supabase using Hugging Face embeddings.

## Setup

1. Change directory:

```bash
cd "C:/Users/Tang/OneDrive/ドキュメント/project/aksaraku-api"
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

3. Copy environment file and fill values:

```bash
copy .env.example .env
```

4. Start the API:

```bash
uvicorn app:app --host 0.0.0.0 --port 3000 --reload
```

## Endpoints

- `POST /embed-upsert`
  - Request body: `{ "chunks": [{ "text": "...", "pdf_name": "...", "metadata": {...}, "embedding": [...] }, ...] }`
  - If `embedding` is missing, the server generates it using Hugging Face.

- `POST /embed-upsert-raw`
  - Request body: `{ "chunks": [{ "text": "...", "pdf_name": "...", "metadata": {...}, "embedding": [...] }, ...] }`
  - Requires precomputed `embedding` values.

- `POST /query-docs`
  - Request body: `{ "question": "..." }`
  - Generates an embedding for the question and calls the Supabase RPC function configured by `SUPABASE_VECTOR_FUNCTION`.

- `GET /documents`
  - Returns documents grouped from `pdf_documents`, including `id`, `name`, `chunk_count`, `created_at`, and `status`.

- `PUT /documents/:id`
  - Request body: `{ "name": "Nama dokumen baru" }`
  - Renames every stored chunk belonging to the document.

- `DELETE /documents/:id`
  - Deletes every stored chunk belonging to the document.

The document `id` is the current `pdf_name`. Since this API file is `app.py`, start it with `uvicorn app:app`; `uvicorn main:app` will fail unless a separate `main.py` is added.

## Environment variables

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `HUGGINGFACE_API_KEY`
- `HUGGINGFACE_EMBEDDING_MODEL`
- `EXPECTED_EMBEDDING_DIMENSION`
- `HUGGINGFACE_EMBEDDING_RETRIES` (default: `3`, retries transient 502/503/504 and timeout errors)
- `SUPABASE_TABLE`
- `SUPABASE_VECTOR_FUNCTION`
- `GROQ_API_KEY`
- `GROQ_CHAT_MODEL` (default: `llama-3.3-70b-versatile`)
- `OPENROUTER_API_KEY`
- `OPENROUTER_CHAT_MODEL` (default: `openrouter/free`)

## Notes

- The API uses `pdf_documents` by default, matching the Node.js reference schema.
- The embedding model must match the vector dimension configured in Supabase. For example, `sentence-transformers/all-MiniLM-L6-v2` produces 384 dimensions, while `BAAI/bge-m3` produces 1024 dimensions.
- Keep your service role key and Hugging Face API key on the server only.
