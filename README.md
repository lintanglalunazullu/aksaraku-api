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

## Environment variables

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `HUGGINGFACE_API_KEY`
- `HUGGINGFACE_EMBEDDING_MODEL`
- `EXPECTED_EMBEDDING_DIMENSION`
- `SUPABASE_TABLE`
- `SUPABASE_VECTOR_FUNCTION`
- `GROQ_API_KEY`
- `GROQ_CHAT_MODEL` (default: `llama-3.3-70b-versatile`)
- `OPENROUTER_API_KEY`
- `OPENROUTER_CHAT_MODEL` (default: `openrouter/free`)

## Notes

- The API uses `pdf_documents` by default, matching the Node.js reference schema.
- Keep your service role key and Hugging Face API key on the server only.
