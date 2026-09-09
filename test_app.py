import unittest
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace

from fastapi import HTTPException

import app


class ChatEndpointTests(unittest.IsolatedAsyncioTestCase):
    def test_unprivileged_retrieval_rejects_uncategorized_rpc_rows(self):
        rpc_response = SimpleNamespace(execute=lambda: {
            "data": [{"pdf_name": "Private.pdf", "content": "Rahasia"}],
            "error": None,
        })
        table_rows = [
            {"pdf_name": "Public.pdf", "content": "Publik", "category": "public", "embedding": [1.0, 0.0]},
            {"pdf_name": "Private.pdf", "content": "Rahasia", "category": "private", "embedding": [1.0, 0.0]},
        ]
        with patch.object(app.supabase, "rpc", return_value=rpc_response) as rpc, \
             patch.object(app, "_fetch_document_rows", return_value=table_rows):
            documents = app.get_similar_documents([1.0, 0.0], "anonymous", 5)

        self.assertEqual([document["pdf_name"] for document in documents], ["Public.pdf"])
        rpc.assert_not_called()

    def test_clean_answer_preserves_markdown_structure(self):
        answer = "## Guru Bahasa Inggris\n\n1. **Santi Komalapuri**\n2. **Arum Nuraeni**"
        self.assertEqual(app._clean_answer(answer), answer)

    def test_clean_answer_strips_chain_of_thought_blocks(self):
        answer = "Here's a thinking process:\nLet's go through the sources...\n\nFinal Answer:\nSMP Negeri 2 Cibungbulang"
        self.assertEqual(app._clean_answer(answer), "SMP Negeri 2 Cibungbulang")

    def test_clean_answer_supports_indonesian_answer_marker(self):
        answer = "Proses berpikir yang panjang tidak berguna\n\nJawaban:\nNama sekolah adalah SMPN 2 Cibungbulang."
        self.assertEqual(app._clean_answer(answer), "Nama sekolah adalah SMPN 2 Cibungbulang.")

    def test_clean_answer_removes_reasoning_tags(self):
        answer = "<reasoning>Panjang</reasoning>Jawaban langsung."
        self.assertEqual(app._clean_answer(answer), "Jawaban langsung.")

    def test_no_information_answer_is_detected(self):
        self.assertTrue(app._answer_claims_no_information("Informasi tidak ditemukan pada dokumen yang tersedia."))
        self.assertFalse(app._answer_claims_no_information("Nama kepala sekolah adalah Rosihan Anwar."))

    def test_long_document_is_split_before_context_compression(self):
        text = ("Informasi umum sekolah. " * 80) + "Nama kepala sekolah adalah Rosihan Anwar."
        segments = app._sentences(text)

        self.assertGreater(len(segments), 1)
        self.assertIn("Rosihan Anwar", segments[-1])

    async def _chat_with_answer(self, answer, docs=None, role="user"):
        docs = docs or [{"pdf_name": "Laporan.pdf", "content": "Isi dokumen"}]
        with patch.object(app, "_authenticated_role", return_value=role), \
             patch.object(app, "embed_text", return_value=[0.1]), \
             patch.object(app, "get_similar_documents", return_value=docs), \
             patch.object(app, "compress_context", return_value=("Isi dokumen", {"original_tokens": 2, "compressed_tokens": 2, "compression_ratio": 1.0, "sentences_kept": 1})), \
             patch.object(app, "generate_with_fallback_with_usage", return_value=(answer, "groq", {})):
            return await app.chat(app.QueryRequest(question="Apa isi dokumen?"))

    async def test_normal_answer_and_source_are_returned(self):
        response = await self._chat_with_answer("Jawaban normal.")
        self.assertEqual(response["answer"], "Jawaban normal.")
        self.assertEqual(response["sources"], [{"pdf_name": "Laporan.pdf", "content": "Isi dokumen"}])

    async def test_accuracy_disclaimer_is_removed(self):
        disclaimer = "Tidak ada nilai akurasi yang tersedia dalam dokumen."
        response = await self._chat_with_answer(f"Jawaban. **{disclaimer}** Selesai.")
        self.assertEqual(response["answer"], "Jawaban. Selesai.")
        self.assertNotIn("akurasi", response["answer"].lower())

    async def test_source_without_accuracy_is_valid(self):
        docs = [{"pdf_name": "Laporan.pdf", "content": "Isi", "accuracy": None}]
        response = await self._chat_with_answer("Jawaban.", docs)
        self.assertEqual(response["sources"], [{"pdf_name": "Laporan.pdf", "content": "Isi"}])

    async def test_provider_error_returns_502(self):
        with patch.object(app, "_authenticated_role", return_value="user"), \
             patch.object(app, "embed_text", return_value=[0.1]), \
             patch.object(app, "get_similar_documents", return_value=[]), \
             patch.object(app, "compress_context", return_value=("", {"original_tokens": 0, "compressed_tokens": 0, "compression_ratio": 0.0, "sentences_kept": 0})), \
             patch.object(app, "generate_with_fallback_with_usage", side_effect=RuntimeError("provider failed")):
            with self.assertRaises(HTTPException) as error:
                await app.chat(app.QueryRequest(question="Apa isi dokumen?"))
        self.assertEqual(error.exception.status_code, 502)

    async def test_empty_question_returns_400(self):
        with patch.object(app, "_authenticated_role", return_value="user"):
            with self.assertRaises(HTTPException) as error:
                await app.chat(app.QueryRequest(question="   "))
        self.assertEqual(error.exception.status_code, 400)

    async def test_request_without_token_returns_public_chat(self):
        with patch.object(app, "embed_text", return_value=[0.1]), \
             patch.object(app, "get_similar_documents", return_value=[]), \
             patch.object(app, "generate_with_fallback_with_usage", return_value=("Jawaban publik.", "groq", {})):
            response = await app.chat(app.QueryRequest(question="Pertanyaan"))
        self.assertEqual(response["answer"], "Jawaban publik.")

    async def test_anonymous_chat_uses_public_documents_without_persisting_session(self):
        docs = [
            {"pdf_name": "Public.pdf", "content": "Publik", "category": "public"},
            {"pdf_name": "Private.pdf", "content": "Rahasia", "category": "private"},
        ]
        with patch.object(app, "embed_text", return_value=[0.1]), \
             patch.object(app, "get_similar_documents", return_value=docs) as search, \
             patch.object(app, "generate_with_fallback_with_usage", return_value=("Jawaban publik.", "groq", {})), \
             patch.object(app, "ensure_session") as ensure, \
             patch.object(app, "save_message") as save, \
             patch.object(app, "compress_context", side_effect=lambda question, selected, max_tokens: (self.assertEqual({doc["pdf_name"] for doc in selected}, {"Public.pdf"}) or ("Publik", {"original_tokens": 1, "compressed_tokens": 1, "compression_ratio": 1.0, "sentences_kept": 1}))):
            response = await app.chat(app.QueryRequest(question="Pertanyaan", session_id="anonymous-session"))

        self.assertEqual(response["sources"], [{"pdf_name": "Public.pdf", "content": "Publik", "category": "public"}])
        search.assert_called_once_with([0.1], "anonymous", app.RAG_TOP_K, "Pertanyaan")
        ensure.assert_not_called()
        save.assert_not_called()

    async def test_invalid_token_returns_401(self):
        with patch.object(app.supabase.auth, "get_user", side_effect=RuntimeError("invalid token")):
            with self.assertRaises(HTTPException) as error:
                await app.chat(app.QueryRequest(question="Pertanyaan"), "Bearer invalid")
        self.assertEqual(error.exception.status_code, 401)

    async def test_user_chat_uses_public_documents_only(self):
        docs = [
            {"pdf_name": "Public.pdf", "content": "Publik", "category": "public"},
            {"pdf_name": "Private.pdf", "content": "Rahasia", "category": "private"},
            {"pdf_name": "Legacy.pdf", "content": "Lama tanpa kategori"},
        ]
        with patch.object(app, "embed_text", return_value=[0.1]), \
             patch.object(app, "get_similar_documents", return_value=docs), \
             patch.object(app, "generate_with_fallback_with_usage", return_value=("Jawaban.", "groq", {})):
            with patch.object(app, "_authenticated_role", return_value="user"), \
                 patch.object(app, "compress_context", side_effect=lambda question, selected, max_tokens: (self.assertEqual({doc["pdf_name"] for doc in selected}, {"Public.pdf", "Legacy.pdf"}) or ("Publik", {"original_tokens": 1, "compressed_tokens": 1, "compression_ratio": 1.0, "sentences_kept": 2}))):
                response = await app.chat(app.QueryRequest(question="Pertanyaan"))
        self.assertEqual({source["pdf_name"] for source in response["sources"]}, {"Public.pdf", "Legacy.pdf"})
        self.assertNotIn("Private.pdf", response["sources"])

    async def test_teacher_chat_can_use_private_documents(self):
        docs = [
            {"pdf_name": "Public.pdf", "content": "Publik", "category": "public"},
            {"pdf_name": "Private.pdf", "content": "Rahasia", "category": "private"},
        ]
        with patch.object(app, "embed_text", return_value=[0.1]), \
             patch.object(app, "get_similar_documents", return_value=docs), \
             patch.object(app, "compress_context", side_effect=lambda question, selected, max_tokens: (self.assertEqual(len(selected), 2) or ("Semua", {"original_tokens": 1, "compressed_tokens": 1, "compression_ratio": 1.0, "sentences_kept": 2}))), \
             patch.object(app, "generate_with_fallback_with_usage", return_value=("Jawaban.", "groq", {})), \
             patch.object(app, "_authenticated_role", return_value="teacher"):
            response = await app.chat(app.QueryRequest(question="Pertanyaan"), "Bearer teacher-token")
        self.assertEqual(len(response["sources"]), 2)

    async def test_admin_chat_can_use_private_documents(self):
        docs = [
            {"pdf_name": "Public.pdf", "content": "Publik", "category": "public"},
            {"pdf_name": "Private.pdf", "content": "Rahasia", "category": "private"},
        ]
        with patch.object(app, "_authenticated_role", return_value="admin"), \
             patch.object(app, "embed_text", return_value=[0.1]), \
             patch.object(app, "get_similar_documents", return_value=docs), \
             patch.object(app, "compress_context", side_effect=lambda question, selected, max_tokens: (self.assertEqual(len(selected), 2) or ("Semua", {"original_tokens": 1, "compressed_tokens": 1, "compression_ratio": 1.0, "sentences_kept": 2}))), \
             patch.object(app, "generate_with_fallback_with_usage", return_value=("Jawaban.", "groq", {})):
            response = await app.chat(app.QueryRequest(question="Pertanyaan"), "Bearer admin-token")
        self.assertEqual({source["pdf_name"] for source in response["sources"]}, {"Public.pdf", "Private.pdf"})


class AuthenticationTests(unittest.TestCase):
    def test_profile_role_is_loaded_from_supabase(self):
        auth_response = SimpleNamespace(user=SimpleNamespace(id="user-1"))
        profile_response = {"data": [{"role": "teacher"}], "error": None}
        profile_table = FakeProfileTable(profile_response)
        with patch.object(app.supabase.auth, "get_user", return_value=auth_response), \
             patch.object(app.supabase, "table", return_value=profile_table):
            self.assertEqual(app._authenticated_role("Bearer valid"), "teacher")

    def test_missing_profile_returns_403(self):
        auth_response = SimpleNamespace(user=SimpleNamespace(id="user-1"))
        with patch.object(app.supabase.auth, "get_user", return_value=auth_response), \
             patch.object(app.supabase, "table", return_value=FakeProfileTable({"data": [], "error": None})):
            with self.assertRaises(HTTPException) as error:
                app._authenticated_role("Bearer valid")
        self.assertEqual(error.exception.status_code, 403)


class EmbeddingTests(unittest.TestCase):
    def test_embedding_retries_gateway_timeout(self):
        gateway_error = RuntimeError("Server error '504 Gateway Time-out'")
        with patch.object(app, "hf_client") as client, \
             patch.object(app, "time") as time_module, \
             patch.object(app, "EXPECTED_EMBEDDING_DIMENSION", None):
            client.feature_extraction.side_effect = [gateway_error, [0.1, 0.2]]

            embedding = app.embed_text("teks")

        self.assertEqual(embedding, [0.1, 0.2])
        self.assertEqual(client.feature_extraction.call_count, 2)
        time_module.sleep.assert_called_once_with(1)


class DocumentEndpointTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.rows = [
            {"pdf_name": "Laporan.pdf", "content": "Bagian 1", "created_at": "2026-01-02T00:00:00Z"},
            {"pdf_name": "Laporan.pdf", "content": "Bagian 2", "created_at": "2026-01-02T00:00:00Z"},
            {"pdf_name": "Panduan.pdf", "content": "Isi", "created_at": "2026-01-03T00:00:00Z"},
        ]

    def _table(self, name):
        self.assertEqual(name, app.SUPABASE_TABLE)
        return FakeDocumentTable(self.rows)

    async def test_list_documents_groups_chunks(self):
        with patch.object(app.supabase, "table", side_effect=self._table):
            response = await app.list_documents()

        self.assertEqual(response["data"][0]["id"], "Laporan.pdf")
        self.assertEqual(response["data"][0]["chunk_count"], 2)
        self.assertEqual(len(response["data"]), 2)

    async def test_rename_updates_all_chunks(self):
        table = FakeDocumentTable(self.rows)
        with patch.object(app.supabase, "table", return_value=table):
            response = await app.rename_document("Laporan.pdf", app.DocumentRenameRequest(name="Laporan Final.pdf"))

        self.assertEqual(response["data"]["name"], "Laporan Final.pdf")
        self.assertEqual(
            [row["pdf_name"] for row in self.rows],
            ["Laporan Final.pdf", "Laporan Final.pdf", "Panduan.pdf"],
        )

    async def test_delete_removes_all_chunks(self):
        table = FakeDocumentTable(self.rows)
        with patch.object(app.supabase, "table", return_value=table):
            response = await app.delete_document("Laporan.pdf")

        self.assertTrue(response["data"]["deleted"])
        self.assertEqual([row["pdf_name"] for row in self.rows], ["Panduan.pdf"])


class FakeDocumentTable:
    def __init__(self, rows):
        self.rows = rows
        self.operation = "select"
        self.payload = None
        self.filter_name = None

    def select(self, _columns):
        self.operation = "select"
        return self

    def limit(self, _count):
        return self

    def update(self, payload):
        self.operation = "update"
        self.payload = payload
        return self

    def delete(self):
        self.operation = "delete"
        return self

    def eq(self, name, value):
        self.filter_name = value
        return self

    def execute(self):
        if self.operation == "update":
            for row in self.rows:
                if row.get("pdf_name") == self.filter_name:
                    row.update(self.payload)
        elif self.operation == "delete":
            self.rows[:] = [row for row in self.rows if row.get("pdf_name") != self.filter_name]
        return {"data": self.rows, "error": None}


class FakeProfileTable:
    def __init__(self, response):
        self.response = response

    def select(self, _columns):
        return self

    def eq(self, _column, _value):
        return self

    def limit(self, _count):
        return self

    def execute(self):
        return self.response


class DashboardAdminTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.table_rows = {
            app.SUPABASE_TABLE: [
                {"pdf_name": "Profil.pdf", "content": "Bagian", "category": "public", "created_at": "2026-09-01T00:00:00Z"},
                {"pdf_name": "Profil.pdf", "content": "Bagian 2", "category": "public", "created_at": "2026-09-01T00:00:00Z"},
                {"pdf_name": "Rahasia.pdf", "content": "Isi", "category": "private", "created_at": "2026-09-02T00:00:00Z"},
            ],
            "profiles": [
                {"id": "u1", "email": "admin@aksaraku.id", "full_name": "Admin Satu", "role": "admin", "provider": "email", "created_at": "2026-08-01T00:00:00Z"},
                {"id": "u2", "email": "guru@aksaraku.id", "full_name": "Guru Dua", "role": "teacher", "provider": "email", "created_at": "2026-09-04T00:00:00Z"},
                {"id": "u3", "email": "murid@aksaraku.id", "full_name": "Murid Tiga", "role": "user", "provider": "google", "created_at": "2026-09-05T00:00:00Z"},
            ],
            app.SESSION_TABLE: [
                {"id": "s1", "title": "Sesi tanya", "created_at": "2026-09-05T01:00:00Z"},
            ],
            app.MESSAGE_TABLE: [
                {"session_id": "s1", "role": "user", "content": "Apa itu NPSN?", "inserted_at": "2026-09-05T01:01:00Z"},
                {"session_id": "s1", "role": "assistant", "content": "NPSN adalah...", "inserted_at": "2026-09-05T01:02:00Z"},
            ],
        }

    def _fake_table(self, name):
        return FakeDocumentTable(self.table_rows[name])

    async def test_non_admin_is_rejected(self):
        with patch.object(app, "_authenticated_user", return_value=("u1", "user")):
            with self.assertRaises(HTTPException) as error:
                await app.admin_dashboard()
        self.assertEqual(error.exception.status_code, 403)

    async def test_dashboard_returns_aggregated_stats(self):
        with patch.object(app, "_authenticated_user", return_value=("u1", "admin")), \
             patch.object(app.supabase, "table", side_effect=self._fake_table):
            response = await app.admin_dashboard()

        stats = response["stats"]
        self.assertEqual(stats["documents"], 2)
        self.assertEqual(stats["chunks"], 3)
        self.assertEqual(stats["users"], 3)
        self.assertEqual(stats["sessions"], 1)
        self.assertEqual(stats["messages"], 2)
        self.assertEqual(stats["public_documents"], 1)
        self.assertEqual(stats["private_documents"], 1)
        self.assertEqual(stats["users_by_role"], {"admin": 1, "teacher": 1, "user": 1, "other": 0})
        self.assertEqual(stats["messages_by_role"], {"user": 1, "assistant": 1})

    async def test_dashboard_returns_recent_users_sorted(self):
        with patch.object(app, "_authenticated_user", return_value=("u1", "admin")), \
             patch.object(app.supabase, "table", side_effect=self._fake_table):
            response = await app.admin_dashboard()

        self.assertEqual(response["recent_users"][0]["full_name"], "Murid Tiga")
        self.assertEqual(response["recent_users"][1]["full_name"], "Guru Dua")
        self.assertEqual(response["recent_users"][2]["full_name"], "Admin Satu")


if __name__ == "__main__":
    unittest.main()
