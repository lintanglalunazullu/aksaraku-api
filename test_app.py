import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

import app


class ChatEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def _chat_with_answer(self, answer, docs=None):
        docs = docs or [{"pdf_name": "Laporan.pdf", "content": "Isi dokumen"}]
        with patch.object(app, "embed_text", return_value=[0.1]), \
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
        with patch.object(app, "embed_text", return_value=[0.1]), \
             patch.object(app, "get_similar_documents", return_value=[]), \
             patch.object(app, "compress_context", return_value=("", {"original_tokens": 0, "compressed_tokens": 0, "compression_ratio": 0.0, "sentences_kept": 0})), \
             patch.object(app, "generate_with_fallback_with_usage", side_effect=RuntimeError("provider failed")):
            with self.assertRaises(HTTPException) as error:
                await app.chat(app.QueryRequest(question="Apa isi dokumen?"))
        self.assertEqual(error.exception.status_code, 502)

    async def test_empty_question_returns_400(self):
        with self.assertRaises(HTTPException) as error:
            await app.chat(app.QueryRequest(question="   "))
        self.assertEqual(error.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
