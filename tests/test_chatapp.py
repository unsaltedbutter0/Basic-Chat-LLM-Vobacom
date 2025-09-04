from unittest.mock import patch
import unittest
from flask import json
from chat_app.chat_app import ChatApp
from chat_app.settings import (
    Settings,
    PathsCfg,
    AppCfg,
    ModelCfg,
    EmbeddingsCfg,
    VectorStoreCfg,
    GuardrailsCfg,
    DataDirCfg,
)


class TestChatApp(unittest.TestCase):
    def setUp(self):
        patcher = patch('chat_app.chat_app.LLMHandler')
        patcher_store = patch('chat_app.chat_app.RAGStore')
        patcher_rag = patch('chat_app.chat_app.RAGRetriever')
        patcher_scan = patch('chat_app.chat_app.Scanner')
        patcher_settings = patch('chat_app.chat_app.load_settings')
        patcher_save = patch('chat_app.chat_app.save_settings')

        self.MockLLMHandler = patcher.start()
        self.MockRAGStore = patcher_store.start()
        self.MockRAGRetriever = patcher_rag.start()
        self.MockScanner = patcher_scan.start()
        self.MockLoadSettings = patcher_settings.start()
        self.MockSaveSettings = patcher_save.start()

        self.addCleanup(patcher.stop)
        self.addCleanup(patcher_store.stop)
        self.addCleanup(patcher_rag.stop)
        self.addCleanup(patcher_scan.stop)
        self.addCleanup(patcher_settings.stop)
        self.addCleanup(patcher_save.stop)

        mock_llm_instance = self.MockLLMHandler.return_value
        mock_rag_instance = self.MockRAGRetriever.return_value
        mock_llm_instance.chat_next.return_value = "Mocked response"
        mock_llm_instance.chat_messages.return_value = "Mocked response"
        mock_rag_instance.build_messages_hybrid.return_value = (["Mocked message"], ["Mocked sources"], {"fmt1"})
        self.MockScanner.return_value.scan.return_value = ["file1", "file2"]
        self.MockRAGStore.return_value.ingest.return_value = ["id1", "id2"]

        cfg = Settings(
            app=AppCfg(),
            paths=PathsCfg(data_dirs=[DataDirCfg(path="/tmp", recursive=True)]),
            model=ModelCfg(),
            embeddings=EmbeddingsCfg(),
            vectorstore=VectorStoreCfg(),
            guardrails=GuardrailsCfg(),
        )
        self.MockLoadSettings.return_value = cfg

        self.chat_app = ChatApp("dummy-model-id")
        self.app = self.chat_app.app
        self.client = self.app.test_client()

    def test_index_route(self):
        agents = {
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36": "desktop",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:116.0) Gecko/20100101 Firefox/116.0": "desktop",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1": "mobile",
            "Mozilla/5.0 (iPad; CPU OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/115.0.5790.71 Mobile/15E148 Safari/604.1": "mobile",
        }
        for agent, expected in agents.items():
            response = self.client.get('/', headers={"User-Agent": agent})
            html = response.data.decode('utf-8')

            if expected == "mobile":
                self.assertIn("<!-- MOBILE VERSION -->", html)
                self.assertNotIn("<!-- DESKTOP VERSION -->", html)
            else:
                self.assertIn("<!-- DESKTOP VERSION -->", html)
                self.assertNotIn("<!-- MOBILE VERSION -->", html)

    def test_chat_route(self):
        response = self.client.post('/chat', json={"message": "Hello"})
        data = json.loads(response.data)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(data['message']['content'].startswith("Mocked response"))
        self.assertEqual(data['message']['format'], "markdown")

        self.MockLLMHandler.return_value.chat_next.assert_called_once_with("Hello")

    def test_rag_route(self):
        response = self.client.post('/rag', json={"message": "Hello"})
        data = json.loads(response.data)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(data['message']['content'].startswith("Mocked response"))
        self.assertEqual(data['message']['format'], "markdown")
        self.assertEqual(data['meta']['sources'], ["Mocked sources"])
        self.MockRAGRetriever.return_value.build_messages_hybrid.assert_called_once_with("Hello")
        self.MockLLMHandler.return_value.chat_messages.assert_called_once_with(["Mocked message"], reset=True)

    def test_ingest_route(self):
        response = self.client.post('/ingest')
        data = json.loads(response.data)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['files_ingested'], 2)
        self.MockScanner.return_value.scan.assert_called_once_with("/tmp", recursively=True)
        self.MockRAGStore.return_value.ingest.assert_called_once_with(["file1", "file2"])

    def test_post_settings_saves_data_dirs(self):
        payload = {
            "paths": {
                "data_dirs": [{"path": "/new", "recursive": True}]
            }
        }
        response = self.client.post('/api/settings', json=payload)
        self.assertEqual(response.status_code, 200)
        args, _ = self.MockSaveSettings.call_args
        saved_cfg = args[0]
        self.assertEqual(saved_cfg.paths.data_dirs, [DataDirCfg(path="/new", recursive=True)])


if __name__ == '__main__':
    unittest.main()

