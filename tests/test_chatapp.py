from unittest.mock import patch
import unittest
from flask import json
from chat_app.chat_app import ChatApp


class TestChatApp(unittest.TestCase):
    def setUp(self):
        patcher = patch('chat_app.chat_app.LLMHandler')
        patcher_store = patch('chat_app.chat_app.RAGStore')
        patcher_rag = patch('chat_app.chat_app.RAGRetriever')
        patcher_scan = patch('chat_app.chat_app.Scanner')

        self.MockLLMHandler = patcher.start()
        self.MockRAGStore = patcher_store.start()
        self.MockRAGRetriever = patcher_rag.start()
        self.MockScanner = patcher_scan.start()

        self.addCleanup(patcher.stop)
        self.addCleanup(patcher_store.stop)
        self.addCleanup(patcher_rag.stop)
        self.addCleanup(patcher_scan.stop)

        mock_llm_instance = self.MockLLMHandler.return_value
        mock_rag_instance = self.MockRAGRetriever.return_value
        mock_llm_instance.chat_next.return_value = "Mocked response"
        mock_llm_instance.chat_messages.return_value = "Mocked response"
        mock_rag_instance.build_messages.return_value = (["Mocked message"], ["Mocked sources"])
        self.MockScanner.return_value.scan.return_value = ["file1", "file2"]
        self.MockRAGStore.return_value.ingest.return_value = ["id1", "id2"]

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
        self.assertEqual(data['message']['content'], "Mocked response")
        self.assertEqual(data['message']['format'], "markdown")
        self.MockLLMHandler.return_value.chat_next.assert_called_once_with("Hello")

    def test_rag_route(self):
        response = self.client.post('/rag', json={"message": "Hello"})
        data = json.loads(response.data)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['message']['content'], "Mocked response")
        self.assertEqual(data['message']['format'], "markdown")
        self.assertEqual(data['meta']['sources'], ["Mocked sources"])
        self.MockRAGRetriever.return_value.build_messages.assert_called_once_with("Hello", n_results=5)
        self.MockLLMHandler.return_value.chat_messages.assert_called_once_with(["Mocked message"], reset=True)

    def test_ingest_route(self):
        response = self.client.post('/ingest', json={"folder": "/tmp", "recursive": True})
        data = json.loads(response.data)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['files_ingested'], 2)
        self.MockScanner.return_value.scan.assert_called_once_with("/tmp", recursively=True)
        self.MockRAGStore.return_value.ingest.assert_called_once_with(["file1", "file2"])


if __name__ == '__main__':
    unittest.main()

