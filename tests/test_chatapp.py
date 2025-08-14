from unittest.mock import patch
import unittest
from flask import json
from chat_app.chat_app import ChatApp

class TestChatApp(unittest.TestCase):
	def setUp(self):
		patcher = patch('chat_app.chat_app.LLMHandler')
		patcher_rag = patch('chat_app.chat_app.RAGStore')
		
		self.MockLLMHandler = patcher.start()
		self.MockRAGStore = patcher_rag.start()
		
		self.addCleanup(patcher.stop)
		self.addCleanup(patcher_rag.stop)

		mock_llm_instance = self.MockLLMHandler.return_value
		mock_rag_instance = self.MockRAGStore.return_value
		mock_llm_instance.chat_next.return_value = "Mocked response"
		mock_rag_instance.new_prompt.return_value = "Mocked prompt"

		self.chat_app = ChatApp("dummy-model-id")
		self.app = self.chat_app.app
		self.client = self.app.test_client()

	def test_index_route(self):
		agents = {
			"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36": "desktop",
			"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:116.0) Gecko/20100101 Firefox/116.0": "desktop",
			"Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1": "mobile",
			"Mozilla/5.0 (iPad; CPU OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/115.0.5790.71 Mobile/15E148 Safari/604.1": "mobile"
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
		self.assertEqual(data['response'], "Mocked response")
		self.MockLLMHandler.return_value.chat_next.assert_called_once_with("Hello")

	def test_rag_route(self):
		response = self.client.post('/rag', json={"message": "Hello"})
		data = json.loads(response.data)

		self.assertEqual(response.status_code, 200)
		self.assertEqual(data['response'], "Mocked response")
		self.MockRAGStore.return_value.new_prompt.assert_called_once_with("Hello")
		self.MockLLMHandler.return_value.chat_next.assert_called_once_with("Mocked prompt")


if __name__ == '__main__':
	unittest.main()
