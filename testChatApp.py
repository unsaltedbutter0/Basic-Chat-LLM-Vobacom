from unittest.mock import patch
import unittest
from flask import json
from ChatApp import ChatApp

class TestChatApp(unittest.TestCase):
	def setUp(self):
		patcher = patch('ChatApp.LLM_handler')
		self.MockLLMHandler = patcher.start()
		self.addCleanup(patcher.stop)

		mock_llm_instance = self.MockLLMHandler.return_value
		mock_llm_instance.chat_next.return_value = "Mocked response"

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

if __name__ == '__main__':
	unittest.main()
