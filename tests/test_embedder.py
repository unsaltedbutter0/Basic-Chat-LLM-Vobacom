import unittest
from unittest.mock import patch, MagicMock
from chat_app.embedder import Embedder

class TestEmbedderMock(unittest.TestCase):
	@patch('chat_app.embedder.AutoTokenizer')
	@patch('chat_app.embedder.AutoModel')
	def test_embed_mocked(self, mock_model_cls, mock_tokenizer_cls):
		mock_tokenizer = MagicMock()
		mock_tokenizer.return_value = {'input_ids': 'mock_ids', 'attention_mask': 'mock_mask'}
		mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

		mock_model = MagicMock()
		mock_model.return_value.last_hidden_state.mean.return_value = 'mock_embedding_tensor'
		mock_model_cls.from_pretrained.return_value = mock_model

		embedder = Embedder()
		with patch.object(embedder, 'embed', return_value=[[0.1, 0.2, 0.3]]) as mock_embed:
			result = embedder.embed(["Hello", "World"])
			self.assertEqual(result, [[0.1, 0.2, 0.3]])
			mock_embed.assert_called_once_with(["Hello", "World"])

if __name__ == "__main__":
	unittest.main()
