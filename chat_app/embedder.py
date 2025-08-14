from transformers import AutoTokenizer, AutoModel
import torch

class Embedder():
	def __init__(self, model_id="sentence-transformers/all-MiniLM-L6-v2"):
		self.tokenizer = AutoTokenizer.from_pretrained(model_id)
		self.model = AutoModel.from_pretrained(model_id)

	def embed(self, texts):
		tokens = self.tokenizer(
			texts,
			padding=True,
			truncation=True,
			return_tensors="pt"
			)

		with torch.no_grad():
			outputs = self.model(**tokens)
			embeddings = outputs.last_hidden_state.mean(dim=1)

		return embeddings.cpu().numpy().tolist()

