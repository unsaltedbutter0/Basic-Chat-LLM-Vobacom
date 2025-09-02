from transformers import AutoTokenizer, AutoModel
from typing import Optional
import torch
from .settings import load_settings

class Embedder():
	def __init__(self, model_id: Optional[str] = None):
		if not model_id:
			cfg = load_settings()
			model_id = cfg.embeddings.model_id
		self.tokenizer = AutoTokenizer.from_pretrained(model_id)
		self.model = AutoModel.from_pretrained(model_id).eval()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model.to(self.device)

	def embed(self, texts, batch_size: int = 32):
		all_vecs = []
		with torch.no_grad():
			for i in range(0, len(texts), batch_size):
				batch = texts[i:i+batch_size]
				tokens = self.tokenizer(
					batch,
					padding=True,
					truncation=True,
					max_length=512,
					return_tensors="pt"
				).to(self.device)
				out = self.model(**tokens).last_hidden_state.mean(dim=1)
				all_vecs.extend(out.detach().cpu().numpy().tolist())
		return all_vecs


