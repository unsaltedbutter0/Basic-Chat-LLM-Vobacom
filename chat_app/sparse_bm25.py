# chat_app/sparse_bm25.py
import json, os, re
from rank_bm25 import BM25Okapi

class BM25Index:
	def __init__(self, persist_path: str):
		self.persist_path = persist_path
		self.docs = []		# token lists
		self.ids = []		# aligned with docs
		self._bm = None

	def _tok(self, text: str):
		return re.findall(r"\w+", (text or "").lower())

	def add(self, ids: list[str], texts: list[str]) -> None:
		toks = [self._tok(t) for t in texts]
		self.ids.extend(ids)
		self.docs.extend(toks)
		self._bm = BM25Okapi(self.docs)
		os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
		with open(self.persist_path, "a", encoding="utf-8") as f:
			for _id, tok in zip(ids, toks):
				f.write(json.dumps({"id": _id, "tokens": tok}, ensure_ascii=False) + "\n")

	def load(self) -> None:
		if not os.path.exists(self.persist_path):
			return
		self.docs, self.ids = [], []
		with open(self.persist_path, "r", encoding="utf-8") as f:
			for line in f:
				rec = json.loads(line)
				self.ids.append(rec["id"])
				self.docs.append(rec["tokens"])
		if self.docs:
			self._bm = BM25Okapi(self.docs)

	def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
		if not self._bm:
			return []
		q = self._tok(query)
		scores = self._bm.get_scores(q)
		ranked = sorted(zip(self.ids, scores), key=lambda x: x[1], reverse=True)
		return ranked[:top_k]
