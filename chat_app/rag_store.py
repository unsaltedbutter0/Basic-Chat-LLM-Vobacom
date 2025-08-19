# rag_store.py
from docling_core.types.doc import DoclingDocument
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from chromadb import PersistentClient
import hashlib
import json
import os
from .embedder import Embedder
from PIL import Image
from pathlib import Path
from uuid import uuid4 

class RAGStore:
	def __init__(self, chroma_dir="chroma_db"):
		self.client = PersistentClient(path=chroma_dir)
		self.collection = self.client.get_or_create_collection(name="documents")
		self.embedder = Embedder()
		self.converter = DocumentConverter()
		self.chunker = HybridChunker()

		self._image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
		self._captioner = None

	# ---------- public API -------------------------------------------------

	def ingest(self, paths: list[str], *, use_vlm: bool=False, ocr: bool=True) -> list[str]:
		"""
		Ingest one or more files via Docling → chunk → embed → add to Chroma.
		- Stable, hashed chunk IDs prevent duplicates on re-ingestion.
		- Stores actual text in Chroma's 'documents' for direct retrieval.
		Returns the list of newly-added chunk IDs.
		"""
		added_ids: list[str] = []

		for file_path in paths:
			if not os.path.exists(file_path):
				raise FileNotFoundError(f"File not found: {file_path}")

			abs_path = os.path.abspath(file_path)

			try:
				result = self.converter.convert(abs_path)
				doc = result.document
			except Exception as e:
				print(f"[WARN] Could not parse {abs_path}: {e}")
				continue

			chunks = list(self.chunker.chunk(dl_doc=doc))

			# Build stable IDs + sanitized metadata for chunks that have non-empty text
			candidate_ids, texts_to_add, metas_to_add = [], [], []
			for idx, ch in enumerate(chunks):
				txt = getattr(ch, "text", "")
				if not isinstance(txt, str) or not txt.strip():
					continue

				meta_raw = {
					"source_file": abs_path,
					"chunk_index": int(idx),
					"page": int(getattr(ch, "page", -1)) if getattr(ch, "page", None) is not None else -1,
					"type": self._safe_str(getattr(ch, "type", None)),
				}
				meta = self._sanitize_metadata(meta_raw)
				ch_id = self._stable_chunk_id(abs_path, meta, txt)

				candidate_ids.append(ch_id)
				texts_to_add.append(txt.strip())
				metas_to_add.append(meta)

			ext = os.path.splitext(abs_path)[1].lower()
			if use_vlm and ext in self._image_exts:
				try:
					if self._captioner is None:
						from .vision_captioner import VisionCaptioner
						self._captioner = VisionCaptioner()

					with Image.open(abs_path) as img:
						img = img.convert("RGB")
						caption = self._captioner.caption(img)

					if caption and isinstance(caption, str) and caption.strip():
						meta_cap_raw = {
							"source_file": abs_path,
							"chunk_index": -1,
							"page": -1,
							"type": "image_caption",
						}
						meta_cap = self._sanitize_metadata(meta_cap_raw)
						ch_id_cap = self._stable_chunk_id(abs_path, meta_cap, caption)
						candidate_ids.append(ch_id_cap)
						texts_to_add.append(caption.strip())
						metas_to_add.append(meta_cap)
				except Exception as e:
					print(f"[WARN] Captioning failed for {abs_path}: {e}")

			if not candidate_ids:
				print(f"[INFO] No text extracted from {abs_path} (maybe empty or image without OCR)")
				continue

			# Filter out any chunks already present
			to_add_mask = self._missing_id_mask(candidate_ids)
			if not any(to_add_mask):
				continue

			ids_to_add = [i for i, keep in zip(candidate_ids, to_add_mask) if keep]
			texts_final = [t for t, keep in zip(texts_to_add, to_add_mask) if keep]
			metas_final = [m for m, keep in zip(metas_to_add, to_add_mask) if keep]

			embeddings = self.embedder.embed(texts_final)

			self.collection.add(
				documents=texts_final,
				embeddings=embeddings,
				metadatas=metas_final,
				ids=ids_to_add
			)

			added_ids.extend(ids_to_add)

		return added_ids

	def add_file_to_store(self, file_path):
		"""
		Backwards-compatible wrapper around ingest(); returns number of chunks added.
		"""
		added_ids = self.ingest([file_path], use_vlm=False, ocr=True)
		return len(added_ids)

	def query(self, query_text, n_results=5, where=None, include=("documents","metadatas","distances")):
		"""
		Query the vector store; returns Chroma results directly.
		- Only pass `where` when provided; empty dicts can error.
		- Do not include 'ids' in include (some backends disallow it for query()).
		"""
		q_emb = self.embedder.embed([query_text])[0]

		kwargs = {
			"query_embeddings": [q_emb],
			"n_results": n_results,
			"include": list(include)
		}
		if where:
			kwargs["where"] = where

		results = self.collection.query(**kwargs)
		return results

	def new_prompt_and_sources(self, prompt: str, n_results: int = 5):
		results = self.query(prompt, n_results, include=("documents","metadatas","distances"))
		texts_nested = results.get("documents", [[]])
		metas_nested = results.get("metadatas", [[]])
		dists_nested = results.get("distances", [[]])

		contex = '\n'.join(texts_nested[0]) if (texts_nested and texts_nested[0]) else ""
		contexted_prompt = f"From User: {prompt}\nContext to base your answer: {contex}"

		sources = []
		for i in range(len(metas_nested[0]) if metas_nested else 0):
			m = metas_nested[0][i] or {}
			d = dists_nested[0][i] if (dists_nested and dists_nested[0] and i < len(dists_nested[0])) else None
			sources.append({
				"source_file": m.get("source_file"),
				"chunk_index": m.get("chunk_index"),
				"page": m.get("page", -1),
				"type": m.get("type", "text"),
				"distance": d
			})
		return contexted_prompt, sources

	# ---------- management helpers ----------------------------------------

	def delete_source(self, file_path: str) -> int:
		"""
		Delete all chunks that came from a given file path.
		Returns the number of deleted chunks.
		"""
		abs_path = os.path.abspath(file_path)
		# NOTE: Do NOT pass include=["ids"]; 'ids' is always returned by get()
		records = self.collection.get(where={"source_file": abs_path})
		ids = records.get("ids", []) or []
		if ids:
			self.collection.delete(ids=ids)
		return len(ids)


	def reingest(self, file_path: str, *, use_vlm: bool=False, ocr: bool=True) -> int:
		"""
		Delete all chunks for a source and ingest it again.
		Returns the number of newly added chunks.
		"""
		self.delete_source(file_path)
		added_ids = self.ingest([file_path], use_vlm=use_vlm, ocr=ocr)
		return len(added_ids)

	def list_sources(self) -> list[str]:
		"""
		Return a sorted list of unique source_file paths currently indexed.
		"""
		records = self.collection.get(include=["metadatas"])
		metas = records.get("metadatas", []) or []
		seen = set()
		for m in metas:
			src = (m or {}).get("source_file")
			if src:
				seen.add(src)
		return sorted(seen)

	def stats(self) -> dict:
		"""
		Simple counts for quick visibility.
		"""
		records = self.collection.get(include=[])
		total_chunks = len(records.get("ids", []))
		return {
			"chunks": total_chunks,
			"sources": len(self.list_sources())
		}

	# ---------- internals --------------------------------------------------

	def _stable_chunk_id(self, source_path: str, meta: dict, text: str) -> str:
		payload = {
			"src": os.path.abspath(source_path),
			"meta": meta,
			"text": text
		}
		key = json.dumps(payload, sort_keys=True, ensure_ascii=False)
		digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
		return f"ch:{digest}"

	def _missing_id_mask(self, ids: list[str]) -> list[bool]:
		"""
		Given a list of candidate IDs, return a boolean mask where True means
		the ID is not present in the collection yet.
		"""
		if not ids:
			return []
		found = self.collection.get(ids=ids, include=[])
		found_ids = set(found.get("ids", []) or [])
		return [(_id not in found_ids) for _id in ids]

	def _safe_str(self, val):
		"""
		Safely stringify enums/objects; return None -> None so sanitizer can drop it.
		"""
		if val is None:
			return None
		try:
			return str(val)
		except Exception:
			return None

	def _sanitize_metadata(self, meta: dict) -> dict:
		"""
		Chroma metadata values must be Bool | Int | Float | Str.
		- Drop None
		- Cast non-primitive objects to str
		"""
		clean = {}
		for k, v in meta.items():
			if v is None:
				continue
			if isinstance(v, (bool, int, float, str)):
				clean[k] = v
				continue
			try:
				clean[k] = str(v)
			except Exception:
				continue
		return clean
