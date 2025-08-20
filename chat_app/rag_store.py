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
			abs_path = self._validate_and_abspath(file_path)
			try:
				ids = self._ingest_one(abs_path, use_vlm=use_vlm, ocr=ocr)
				added_ids.extend(ids)
			except Exception as e:
				print(f"[WARN] Failed to ingest {abs_path}: {e}")
				continue
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

	# ---------- internals (ingest helpers) ---------------------------------

	def _ingest_one(self, abs_path: str, *, use_vlm: bool, ocr: bool) -> list[str]:
		"""Ingest a single absolute path and return added chunk IDs."""
		# 1) Convert document with Docling
		result = self._convert_with_docling(abs_path, ocr=ocr)
		doc = result.document

		# 2) Chunk
		chunks = self._chunk_doc(doc)

		# 3) Build candidates from text chunks
		cand_ids, texts, metas = self._build_text_chunks(abs_path, chunks)

		# 4) Optionally caption images (VLM)
		if use_vlm:
			cap_id, cap_text, cap_meta = self._maybe_caption_image(abs_path)
			if cap_id is not None:
				cand_ids.append(cap_id)
				texts.append(cap_text)
				metas.append(cap_meta)

		# 5) Filter out existing IDs
		mask = self._missing_id_mask(cand_ids)
		if not any(mask):
			return []
		ids_to_add = [i for i, keep in zip(cand_ids, mask) if keep]
		texts_final = [t for t, keep in zip(texts, mask) if keep]
		metas_final = [m for m, keep in zip(metas, mask) if keep]

		# 6) Embed + add to Chroma
		embeddings = self._embed_texts(texts_final)
		self._add_to_collection(ids_to_add, texts_final, metas_final, embeddings)
		return ids_to_add

	def _validate_and_abspath(self, file_path: str) -> str:
		"""Ensure path exists and return absolute path."""
		if not os.path.exists(file_path):
			raise FileNotFoundError(f"File not found: {file_path}")
		return os.path.abspath(file_path)

	def _convert_with_docling(self, abs_path: str, *, ocr: bool):
		"""
		Run Docling conversion.
		If your Docling version exposes configuration for OCR (e.g., pipeline options),
		you can wire it here with the `ocr` flag.
		"""
		return self.converter.convert(abs_path)

	def _chunk_doc(self, dl_doc: DoclingDocument):
		"""Chunk a DoclingDocument using HybridChunker (titles + semantic)."""
		return list(self.chunker.chunk(dl_doc=dl_doc))

	def _build_text_chunks(self, abs_path: str, chunks):
		"""
		From Docling chunks, produce candidate IDs, texts, and metadata.
		Ignores empty / whitespace-only text chunks.
		"""
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
		return candidate_ids, texts_to_add, metas_to_add

	def _maybe_caption_image(self, abs_path: str):
		"""
		If `abs_path` is an image and VLM is enabled, run captioning and return a
		single (id, text, meta) tuple; otherwise (None, None, None).
		"""
		ext = os.path.splitext(abs_path)[1].lower()
		if ext not in self._image_exts:
			return None, None, None
		try:
			if self._captioner is None:
				from .vision_captioner import VisionCaptioner
				self._captioner = VisionCaptioner()
			from PIL import Image
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
				return ch_id_cap, caption.strip(), meta_cap
		except Exception as e:
			print(f"[WARN] Captioning failed for {abs_path}: {e}")
		return None, None, None

	def _embed_texts(self, texts: list[str]):
		"""Return embeddings for the given list of texts using self.embedder."""
		return self.embedder.embed(texts)

	def _add_to_collection(self, ids, texts, metas, embeddings):
		"""Add records to Chroma collection (documents + embeddings + metadata)."""
		self.collection.add(
			documents=texts,
			embeddings=embeddings,
			metadatas=metas,
			ids=ids,
		)

	# ---------- internals (generic) ---------------------------------------

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
