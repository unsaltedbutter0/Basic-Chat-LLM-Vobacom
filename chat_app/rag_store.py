# rag_store.py
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

from chromadb import PersistentClient
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
	ThreadedPdfPipelineOptions as PdfPipelineOptions,
	TesseractCliOcrOptions,
)
from PIL import Image

from .embedder import Embedder
from .sparse_bm25 import BM25Index
from .settings import load_settings


class RAGStore:
	"""
	Vector store + ingest pipeline powered by Docling (convert + chunk),
	Tesseract OCR (when truly needed), and Chroma.

	Notes:
	- OCR config follows Docling's current TesseractCliOcrOptions (no extra_args).
	- Stable SHA1-based chunk IDs prevent duplicates on re-ingest.
	"""

	def __init__(
		self,
		chroma_dir: Optional[str] = None,
		tesseract_dir: Optional[str] = None,		# e.g. r"C:\Program Files\Tesseract-OCR"
	):
		cfg = load_settings()
		if not chroma_dir:
			chroma_dir = cfg.vectorstore.persist_dir

		# --- storage
		self.client = PersistentClient(path=chroma_dir)
		self.collection = self.client.get_or_create_collection(name=cfg.vectorstore.collection)

		# --- NLP
		self.embedder = Embedder()
		self.chunker = HybridChunker()

		# --- conversion / OCR
		self.converter: DocumentConverter | None = None
		self._captioner = None  # lazy-load when needed

		# Where is tesseract.exe? No PATH required.
		self.tesseract_dir = self._resolve_tesseract_dir(tesseract_dir)
		self.tessdata_dir = self._resolve_tessdata_dir(self.tesseract_dir)
		if self.tessdata_dir:
			os.environ.setdefault("TESSDATA_PREFIX", str(self.tessdata_dir) + os.sep)

		self.tesseract_cmd = self._resolve_tesseract_cmd(self.tesseract_dir)
		self._maybe_set_tessdata_prefix(self.tesseract_dir)

		# supported image types (for VisionCaptioner)
		self._image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

		# load sparse BM25
		self.bm25 = BM25Index(persist_path=os.path.join(chroma_dir, "bm25_corpus.jsonl"))
		self.bm25.load()

	# ---------------------------
	# Public API
	# ---------------------------

	def ingest(self, file_paths: Iterable[str] | str, use_vlm: bool = True, ocr: bool = True) -> List[str]:
		"""
		Ingest one or many paths. Returns list of *added* chunk IDs.
		- PDFs: auto-OCR (we'll *ignore* the 'ocr' flag and decide per file).
		- Images: if `use_vlm=True`, add a caption chunk (VisionCaptioner).
		"""
		if isinstance(file_paths, (str, os.PathLike)):
			file_paths = [str(file_paths)]

		sorted_paths = self._sort_flag_paths(file_paths) # -> [(path, is_image)]

		added_ids: List[str] = []
		for fp, is_image in sorted_paths:
			try:
				abs_path = self._validate_and_abspath(fp)
			except Exception as e:
				self._debug(f"[WARN] Skipping {fp}: {e}")
				continue

			try:
				added_ids.extend(self._ingest_one(abs_path, use_vlm=use_vlm))
			except Exception as e:
				self._debug(f"[ERROR] Failed to ingest {abs_path}: {e}")

		if self._captioner is not None:
			self._captioner.unload()
			self._captioner = None

		return added_ids

	def add_file_to_store(self, file_path: str) -> int:
		"""
		Backwards-compatible wrapper around ingest(); returns number of chunks added.
		"""
		added_ids = self.ingest([file_path], use_vlm=False, ocr=True)
		return len(added_ids)

	def query(
		self,
		query_text: str,
		n_results: int = 5,
		where: dict | None = None,
		include: Tuple[str, ...] = ("documents", "metadatas", "distances"),
	):
		"""
		Query the vector store; returns Chroma results directly.
		- Only pass `where` when provided; empty dicts can error.
		- Avoid including 'ids' in include for query() to keep it portable.
		"""
		q_emb = self.embedder.embed([query_text])[0]
		kwargs = {
			"query_embeddings": [q_emb],
			"n_results": int(n_results),
			"include": list(include),
		}
		if where:
			kwargs["where"] = where
		return self.collection.query(**kwargs)

	def sparse_query(self, query_text: str, n_results: int = 20):
		hits = self.bm25.search(query_text, top_k=int(n_results))
		if not hits:
			return {"ids": [[]], "documents": [[]], "metadatas": [[]], "scores": [[]]}
		ids = [h[0] for h in hits]
		got = self.collection.get(ids=ids, include=["documents", "metadatas"])
		# keep BM25 order
		id_to_i = {i: k for k, i in enumerate(got.get("ids", []))}
		ordered = [got["ids"][id_to_i[i]] for i in ids if i in id_to_i]
		docs = [got["documents"][id_to_i[i]] for i in ordered]
		metas = [got["metadatas"][id_to_i[i]] for i in ordered]
		scores = [s for (_, s) in hits if _ in id_to_i]
		return {"ids": [ordered], "documents": [docs], "metadatas": [metas], "scores": [scores]}


	def new_prompt(self, prompt: str, n_results: int = 5) -> str:
		"""
		Build a retrieval-augmented prompt expected by your /rag route.
		"""
		res = self.query(prompt, n_results=n_results, include=("documents",))
		docs_nested = res.get("documents") or [[]]
		context = "\n".join(docs_nested[0]) if docs_nested and docs_nested[0] else ""
		return f"From User: {prompt}\nContext to base your answer: {context}"

	def new_prompt_and_sources(self, prompt: str, n_results: int = 5):
		"""
		Like new_prompt(), but also returns a lightweight sources list for logging.
		"""
		res = self.query(prompt, n_results=n_results, include=("documents", "metadatas", "distances"))
		docs_nested = res.get("documents") or [[]]
		metas_nested = res.get("metadatas") or [[]]
		dists_nested = res.get("distances") or [[]]

		context = "\n".join(docs_nested[0]) if docs_nested and docs_nested[0] else ""
		contexted_prompt = f"From User: {prompt}\nContext to base your answer: {context}"

		sources = []
		for i, m in enumerate(metas_nested[0] if metas_nested else []):
			m = m or {}
			d = dists_nested[0][i] if dists_nested and dists_nested[0] and i < len(dists_nested[0]) else None
			sources.append({
				"source_file": m.get("source_file"),
				"chunk_index": m.get("chunk_index"),
				"page": m.get("page", -1),
				"type": m.get("type", "text"),
				"distance": d,
			})
		return contexted_prompt, sources

	# ---------------------------
	# Ingest helpers
	# ---------------------------

	def _ingest_one(self, abs_path: str, *, use_vlm: bool) -> List[str]:
		added_ids: List[str] = []

		ext = Path(abs_path).suffix.lower()
		self._debug(f"[INFO] Ingesting file with a path: {abs_path}")

		# --- Pure image file → optional caption
		if ext in self._image_exts:
			if use_vlm:
				try:
					if self._captioner is None:
						from .vision_captioner import VisionCaptioner
						self._captioner = VisionCaptioner()
					caption = self._captioner.caption(Image.open(abs_path))
					if caption and caption.strip():
						ch_id = self._stable_chunk_id(
							abs_path,
							{"source_file": abs_path, "chunk_index": -1, "page": -1, "type": "image_caption"},
							caption,
						)
						# dedup
						if self._ids_absent([ch_id])[0]:
							# add to chroma
							self.collection.add(
								documents=[caption.strip()],
								embeddings=self.embedder.embed([caption.strip()]),
								metadatas=[{"source_file": abs_path, "chunk_index": -1, "page": -1, "type": "image_caption"}],
								ids=[ch_id],
							)
							added_ids.append(ch_id)
							# add to BM25
							self.bm25.add([ch_id], [caption.strip()])
				except Exception as e:
                    # don't fail ingestion on caption hiccups
					self._debug(f"[WARN] VisionCaptioner failed: {e}")

			# even if no caption, nothing else to do for images
			return added_ids

		# --- Document file (e.g., PDF)
		result = self._convert_auto_ocr(abs_path)
		doc = result.document

		# Build chunks from Docling
		raw_chunks = list(self.chunker.chunk(dl_doc=doc))
		candidate_ids, texts_to_add, metas_to_add = self._build_text_chunks(abs_path, raw_chunks)

		# Filter out IDs that already exist in Chroma
		mask_new = self._ids_absent(candidate_ids)
		ids_to_add = [cid for cid, keep in zip(candidate_ids, mask_new) if keep]
		texts_final = [t for t, keep in zip(texts_to_add, mask_new) if keep]
		metas_final = [m for m, keep in zip(metas_to_add, mask_new) if keep]

		if not ids_to_add:
			return added_ids

		embeddings = self.embedder.embed(texts_final)
		# add to chroma
		self.collection.add(
			documents=texts_final,
			embeddings=embeddings,
			metadatas=metas_final,
			ids=ids_to_add,
		)
		# add to BM25
		self.bm25.add(ids_to_add, texts_final)
		added_ids.extend(ids_to_add)
		return added_ids

	def _build_text_chunks(self, abs_path: str, chunks) -> Tuple[List[str], List[str], List[dict]]:
		candidate_ids: List[str] = []
		texts_to_add: List[str] = []
		metas_to_add: List[dict] = []

		for idx, ch in enumerate(chunks):
			txt = getattr(ch, "text", "")
			if not isinstance(txt, str) or not txt.strip():
				continue

			# Normalize text artifacts from PDFs (ligatures, soft hyphens, split words, ws)
			txt = self._normalize_text(txt)
			if not txt or len(txt) < 10:	# skip junky/ultra-short chunks
				continue

			meta_raw = {
				"source_file": abs_path,
				"chunk_index": int(idx),
				"page": int(getattr(ch, "page", -1)) if getattr(ch, "page", None) is not None else -1,
				"type": self._safe_str(getattr(ch, "type", None)) or "text",
			}
			meta = self._sanitize_metadata(meta_raw)
			ch_id = self._stable_chunk_id(abs_path, meta, txt)

			candidate_ids.append(ch_id)
			texts_to_add.append(txt)
			metas_to_add.append(meta)

		return candidate_ids, texts_to_add, metas_to_add

	# ---------------------------
	# Docling conversion & OCR
	# ---------------------------

	def _convert_auto_ocr(self, abs_path: str):
		"""
		Two-pass strategy:
		1) Try without OCR (fastest for text-native PDFs).
		2) If extracted text is sparse, retry with OCR enabled.
		"""
		res1 = self._docling_convert(abs_path, ocr=False, do_picture_description=False)
		doc1 = res1.document
		total_len = sum(len(getattr(p, "text", "") or "") for p in getattr(doc1, "pages", []))

		# Heuristic: enough text → keep no-OCR result
		if total_len >= 2000 or any(len(getattr(p, "text", "") or "") > 200 for p in getattr(doc1, "pages", [])):
			return res1

		# Otherwise OCR (page-selective; not full-page)
		return self._docling_convert(abs_path, ocr=True, do_picture_description=False)

	def _docling_convert(self, abs_path: str, *, ocr: bool, do_picture_description: bool = False):
		popts = PdfPipelineOptions()
		popts.do_picture_description = bool(do_picture_description)
		popts.images_scale = 3.0
		popts.ocr_batch_size = 1

		popts.do_ocr = bool(ocr)
		if popts.do_ocr:
			tdir = str(self.tessdata_dir) if self.tessdata_dir else None
			desired_langs = ["eng","pol"]
			use_langs = [l for l in desired_langs if l in self._installed_tess_langs()] or ["eng"]
			self._debug(f"[OCR] Using langs={use_langs}, tessdata_dir={tdir}")

			popts.ocr_options = TesseractCliOcrOptions(
				tesseract_cmd=self.tesseract_cmd,
				path=tdir,								# <- pass tessdata folder, not install root
				lang=use_langs,
				force_full_page_ocr=True,
				bitmap_area_threshold=0.0,
			)

		self.converter = DocumentConverter(format_options={
			InputFormat.PDF: PdfFormatOption(pipeline_options=popts)
		})
		return self.converter.convert(abs_path)


	# ---------------------------
	# Utilities
	# ---------------------------

	def _validate_and_abspath(self, file_path: str) -> str:
		if not file_path:
			raise ValueError("Empty file path.")
		p = Path(file_path)
		if not p.exists():
			raise FileNotFoundError(f"File not found: {file_path}")
		if not p.is_file():
			raise IsADirectoryError(f"Not a file: {file_path}")
		return str(p.resolve())

	def _sort_flag_paths(self, file_paths: list[str]) -> List[dir]:
		sorted_and_flaged = []
		images = []
		docs = []
		for fp in file_paths:
			ext = Path(fp).suffix.lower()
			if ext in self._image_exts:
				images.append((fp, True))
			else:
				docs.append((fp, False))

		return images + docs


	def _ids_absent(self, ids: List[str]) -> List[bool]:
		"""
		Returns a boolean mask: True where id is NOT present in the collection.
		"""
		try:
			found = self.collection.get(ids=ids)
		except Exception:
			found = {"ids": []}
		found_ids = set(found.get("ids", []) or [])
		return [(_id not in found_ids) for _id in ids]

	def _stable_chunk_id(self, abs_path: str, meta: dict, text: str) -> str:
		payload = json.dumps(
			{
				"src": abs_path,
				"meta": meta,
				"sha1": hashlib.sha1(text.encode("utf-8")).hexdigest(),
			},
			sort_keys=True,
			ensure_ascii=False,
		)
		return hashlib.sha1(payload.encode("utf-8")).hexdigest()

	def _normalize_text(self, t: str) -> str:
		# Basic fixes for PDF artifacts (ligatures, soft hyphen, intra-word spaces)
		try:
			import unicodedata, re
			_SOFT_HYPHEN = "\u00AD"
			_LIG_MAP = {
				"\uFB00": "ff",	 # ﬀ
				"\uFB01": "fi",	 # ﬁ
				"\uFB02": "fl",	 # ﬂ
				"\uFB03": "ffi", # ﬃ
				"\uFB04": "ffl", # ﬄ
			}
			t = unicodedata.normalize("NFKC", t or "")
			for k, v in _LIG_MAP.items():
				t = t.replace(k, v)
			t = t.replace(_SOFT_HYPHEN, "")
			# Collapse weird intra-word splits occasionally seen in PDFs
			t = re.sub(r"([A-Za-z])\s+([A-Za-z])", r"\1 \2", t)
			# Tidy whitespace
			t = re.sub(r"[ \t]+", " ", t)
			t = re.sub(r"\s*\n\s*", "\n", t)
			return t.strip()
		except Exception:
			return (t or "").strip()

	def _sanitize_metadata(self, meta: dict) -> dict:
		"""
		Chroma metadata must be Bool | Int | Float | Str.
		- Drop None
		- Cast non-primitive objects to str
		"""
		clean = {}
		for k, v in (meta or {}).items():
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

	def _resolve_tessdata_dir(self, tesseract_dir: Path | None) -> Path | None:
		if not tesseract_dir:
			return None
		td = tesseract_dir / "tessdata"
		return td if td.exists() else None


	def _resolve_tesseract_dir(self, preferred: str | None) -> Path | None:
		if preferred:
			p = Path(preferred)
			return p if p.exists() else None
		# Try common locations on Windows; noop elsewhere
		candidates = []
		if sys.platform.startswith("win"):
			candidates = [
				Path(r"C:\Program Files\Tesseract-OCR"),
				Path(r"C:\Program Files (x86)\Tesseract-OCR"),
				Path.home() / "AppData" / "Local" / "Tesseract-OCR",
			]
		for d in candidates:
			if (d / "tesseract.exe").exists():
				return d
		return None

	def _resolve_tesseract_cmd(self, tesseract_dir: Path | None) -> str:
		if tesseract_dir:
			cmd = (tesseract_dir / "tesseract.exe").resolve()
			return str(cmd)
		# Fallback: let system PATH resolve it (may fail, but Docling will surface error)
		return "tesseract"

	def _maybe_set_tessdata_prefix(self, tesseract_dir: Path | None) -> None:
		if not tesseract_dir:
			return
		td = tesseract_dir / "tessdata"
		if td.exists():
			os.environ.setdefault("TESSDATA_PREFIX", str(td) + os.sep)

	def _installed_tess_langs(self) -> set[str]:
		"""
		Return set of installed Tesseract language codes, e.g. {'eng','pol'}.
		"""
		try:
			import subprocess, shlex
			cmd = f"\"{self.tesseract_cmd}\" --list-langs"
			out = subprocess.check_output(shlex.split(cmd), stderr=subprocess.STDOUT, text=True)
			langs = {ln.strip() for ln in out.splitlines() if ln.strip() and not ln.lower().startswith("list of")}
			return langs
		except Exception as e:
			self._debug(f"[OCR] Couldn't list Tesseract languages: {e}")
			return set()


	def _debug(self, msg: str) -> None:
		try:
			print(msg, flush=True)
		except Exception:
			pass
