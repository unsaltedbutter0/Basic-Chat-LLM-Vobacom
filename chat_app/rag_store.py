# rag_store.py
from docling_core.types.doc import DoclingDocument
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from chromadb import PersistentClient
from chromadb.config import Settings
from uuid import uuid4
from pathlib import Path
import os
from .embedder import Embedder

class RAGStore:
	def __init__(self, chroma_dir="chroma_db"):
		self.client = PersistentClient(path=chroma_dir)
		self.collection = self.client.get_or_create_collection(name="documents")
		self.TEXT_FORMATS = {".txt", ".log", ".csv", ".json", ".xml", ".yaml"}
		self.embedder = Embedder()
		self.converter = DocumentConverter()
		self.chunker = HybridChunker()

	def add_document(self, file_path):
		if not os.path.exists(file_path):
			raise FileNotFoundError(f"File not found: {file_path}")

		doc = self.converter.convert(file_path).document

		chunks = list(self.chunker.chunk(dl_doc=doc))
		# print(chunks)

		texts = [chunk.text for chunk in chunks if getattr(chunk, "text", "").strip()]			
		if not texts:
			return 0

		embeddings = self.embedder.embed(texts)

		for i, (text, emb) in enumerate(zip(texts, embeddings)):
			self.collection.add(
				documents="<stored externally>",
				embeddings=emb,
				metadatas={
					"source_file": os.path.abspath(file_path),
					"chunk_index": i,
					"page": getattr(chunks[i], 'page', -1)
				},
				ids=f"{file_path}_chunk_{i}"
			)

	# def _is_text_file(self, file_path):
	# 	try:
	# 		with open(file_path, "r") as f:
	# 			f.read(1024)
	# 		return True
	# 	except UnicodeDecodeError:
	# 		return False

	# def _load_file(self, file_path):
	# 	if Path(file_path).suffix.lower() in self.TEXT_FORMATS and self._is_text_file(file_path):
	# 		with open(file_path, "r", encoding="utf-8") as f:
	# 			return DoclingDocument(
	# 				name=Path(file_path).stem,
	# 				text=f.read(),
	# 				metadata={"source": file_path}
	# 			)
	# 	else:
	# 		converter = DocumentConverter()
	# 		return converter.convert(file_path).document


	def query(self, query_text, n_results=5):
		query_embeddings = self.embedder.embed(query_text)[0]

		results = self.collection.query(
			query_embeddings=[query_embeddings],
			n_results=n_results
		)

		for i, meta_list in enumerate(results['metadatas']):
			for j, meta in enumerate(meta_list):
				results["documents"][i][j] = self._get_chunk_text(meta)

		return results

	def _get_chunk_text(self, metadata):
		file_path = metadata['source_file']
		result = self.converter.convert(file_path)

		doc = result.document
		chunks = list(self.chunker.chunk(dl_doc=doc))

		return chunks[metadata['chunk_index']].text
