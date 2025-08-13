# rag_store.py
from docling_core.types.doc import DoclingDocument
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from chromadb import Client
from chromadb.config import Settings
from uuid import uuid4
from pathlib import Path
import os

class RAGStore:
	def __init__(self, chroma_dir="chroma_db"):
		self.client = Client(Settings(persist_directory=chroma_dir))
		self.collection = self.client.get_or_create_collection(name="documents")
		self.TEXT_FORMATS = {".txt", ".log", ".md", ".csv", ".json", ".xml", ".yaml"}

	def add_document(self, file_path):
	    if not os.path.exists(file_path):
	        raise FileNotFoundError(f"File not found: {file_path}")

	    doc = self._load_file(file_path)
	    print("Loaded doc id:", doc.id if hasattr(doc, "id") else None)

	    chunker = HybridChunker()
	    chunks = chunker.chunk(dl_doc=doc)
	    print("Number of chunks:", len(chunks))
	    # for i, chunk in enumerate(chunker.chunk(dl_doc=doc)):
	    #     self.collection.add(
	    #         documents=[],
	    #         embeddings=[chunk.embedding],
	    #         metadatas=[{
	    #             "source_file": file_path,
	    #             "chunk_index": i,
	    #             "page": getattr(chunk, 'page', None)
	    #         }],
	    #         ids=[f"{doc.id}_chunk_{i}"]
	    #     )

	def _is_text_file(self, file_path):
	    try:
	        with open(file_path, "r") as f:
	            f.read(1024)
	        return True
	    except UnicodeDecodeError:
	        return False

	def _load_file(self, file_path):
	    if Path(file_path).suffix.lower() in self.TEXT_FORMATS and self._is_text_file(file_path):
	        with open(file_path, "r", encoding="utf-8") as f:
	            return DoclingDocument(
	            	id=str(uuid4()),
                	name=Path(file_path).stem,
                	text=f.read(),
                	metadata={"source": file_path}
            	)
	    else:
	        converter = DocumentConverter()
	        return converter.convert(file_path).document


	def query(self, query_text, n_results=5):
		"""
		Search similar chunks in Chroma using the query.
		Returns metadata including file path and chunk index.
		"""
		results = self.collection.query(
			query_texts=[query_text],
			n_results=n_results
		)
		return results['metadatas'][0]

	def get_chunk_text(self, metadata):
		"""
		Retrieve the full chunk text from original file using metadata.
		"""
		file_path = metadata['source_file']
		result = self.converter.convert(file_path, InputFormat.detect(file_path))
		doc = result.document
		chunks = [p.text for p in doc.paragraphs if p.text.strip()]
		return chunks[metadata['chunk_index']]
