from unittest.mock import patch
import unittest
from chat_app.rag_store import RAGStore
import os

class TestRAGStore(unittest.TestCase):
	def setUp(self):
		self.documents_paths = [
		os.path.join("tests", "testDoc1.txt"),
		os.path.join("tests", "testDoc2.txt")
		]
		for doc in self.documents_paths:
			with open(doc, "w") as f:
				f.write(f"This is a test file {self.documents_paths.index(doc) + 1}")

		self.testRAGStore = RAGStore(os.path.join("tests", "test_chroma_db"))

		for doc in self.documents_paths:
			self.testRAGStore.add_document(doc)

	def test_adding_documents(self):
		count = len(self.testRAGStore.collection.get(include=["documents"])["documents"])
		self.assertEqual(count, len(self.documents_paths))


if __name__ == '__main__':
	unittest.main()
