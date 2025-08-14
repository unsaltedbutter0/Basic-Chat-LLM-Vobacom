from unittest.mock import patch
import unittest
from chat_app.rag_store import RAGStore
import os

class TestRAGStore(unittest.TestCase):
	def setUp(self):
		self.documents_paths = [
		os.path.join("tests", "testDoc1.md"),
		os.path.join("tests", "testDoc2.md")
		]
		for doc in self.documents_paths:
			with open(doc, "w") as f:
				f.write(f"This is a test file {self.documents_paths.index(doc) + 1}")

		self.testRAGStore = RAGStore(os.path.join("tests", "test_chroma_db"))

		for doc in self.documents_paths:
			self.testRAGStore.add_document(doc)

	def test_adding_documents(self):
		docs = self.testRAGStore.collection.get(include=["embeddings"])["embeddings"]
		# print(docs)
		count = len(docs)
		self.assertEqual(count, len(self.documents_paths))

		for file in self.documents_paths:
			os.remove(file)

	def test_quering(self):
		for i in range(2):
			result = self.testRAGStore.query(f"Test{i+1}", 1)
			text = result['documents'][0][0]
			self.assertEqual(text, f"This is a test file {i+1}")

if __name__ == '__main__':
	unittest.main()
