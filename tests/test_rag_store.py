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

		self.testRAGStore = RAGStore(os.path.join("tests", "test_chroma_db"))
		
		for doc in self.documents_paths:
			with open(doc, "w") as f:
				f.write(f"This is a test file {self.documents_paths.index(doc) + 1}")

	def tearDown(self):
		for file in self.documents_paths:
			if os.path.exists(file):
				os.remove(file)

	def test_adding_documents(self):

		for doc in self.documents_paths:
			self.testRAGStore.add_document(doc)

		docs = self.testRAGStore.collection.get(include=["embeddings"])["embeddings"]
		print(docs)
		count = len(docs)
		self.assertEqual(count, len(self.documents_paths))

	def test_quering(self):
		for i in range(len(self.documents_paths)):
			result = self.testRAGStore.query(f"Test{i+1}", 1)
			text = result['documents'][0][0]
			self.assertEqual(text, f"This is a test file {i+1}")
		print(result)
		
	def test_adding_context(self):
		prompt = "What's in Test2?"
		contexts = ["This is a test file 2", "This is a test file 1"]
		for i in range(2):
			expected_result = f"From User: {prompt}\nContext: {'\n'.join(contexts[:i+1])}"
			contexted_prompt = self.testRAGStore.new_prompt(prompt, i+1)
			# print(contexted_prompt)

		self.assertEqual(contexted_prompt, expected_result)

if __name__ == '__main__':
	unittest.main()
