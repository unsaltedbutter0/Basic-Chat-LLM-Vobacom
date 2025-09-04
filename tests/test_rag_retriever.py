import unittest
from chat_app.rag_retriever import RAGRetriever

class DummyCollection:
    def get(self, ids, include):
        return {"ids": [], "documents": [], "metadatas": []}

class DummyStore:
    def __init__(self):
        self.collection = DummyCollection()
    def query(self, query_text, n_results, include):
        return {
            "ids": [["1"]],
            "documents": [["alpha context"]],
            "metadatas": [[{"source_file": "a.txt", "chunk_index": 0}]],
            "distances": [[0.1]],
        }
    def sparse_query(self, query_text, n_results):
        return {
            "ids": [["1"]],
            "documents": [["alpha context"]],
            "metadatas": [[{"source_file": "a.txt", "chunk_index": 0}]],
            "scores": [[1.0]],
        }

class TestRAGRetriever(unittest.TestCase):
    def test_build_messages_hybrid_has_context(self):
        rag = RAGRetriever(DummyStore())
        messages, sources, fmt_ids = rag.build_messages_hybrid("question", top_k=1)
        self.assertIn("alpha context", messages[1]["content"])
        self.assertEqual(len(sources), 1)
        self.assertEqual(fmt_ids, ["a.txt#0"])

if __name__ == '__main__':
    unittest.main()
