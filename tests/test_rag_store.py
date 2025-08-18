# tests/test_rag_store.py
import unittest
import tempfile
import shutil
import os
from chat_app.rag_store import RAGStore

# --- tiny deterministic embedder so we don't hit transformers ---
class _FakeEmbedder:
	def __init__(self):
		self.dim = 1  # not used, but harmless

	def embed(self, texts):
		# Return a 1D embedding that maps strings containing "2" closer to query "Test2"
		# Simple rule:
		#  - contains "2" -> [2.0]
		#  - contains "1" -> [1.0]
		#  - else -> [0.0]
		embs = []
		if isinstance(texts, str):
			texts = [texts]
		for t in texts:
			val = 2.0 if "2" in t else (1.0 if "1" in t else 0.0)
			embs.append([val])
		return embs


class TestRAGStore(unittest.TestCase):
	def setUp(self):
		# temp chroma dir per test, so no cross-test leakage
		self._tmpdir = tempfile.mkdtemp(prefix="chroma_test_")

		# make sure tests/ exists (Windows-safe)
		self.tests_dir = os.path.abspath("tests")
		os.makedirs(self.tests_dir, exist_ok=True)

		# simple markdown docs; one mentions "1", the other "2"
		self.documents_paths = [
			os.path.join(self.tests_dir, "testDoc1.md"),
			os.path.join(self.tests_dir, "testDoc2.md"),
		]
		with open(self.documents_paths[0], "w", encoding="utf-8") as f:
			f.write("This is a test file 1")
		with open(self.documents_paths[1], "w", encoding="utf-8") as f:
			f.write("This is a test file 2")

		# fresh store; our RAGStore already clears its collection at init
		self.store = RAGStore(self._tmpdir)

		# swap in a fake embedder to avoid heavy model downloads and make ranking deterministic
		self.store.embedder = _FakeEmbedder()

	def tearDown(self):
		# clean test docs
		for file in self.documents_paths:
			try:
				os.remove(file)
			except FileNotFoundError:
				pass
		# nuke temp chroma dir
		shutil.rmtree(self._tmpdir, ignore_errors=True)

	# ---------------------- tests -------------------------------

	def test_ingest_and_sources_count(self):
		added1 = self.store.add_file_to_store(self.documents_paths[0])
		added2 = self.store.add_file_to_store(self.documents_paths[1])

		# At least one chunk per file (Docling chunking may produce >1)
		self.assertGreaterEqual(added1, 1)
		self.assertGreaterEqual(added2, 1)

		stats = self.store.stats()
		self.assertEqual(stats["sources"], 2)
		self.assertGreaterEqual(stats["chunks"], 2)

	def test_query_top1(self):
		self.store.add_file_to_store(self.documents_paths[0])
		self.store.add_file_to_store(self.documents_paths[1])

		# Deterministic because of _FakeEmbedder
		res = self.store.query("What's in Test2?", n_results=1)
		doc = res["documents"][0][0]
		self.assertIn("This is a test file 2", doc)

	def test_new_prompt_shape_and_order(self):
		self.store.add_file_to_store(self.documents_paths[0])
		self.store.add_file_to_store(self.documents_paths[1])

		prompt = "What's in Test2?"

		# top-1 context: only the "2" file
		contexted_1 = self.store.new_prompt(prompt, 1)
		self.assertEqual(
			contexted_1,
			"From User: What's in Test2?\nContext to base your answer: This is a test file 2"
		)

		# top-2 context: "2" first, then "1" (by our fake embedding distances)
		contexted_2 = self.store.new_prompt(prompt, 2)
		self.assertEqual(
			contexted_2,
			"From User: What's in Test2?\nContext to base your answer: This is a test file 2\nThis is a test file 1"
		)

	def test_dedup_on_reingest(self):
		# first ingest
		first_added = self.store.ingest(self.documents_paths)
		self.assertGreaterEqual(len(first_added), 2)

		# re-ingesting the same files shouldn't add any new chunks (stable hashed IDs)
		second_added = self.store.ingest(self.documents_paths)
		self.assertEqual(len(second_added), 0)

	def test_delete_and_reingest(self):
		self.store.ingest(self.documents_paths)
		stats = self.store.stats()
		self.assertEqual(stats["sources"], 2)

		# delete file 1 chunks
		deleted = self.store.delete_source(self.documents_paths[0])
		self.assertGreater(deleted, 0)
		self.assertEqual(self.store.stats()["sources"], 1)

		# reingest file 1
		re_added = self.store.reingest(self.documents_paths[0])
		self.assertGreater(re_added, 0)
		self.assertEqual(self.store.stats()["sources"], 2)


if __name__ == '__main__':
	unittest.main()
