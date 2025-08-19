# tests/test_scanner.py
import unittest, tempfile, shutil
from pathlib import Path
from chat_app.scanner import Scanner

class TestScanner(unittest.TestCase):
	def setUp(self):
		self.tmp = tempfile.mkdtemp(prefix="scan_")

		Path(self.tmp, "keep").mkdir()
		Path(self.tmp, "skipme").mkdir()
		Path(self.tmp, ".hidden").mkdir()

		Path(self.tmp, "keep", "doc1.pdf").write_text("dummy")
		Path(self.tmp, "keep", "notes.md").write_text("dummy")
		Path(self.tmp, "skipme", "img.png").write_text("dummy")
		Path(self.tmp, ".hidden", "secret.docx").write_text("dummy")

		Path(self.tmp, "root1.txt").write_text("dummy")
		Path(self.tmp, "root2.pdf").write_text("dummy")

	def tearDown(self):
		shutil.rmtree(self.tmp)

	def test_scan_shallow(self):
		sc = Scanner(skip_dirs={"skipme"})
		found = sc.scan(self.tmp, recursively=False)

		# Should include the two root files
		self.assertIn(str(Path(self.tmp, "root1.txt").resolve()), found)
		self.assertIn(str(Path(self.tmp, "root2.pdf").resolve()), found)

		# Should not include nested ones (keep/ or skipme/)
		self.assertNotIn(str(Path(self.tmp, "keep", "doc1.pdf").resolve()), found)
		self.assertNotIn(str(Path(self.tmp, "skipme", "img.png").resolve()), found)

	def test_scan_recursive(self):
		sc = Scanner(skip_dirs={"skipme"})
		found = sc.scan(self.tmp, recursively=True)

		# Should include root + keep/
		self.assertIn(str(Path(self.tmp, "root1.txt").resolve()), found)
		self.assertIn(str(Path(self.tmp, "keep", "doc1.pdf").resolve()), found)

		# Should skip skipme/
		self.assertNotIn(str(Path(self.tmp, "skipme", "img.png").resolve()), found)

if __name__ == "__main__":
	unittest.main()
