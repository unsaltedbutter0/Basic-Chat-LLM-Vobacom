# supported formats:
# PDF	
# DOCX, XLSX, PPTX
# Markdown	
# AsciiDoc	
# HTML, XHTML	
# CSV	
# PNG, JPEG, TIFF, BMP, WEBP	
# scanner.py
from pathlib import Path
import os

class Scanner:
	def __init__(
		self,
		supported_extensions=None,
		skip_dirs=None,
		follow_symlinks=False
	):
		# Docling-friendly defaults
		self.SUPPORTED_EXTENSIONS = set(
			ext.lower() for ext in (
				supported_extensions
				or [
					".pdf",
					".docx", ".xlsx", ".pptx",
					".md", ".markdown",
					".adoc", ".asciidoc",
					".html", ".htm", ".xhtml",
					".csv",
					".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp",
					".xml", ".json", ".txt", ".log", ".yaml", ".yml"
				]
			)
		)
		self.SKIP_DIRS = set(skip_dirs or [])
		self.follow_symlinks = bool(follow_symlinks)

	def scan(
		self,
		folder_path: str,
		recursively: bool=False,
		include_hidden: bool=False
	) -> list[str]:
		"""
		Return a list of absolute file paths under `folder_path` that match supported extensions.
		- Skips any path containing a directory listed in self.SKIP_DIRS
		- Optionally recurses
		- Optionally includes dotfiles/hidden paths
		"""
		root = Path(folder_path).resolve()
		if not root.exists():
			raise FileNotFoundError(f"Folder not found: {folder_path}")
		if not root.is_dir():
			raise NotADirectoryError(f"Not a directory: {folder_path}")

		if recursively:
			return self._scan_walk(root, include_hidden)
		else:
			return self._scan_shallow(root, include_hidden)

	def _scan_shallow(self, root: Path, include_hidden: bool) -> list[str]:
		results = []
		for p in root.iterdir():
			# skip directories from SKIP_DIRS
			if p.is_dir():
				continue

			if not include_hidden and self._is_hidden(root, p):
				continue

			# symlink policy
			if p.is_symlink() and not self.follow_symlinks:
				continue

			# extension filter
			if p.suffix.lower() in self.SUPPORTED_EXTENSIONS:
				results.append(str(p.resolve()))
		return sorted(set(results))

	def _scan_walk(self, root: Path, include_hidden: bool) -> list[str]:
		results = []
		for dirpath, dirnames, filenames in os.walk(root, followlinks=self.follow_symlinks):
			dirnames[:] = [
				d for d in dirnames
				if not self._should_skip_dir(Path(dirpath) / d)
			]

			for fname in filenames:
				p = Path(dirpath) / fname

				if not include_hidden and self._is_hidden(root, p):
					continue

				# os.walk followlinks covers dirs only
				try:
					if p.is_symlink() and not self.follow_symlinks:
						continue
				except OSError:
					continue

				if p.suffix.lower() in self.SUPPORTED_EXTENSIONS:
					results.append(str(p.resolve()))
		return sorted(set(results))

	# ----------------- helpers -----------------

	def _should_skip_dir(self, p: Path) -> bool:
		# If any path component matches a configured skip dir (exact match), skip
		parts = set(p.resolve().parts)
		return any(skip in parts for skip in self.SKIP_DIRS)

	def _is_hidden(self, root: Path, p: Path) -> bool:
		# A path is considered hidden if any relative component starts with a dot.
		try:
			rel = p.resolve().relative_to(root.resolve())
		except Exception:
			# If not under root, treat cautiously as hidden
			return True
		return any(part.startswith(".") for part in rel.parts)
