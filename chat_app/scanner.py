# supported formats:
# PDF	
# DOCX, XLSX, PPTX
# Markdown	
# AsciiDoc	
# HTML, XHTML	
# CSV	
# PNG, JPEG, TIFF, BMP, WEBP	
from pathlib import Path

class Scanner():
	def __init__(self, SKIP_DIRS=[]):
		self.SUPPORTED_EXTENSIONS = {
		".pdf",
		".docx",
		".xlsx",
		".pptx",
		".md", ".markdown",
		".adoc", ".asciidoc",
		".html", ".htm", ".xhtml",
		".csv",
		".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"
		}

	def scan(self,  folder_path, recursively=False, SKIP_DIRS=self.SKIP_DIRS):
		found_files = []
		self.folder = Path(folder)

		if self.recursively:
			indexed_files = self.folder.rglob("*")
		else:
			indexed_files = self.folder.glob("*")

		for file in indexed_files:
			if set(file.parts).isdisjoint(self.SKIP_DIRS):
				continue

			if file.suffix in SUPPORTED_EXTENSIONS:
				found_files.append(str(file.resolve()))

		return found_files
