from .chat_app import ChatApp
from .rag_store import RAGStore

if __name__ == "__main__":
	# storage = RAGStore()
	# storage.ingest(["C:\\Users\\wikto\\Documents\\Vobacom\\Files\\git-cheat-sheet-education.pdf"])
	app = ChatApp("google/gemma-7b-it")
	app.run(debug=True, host="127.0.0.1", port=5000, use_reloader=False)
