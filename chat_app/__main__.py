from .chat_app import ChatApp
from .rag_store import RAGStore
from .scanner import Scanner

if __name__ == "__main__":
	storage = RAGStore()
	# paths = Scanner().scan("C:\\Users\\wikto\\Documents\\Vobacom\\Files")
	# storage.ingest(paths)
	app = ChatApp("NousResearch/Hermes-3-Llama-3.1-8B")
	app.run(debug=True, host="127.0.0.1", port=5000, use_reloader=False)
