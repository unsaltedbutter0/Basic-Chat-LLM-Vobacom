# chat_app/__init__.py
__all__ = ["ChatApp", "LLMHandler", "RAGStore", "RAGRetriever", "Embedder"]

def __getattr__(name):
	if name == "ChatApp":
		from .chat_app import ChatApp
		return ChatApp
	if name == "LLMHandler":
		from .llm_handler import LLMHandler
		return LLMHandler
	if name == "RAGStore":
		from .rag_store import RAGStore
		return RAGStore
	if name == "RAGRetriever":
		from .rag_store import RAGRetriever
		return RAGRetriever
	if name == "Embedder":
		from .embedder import Embedder
		return Embedder
	raise AttributeError(name)
