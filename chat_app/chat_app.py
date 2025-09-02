# chat_app.py
import logging
from flask import Flask, render_template, request, jsonify
from user_agents import parse
from .llm_handler import LLMHandler
from .rag_store import RAGStore
from .rag_retriever import RAGRetriever
from .scanner import Scanner
from .guardrails import Guardrails
from .disk_cache import DiskCache

logger = logging.getLogger(__name__)


class ChatApp:
	def __init__(self, model_id):
		if not logging.getLogger().hasHandlers():
			logging.basicConfig(level=logging.INFO)
		self.app = Flask(__name__)
		self.llm = LLMHandler(model_id)
		self.store = RAGStore()
		self.rag = RAGRetriever(self.store)
		self.scanner = Scanner()
		self.guard = Guardrails()
		self.cache = DiskCache("cache")
		logger.info("ChatApp initialized with model %s", model_id)

		self.app.add_url_rule('/', view_func=self.index, methods=['GET'])
		self.app.add_url_rule('/chat', view_func=self.chat, methods=['POST'])
		self.app.add_url_rule('/rag', view_func=self.rag_chat, methods=['POST'])
		self.app.add_url_rule('/ingest', view_func=self.ingest_folder, methods=['POST'])

	def index(self):
		try:
			user_agent_string = request.headers.get('User-Agent', '')
			logger.info("Serving index for user agent: %s", user_agent_string)
			user_agent = parse(user_agent_string)
			if user_agent.is_mobile:
				return render_template("index_mobile.html")
			else:
				return render_template("index_desktop.html")
		except Exception as e:
			logger.exception("Error in index route: %s", e)
			return jsonify({'error': str(e)})

	def chat(self):
		try:
			user_message = request.json['message']
			logger.info("Chat message received: %s", user_message)
			key = self._cache_key(user_message, k=3)
			cached = self.cache.get(key)
			if cached:
				processed_response = cached
			else:
				llm_response = self.llm.chat_next(user_message)
				processed_response = self.guard.post_processing(llm_response) 
				self.cache.add(key, llm_response)
			return jsonify({
				'message': {
					'format': 'markdown',
					'content': processed_response,
				},
				'meta': {
					'mode': 'llm',
				}
			}), 200
		except Exception as e:
			logger.exception("Error in chat route: %s", e)
			return jsonify({'error': str(e)})

	def rag_chat(self):
		try:
			processed_response = None
			user_message = request.json['message']
			logger.info("RAG chat message received: %s", user_message)
			key = self._cache_key(user_message, k=3)

			rec = self.cache.get(key, get_extra=True)
			cached, cache_meta = rec if rec else (None, None)			
			messages, sources, fmt_ids_new = self.rag.build_messages_hybrid(user_message, top_k=3)
			if cached and cache_meta and cache_meta["fmt_ids"]:
				fmt_ids_old = cache_meta["fmt_ids"]
				if self._is_similar_jaccard(fmt_ids_new, fmt_ids_old):
					processed_response = cached
			if not processed_response:
				logger.info("Built RAG messages with %d sources", len(sources))

				# Stateless RAG turn: reset history so prior chit-chat doesn't leak
				llm_response = self.llm.chat_messages(messages, reset=True)
				processed_response = self.guard.post_processing(llm_response) 
				self.cache.add(key, llm_response, extra_meta={"fmt_ids": fmt_ids_new})

			return jsonify({
				'message': {
					'format': 'markdown',
					'content': processed_response,
				},
				'meta': {
					'sources': sources,
					'mode': 'rag',
				}
			}), 200
		except Exception as e:
			logger.exception("Error in rag route: %s", e)
			return jsonify({'error': str(e)})

	def ingest_folder(self):
		try:
			data = request.json or {}
			folder = data.get('folder')
			recursive = bool(data.get('recursive', False))
			if not folder:
				raise ValueError('folder is required')
			files = self.scanner.scan(folder, recursively=recursive)
			added = self.store.ingest(files)
			return jsonify({'files_ingested': len(added)}), 200
		except Exception as e:
			logger.exception("Error in ingest route: %s", e)
			return jsonify({'error': str(e)})

	def run(self, **kwargs):
		logger.info("Starting ChatApp server with args: %s", kwargs)
		self.app.run(**kwargs)

	def _is_similar_jaccard(self, a: set[str], b: set[str], tau: float=0.8) -> bool:
		A, B = set(a or []), set(b or [])
		if not A and not B: 
			return True
		similarity = len(A & B) / len(A | B)
		if similarity >= tau:
			return True
		return False

	def _cache_key(self, question: str, k: int, model_id: str | None = None, prompt_ver: str = "v1") -> str:
		model_id = model_id or getattr(self.llm, "model_id", "unknown")
		return f"{prompt_ver}|{model_id}|k{k}|{qn}"

if __name__ == '__main__':
	chat_app = ChatApp("NousResearch/Hermes-3-Llama-3.1-8B")
	chat_app.run(debug=True, use_reloader=False)

