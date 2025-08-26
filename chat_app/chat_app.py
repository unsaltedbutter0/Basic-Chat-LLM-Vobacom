# chat_app.py
import logging
from flask import Flask, render_template, request, jsonify
from user_agents import parse
from .llm_handler import LLMHandler
from .rag_store import RAGStore
from .rag_retriever import RAGRetriever
from .scanner import Scanner


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
            llm_response = self.llm.chat_next(user_message)
            return jsonify({
                'message': {
                    'format': 'markdown',
                    'content': llm_response,
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
            user_message = request.json['message']
            logger.info("RAG chat message received: %s", user_message)

            messages, sources = self.rag.build_messages_hybrid(user_message, top_k=3)
            logger.info("Built RAG messages with %d sources", len(sources))

            # Stateless RAG turn: reset history so prior chit-chat doesn't leak
            llm_response = self.llm.chat_messages(messages, reset=True)
            return jsonify({
                'message': {
                    'format': 'markdown',
                    'content': llm_response,
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


if __name__ == '__main__':
    chat_app = ChatApp("NousResearch/Hermes-3-Llama-3.1-8B")
    chat_app.run(debug=True, use_reloader=False)

