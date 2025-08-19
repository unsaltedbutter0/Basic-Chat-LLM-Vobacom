from flask import Flask, render_template, request, jsonify
from user_agents import parse
from .llm_handler import LLMHandler
from .rag_store import RAGStore

class ChatApp:
	def __init__(self, model_id):
		self.app = Flask(__name__)
		self.llm = LLMHandler(model_id)
		self.rag = RAGStore()

		# self.rag.add_file_to_store("C:\\Users\\wikto\\Documents\\Vobacom\\ChatVobacom\\What is RAG (Retrieval Augmented Generation)_ _ IBM.html")

		# Register routes
		self.app.add_url_rule('/', view_func=self.index, methods=['GET'])
		self.app.add_url_rule('/chat', view_func=self.chat, methods=['POST'])
		self.app.add_url_rule('/rag', view_func=self.rag_chat, methods=['POST'])

	def index(self):
		try:
			user_agent_string = request.headers.get('User-Agent', '')
			user_agent = parse(user_agent_string)

			if user_agent.is_mobile:
				return render_template("index_mobile.html")
			else :
				return render_template("index_desktop.html")
		except Exception as e:
			return jsonify({'error': str(e)})

	def chat(self):
		try:
			user_message = request.json['message']
			llm_response = self.llm.chat_next(user_message)
			return jsonify({'response': llm_response}), 200
		except Exception as e:
			return jsonify({'error': str(e)})

	def rag_chat(self):
		try:
			user_message = request.json['message']
			# do smth about sources
			user_message, sources = self.rag.new_prompt_and_sources(user_message)
			llm_response = self.llm.chat_next(user_message)
			return jsonify({'response': llm_response})
		except Exception as e:
			return jsonify({'error': str(e)})

	def run(self, **kwargs):
		self.app.run(**kwargs)

if __name__ == '__main__':
	chat_app = ChatApp("google/gemma-7b-it")
	chat_app.run(debug=True, use_reloader=False)
