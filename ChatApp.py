from flask import Flask, render_template, request, jsonify
from user_agents import parse
from LLM_handler import LLM_handler

class ChatApp:
	def __init__(self, model_id):
		self.app = Flask(__name__)
		self.llm = LLM_handler(model_id)

		# Register routes
		self.app.add_url_rule('/', view_func=self.index, methods=['GET'])
		self.app.add_url_rule('/chat', view_func=self.chat, methods=['POST'])

	def index(self):
		user_agent_string = request.headers.get('User-Agent', '')
		user_agent = parse(user_agent_string)

		if user_agent.is_mobile:
			return render_template("index_mobile.html")
		else :
			return render_template("index_desktop.html")

	def chat(self):
		try:
			user_message = request.json['message']
			llm_response = self.llm.chat_next(user_message)
			return jsonify({'response': llm_response})
		except Exception as e:
			return jsonify({'error': str(e)})

	def run(self, **kwargs):
		self.app.run(**kwargs)


if __name__ == '__main__':
	chat_app = ChatApp("google/gemma-7b-it")
	chat_app.run(debug=True, use_reloader=False)
