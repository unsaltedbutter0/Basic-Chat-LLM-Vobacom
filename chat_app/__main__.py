from .chat_app import ChatApp

if __name__ == "__main__":
	app = ChatApp("google/gemma-7b-it")
	app.run(debug=True, host="127.0.0.1", port=5000, use_reloader=False)
