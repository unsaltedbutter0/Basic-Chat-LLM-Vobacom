from .chat_app import ChatApp
from .rag_store import RAGStore
from .scanner import Scanner
from .vision_captioner import VisionCaptioner

if __name__ == "__main__":
	# vc = VisionCaptioner()
	# cap = vc.caption("C:\\Users\\wikto\\Documents\\Vobacom\\Files\\images\\lena.png")
	# vc.unload()
	# with VisionCaptioner(device_map="auto") as vc:
	# 	cap = vc.caption("C:\\Users\\wikto\\Documents\\Vobacom\\Files\\images\\lena.png")

	# print(cap)
	# import torch, gc
	# gc.collect()
	# if torch.cuda.is_available():
	# 	print("allocated MB:", torch.cuda.memory_allocated()//(1024**2))
	# 	print("reserved MB:",  torch.cuda.memory_reserved()//(1024**2))
	# while True:
	# 	continue

	# storage = RAGStore()
	# scanner = Scanner()
	# paths = scanner.scan("C:\\Users\\wikto\\Documents\\Vobacom\\Files")
	# paths = paths + scanner.scan("C:\\Users\\wikto\\Documents\\Vobacom\\Files\\images")
	# storage.ingest(paths)
	app = ChatApp("NousResearch/Hermes-3-Llama-3.1-8B")
	app.run(debug=True, host="127.0.0.1", port=5000, use_reloader=False)