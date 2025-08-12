from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# use google/gemma-7b-it

class LLM_handler():

	def __init__(self, model_id):

		self.tokenizer = AutoTokenizer.from_pretrained(model_id)

		self.model = AutoModelForCausalLM.from_pretrained(
			model_id,
			device_map="auto",
			offload_folder="offload",
			# load_in_4bit=True,	# 4bit quantization
			torch_dtype=torch.float16
		)

		self.device = next(self.model.parameters()).device

		self.conversation = []

	def chat_next(self, prompt):
		self.conversation.append({"role": "user", "content": prompt})

		input_ids = self.tokenizer.apply_chat_template(
			self.conversation,
			add_generation_prompt=True,
			tokenize=True,
			return_dict=True,
			return_tensors="pt",
		)
		input_ids = {k: v.to(self.device) for k, v in input_ids.items()}

		outputs = self.model.generate(**input_ids, max_new_tokens=100, eos_token_id=self.tokenizer.eos_token_id)
		reply = self.tokenizer.decode(outputs[0][input_ids["input_ids"].shape[-1]:])

		eos_pos = reply.find(self.tokenizer.eos_token)
		if eos_pos != -1:
			reply = reply[:eos_pos]

		self.conversation.append({"role": "assistant", "content": reply})

		return reply

gemma = LLM_handler("google/gemma-7b-it")

while True:
	user_text = input("You: ")
	print
	if user_text.lower() in ["end", "exit"]:
		print("\nConversation ended.")
		break
	print("Chat: " + gemma.chat_next(user_text) + "\n")

