from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import torch
import os

# use google/gemma-7b-it

class LLMHandler():
	def __init__(self, model_id="google/gemma-7b-it"):
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
		
		filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
		path = os.path.join("conversation_logs", filename)
		self.convo_log_file = open(path, 'w')

	def add_user_message(self, message: str):
		self.conversation.append({"role": "user", "content": message})
		self.convo_log_file.write(f'{{"role": "user", "content": "{message}"}}\n')

	def add_assistant_message(self, message: str):
		self.conversation.append({"role": "assistant", "content": message})
		self.convo_log_file.write(f'{{"role": "assistant", "content": "{message}"}}\n')

	def prepare_inputs(self):
		input_ids = self.tokenizer.apply_chat_template(
			self.conversation,
			add_generation_prompt=True,
			tokenize=True,
			return_dict=True,
			return_tensors="pt",
		)
		input_ids = {k: v.to(self.device) for k, v in input_ids.items()}

		return input_ids

	def generate_response(self, input_ids):
		return self.model.generate(**input_ids,
			max_new_tokens=100,
			eos_token_id=self.tokenizer.eos_token_id)
	
	def format_reply(self, input_ids, generated_response):
		reply = self.tokenizer.decode(generated_response[0][input_ids["input_ids"].shape[-1]:])

		eos_pos = reply.find(self.tokenizer.eos_token)
		if eos_pos != -1:
			reply = reply[:eos_pos]
		return reply

	def chat_next(self, prompt):
		self.add_user_message(prompt)

		inputs = self.prepare_inputs()
		response = self.generate_response(inputs)
		reply = self.format_reply(inputs, response)

		self.add_assistant_message(reply)

		return reply

if __name__ == "__main__":

	gemma = LLMHandler("google/gemma-7b-it")

	while True:
		user_text = input("You: ")
		print
		if user_text.lower() in ["end", "exit"]:
			print("\nConversation ended.")
			break
		print("Chat: " + gemma.chat_next(user_text) + "\n")

