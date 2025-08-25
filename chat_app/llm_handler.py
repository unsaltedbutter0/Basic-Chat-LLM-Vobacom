# llm_handler.py
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datetime import datetime
import os, torch, json

class LLMHandler():
	def __init__(self, model_id="NousResearch/Hermes-3-Llama-3.1-8B"):
		self.tokenizer = AutoTokenizer.from_pretrained(model_id)

		bnb_cfg = BitsAndBytesConfig(
			load_in_4bit=True,
			bnb_4bit_use_double_quant=True,
			bnb_4bit_quant_type="nf4",
			bnb_4bit_compute_dtype=torch.bfloat16
		)

		self.model = AutoModelForCausalLM.from_pretrained(
			model_id,
			device_map={"": 0},
			torch_dtype=torch.float16,
			low_cpu_mem_usage=True,
			attn_implementation="sdpa",
			quantization_config=bnb_cfg,
		).eval()

		if self.tokenizer.pad_token_id is None:
			self.tokenizer.pad_token = self.tokenizer.eos_token
		self.model.config.pad_token_id = self.tokenizer.pad_token_id

		self.device = next(self.model.parameters()).device
		self.system_preamble = (
			"You are a concise, helpful assistant. "
			"Answer clearly, avoid speculation, and keep responses brief unless asked."
		)
		self.conversation = [{"role": "system", "content": self.system_preamble}]

		filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
		os.makedirs("conversation_logs", exist_ok=True)
		path = os.path.join("conversation_logs", filename)
		self.convo_log_file = open(path, 'w', encoding='utf-8', newline='\n')
		self.convo_log_file_reset_tag = '######################## Conversation Restarted ########################\n'

	def reset(self):
		self.convo_log_file.write(self.convo_log_file_reset_tag)
		self.convo_log_file.flush()
		self.conversation = [{"role": "system", "content": self.system_preamble}]


	def add_message(self, role: str, content: str):
		self.conversation.append({"role": role, "content": content})
		if self.convo_log_file:
			rec = {"role": role, "content": content}
			self.convo_log_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
			self.convo_log_file.flush()

	def add_user_message(self, message: str):
		self.add_message("user", message)

	def add_assistant_message(self, message: str):
		self.add_message("assistant", message)

	def prepare_inputs(self):
		# Build a single prompt string using the model's chat template
		prompt = self.tokenizer.apply_chat_template(
			self.conversation,
			add_generation_prompt=True,
			tokenize=False,	# return a string, not tensors
		)

		# Tokenize to get a dict
		enc = self.tokenizer(
			prompt,
			return_tensors="pt",
		)

		# Move to the model device
		return {k: v.to(self.device) for k, v in enc.items()}


	def generate_response(self, input_ids):
		return self.model.generate(
			**input_ids,
			max_new_tokens=500,
			eos_token_id=self.tokenizer.eos_token_id
		)

	def format_reply(self, input_ids, generated_response):
		reply = self.tokenizer.decode(generated_response[0][input_ids["input_ids"].shape[-1]:])
		eos_pos = reply.find(self.tokenizer.eos_token)
		if eos_pos != -1:
			reply = reply[:eos_pos]
		return reply

	def ensure_system(self):
		# Make sure a system message exists (used by /chat route)
		if not self.conversation or self.conversation[0].get("role") != "system":
			self.conversation.insert(0, {"role": "system", "content": self.system_preamble})

	def chat_next(self, prompt):
		self.ensure_system()
		self.add_user_message(prompt)
		inputs = self.prepare_inputs()
		response = self.generate_response(inputs)
		reply = self.format_reply(inputs, response)
		self.add_assistant_message(reply)
		return reply

	def chat_messages(self, messages: list[dict], reset: bool = True):
		# For RAG: when reset=True you pass your own system+user,
		# so do NOT inject the default preamble here.
		if reset:
			self.convo_log_file.write(self.convo_log_file_reset_tag)
			self.conversation = []	# start exactly with provided messages
		for m in messages:
			self.add_message(m["role"], m["content"])
		inputs = self.prepare_inputs()
		response = self.generate_response(inputs)
		reply = self.format_reply(inputs, response)
		self.add_assistant_message(reply)
		return reply
