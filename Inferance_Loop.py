from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Setup
tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it")

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-7b-it",
    device_map="auto",
    offload_folder="offload",
    # load_in_4bit=True,	# 4bit quantization
    torch_dtype=torch.float16
)

device = next(model.parameters()).device

conversation = []

# Generate next response
def chat_next(prompt):
	conversation.append({"role": "user", "content": prompt})

	input_ids = tokenizer.apply_chat_template(
	    conversation,
	    add_generation_prompt=True,
	    tokenize=True,
	    return_dict=True,
	    return_tensors="pt",
	)
	input_ids = {k: v.to(device) for k, v in input_ids.items()}

	outputs = model.generate(**input_ids, max_new_tokens=100, eos_token_id=tokenizer.eos_token_id)
	reply = tokenizer.decode(outputs[0][input_ids["input_ids"].shape[-1]:])
	eos_pos = reply.find(tokenizer.eos_token)

	if eos_pos != -1:
		reply = reply[:eos_pos]

	conversation.append({"role": "assistant", "content": reply})

	return reply

while True:
	user_text = input("You: ")
	print
	if user_text.lower() in ["end", "exit"]:
		print("\nConversation ended.")
		break
	print("Chat: " + chat_next(user_text) + "\n")

