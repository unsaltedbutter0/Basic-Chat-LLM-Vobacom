import unittest
from unittest.mock import MagicMock
import torch
from Inferance_Loop_with_Classes_Testable import LLM_handler

class TestLLMHandler(unittest.TestCase):
	def setUp(self):
		self.handler = LLM_handler.__new__(LLM_handler)
		
		self.handler.tokenizer = MagicMock()
		self.handler.model = MagicMock()
		self.handler.device = "cpu"
		self.handler.conversation = [{"role": "user", "content": "Hello"}]
		self.handler.tokenizer.eos_token = "<eos>"
		self.handler.tokenizer.eos_token_id = 99

		self.fake_tensor_dict = {
			"input_ids": torch.tensor([[1, 2, 3]]),
			"attention_mask": torch.tensor([[1, 1, 1]])
		}

	def test_add_user_message(self):
		fake_expected_conversation = [
			{"role": "user", "content": "Hello"}, 
			{"role": "user", "content": "Hello?!"}
		]
		self.handler.add_user_message("Hello?!")
		self.assertEqual(self.handler.conversation, fake_expected_conversation)

	def test_add_assistant_message(self):
		fake_expected_conversation = [
			{"role": "user", "content": "Hello"}, 
			{"role": "assistant", "content": "Hi, how can I help you?"}
		]
		self.handler.add_assistant_message("Hi, how can I help you?")
		self.assertEqual(self.handler.conversation, fake_expected_conversation)

	def test_prepare_inputs(self):
		self.handler.tokenizer.apply_chat_template.return_value = self.fake_tensor_dict
		
		inputs = self.handler.prepare_inputs()

		self.handler.tokenizer.apply_chat_template.assert_called_once_with(
			self.handler.conversation,
			add_generation_prompt=True,
			tokenize=True,
			return_dict=True,
			return_tensors="pt",
		)
		for key, tensor in inputs.items():
			self.assertTrue(tensor.device.type == "cpu")

		self.assertSetEqual(set(inputs.keys()), set(self.fake_tensor_dict.keys()))

	def test_generate_response(self):
		fake_output = torch.tensor([[1, 2, 3, 4, 5]])
		self.handler.model.generate.return_value = fake_output

		output = self.handler.generate_response(self.fake_tensor_dict)
		self.assertTrue(torch.equal(output, self.handler.model.generate.return_value))

	def test_format_reply_no_eos(self):
		self.handler.tokenizer.decode.return_value = "Hello World!"

		fake_generated_response = torch.tensor([[1, 2, 3, 4, 5]])
		output = self.handler.format_reply(self.fake_tensor_dict, fake_generated_response)

		self.assertEqual(output, "Hello World!")

	def test_format_reply_with_eos(self):
		self.handler.tokenizer.decode.return_value = "Hello World!<eos>More stuff"

		fake_generated_response = torch.tensor([[1, 2, 3, 4, 5]])
		output = self.handler.format_reply(self.fake_tensor_dict, fake_generated_response)
		
		self.assertEqual(output, "Hello World!")


if __name__ == "__main__":
	unittest.main()
