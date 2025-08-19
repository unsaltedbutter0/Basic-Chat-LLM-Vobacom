import unittest
from unittest.mock import patch, MagicMock
from PIL import Image
import torch

from chat_app.vision_captioner import VisionCaptioner

class TestVisionCaptioner(unittest.TestCase):
	@patch("chat_app.vision_captioner.LlavaForConditionalGeneration")
	@patch("chat_app.vision_captioner.AutoProcessor")
	def test_caption_happy_path(self, mock_proc_cls, mock_model_cls):
		mock_proc = MagicMock()
		mock_proc.apply_chat_template.return_value = "CHAT_PROMPT"

		fake_inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
		class _ProcCallResult(dict):
			def to(self, *_args, **_kwargs):
				return fake_inputs
		mock_proc.return_value = _ProcCallResult(fake_inputs)

		mock_proc.batch_decode.return_value = ["ASSISTANT: A cat on a wooden table."]
		mock_proc_cls.from_pretrained.return_value = mock_proc

		mock_model = MagicMock()
		mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
		mock_model.device = torch.device("cpu")
		mock_model_cls.from_pretrained.return_value = mock_model

		vc = VisionCaptioner()

		img = Image.new("RGB", (2, 2), color=(255, 0, 0))

		caption = vc.caption(img)

		self.assertEqual(caption, "A cat on a wooden table.")
		mock_proc.apply_chat_template.assert_called_once()
		mock_model.generate.assert_called_once()
		mock_proc.batch_decode.assert_called_once()

	@patch("chat_app.vision_captioner.LlavaForConditionalGeneration")
	@patch("chat_app.vision_captioner.AutoProcessor")
	def test_caption_uses_custom_prompt(self, mock_proc_cls, mock_model_cls):
		mock_proc = MagicMock()
		mock_proc.apply_chat_template.return_value = "CHAT_PROMPT"

		class _ProcCallResult(dict):
			def to(self, *_args, **_kwargs):
				return {"input_ids": torch.tensor([[1]])}
		mock_proc.return_value = _ProcCallResult({})

		mock_proc.batch_decode.return_value = ["ASSISTANT: Custom prompt respected."]
		mock_proc_cls.from_pretrained.return_value = mock_proc

		mock_model = MagicMock()
		mock_model.generate.return_value = torch.tensor([[1, 2]])
		mock_model.device = torch.device("cpu")
		mock_model_cls.from_pretrained.return_value = mock_model

		vc = VisionCaptioner()
		img = Image.new("RGB", (2, 2))
		_ = vc.caption(img, prompt="Describe briefly.")
		args, kwargs = mock_proc.apply_chat_template.call_args
		messages = args[0]
		self.assertEqual(messages[0]["content"][0]["text"], "Describe briefly.")

	def test_caption_none_image_raises(self):
		vc = VisionCaptioner.__new__(VisionCaptioner)
		with self.assertRaises(ValueError):
			vc.caption(None)


if __name__ == "__main__":
	unittest.main()
