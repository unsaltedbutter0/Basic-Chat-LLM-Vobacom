import os
import unittest
from unittest.mock import patch, MagicMock
import torch

class _ProcCallResult(dict):
	def to(self, *_args, **_kwargs):
		return self

class TestVisionCaptioner(unittest.TestCase):
	@patch("chat_app.vision_captioner.Image")
	@patch("chat_app.vision_captioner.LlavaForConditionalGeneration")
	@patch("chat_app.vision_captioner.AutoProcessor")
	def test_caption_returns_string_only(self, mock_processor_cls, mock_model_cls, mock_image_cls):
		# Mock image
		mock_img = MagicMock()
		mock_image_cls.open.return_value = mock_img
		mock_img.convert.return_value = mock_img

		# Mock processor
		mock_processor = MagicMock()
		mock_processor.batch_decode.return_value = ["A cat sitting on a chair."]
		# When the processor is called, it should return a tensor dict with attribute 'to'
		mock_processor.return_value = _ProcCallResult({"input_ids": torch.tensor([[1, 2, 3]])})
		mock_processor_cls.from_pretrained.return_value = mock_processor

		# Mock model
		mock_model = MagicMock()
		mock_model.device = "cpu"
		mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
		mock_model.eval.return_value = mock_model
		mock_model_cls.from_pretrained.return_value = mock_model

		from chat_app.vision_captioner import VisionCaptioner
		captioner = VisionCaptioner("llava-hf/llava-1.5-7b-hf")
		result = captioner.caption("fake_path.jpg")

		self.assertIsInstance(result, str)
		self.assertEqual(result, "A cat sitting on a chair.")

if __name__ == "__main__":
	unittest.main()
