import torch
import os
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


class VisionCaptioner:
	def __init__(
		self,
		model_id: str = "llava-hf/llava-1.5-7b-hf",
		torch_dtype: torch.dtype = torch.float16,
		device_map: str | dict | None = "auto",
		max_new_tokens: int = 64,
	):
		self.model_id = model_id
		self.max_new_tokens = int(max_new_tokens)
		self.processor = AutoProcessor.from_pretrained(self.model_id)
		self.model = LlavaForConditionalGeneration.from_pretrained(
			self.model_id,
			torch_dtype=torch_dtype,
			device_map=device_map,
		)

		self._device = getattr(self.model, "device", torch.device("cpu"))

	def caption(self, image, prompt: str = None) -> str:
		if isinstance(image, (str, os.PathLike)):
			img = Image.open(image)
		elif isinstance(image, Image.Image):
			img = image
		else:
			raise ValueError("image must be a PIL.Image or a path")

		img_rgb = img.convert("RGB")

		instruction = (
			prompt
			or "Provide a brief, faithful caption of this image. Be specific."
		)

		messages = [
			{
				"role": "user",
				"content": [
					{"type": "text", "text": instruction},
					{"type": "image"},
				],
			}
		]

		chat = self.processor.apply_chat_template(
			messages, add_generation_prompt=True
		)

		inputs = self.processor(
			images=img_rgb,
			text=chat,
			return_tensors="pt",
		).to(self._device)

		with torch.inference_mode():
			gen_ids = self.model.generate(
				**inputs,
				max_new_tokens=self.max_new_tokens,
				do_sample=False,
			)

		# Decode and post-process to return only the assistant text
		text = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
		# Heuristic: if template markers remain, strip up to the last 'ASSISTANT:'
		lower = text.lower()
		marker = "assistant:"
		if marker in lower:
			idx = lower.rfind(marker)
			text = text[idx + len(marker):]

		return text.strip()


__all__ = ["VisionCaptioner"]
