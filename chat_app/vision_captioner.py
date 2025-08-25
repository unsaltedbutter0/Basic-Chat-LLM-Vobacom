import os
import torch
from PIL import Image, ImageOps
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
			low_cpu_mem_usage=True,
		)

		# Fallback single-device hint for token tensors passed to generate(...)
		self._device = "cuda" if torch.cuda.is_available() else "cpu"

	def _prepare_image(self, img: Image.Image, max_side: int = 1536) -> Image.Image:
		# Normalize orientation and color; guard giant images to avoid int overflows
		img = ImageOps.exif_transpose(img).convert("RGB")
		w, h = img.size
		long_side = max(w, h)
		if long_side > max_side:
			scale = max_side / float(long_side)
			new_w = max(1, int(w * scale))
			new_h = max(1, int(h * scale))
			img = img.resize((new_w, new_h), Image.BICUBIC)
		return img

	def caption(self, image, prompt: str = None, max_side: int = 1536, max_new_tokens: int | None = None) -> str:
		if isinstance(image, (str, os.PathLike)):
			img = Image.open(image)
		elif isinstance(image, Image.Image):
			img = image
		else:
			raise ValueError("image must be a PIL.Image or a path")

		img_rgb = self._prepare_image(img, max_side=max_side)

		instruction = prompt or "Provide a brief, faithful caption of this image. Be specific."

		messages = [
			{
				"role": "user",
				"content": [
					{"type": "text", "text": instruction},
					{"type": "image"},
				],
			}
		]

		chat = self.processor.apply_chat_template(messages, add_generation_prompt=True)

		def _encode(img_for_encoder: Image.Image):
			enc = self.processor(images=img_for_encoder, text=chat, return_tensors="pt")
			# tests may mock .to(...); be permissive
			if hasattr(enc, "to"):
				return enc.to(self._device)
			return enc

		use_max_new = int(max_new_tokens) if max_new_tokens is not None else self.max_new_tokens

		with torch.inference_mode():
			try:
				inputs = _encode(img_rgb)
				gen_ids = self.model.generate(
					**inputs,
					max_new_tokens=use_max_new,
					do_sample=False,
				)
			except OverflowError:
				# Retry with a smaller image to avoid "int too big to convert" on some backends
				img_small = self._prepare_image(img_rgb, max_side=1024)
				inputs = _encode(img_small)
				gen_ids = self.model.generate(
					**inputs,
					max_new_tokens=use_max_new,
					do_sample=False,
				)

		# Decode and post-process to return only the assistant text
		text = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
		lower = text.lower()
		marker = "assistant:"
		if marker in lower:
			idx = lower.rfind(marker)
			text = text[idx + len(marker):]

		return text.strip()


__all__ = ["VisionCaptioner"]
