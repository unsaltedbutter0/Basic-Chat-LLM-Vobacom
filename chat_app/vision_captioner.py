# chat_app/vision_captioner.py
import os, gc, torch, contextlib
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
		self.model.eval()

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
			# Let Accelerate/hf hooks place tensors; don't force .to(device)
			return self.processor(images=img_for_encoder, text=chat, return_tensors="pt")

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

		# Explicitly drop large temporary tensors before returning
		with contextlib.suppress(Exception):
			del inputs, gen_ids

		return text.strip()

	def _has_accelerate_hooks(self) -> bool:
		m = getattr(self, "model", None)
		return m is not None and (
			hasattr(m, "hf_device_map")
			or hasattr(m, "is_loaded_in_8bit")
			or hasattr(m, "is_loaded_in_4bit")
		)

	def unload(self, *, clear_hf_cache: bool = False) -> None:
		"""
		Free VLM memory safely. Avoid .to('cpu') when Accelerate/quantization hooks are present.
		Optionally clear the local Hugging Face cache if ``clear_hf_cache=True``.
		"""
		# 1) Try to move off accelerator only if it's a plain torch model.
		if getattr(self, "model", None) is not None and not self._has_accelerate_hooks():
			with contextlib.suppress(Exception):
				self.model.to("cpu")

		# 2) Break internal references on the processor to help GC
		proc = getattr(self, "processor", None)
		if proc is not None:
			with contextlib.suppress(Exception):
				if hasattr(proc, "image_processor"):
					proc.image_processor = None
			with contextlib.suppress(Exception):
				if hasattr(proc, "tokenizer"):
					proc.tokenizer = None

		# 3) Drop strong refs

		with contextlib.suppress(Exception):
			_force_model_to_cpu(self.model)
		del self.model
		self.model = None
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.synchronize()
			torch.cuda.empty_cache()
			torch.cuda.ipc_collect()
			torch.cuda.reset_peak_memory_stats()
			
		if hasattr(torch, "mps") and torch.backends.mps.is_available():  # Apple Silicon
			with contextlib.suppress(Exception):
				torch.mps.empty_cache()

		# 5) Optional: clear local HF cache (disk), not required for RAM/VRAM
		if clear_hf_cache:
			with contextlib.suppress(Exception):
				from huggingface_hub import scan_cache_dir, delete_cache_entries
				info = scan_cache_dir()
				delete_cache_entries(info.references())

	def _force_model_to_cpu(m):
		for p in m.parameters(recurse=True):
			if hasattr(p, "data") and p.device.type == "cuda":
				p.data = p.data.cpu()
		for b in m.buffers(recurse=True):
			if hasattr(b, "data") and b.device.type == "cuda":
				b.data = b.data.cpu()


	def __enter__(self):
		return self

	def __del__(self):
		# Ensure resources are released if the object falls out of scope
		with contextlib.suppress(Exception):
			self.unload()

	def __exit__(self, exc_type, exc, tb):
		self.unload()
		return False


__all__ = ["VisionCaptioner"]
