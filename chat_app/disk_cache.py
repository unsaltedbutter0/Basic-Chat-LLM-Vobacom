# chat_app/disk_cache.py
import json, os, time, tempfile
from hashlib import sha1
from pathlib import Path
from typing import Optional, Tuple, Union
from .settings import load_settings

class DiskCache:
	"""
	Simple on-disk cache for responses.
	- One file per entry: first line = JSON meta, remaining = payload.
	- LRU-ish purge by mtime when size exceeds threshold.
	"""
	def __init__(self, type: str = "text", cache_folder_path: Optional[str] = None, max_size_Gb: int = 2):
		match type:
			case "text":
				self.ext = ".txt"
			case "json":
				self.ext = ".json"
			case "npy":
				self.ext = ".npy"
			case _:
				raise Exception("TypeError: provided type not supported.\nTry: [text, json, npy]")

		self.type_name = type

		if not cache_folder_path:
			cfg = load_settings()
			cache_folder_path = cfg.paths.cache_dir

		self.root = Path(cache_folder_path)
		self.root.mkdir(parents=True, exist_ok=True)
		self.type_dir = self.root / type
		self.type_dir.mkdir(parents=True, exist_ok=True)
		self.type_meta = self.type_dir / "meta.json"

		# Initialize or read meta.json
		if not self.type_meta.exists():
			self._approx_bytes = self._count_bytes()
			with self.type_meta.open("w", encoding="utf-8") as f:
				meta = {"cache_size": self._approx_bytes, "cache_type": self.type_name}
				f.write(json.dumps(meta))
		else:
			try:
				with self.type_meta.open("r", encoding="utf-8") as f:
					header = f.readline().strip()
					if header:
						meta = json.loads(header)
						self._approx_bytes = int(meta.get("cache_size", 0))
					else:
						self._approx_bytes = self._count_bytes()
			except Exception:
				self._approx_bytes = self._count_bytes()

		self.max_bytes = int(max_size_Gb * 1024**3)

	def add(self, key: str, value: str, *, ttl: float | None = None, extra_meta: dict | None = None) -> str:
		"""
		Add value under key. Returns the hashed key actually used for the filename.
		"""
		raw_key = key
		key = self._normalize_key(key)

		hashed_key = key if self._is_sha1(key) else self._sha1(key)
		p = self._path_for(hashed_key)
		p.parent.mkdir(parents=True, exist_ok=True)

		meta = {
			"created": time.time(),
			"last": time.time(),
			"ttl": ttl,
			"orig_key": raw_key,
			"extra_meta": extra_meta,
		}

		tmp = tempfile.NamedTemporaryFile("w", delete=False, dir=p.parent, encoding="utf-8")
		try:
			tmp.write(json.dumps(meta) + "\n")
			tmp.write(value)
			tmp.flush()
			os.fsync(tmp.fileno())
		finally:
			tmp.close()
		Path(tmp.name).replace(p)

		# Keep approximate size fresh
		try:
			self._approx_bytes += p.stat().st_size
		except Exception:
			self._approx_bytes = self._count_bytes()

		if self._should_purge(high=0.9):
			self.purge_size(low=0.7)

		return hashed_key

	def get(self, key: str, get_extra: bool = False) -> Union[str, None, Tuple[str, dict]]:
		"""
		Get cached value by key. When get_extra=True, returns (value, extra_meta).
		"""
		key = self._normalize_key(key)
		hashed_key = key if self._is_sha1(key) else self._sha1(key)
		p = self._path_for(hashed_key)
		if not p.exists():
			return None

		with p.open("r", encoding="utf-8") as f:
			header = f.readline()
			if not header:
				return None
			meta = json.loads(header)
			now = time.time()
			if meta.get("ttl") is not None and now > meta["created"] + float(meta["ttl"]):
				return None
			meta["last"] = now
			rest = f.read()

		# Update last access in-place
		try:
			with p.open("r+", encoding="utf-8") as f2:
				payload = json.dumps(meta) + "\n" + rest
				f2.seek(0)
				f2.write(payload)
				f2.truncate()
		except Exception:
			pass

		if get_extra:
			return (rest, meta.get("extra_meta"))
		return rest

	def purge_size(self, low: float = 0.7) -> None:
		"""
		Purge oldest files until total size <= low * max_bytes.
		"""
		target = int(self.max_bytes * low)
		total = getattr(self, "_approx_bytes", None)
		if total is None:
			total = self._count_bytes()

		files = []
		for p in self.type_dir.rglob("*" + self.ext):
			try:
				st = p.stat()
				files.append((st.st_mtime, st.st_size, p))
			except Exception:
				continue

		files.sort(key=lambda t: t[0])  # oldest first

		for _, size, path in files:
			try:
				path.unlink()
				total -= size
				if total <= target:
					break
			except Exception:
				continue

		self._approx_bytes = max(int(total), 0)

	# ---------- helpers ----------
	def _normalize_key(self, text: str) -> str:
		return " ".join((text or "").strip().split()).lower()

	def _count_bytes(self) -> int:
		total = 0
		for p in self.type_dir.rglob("*" + self.ext):
			try:
				total += p.stat().st_size
			except Exception:
				pass
		return int(total)

	def _should_purge(self, high: float = 0.9) -> bool:
		try:
			return self._approx_bytes >= (high * self.max_bytes)
		except Exception:
			return False

	def _sha1(self, s: str) -> str:
		return sha1(s.encode("utf-8")).hexdigest()

	def _is_sha1(self, maybe_sha: str) -> bool:
		if len(maybe_sha) != 40:
			return False
		try:
			int(maybe_sha, 16)
		except ValueError:
			return False
		return True

	def _path_for(self, key_hash: str) -> Path:
		return self.type_dir / key_hash[:2] / key_hash[2:4] / (key_hash + self.ext)

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc, tb):
		# Persist the approximate size back to meta.json
		try:
			with self.type_meta.open("w", encoding="utf-8") as f:
				meta = {"cache_size": int(getattr(self, "_approx_bytes", 0)), "cache_type": self.type_name}
				f.write(json.dumps(meta))
		except Exception:
			pass
