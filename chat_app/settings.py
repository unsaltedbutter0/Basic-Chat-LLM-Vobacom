# settings.py
# Tabs for indentation only.

from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

try:
	import tomllib as toml  # py311+
except ModuleNotFoundError:
	import tomli as toml  # type: ignore

try:
	import tomli_w  # tiny writer; if unavailable we fall back to JSON
except Exception:
	tomli_w = None  # type: ignore


_DEFAULT_PATH = os.path.join("config", "settings.toml")
_ENV_PREFIX = "VOBA"  # change if you like


@dataclass
class ModelCfg:
	provider: str = "openai"
	model_name: str = "gpt-4o-mini"

@dataclass
class EmbeddingsCfg:
	provider: str = "openai"
	dim: int = 1536

@dataclass
class VectorStoreCfg:
	backend: str = "chroma"
	persist_dir: str = "./chroma_store"
	collection: str = "vobachat"

@dataclass
class PathsCfg:
	data_dir: str = "./data"
	temp_dir: str = "./tmp"

@dataclass
class AppCfg:
	env: str = "dev"
	port: int = 8000
	host: str = "127.0.0.1"

@dataclass
class Settings:
	app: AppCfg = AppCfg()
	paths: PathsCfg = PathsCfg()
	model: ModelCfg = ModelCfg()
	embeddings: EmbeddingsCfg = EmbeddingsCfg()
	vectorstore: VectorStoreCfg = VectorStoreCfg()

	def to_dict(self) -> Dict[str, Any]:
		return {
			"app": asdict(self.app),
			"paths": asdict(self.paths),
			"model": asdict(self.model),
			"embeddings": asdict(self.embeddings),
			"vectorstore": asdict(self.vectorstore),
		}


def _env_override(d: Dict[str, Any]) -> Dict[str, Any]:
	"""
	Allow env overrides like:
	VOBA_APP_PORT=9001
	VOBA_VECTORSTORE_PERSIST_DIR=/mnt/chroma
	"""
	out = json.loads(json.dumps(d))  # deep copy
	for section, section_dict in d.items():
		for key, val in section_dict.items():
			env_key = f"{_ENV_PREFIX}_{section}_{key}".upper()
			if env_key in os.environ:
				raw = os.environ[env_key]
				if isinstance(val, bool):
					out[section][key] = raw.lower() in ("1", "true", "yes", "on")
				elif isinstance(val, int):
					out[section][key] = int(raw)
				else:
					out[section][key] = raw
	return out


def _dict_to_settings(d: Dict[str, Any]) -> Settings:
	# minimal "validation": ensure sections/keys exist; fill defaults if missing
	def get(section: str, defaults: Dict[str, Any]) -> Dict[str, Any]:
		blob = d.get(section, {})
		out = dict(defaults); out.update(blob)
		return out

	return Settings(
		app=AppCfg(**get("app", asdict(AppCfg()))),
		paths=PathsCfg(**get("paths", asdict(PathsCfg()))),
		model=ModelCfg(**get("model", asdict(ModelCfg()))),
		embeddings=EmbeddingsCfg(**get("embeddings", asdict(EmbeddingsCfg()))),
		vectorstore=VectorStoreCfg(**get("vectorstore", asdict(VectorStoreCfg()))),
	)


def load_settings(path: Optional[str] = None) -> Settings:
	if path is None:
		path = os.environ.get(f"{_ENV_PREFIX}_CONFIG") or _DEFAULT_PATH

	if not os.path.exists(path):
		# ensure dir exists and write defaults
		os.makedirs(os.path.dirname(path), exist_ok=True)
		save_settings(Settings(), path)
		return Settings()

	with open(path, "rb") as f:
		raw = toml.load(f)

	raw = _env_override(raw)
	return _dict_to_settings(raw)


def save_settings(s: Settings, path: Optional[str] = None) -> None:
	if path is None:
		path = os.environ.get(f"{_ENV_PREFIX}_CONFIG") or _DEFAULT_PATH
	os.makedirs(os.path.dirname(path), exist_ok=True)

	data = s.to_dict()
	if tomli_w:
		with open(path, "wb") as f:
			f.write(tomli_w.dumps(data).encode("utf-8"))
	else:
		# fallback to json if tomli_w not installed
		json_path = os.path.splitext(path)[0] + ".json"
		with open(json_path, "w", encoding="utf-8") as f:
			json.dump(data, f, indent=2)
