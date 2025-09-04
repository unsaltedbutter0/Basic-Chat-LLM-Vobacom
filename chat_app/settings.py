# settings.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional

try:
    import tomllib as toml  # py311+
except ModuleNotFoundError:
    import tomli as toml  # type: ignore

try:
    import tomli_w
except Exception:
    tomli_w = None  # type: ignore


_DEFAULT_PATH = os.path.join("config", "settings.toml")
_ENV_PREFIX = "VOBA"  # change if you like


@dataclass
class ModelCfg:
    model_id: str = "NousResearch/Hermes-3-Llama-3.1-8B"
    provider: str = "NousResearch"
    model_name: str = "Hermes-3-Llama-3.1-8B"


@dataclass
class EmbeddingsCfg:
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    provider: str = "sentence-transformers"
    model_name: str = "all-MiniLM-L6-v2"


@dataclass
class VectorStoreCfg:
    persist_dir: str = "./chroma_db"
    collection: str = "documents"


@dataclass
class DataDirCfg:
    path: str
    recursive: bool = False


@dataclass
class PathsCfg:

    cache_dir: str = "cache"
    secret_dirs: list[str] = field(default_factory=lambda: [
                                   "private", "private", "secrets", "secrets", ".ssh", ".ssh"])
    data_dirs: list[DataDirCfg] = field(
        default_factory=lambda: [DataDirCfg(path="./data", recursive=False)])

      
@dataclass
class AppCfg:
    env: str = "dev"
    port: int = 8000
    host: str = "127.0.0.1"
    max_context: int = 3


@dataclass
class GuardrailsCfg:
    BLOCK_PRIVATE: bool = False
    ALLOW_ONLY_TECH: bool = False


@dataclass
class Settings:

    app: AppCfg = field(default_factory=AppCfg)
    paths: PathsCfg = field(default_factory=PathsCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    embeddings: EmbeddingsCfg = field(default_factory=EmbeddingsCfg)
    vectorstore: VectorStoreCfg = field(default_factory=VectorStoreCfg)
    guardrails: GuardrailsCfg = field(default_factory=GuardrailsCfg)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "app": asdict(self.app),
            "paths": asdict(self.paths),
            "model": asdict(self.model),
            "embeddings": asdict(self.embeddings),
            "vectorstore": asdict(self.vectorstore),
            "guardrails": asdict(self.guardrails),
        }


def _dict_to_settings(d: Dict[str, Any]) -> Settings:
    # minimal "validation": ensure sections/keys exist; fill defaults if missing
    def get(section: str, defaults: Dict[str, Any]) -> Dict[str, Any]:
        blob = d.get(section, {})
        out = dict(defaults)
        out.update(blob)
        return out

    paths_dict = get("paths", asdict(PathsCfg()))
    raw_dirs = paths_dict.get("data_dirs", [])
    norm_dirs = []
    for item in raw_dirs:
        if isinstance(item, str):
            norm_dirs.append(DataDirCfg(path=item, recursive=False))
        elif isinstance(item, dict):
            path = item.get("path")
            if path:
                norm_dirs.append(DataDirCfg(
                    path=path, recursive=bool(item.get("recursive", False))))
    paths_dict["data_dirs"] = norm_dirs

    return Settings(
        app=AppCfg(**get("app", asdict(AppCfg()))),
        paths=PathsCfg(**paths_dict),
        model=ModelCfg(**get("model", asdict(ModelCfg()))),
        embeddings=EmbeddingsCfg(**get("embeddings", asdict(EmbeddingsCfg()))),
        vectorstore=VectorStoreCfg(
            **get("vectorstore", asdict(VectorStoreCfg()))),
        guardrails=GuardrailsCfg(**get("guardrails", asdict(GuardrailsCfg())))
    )


def merge_settings(cfg: Settings, patch: Dict[str, Any]) -> Settings:
    """Merge a patch dict into an existing Settings instance."""

    def _merge_dict(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                _merge_dict(dst[k], v)
            else:
                dst[k] = v

    current = cfg.to_dict()
    _merge_dict(current, patch)
    return _dict_to_settings(current)


def load_settings(path: Optional[str] = None) -> Settings:
        if path is None:
                path = os.environ.get(f"{_ENV_PREFIX}_CONFIG") or _DEFAULT_PATH

        json_path = os.path.splitext(path)[0] + ".json"
        ext = os.path.splitext(path)[1].lower()

        if os.path.exists(path):
                if ext == ".json":
                        with open(path, "r", encoding="utf-8") as f:
                                raw = json.load(f)
                else:
                        with open(path, "rb") as f:
                                raw = toml.load(f)
        elif os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                        raw = json.load(f)
        else:
                # ensure dir exists and write defaults
                os.makedirs(os.path.dirname(path), exist_ok=True)
                save_settings(Settings(), path)
                return load_settings(path)

        raw = _env_override(raw)
        return _dict_to_settings(raw)


def save_settings(s: Settings|dict, path: Optional[str] = None) -> None:
	if isinstance(s, dict):
		s = _dict_to_settings(s)
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


# for later maybe


def _env_override(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Allow env overrides like:
    VOBA_APP_PORT=9001
    VOBA_VECTORSTORE_PERSIST_DIR=/mnt/chroma
    """
    out = json.loads(json.dumps(d))	 # deep copy
    for section, section_dict in d.items():
        for key, val in section_dict.items():
            env_key = f"{_ENV_PREFIX}_{section}_{key}".upper()
            if env_key in os.environ:
                raw = os.environ[env_key]
                if isinstance(val, bool):
                    out[section][key] = raw.lower() in (
                        "1", "true", "yes", "on")
                elif isinstance(val, int):
                    out[section][key] = int(raw)
                else:
                    out[section][key] = raw
    return out
