# chat_app/guardrails.py
from typing import List, Dict, Tuple, Optional
import math, os, re
from .settings import load_settings

SUSPICIOUS_PATTERNS = (
    "ignore previous instructions",
    "please ignore everything above",
    "disregard the system prompt",
    "extract the system prompt",
    "reveal system instructions",
    "as the system administrator",
    "reply with 'your security code'",
    "your api key is",
    "if this is true, output",
    "answer.*but also print",
)

_TECH_SCI_KEYWORDS = (
	"api","sdk","python","java","c++","c#","javascript","typescript","sql","linux","docker","kubernetes","k8s",
	"linux","bash","zsh","git","ci/cd","devops","ml","ai","nn","nlp","cv","tensor","pytorch","transformer",
	"gpu","cuda","memory","latency","throughput","benchmark","regex","http","rest","grpc","json","yaml","xml",
	"encryption","hash","jwt","oauth","tls","ssl","cipher","key","token",
	"math","algebra","calculus","statistics","probability","combinatorics","physics","chemistry","biology",
	"embeddings","vector","cosine","semantic","retrieval","index","chroma","docling","ocr","tesseract","space"
)

cfg = load_settings

_SECRET_REGEXES = (
	re.compile(r"\b[A-Za-z0-9_]{16,}\.[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}"),	# jwt-ish
	re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),									# aws access key
	re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),											# api keys (generic)
)
_PII_REGEXES = (
	re.compile(r"\b\d{2,4}[-\s]?\d{2,4}[-\s]?\d{2,4}[-\s]?\d{2,4}\b"),				# generic long numbers
	re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b"),										# emails
	re.compile(r"\b\+?\d{1,3}[-\s]?\(?\d{2,4}\)?[-\s]?\d{3,4}[-\s]?\d{3,4}\b"),		# phones
)

_ADDRESS_REGEXES = (
	re.compile(r"(?i)\b(address|adres)\s*:\s*.+"),  # 'address: ...' / 'adres: ...'
	re.compile(r"(?i)\b(ul\.|al\.)\s+[^\n,]+?\s+\d+[^\n,]*"),  # 'ul. Foobar 12' / 'al. ...'
	re.compile(r"\b\d{2}-\d{3}\b"),  # PL postal code like 00-071
)

class Guardrails():
	"""Simple and lightweight guardrails with relevant chunk selection"""
	def __init__(self, dense_metric: str = "l2", alpha: float = 0.5):
		"""
		dense_metric: 'l2' or 'cosine' (Chroma collection metric)
		alpha: weight for BM25 vs dense in score blending (0..1)
		"""
		self.dense_metric = dense_metric
		self.alpha = alpha

	# ---------- public API ----------

	def normalized_scores(
		self,
		bm25_scores: List[float],
		dense_distances: List[float],
		dense_metric: Optional[str] = None,
		alpha: Optional[float] = None,
	) -> List[float]:
		"""
		Return fused, per-query normalized scores in [0,1], aligned with inputs.
		"""
		dmetric = dense_metric or self.dense_metric
		w = self._clamp01(alpha if alpha is not None else self.alpha)

		# Dense: distances -> similarities (bigger is better)
		dense_sims = self._dense_dist_to_sim(dmetric, dense_distances)

		# BM25: log1p then min-max
		b_norm = self._norm([math.log1p(max(0.0, s)) for s in bm25_scores])

		# Dense: if already in [0,1], keep; else min-max
		d_norm = dense_sims if self._is_01_range(dense_sims) else self._norm(dense_sims)

		# Blend
		n = min(len(b_norm), len(d_norm))
		return [w * b_norm[i] + (1.0 - w) * d_norm[i] for i in range(n)]

	def looks_sus(self, t: str) -> bool:
		text = (t or "").lower()
		return any(pattern in text for pattern in SUSPICIOUS_PATTERNS)

	def post_processing(self, llm_response: str, is_sus: bool, was_redacted: bool) -> str:
		to_add = "\n"
		if was_redacted:
			to_add += "\nWarning!:**Redacted Private Information**"
		if is_sus:
			to_add += "\nWarning!:**Malicous prompt detected**, skiping infected context"
		if not self._has_citation(llm_response):
			to_add += "\n_Note: no source snippets were cited for this answer._"
		if to_add != "\n":
			llm_response += to_add
		return llm_response

	def is_tech_science(self, q: str) -> bool:
		if not cfg().guardrails.ALLOW_ONLY_TECH:
			return True
		if not q or len(q) < 5:
			return False
		lq = q.lower()
		# fast allow if code-ish content present
		if "```" in lq or any(sym in lq for sym in ("{", "}", "();", "=>", "import ", "def ")):
			return True
		# keyword hit ratio
		hits = sum(1 for k in _TECH_SCI_KEYWORDS if k in lq)
		return hits >= 1

	def redact_private(self, text: str) -> str:
		was_redacted = False
		redacted = text
		if not cfg().guardrails.BLOCK_PRIVATE:
			return redacted, was_redacted
		if self._text_has_private_bits(redacted):
			was_redacted = True
			redacted = re.sub(_SECRET_REGEXES[0], "[REDACTED-JWT]", redacted)
			for rx in _SECRET_REGEXES[1:]:
				redacted = rx.sub("[REDACTED-SECRET]", redacted)
			for rx in _PII_REGEXES:
				redacted = rx.sub("[REDACTED-PII]", redacted)
			for rx in _ADDRESS_REGEXES:
				redacted = rx.sub("[REDACTED-ADDRESS]", redacted)
		
		return redacted, was_redacted

	# ---------- helpers ----------
	def _text_has_private_bits(self, text: str) -> bool:
		# TODO: check for private atribute or do it in ret
		if not text:
			return False
		for rx in _SECRET_REGEXES + _PII_REGEXES + _ADDRESS_REGEXES:
			if rx.search(text):
				return True
		return False

	def _has_citation(self, text: str) -> bool:
		return ("[" in text and "#" in text and "]" in text)

	def _dense_dist_to_sim(self, metric: str, distances: List[float]) -> List[float]:
		if metric == "cosine":
			# cosine distance d∈[0,2] -> similarity in [0,1]
			return [self._clip01((1.0 - self._clip(d, 0.0, 2.0) + 1.0) / 2.0) for d in distances]
		if metric == "l2":
			# monotone map to (0,1]; larger is better
			return [1.0 / (1.0 + max(0.0, d)) for d in distances]
		# fallback: invert, will be min-maxed later
		return [-d for d in distances]

	def _norm(self, xs: List[float]) -> List[float]:
		if not xs:
			return []
		lo, hi = min(xs), max(xs)
		span = (hi - lo) or 1e-9
		return [(x - lo) / span for x in xs]

	def _is_01_range(self, xs: List[float]) -> bool:
		if not xs:
			return False
		mn, mx = min(xs), max(xs)
		return -1e-9 <= mn <= 1.0 + 1e-9 and -1e-9 <= mx <= 1.0 + 1e-9

	def _clip(self, x: float, lo: float, hi: float) -> float:
		return lo if x < lo else hi if x > hi else x

	def _clip01(self, x: float) -> float:
		return self._clip(x, 0.0, 1.0)

	def _clamp01(self, x: float) -> float:
		return self._clip01(x)
	