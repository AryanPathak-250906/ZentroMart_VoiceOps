"""Vector backends.

* ``fastembed`` (preferred): a real multilingual sentence-embedding model
  (default ``sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2``, ONNX, CPU,
  ~220 MB, downloaded once). Handles English, Hindi and Hinglish queries against English
  documents.
* ``tfidf`` (fallback only): character n-gram TF-IDF vectors. This is a *lexical-statistical*
  representation, NOT a semantic embedding. It is used only when the embedding model cannot
  be installed or downloaded, and every index / response records which backend was used.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np


class EmbeddingUnavailable(RuntimeError):
    pass


def _l2(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class FastEmbedBackend:
    kind = "fastembed"
    semantic = True

    def __init__(self, model_name: str, cache_dir: Path):
        try:
            from fastembed import TextEmbedding  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise EmbeddingUnavailable("fastembed is not installed (pip install fastembed)") from exc
        try:
            self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))
        except Exception as exc:  # pragma: no cover - network/model download failures
            raise EmbeddingUnavailable(f"could not load embedding model {model_name!r}: {exc}") from exc
        self.model_name = model_name

    @property
    def description(self) -> str:
        return f"fastembed:{self.model_name} (semantic multilingual embedding)"

    def fit(self, texts: list[str]) -> None:  # stateless
        return None

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.array([np.asarray(v, dtype=np.float32) for v in self._model.embed(texts)])
        return _l2(vecs)

    def state(self) -> dict:
        return {"model_name": self.model_name}


class TfidfCharBackend:
    """Character 3–5-gram TF-IDF (word-boundary aware). Pure numpy; deterministic."""

    kind = "tfidf"
    semantic = False

    def __init__(self, ngram: tuple[int, int] = (3, 5)):
        self.ngram = ngram
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.zeros(0, dtype=np.float32)

    @property
    def description(self) -> str:
        return "tfidf-char-ngram (LEXICAL FALLBACK — not a semantic embedding)"

    def _grams(self, text: str) -> Counter:
        text = unicodedata.normalize("NFKC", text).lower()
        grams: Counter = Counter()
        for word in re.findall(r"[\wऀ-෿]+", text):
            padded = f" {word} "
            for n in range(self.ngram[0], self.ngram[1] + 1):
                for i in range(max(1, len(padded) - n + 1)):
                    grams[padded[i:i + n]] += 1
        return grams

    def fit(self, texts: list[str]) -> None:
        df: Counter = Counter()
        for t in texts:
            df.update(set(self._grams(t)))
        self.vocab = {g: i for i, g in enumerate(sorted(df))}
        n = len(texts)
        self.idf = np.array([math.log((1 + n) / (1 + df[g])) + 1.0 for g in sorted(df)], dtype=np.float32)

    def embed(self, texts: list[str]) -> np.ndarray:
        mat = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for row, t in enumerate(texts):
            for g, c in self._grams(t).items():
                j = self.vocab.get(g)
                if j is not None:
                    mat[row, j] = (1.0 + math.log(c)) * self.idf[j]
        return _l2(mat)

    def state(self) -> dict:
        return {"ngram": list(self.ngram)}

    def save(self, folder: Path) -> None:
        (folder / "tfidf_vocab.json").write_text(json.dumps(self.vocab, ensure_ascii=False), encoding="utf-8")
        np.save(folder / "tfidf_idf.npy", self.idf)

    def load(self, folder: Path) -> None:
        self.vocab = json.loads((folder / "tfidf_vocab.json").read_text(encoding="utf-8"))
        self.idf = np.load(folder / "tfidf_idf.npy")


def make_backend(kind: str, model_name: str, cache_dir: Path) -> tuple[object, list[str]]:
    """Return (backend, warnings). ``auto`` tries fastembed first, then the labelled fallback."""
    warnings: list[str] = []
    if kind in ("auto", "fastembed"):
        try:
            return FastEmbedBackend(model_name, cache_dir), warnings
        except EmbeddingUnavailable as exc:
            if kind == "fastembed":
                raise
            warnings.append(f"Embedding model unavailable ({exc}); using lexical TF-IDF fallback.")
    return TfidfCharBackend(), warnings
