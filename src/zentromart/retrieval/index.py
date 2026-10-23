"""Persisted hybrid index: BM25 (rebuilt in memory, cheap) + vectors (persisted on disk).

The on-disk index is invalidated automatically when any KB file, the chunker version, the
embedding backend or the embedding model changes (fingerprint stored in manifest.json).
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .embeddings import TfidfCharBackend, make_backend
from .ingest import Chunk, chunk_documents, kb_fingerprint, tokenize


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.doc_len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if docs else 0.0
        self.tf = [Counter(d) for d in docs]
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        out = np.zeros(len(self.docs), dtype=np.float32)
        for i, tf in enumerate(self.tf):
            s = 0.0
            for t in query_tokens:
                f = tf.get(t)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1))
                s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / denom
            out[i] = s
        return out


class KBIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, backend, manifest: dict):
        self.chunks = chunks
        self.vectors = vectors
        self.backend = backend
        self.manifest = manifest
        self.bm25 = BM25([tokenize(c.embed_text()) for c in chunks])
        self.by_section: dict[str, list[int]] = {}
        for i, c in enumerate(chunks):
            self.by_section.setdefault(c.section_id, []).append(i)

    @property
    def backend_description(self) -> str:
        return self.manifest.get("backend_description", "unknown")

    @property
    def semantic(self) -> bool:
        return bool(self.manifest.get("semantic"))

    def embed_query(self, text: str) -> np.ndarray:
        return self.backend.embed([text])[0]

    @classmethod
    def load_or_build(cls, kb_dir: Path, index_dir: Path, backend_kind: str = "auto",
                      model_name: str = "", force: bool = False) -> "KBIndex":
        index_dir.mkdir(parents=True, exist_ok=True)
        backend, warnings = make_backend(backend_kind, model_name, index_dir.parent / "models")
        fp = kb_fingerprint(kb_dir, extra=f"{backend.kind}|{getattr(backend, 'model_name', '')}")
        manifest_path = index_dir / "manifest.json"
        if not force and manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("fingerprint") == fp:
                    chunks = [Chunk(**c) for c in json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))]
                    vectors = np.load(index_dir / "vectors.npy")
                    if isinstance(backend, TfidfCharBackend):
                        backend.load(index_dir)
                    manifest["loaded_from_disk"] = True
                    manifest["warnings"] = warnings
                    return cls(chunks, vectors, backend, manifest)
            except (OSError, ValueError, KeyError, TypeError):
                pass  # corrupted/partial index -> rebuild below
        chunks = chunk_documents(kb_dir)
        texts = [c.embed_text() for c in chunks]
        backend.fit(texts)
        vectors = backend.embed(texts).astype(np.float32)
        manifest = {
            "fingerprint": fp, "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "backend": backend.kind, "backend_description": backend.description,
            "semantic": backend.semantic, "model_name": getattr(backend, "model_name", ""),
            "chunk_count": len(chunks), "dimensions": int(vectors.shape[1]) if vectors.size else 0,
            "kb_files": sorted({c.file for c in chunks}),
        }
        (index_dir / "chunks.json").write_text(json.dumps([c.to_dict() for c in chunks], ensure_ascii=False, indent=1), encoding="utf-8")
        np.save(index_dir / "vectors.npy", vectors)
        if isinstance(backend, TfidfCharBackend):
            backend.save(index_dir)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest["loaded_from_disk"] = False
        manifest["warnings"] = warnings
        return cls(chunks, vectors, backend, manifest)
