"""Precedence-aware hybrid retriever.

Pipeline for one turn:
1. For every query (customer text + planner rewrites) score all chunks with BM25 and with
   the vector backend; fuse the rankings with Reciprocal Rank Fusion (RRF).
2. Multiply by an authority prior that follows KB-00.1 (policy > agent guide > help > FAQ
   > marketing). Superseded chunks are demoted unless the question refers to a date before
   the successor's effective date (their documented historical applicability).
3. Back-fill: when a lower-authority source (help/FAQ/marketing) is retrieved, the
   best-matching *current policy* chunk on the same topic is added so conflicts are always
   resolved against the governing document; when a superseded chunk is retrieved its
   successor's matching chunk is added; when a historical date is mentioned, the
   predecessor's matching chunk is added for the preserved exception.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from .glossary import expand
from .index import KBIndex
from .ingest import Chunk, tokenize

RRF_K = 60
AUTHORITY_WEIGHT = {0: 1.0, 1: 1.0, 2: 0.97, 3: 0.93, 4: 0.88, 5: 0.80, 9: 0.70}
MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september",
     "october", "november", "december"])}
MONTHS.update({m[:3]: v for m, v in list(MONTHS.items())})


@dataclass
class Hit:
    chunk: Chunk
    score: float
    bm25: float = 0.0
    vector: float = 0.0
    via: str = "search"
    rank: int = 0

    def to_dict(self) -> dict:
        c = self.chunk
        return {
            "section_id": c.section_id, "chunk_id": c.chunk_id, "doc_id": c.doc_id,
            "file": c.file, "doc_title": c.doc_title, "section_title": c.section_title,
            "doc_type": c.doc_type, "status": c.status, "authority": c.authority_label,
            "effective_from": c.effective_from, "score": round(float(self.score), 5),
            "bm25": round(float(self.bm25), 4), "vector": round(float(self.vector), 4),
            "via": self.via, "rank": self.rank,
        }


@dataclass
class RetrievalResult:
    hits: list[Hit]
    queries: list[str]
    historical_dates: list[str] = field(default_factory=list)
    backend: str = ""

    @property
    def section_ids(self) -> list[str]:
        seen: list[str] = []
        for h in self.hits:
            if h.chunk.section_id not in seen:
                seen.append(h.chunk.section_id)
        return seen


def extract_dates(text: str, reference: date) -> list[date]:
    """Find explicit calendar dates in free text (ISO, '20 July', 'July 20, 2026')."""
    out: list[date] = []
    for y, m, d in re.findall(r"\b(20\d\d)-(\d\d)-(\d\d)\b", text):
        try:
            out.append(date(int(y), int(m), int(d)))
        except ValueError:
            pass
    low = text.lower()
    for d, mon, y in re.findall(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([a-z]{3,9})\.?(?:,?\s+(20\d\d))?", low):
        if mon in MONTHS:
            try:
                out.append(date(int(y) if y else reference.year, MONTHS[mon], int(d)))
            except ValueError:
                pass
    for mon, d, y in re.findall(r"\b([a-z]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(20\d\d))?", low):
        if mon in MONTHS:
            try:
                out.append(date(int(y) if y else reference.year, MONTHS[mon], int(d)))
            except ValueError:
                pass
    return out


class Retriever:
    def __init__(self, index: KBIndex, reference_date: date):
        self.index = index
        self.reference_date = reference_date
        self.docs_by_id: dict[str, list[int]] = {}
        for i, c in enumerate(index.chunks):
            self.docs_by_id.setdefault(c.doc_id, []).append(i)

    # ----------------------------------------------------------------- scoring helpers
    def _rank_lists(self, query: str, allowed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        bm = self.index.bm25.scores(tokenize(expand(query)))  # glossary expansion: lexical channel only
        vec = self.index.vectors @ self.index.embed_query(query)
        bm = np.where(allowed, bm, -np.inf)
        vec = np.where(allowed, vec, -np.inf)
        rrf = np.zeros(len(bm), dtype=np.float64)
        for scores in (bm, vec):
            order = np.argsort(-scores, kind="stable")
            for r, i in enumerate(order):
                if not np.isfinite(scores[i]):
                    break
                if scores is bm and scores[i] <= 0:
                    break  # no lexical overlap: contributes nothing
                rrf[i] += 1.0 / (RRF_K + r + 1)
        return rrf, bm, vec

    def _search(self, queries: list[str], allowed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        total = np.zeros(len(self.index.chunks))
        best_bm = np.zeros(len(self.index.chunks))
        best_vec = np.full(len(self.index.chunks), -1.0)
        for q in queries:
            if not q.strip():
                continue
            rrf, bm, vec = self._rank_lists(q, allowed)
            total += rrf
            best_bm = np.maximum(best_bm, np.where(np.isfinite(bm), bm, 0))
            best_vec = np.maximum(best_vec, np.where(np.isfinite(vec), vec, -1))
        return total, best_bm, best_vec

    def _historical(self, chunk: Chunk, dates: list[date]) -> bool:
        """A superseded chunk is historically applicable if a mentioned date precedes the
        successor's effective date (successor = the doc that lists this one in `supersedes`)."""
        if chunk.status != "superseded" or not dates:
            return False
        successor_eff = None
        for c in self.index.chunks:
            if c.doc_id == chunk.superseded_by and c.effective_from:
                successor_eff = date.fromisoformat(c.effective_from)
                break
        return bool(successor_eff and any(d < successor_eff for d in dates))

    def _best_in(self, text: str, predicate) -> int | None:
        allowed = np.array([predicate(c) for c in self.index.chunks])
        if not allowed.any():
            return None
        total, _, _ = self._search([text], allowed)
        if total.max() <= 0:
            return None
        return int(np.argmax(total))

    # --------------------------------------------------------------------- main entry
    def retrieve(self, queries: list[str], k: int = 6, exclude_doc_ids: set[str] | None = None,
                 mentioned_dates: list[str] | None = None) -> RetrievalResult:
        exclude_doc_ids = exclude_doc_ids or set()
        queries = [q for q in dict.fromkeys(q.strip() for q in queries) if q]
        allowed = np.array([c.doc_id not in exclude_doc_ids for c in self.index.chunks])
        dates: list[date] = []
        for q in queries:
            dates += extract_dates(q, self.reference_date)
        for d in mentioned_dates or []:
            try:
                dates.append(date.fromisoformat(d))
            except ValueError:
                pass
        total, best_bm, best_vec = self._search(queries, allowed)
        final = np.zeros_like(total)
        for i, c in enumerate(self.index.chunks):
            w = AUTHORITY_WEIGHT.get(c.authority_rank, 0.85)
            if c.status == "superseded" and self._historical(c, dates):
                w = 1.0
            final[i] = total[i] * w
        order = [int(i) for i in np.argsort(-final, kind="stable") if final[i] > 0][:k]
        hits = [Hit(self.index.chunks[i], float(final[i]), float(best_bm[i]), float(best_vec[i])) for i in order]
        present = {h.chunk.chunk_id for h in hits}

        def add(idx: int | None, via: str) -> None:
            if idx is None:
                return
            ch = self.index.chunks[idx]
            if ch.chunk_id in present or ch.doc_id in exclude_doc_ids:
                return
            present.add(ch.chunk_id)
            hits.append(Hit(ch, float(final[idx]), float(best_bm[idx]), float(best_vec[idx]), via=via))

        for h in list(hits):
            c = h.chunk
            if c.authority_rank in (3, 4, 5):  # help / faq / marketing → governing policy
                add(self._best_in(c.embed_text() + " " + " ".join(queries),
                                  lambda x: x.doc_type == "policy" and x.status == "current"),
                    f"authority_backfill:{c.section_id}")
            if c.status == "superseded" and c.superseded_by:
                add(self._best_in(c.embed_text(), lambda x, s=c.superseded_by: x.doc_id == s),
                    f"successor_backfill:{c.section_id}")
            if c.status == "current" and c.supersedes and dates:
                pred = c.supersedes
                pred_chunks = [self.index.chunks[i] for i in self.docs_by_id.get(pred, [])]
                if pred_chunks and self._historical(pred_chunks[0], dates):
                    add(self._best_in(" ".join(queries), lambda x, p=pred: x.doc_id == p),
                        f"historical_backfill:{c.section_id}")
        # cross-reference back-fill: a retrieved section that names another section ID
        # (e.g. CAN-2 "... the refund follows REF-1") brings that section into context
        for h in list(hits):
            for ref in set(re.findall(r"\b([A-Z]{2,4}-\d+(?:\.\d+)?)\b", h.chunk.text)):
                if ref != h.chunk.section_id and ref in self.index.by_section:
                    add(self.index.by_section[ref][0], f"reference_backfill:{h.chunk.section_id}")
        for r, h in enumerate(hits, 1):
            h.rank = r
        return RetrievalResult(hits=hits, queries=queries,
                               historical_dates=sorted({d.isoformat() for d in dates}),
                               backend=self.index.backend_description)
