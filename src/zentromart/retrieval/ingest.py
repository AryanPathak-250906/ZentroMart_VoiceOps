"""Knowledge-base ingestion: front matter, [SECTION-ID] headings, section-aligned chunks."""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

CHUNKER_VERSION = "2026-10-21.1"
MAX_CHUNK_CHARS = 1400

SECTION_RE = re.compile(r"^##\s*\[([^\]]+)\]\s*(.*)$")
# Precedence from KB-00.1 (lower rank = higher authority). Meta doc defines the rules.
AUTHORITY_RANK = {"meta": 0, "policy": 1, "agent_guide": 2, "help": 3, "faq": 4, "marketing": 5}
SUPERSEDED_RANK = 9
AUTHORITY_LABEL = {
    0: "knowledge-base rules (meta)",
    1: "current policy (highest authority)",
    2: "agent guide",
    3: "help article",
    4: "FAQ (lower authority than policies)",
    5: "marketing page (never overrides a policy)",
    9: "SUPERSEDED policy (reference only)",
}


@dataclass
class Chunk:
    chunk_id: str
    section_id: str
    section_title: str
    doc_id: str
    doc_title: str
    file: str
    doc_type: str
    status: str
    version: str
    effective_from: str
    language: str
    supersedes: str
    superseded_by: str
    applicability_note: str
    text: str
    part: int = 0
    authority_rank: int = 1

    @property
    def authority_label(self) -> str:
        return AUTHORITY_LABEL.get(self.authority_rank, "unknown")

    def embed_text(self) -> str:
        note = f"\n{self.applicability_note}" if self.applicability_note else ""
        return f"{self.doc_title} — {self.section_title}{note}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Document:
    file: str
    meta: dict
    title: str
    preamble: str
    sections: list[tuple[str, str, str]] = field(default_factory=list)  # (id, title, body)


def parse_front_matter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    meta = {}
    for line in text[3:end].strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, text[end + 4:].lstrip("\n")


def parse_document(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    meta, body = parse_front_matter(raw)
    title = meta.get("title", path.stem)
    preamble_lines: list[str] = []
    sections: list[tuple[str, str, list[str]]] = []
    for line in body.splitlines():
        m = SECTION_RE.match(line.strip())
        if m:
            sections.append((m.group(1).strip(), m.group(2).strip(), []))
            continue
        if line.startswith("# ") and not sections:
            title = line[2:].strip() or title
            continue
        if sections:
            sections[-1][2].append(line)
        else:
            preamble_lines.append(line)
    preamble = "\n".join(l for l in preamble_lines).strip()
    return Document(
        file=path.name, meta=meta, title=title, preamble=preamble,
        sections=[(sid, st, "\n".join(lines).strip()) for sid, st, lines in sections],
    )


def _split_long(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts, current = [], ""
    for para in re.split(r"\n\s*\n", text):
        if current and len(current) + len(para) + 2 > limit:
            parts.append(current.strip())
            current = ""
        current += para + "\n\n"
    if current.strip():
        parts.append(current.strip())
    return parts


def chunk_documents(kb_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        doc = parse_document(path)
        meta = doc.meta
        doc_type = meta.get("doc_type", "help")
        status = meta.get("status", "current")
        rank = SUPERSEDED_RANK if status == "superseded" else AUTHORITY_RANK.get(doc_type, 4)
        # A superseded document's preamble says which cases it still covers; keep it with
        # every chunk so the applicability is never separated from the old rule.
        note = ""
        if status == "superseded":
            note = re.sub(r"^>\s*", "", doc.preamble, flags=re.M).strip()
        intro = "" if status == "superseded" else doc.preamble
        for sid, stitle, body in doc.sections:
            text = body
            if intro and len(intro) < 300:
                text = f"{intro}\n{body}" if body else intro
            for i, part in enumerate(_split_long(text)):
                chunks.append(Chunk(
                    chunk_id=sid if i == 0 else f"{sid}#p{i}", section_id=sid, section_title=stitle,
                    doc_id=meta.get("doc_id", path.stem), doc_title=doc.title, file=path.name,
                    doc_type=doc_type, status=status, version=meta.get("version", ""),
                    effective_from=meta.get("effective_from", ""), language=meta.get("language", "en"),
                    supersedes=meta.get("supersedes", ""), superseded_by=meta.get("superseded_by", ""),
                    applicability_note=note, text=part, part=i, authority_rank=rank,
                ))
    return chunks


def kb_fingerprint(kb_dir: Path, extra: str = "") -> str:
    h = hashlib.sha256()
    h.update(CHUNKER_VERSION.encode())
    h.update(extra.encode())
    for path in sorted(kb_dir.glob("*.md")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


TOKEN_RE = re.compile(r"[\wऀ-෿]+", re.UNICODE)
STOPWORDS = set("""a an the and or of to in on for is are was were be been it this that with as at by from
i me my we our you your can do does did will would should could if not no yes so but about what when how
which who whom there their they them he she his her its into than then too very just also any all some
hai hain ka ki ke ko se mein me kya main mujhe mera meri""".split())


def _stem(tok: str) -> str:
    if not tok.isascii() or len(tok) <= 3:
        return tok
    for suf in ("ings", "ing", "ies", "ed", "es", "s", "ly"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            base = tok[: -len(suf)]
            return base + "y" if suf == "ies" else base
    return tok


def tokenize(text: str) -> list[str]:
    import unicodedata

    text = unicodedata.normalize("NFKC", text).lower()
    return [_stem(t) for t in TOKEN_RE.findall(text) if t not in STOPWORDS]
