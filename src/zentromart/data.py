"""Structured data access.

All CSV values are kept as *strings* (IDs, pincodes, phone numbers, refund references never
become numbers). Empty cells stay ''. This module is used for indexing, the public product
catalog and evaluation; customer-facing order lookups go exclusively through the supplied
``lookup_order`` tool (see tools_adapter.py).
"""
from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from functools import cached_property
from pathlib import Path

DATA_TABLES = {
    "customers": "customers.csv",
    "orders": "orders.csv",
    "order_items": "order_items.csv",
    "products": "products.csv",
    "pincodes": "pincodes.csv",
    "holidays": "holidays.csv",
    "support_logs": "support_logs.csv",
}
SKU_RE = re.compile(r"\b([A-Z]{3})-(\d{4})\b", re.I)


def read_csv_str(path: Path) -> list[dict[str, str]]:
    """Read a CSV with every value as str (equivalent to pandas dtype=str, keep_default_na=False)."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[^\w\s]", " ", text)


_SPOKEN_DIGITS = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "to": "2", "too": "2", "three": "3",
    "four": "4", "for": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}


def normalise_spoken_sku(text: str) -> str:
    """Turn ASR forms like 'sku ele-zero zero one two' into 'ELE-0012' where possible."""
    def repl(match: re.Match) -> str:
        prefix = match.group(1).upper()
        words = re.findall(r"[a-z]+|\d", match.group(2).lower())
        digits = "".join(_SPOKEN_DIGITS.get(w, w if w.isdigit() else "") for w in words)
        return f"{prefix}-{digits}" if len(digits) == 4 else match.group(0)

    tok = r"(?:zero|oh|one|two|to|too|three|four|for|five|six|seven|eight|nine|\d)"
    return re.sub(
        rf"\b(ele|fsh|hom|bty|bks|gro|inr|dgt)[\s-]+((?:{tok}[\s-]*){{3}}{tok})(?![a-z])",
        repl, text, flags=re.I)


@dataclass
class Dataset:
    root: Path

    def table(self, name: str) -> list[dict[str, str]]:
        return read_csv_str(self.root / "data" / DATA_TABLES[name])

    @cached_property
    def products(self) -> dict[str, dict[str, str]]:
        return {row["sku"]: row for row in self.table("products")}

    @cached_property
    def holidays(self) -> dict[date, str]:
        return {date.fromisoformat(r["date"]): r["name"] for r in self.table("holidays")}

    @cached_property
    def pincodes(self) -> dict[str, dict[str, str]]:
        return {r["pincode"]: r for r in self.table("pincodes")}

    def practice_questions(self) -> list[dict[str, str]]:
        return read_csv_str(self.root / "practice" / "questions.csv")

    def practice_conversations(self) -> list[dict[str, str]]:
        return read_csv_str(self.root / "practice" / "conversations.csv")

    # ---------------------------------------------------------------- catalog matching
    def match_products(self, text: str, extra_mentions: list[str] | None = None, limit: int = 4) -> list[dict[str, str]]:
        """Return catalog rows referenced by an explicit SKU or by product name.

        Name matching compares the customer's words with each product's *base title*
        (the part before the colour, e.g. 'Kestrel K7 Smartphone 256 GB'). When a base title
        matches several colour variants, all variants are returned (they share policy fields).
        """
        haystacks = [text] + list(extra_mentions or [])
        found: dict[str, dict[str, str]] = {}
        for hay in haystacks:
            for m in SKU_RE.finditer(normalise_spoken_sku(hay)):
                sku = f"{m.group(1).upper()}-{m.group(2)}"
                if sku in self.products:
                    found[sku] = self.products[sku]
        if found:
            return list(found.values())[:limit]

        query_tokens = set(_norm(" ".join(haystacks)).split())
        scored: list[tuple[tuple, str]] = []
        for sku, row in self.products.items():
            base = row["title"].split(",")[0]
            tokens = [t for t in _norm(base).split() if t not in {"of", "the", "a", "pack"}]
            words = [t for t in tokens if not t.isdigit() and len(t) > 1]   # name words (incl. 'k7', '4k')
            numbers = [t for t in tokens if t.isdigit()]                     # sizes such as 256 / 1 5
            if not words:
                continue
            hits = sum(1 for t in words if t in query_tokens)
            ratio = hits / len(words)
            # require most of the product's name words, not just one generic word
            if hits >= 2 and ratio >= 0.75:
                num_hits = sum(1 for t in numbers if t in query_tokens)
                scored.append(((round(ratio, 3), num_hits, hits), sku))
        if not scored:
            return self._fuzzy_products(query_tokens, limit)
        scored.sort(reverse=True)
        top = scored[0][0]
        best = [self.products[sku] for score, sku in scored if score == top]
        coloured = [r for r in best if "," in r["title"]
                    and r["title"].split(",")[1].strip().lower() in query_tokens]
        return (coloured or best)[:limit]

    def _fuzzy_products(self, query_tokens: set[str], limit: int) -> list[dict[str, str]]:
        """Fallback for speech-recognition misspellings ('bruli' for 'Brewly'): used ONLY when the
        exact pass found nothing. A name word counts if it is similar (same first letter,
        SequenceMatcher >= 0.72); at least two name words and 60% of them must match, and the
        best product must be unambiguous (all winners share one base title)."""
        qt = [t for t in query_tokens if len(t) > 1]
        scored = []
        for sku, row in self.products.items():
            base = row["title"].split(",")[0]
            words = [t for t in _norm(base).split() if not t.isdigit() and len(t) > 1 and t not in {"of", "the", "pack"}]
            if not words:
                continue
            hits = sum(1 for w in words if any(_similar(w, q) for q in qt))
            if hits >= 2 and hits / len(words) >= 0.6:
                scored.append(((round(hits / len(words), 3), hits), sku))
        if not scored:
            return []
        scored.sort(reverse=True)
        top = scored[0][0]
        best = [self.products[sku] for score, sku in scored if score == top]
        if len({r["title"].split(",")[0] for r in best}) != 1:
            return []  # ambiguous: do not guess
        return best[:limit]


def _similar(a: str, b: str) -> bool:
    from difflib import SequenceMatcher
    if a == b:
        return True
    if len(a) < 4 or len(b) < 4 or a[0] != b[0]:
        return False
    return SequenceMatcher(None, a, b).ratio() >= 0.72


def product_evidence_text(row: dict[str, str]) -> str:
    return (
        f"Catalog row products.csv:{row['sku']} — title: {row['title']}; brand: {row['brand']}; "
        f"return_category: {row['return_category']}; price_inr: {row['price_inr']}; "
        f"fulfilled_by: {row['fulfilled_by']}; assured (listing badge): {row['assured']}; "
        f"warranty_months (brand): {row['warranty_months']}"
    )
