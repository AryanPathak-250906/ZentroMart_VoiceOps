"""Deterministic scoring rules for the practice sets (evaluation only — never used at runtime).

must_include: facts separated by ' ; ' are ALL required; ' / ' separates acceptable alternatives.
must_not_include: any listed phrase present = violation.
Matching is case-insensitive after Unicode/digit normalisation. Dates match in any common
format (ISO, '27 October 2026', '27th of October', 'October 27', Hindi month names; a missing
year is read as 2026). Numbers match as whole numbers or as number words (English, Hindi,
Hinglish). Short tokens (e.g. 'AI', 'QC', 'ek') match on word boundaries.
Limitation: paraphrases of forbidden claims and semantic equivalence beyond these rules are not
detected — see TEST_REPORT.md.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date

DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
MONTHS = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7, "august": 8,
          "september": 9, "october": 10, "november": 11, "december": 12,
          "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
          "जनवरी": 1, "फ़रवरी": 2, "फरवरी": 2, "मार्च": 3, "अप्रैल": 4, "मई": 5, "जून": 6, "जुलाई": 7, "अगस्त": 8,
          "सितंबर": 9, "सितम्बर": 9, "अक्टूबर": 10, "अक्तूबर": 10, "नवंबर": 11, "नवम्बर": 11, "दिसंबर": 12, "दिसम्बर": 12}
NUMBER_WORDS = {
    1: ["one", "ek", "एक"], 2: ["two", "do", "दो"], 3: ["three", "teen", "तीन"], 4: ["four", "char", "chaar", "चार"],
    5: ["five", "paanch", "panch", "पाँच", "पांच"], 6: ["six", "chhe", "chah", "छह", "छः"], 7: ["seven", "saat", "सात"],
    8: ["eight", "aath", "आठ"], 9: ["nine", "nau", "नौ"], 10: ["ten", "das", "दस"], 12: ["twelve", "barah", "baarah", "बारह"],
    15: ["fifteen", "pandrah", "पंद्रह"], 30: ["thirty", "tees", "तीस"], 48: ["forty-eight", "forty eight", "अड़तालीस"],
    49: ["forty-nine", "forty nine", "उनचास"], 79: ["seventy-nine", "seventy nine", "उन्यासी"],
}
YEAR = 2026


def norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "").translate(DEV_DIGITS).lower()
    t = t.replace("–", "-").replace("—", "-").replace("‑", "-")
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)
    return re.sub(r"\s+", " ", t).strip()


def parse_date(text: str) -> date | None:
    t = norm(text)
    m = re.fullmatch(r"(20\d\d)-(\d\d)-(\d\d)", t)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.fullmatch(r"(\d{1,2})\s+(\S+)\s+(20\d\d)", t)
    if m and m.group(2) in MONTHS:
        return date(int(m.group(3)), MONTHS[m.group(2)], int(m.group(1)))
    return None


def dates_in(text: str) -> set[date]:
    t = norm(text)
    found: set[date] = set()

    def add(y, mo, d):
        try:
            found.add(date(int(y) if y else YEAR, mo, int(d)))
        except ValueError:
            pass

    for y, mo, d in re.findall(r"\b(20\d\d)-(\d\d)-(\d\d)\b", t):
        add(y, int(mo), d)
    for d, mo, y in re.findall(r"\b(\d{1,2})/(\d{1,2})/(20\d\d)\b", t):
        add(y, int(mo), d)
    month_alt = "|".join(sorted(map(re.escape, MONTHS), key=len, reverse=True))
    for d, mo, y in re.findall(rf"(?<!\d)(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({month_alt})\b\.?,?(?:\s+(20\d\d))?", t):
        add(y, MONTHS[mo], d)
    for mo, d, y in re.findall(rf"\b({month_alt})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b,?(?:\s+(20\d\d))?", t):
        add(y, MONTHS[mo], d)
    return found


def _is_number(alt: str) -> bool:
    return bool(re.fullmatch(r"\d+", norm(alt)))


def contains_fact(reply: str, alt: str) -> bool:
    a = alt.strip()
    if not a:
        return True
    d = parse_date(a)
    if d is not None:
        return d in dates_in(reply)
    r = norm(reply)
    na = norm(a)
    if _is_number(na):
        n = int(na)
        if re.search(rf"(?<![\d.]){n}(?![\d])", r):
            return True
        return any(re.search(rf"(?<![\w]){re.escape(w)}(?![\w])", r) for w in NUMBER_WORDS.get(n, []))
    if len(na) <= 3 and na.isascii():
        return re.search(rf"(?<![a-z0-9]){re.escape(na)}(?![a-z0-9])", r) is not None
    return na in r


def split_groups(field: str) -> list[list[str]]:
    groups = []
    for g in (field or "").split(" ; "):
        g = g.strip()
        if g:
            groups.append([alt.strip() for alt in g.split(" / ") if alt.strip()])
    return groups


def check_must_include(reply: str, field: str) -> tuple[bool, list[str]]:
    missing = []
    for alts in split_groups(field):
        if not any(contains_fact(reply, a) for a in alts):
            missing.append(" / ".join(alts))
    return not missing, missing


def check_must_not(reply: str, field: str) -> tuple[bool, list[str]]:
    hits = []
    for phrase in [p.strip() for p in (field or "").split(" ; ") if p.strip()]:
        if contains_fact(reply, phrase):
            hits.append(phrase)
    return not hits, hits


def split_list(field: str) -> list[str]:
    return [x.strip() for x in (field or "").split(" ; ") if x.strip()]


def parse_expected_tools(field: str) -> list[dict]:
    calls = []
    for m in re.finditer(r"(\w+)\(([^)]*)\)", field or ""):
        args = {}
        for part in [p.strip() for p in m.group(2).split(",") if p.strip()]:
            if "=" in part:
                k, v = part.split("=", 1)
                args[k.strip()] = v.strip()
        calls.append({"name": m.group(1), "args": args})
    return calls


def check_tools(expected: list[dict], actual_internal: list[dict]) -> dict:
    """actual_internal: [{'name', 'order_id', 'verification_value', 'created'}] for THIS turn."""
    missing, matched = [], set()
    for exp in expected:
        ok = False
        for i, act in enumerate(actual_internal):
            if i in matched or act["name"] != exp["name"]:
                continue
            if exp["name"] == "lookup_order":
                if act.get("order_id") != exp["args"].get("order_id"):
                    continue
                want = exp["args"].get("verification_value")
                if (want or None) != (act.get("verification_value") or None):
                    continue
            if exp["name"] == "create_ticket" and not act.get("created"):
                continue
            matched.add(i)
            ok = True
            break
        if not ok:
            missing.append(exp)
    unexpected = [a for i, a in enumerate(actual_internal) if i not in matched]
    return {"pass": not missing and not unexpected, "missing": missing,
            "unexpected": [{k: ("[redacted]" if k == "verification_value" and v else v) for k, v in a.items()} for a in unexpected]}
