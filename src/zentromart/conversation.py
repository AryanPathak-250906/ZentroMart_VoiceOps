"""Conversation state (per session) and an in-memory session store.

Verified order records live only inside the Session object of the conversation that verified
them; they are never serialised to clients, logs or evaluation artefacts.
"""
from __future__ import annotations

import re
import secrets
import threading
import time
from dataclasses import dataclass, field

DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
MAX_VERIFICATION_ATTEMPTS = 2  # first try + one retry (IDV-3)


def normalise_digits(text: str) -> str:
    t = (text or "").translate(DEV_DIGITS)
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)  # '7,936' -> '7936' (ITN thousands separators)


def digit_runs(text: str) -> list[str]:
    """Digit groups, merging groups separated by a single space or hyphen ('8978 2837')."""
    t = normalise_digits(text)
    runs = []
    for m in re.finditer(r"\d+(?:[ \-]\d+)*", t):
        runs.append(m.group(0))
    return runs


def order_id_candidates(text: str) -> list[str]:
    out = []
    for run in digit_runs(text):
        digits = re.sub(r"\D", "", run)
        if len(digits) == 8:
            out.append(digits)
        else:
            out += [g for g in re.split(r"[ \-]", run) if len(g) == 8]
    return list(dict.fromkeys(out))


def verification_candidates(text: str) -> list[str]:
    out = []
    for run in digit_runs(text):
        digits = re.sub(r"\D", "", run)
        if len(digits) in (4, 6):
            out.append(digits)
        else:
            out += [g for g in re.split(r"[ \-]", run) if len(g) in (4, 6)]
    return list(dict.fromkeys(out))


@dataclass
class Session:
    session_id: str
    channel: str = "chat"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    turns: list[dict] = field(default_factory=list)          # {"role": customer|assistant, "text": str}
    language: str = "en"
    current_order_id: str | None = None
    awaiting: str | None = None                               # None | "order_id" | "verification"
    lookups_started: set[str] = field(default_factory=set)
    failed_attempts: dict[str, int] = field(default_factory=dict)
    locked_orders: set[str] = field(default_factory=set)
    verified: dict[str, dict] = field(default_factory=dict)  # PRIVATE: order_id -> lookup result
    pending_verification_value: str | None = None
    secrets: list[str] = field(default_factory=list)         # values to scrub from any output
    tickets: list[dict] = field(default_factory=list)        # {"ticket_id", "reason", "order_id"}
    issue_summary: str = ""
    last_intent: str = ""
    last_products: list[str] = field(default_factory=list)
    assistant_turns: int = 0

    # ------------------------------------------------------------------ helpers
    def verification_state(self, order_id: str | None = None) -> str:
        oid = order_id or self.current_order_id
        if not oid:
            return "no_order"
        if oid in self.verified:
            return "verified"
        if oid in self.locked_orders:
            return "locked"
        fails = self.failed_attempts.get(oid, 0)
        if fails:
            return "failed_once"
        if oid in self.lookups_started:
            return "awaiting_verification"
        return "not_started"

    @property
    def verified_order(self) -> dict | None:
        return self.verified.get(self.current_order_id or "")

    def summary_for_model(self) -> str:
        parts = [f"assistant replies so far: {self.assistant_turns}",
                 f"order being discussed: {self.current_order_id or 'none'}",
                 f"identity verification for that order: {self.verification_state()}",
                 f"waiting for: {self.awaiting or 'nothing'}"]
        if self.tickets:
            parts.append("tickets already created: " + ", ".join(f"{t['ticket_id']} ({t['reason']})" for t in self.tickets))
        if self.issue_summary:
            parts.append(f"issue so far: {self.issue_summary}")
        if self.last_products:
            parts.append("products discussed: " + ", ".join(self.last_products))
        return "; ".join(parts)

    def history_text(self, limit: int = 10) -> str:
        lines = []
        for t in self.turns[-limit:]:
            who = "Customer" if t["role"] == "customer" else "Assistant"
            lines.append(f"{who}: {t['text']}")
        return "\n".join(lines) if lines else "(start of conversation)"

    def public_state(self) -> dict:
        return {
            "session_id": self.session_id,
            "order_id": self.current_order_id,
            "verification": self.verification_state(),
            "verification_failures": self.failed_attempts.get(self.current_order_id or "", 0),
            "awaiting": self.awaiting,
            "tickets": [dict(t) for t in self.tickets],
            "language": self.language,
            "turns": len([t for t in self.turns if t["role"] == "customer"]),
        }


class SessionStore:
    """Thread-safe in-memory store with idle expiry. One Session per conversation."""

    def __init__(self, ttl_s: int = 3600, max_sessions: int = 500):
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self.ttl_s = ttl_s
        self.max_sessions = max_sessions

    def create(self, channel: str = "chat") -> Session:
        with self._lock:
            self._expire()
            sid = secrets.token_urlsafe(16)
            s = Session(session_id=sid, channel=channel)
            self._sessions[sid] = s
            return s

    def get(self, sid: str) -> Session | None:
        with self._lock:
            self._expire()
            s = self._sessions.get(sid)
            if s:
                s.updated_at = time.time()
            return s

    def delete(self, sid: str) -> bool:
        with self._lock:
            return self._sessions.pop(sid, None) is not None

    def _expire(self) -> None:
        now = time.time()
        for sid in [k for k, s in self._sessions.items() if now - s.updated_at > self.ttl_s]:
            del self._sessions[sid]
        while len(self._sessions) >= self.max_sessions:
            oldest = min(self._sessions.values(), key=lambda s: s.updated_at)
            del self._sessions[oldest.session_id]
