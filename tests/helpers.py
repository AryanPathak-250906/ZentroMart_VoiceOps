"""Shared offline fixtures. Every LLM here is a ScriptedLLM (mock=True) — never a model score."""
from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zentromart.agent import SupportAgent, guess_language  # noqa: E402
from zentromart.config import discover_dataset_dir, get_settings  # noqa: E402
from zentromart.conversation import order_id_candidates, verification_candidates  # noqa: E402
from zentromart.llm import ScriptedLLM  # noqa: E402

DATASET = discover_dataset_dir()


def require_dataset(test):
    import unittest
    return unittest.skipIf(DATASET is None, "dataset not found (set ZENTROMART_DATASET_DIR)")(test)


def rule_planner(user: str) -> dict:
    """Deterministic stand-in for the planner LLM (keyword rules; for plumbing tests only)."""
    msg = user.split("CURRENT customer message:\n<<<", 1)[-1].rsplit(">>>", 1)[0]
    low = msg.lower()
    awaiting_ver = "waiting for: verification" in user
    oids = order_id_candidates(msg)
    vals = verification_candidates(msg) if not oids else []
    signals = []
    if any(w in low for w in ("human", "person", "insaan", "इंसान", "agent")):
        signals.append("human_requested")
    if any(w in low for w in ("never placed", "did not make", "didn't make")):
        signals.append("unauthorized_activity")
    if any(w in low for w in ("crack", "damage", "broken", "टूटा")):
        signals.append("damaged_wrong_or_missing_item")
    if any(w in low for w in ("third time", "teesri baar")):
        signals.append("repeat_contact")
    if any(w in low for w in ("battery", "burning", "sparks")):
        signals.append("safety_risk")
    order_words = ("order", "ऑर्डर", "refund", "रिफ़ंड", "cancel", "shipped", "arrived", "return")
    needs = bool(oids) or ("my order" in low) or ("मेरा" in msg and "ऑर्डर" in msg) or ("mera order" in low)
    return {
        "language": guess_language(msg), "intent": "verification" if vals else "other",
        "order_id": oids[0] if oids else None,
        "verification_value": vals[0] if (vals and awaiting_ver) else None,
        "needs_order_details": needs,
        "is_general_policy_question": not needs, "search_queries": [msg],
        "product_mentions": re.findall(r"[A-Z]{3}-\d{4}", msg), "mentioned_dates": [],
        "escalation_signals": signals, "requests_exception_or_compensation": "voucher" in low or "compensation" in low,
        "issue_summary": msg[:120],
    }


def echo_responder(user: str, tools) -> dict:
    """Mock responder: cites the first evidence section; replies with a fixed MOCK text."""
    ids = re.findall(r'<source id="([^"]+)"', user.split("<evidence>", 1)[-1])
    actions = user.split("<turn_actions>", 1)[-1].split("</turn_actions>", 1)[0]
    decision = "answer"
    if "ASK_ORDER_ID" in actions or "ASK_VERIFICATION" in actions or "FAILED_ONCE" in actions:
        decision = "clarify"
    if "TICKET_CREATED" in actions:
        decision = "escalate"
    return {"reply": "[MOCK REPLY] " + actions.strip()[:80], "decision": decision,
            "used_section_ids": ids[:2], "answer_supported_by_documents": True}


def make_agent(planner=rule_planner, responder=echo_responder, llm=None):
    tmp = Path(tempfile.mkdtemp(prefix="zm_test_"))
    os.environ["ZENTROMART_RUNTIME_DIR"] = str(tmp)
    os.environ.setdefault("EMBEDDING_BACKEND", "tfidf")  # offline, deterministic
    settings = get_settings(load_env=False)
    settings.embedding_backend = os.environ["EMBEDDING_BACKEND"]
    llm = llm or ScriptedLLM(planner, responder)
    agent = SupportAgent.from_settings(settings, llm=llm, build_llm_if_missing=False)
    return agent, tmp


def cleanup(tmp: Path) -> None:
    shutil.rmtree(tmp, ignore_errors=True)
