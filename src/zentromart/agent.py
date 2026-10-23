"""The ZentroMart support agent (Steps 2 and 3). One class powers chat, voice and batch.

Per turn:
  1. Planner LLM call → validated PlannerOutput (fallback to regex extraction if invalid).
  2. Code-enforced identity workflow with the supplied ``lookup_order`` tool.
  3. Code-mandated escalations (ESC-1) → supplied ``create_ticket`` tool, before any claim.
  4. Precedence-aware hybrid retrieval + catalog rows.
  5. Responder LLM call with calculator tools and ``create_ticket``; validated JSON output.
  6. Guardrails: AI disclosure, decision consistency, secret scrubbing, honest failures.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from . import prompts
from .config import Settings
from .conversation import Session, order_id_candidates, verification_candidates, MAX_VERIFICATION_ATTEMPTS
from .data import Dataset, product_evidence_text
from .dates import Calendar
from .llm import LLMError, ToolSpec
from .policy_tables import parse_refund_table, refund_checks
from .retrieval import KBIndex, Retriever
from .retrieval.ingest import AUTHORITY_LABEL
from .schemas import (BASELINE_SCHEMA, PLANNER_SCHEMA, RESPONDER_SCHEMA, BaselineOutput, PlannerOutput,
                      ResponderOutput)
from .tools_adapter import ToolBox, ToolCall, scrub

STANDING_DOCS = {"KB-00", "AGT-STY", "AGT-IDV", "AGT-ESC"}
HIGH_VALUE_INR = 10000
TICKET_REASONS = ["customer_requested_human", "unauthorized_activity", "legal_complaint", "safety_risk",
                  "high_value_damage", "repeat_contact", "overdue_refund", "policy_requires_human", "agent_judgement"]
REASON_TEXT = {
    "customer_requested_human": "customer asked for a human agent (ESC-1 rule 6)",
    "unauthorized_activity": "customer reports order/payment/account activity they did not make (ESC-1 rule 2)",
    "legal_complaint": "customer mentions legal action / consumer court / police (ESC-1 rule 3)",
    "safety_risk": "product safety risk reported (ESC-1 rule 7, DWM-5)",
    "high_value_damage": "damaged/defective/wrong item worth more than ₹10,000 (ESC-1 rule 1, DWM-3)",
    "repeat_contact": "third or later contact about the same issue within 7 days (ESC-1 rule 4)",
    "overdue_refund": "refund more than 3 business days past its maximum timeline (ESC-1 rule 5)",
}
AI_RE = re.compile(r"\bAI\b|artificial intelligence|एआई|कृत्रिम", re.I)
INTRO = {
    "en": "Hi, I'm ZentroMart's AI support assistant.",
    "hi": "नमस्ते, मैं ZentroMart की AI सपोर्ट असिस्टेंट हूँ।",
    "hi-en": "Namaste, main ZentroMart ki AI support assistant hoon.",
}
FALLBACK_TEXT = {
    "llm_down": {
        "en": "Sorry, I'm having a technical problem right now. Please try again in a moment, or ask me to connect you to a human colleague.",
        "hi": "माफ़ कीजिए, अभी तकनीकी समस्या है। कृपया थोड़ी देर बाद फिर कोशिश करें, या किसी सहयोगी से बात करने के लिए कहें।",
        "hi-en": "Sorry, abhi technical problem aa rahi hai. Thodi der baad try karein, ya kisi colleague se baat karne ke liye kahein.",
    },
    "ticket_failed": {
        "en": "I tried to hand this over to a human colleague, but the ticket could not be created. Please try again in a moment or contact support from your registered phone number.",
        "hi": "मैंने इसे सहयोगी को सौंपने की कोशिश की, लेकिन टिकट नहीं बन पाया। कृपया थोड़ी देर बाद फिर कोशिश करें या अपने रजिस्टर्ड फ़ोन नंबर से सपोर्ट से संपर्क करें।",
        "hi-en": "Maine ise colleague ko handover karne ki koshish ki, lekin ticket create nahi ho paya. Thodi der baad try karein ya registered phone number se support se contact karein.",
    },
    "empty": {
        "en": "Sorry, I didn't catch that. Could you say it again?",
        "hi": "माफ़ कीजिए, मैं सुन नहीं पाई। क्या आप दोबारा बोल सकते हैं?",
        "hi-en": "Sorry, main sun nahi payi. Kya aap dobara bol sakte hain?",
    },
}
HINGLISH_MARKERS = {"hai", "hain", "kya", "mera", "meri", "mujhe", "nahi", "kab", "kitne", "kaise", "karna",
                    "kar", "aur", "wala", "wali", "chahiye", "tha", "thi", "hoon", "ho", "gaya", "abhi", "kyun"}


def guess_language(text: str) -> str:
    if re.search(r"[ऀ-ॿ]", text):
        return "hi"
    words = set(re.findall(r"[a-z]+", text.lower()))
    return "hi-en" if len(words & HINGLISH_MARKERS) >= 2 else "en"


def localized(kind: str, lang: str) -> str:
    table = FALLBACK_TEXT[kind]
    return table.get(lang, table["en"])


@dataclass
class TurnResult:
    session_id: str
    turn: int
    reply: str
    decision: str
    intent: str = ""
    language: str = "en"
    channel: str = "chat"
    sources: list[dict] = field(default_factory=list)
    used_sections: list[str] = field(default_factory=list)
    retrieved_sections: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    verification: dict = field(default_factory=dict)
    escalation: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    planner: dict = field(default_factory=dict)  # internal (contains customer-supplied digits)

    def public(self) -> dict:
        """Safe for the web client, batch outputs and logs (no secrets, no private records)."""
        return {
            "session_id": self.session_id, "turn": self.turn, "reply": self.reply,
            "decision": self.decision, "intent": self.intent, "language": self.language,
            "channel": self.channel, "used_sections": self.used_sections,
            "retrieved_sections": self.retrieved_sections, "sources": self.sources,
            "tool_calls": [c.public() for c in self.tool_calls],
            "verification": self.verification, "escalation": self.escalation,
            "errors": self.errors, "diagnostics": self.diagnostics,
        }


class SupportAgent:
    def __init__(self, settings: Settings, llm, dataset: Dataset, retriever: Retriever, toolbox: ToolBox):
        self.settings = settings
        self.llm = llm
        self.dataset = dataset
        self.retriever = retriever
        self.toolbox = toolbox
        self.calendar = Calendar(dataset.holidays, settings.reference_date)
        index = retriever.index
        self.standing = [c for c in index.chunks if c.doc_id in STANDING_DOCS]
        self.refund_section, self.refund_table = parse_refund_table(index.chunks)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_settings(cls, settings: Settings, llm=None, build_llm_if_missing: bool = True) -> "SupportAgent":
        if settings.dataset_dir is None:
            raise FileNotFoundError("ZentroMart dataset not found. Extract Capstone_Project_Dataset_FTTT_2026.zip into "
                                    "./dataset/ or set ZENTROMART_DATASET_DIR.")
        dataset = Dataset(settings.dataset_dir)
        index = KBIndex.load_or_build(settings.dataset_dir / "kb", settings.index_dir,
                                      settings.embedding_backend, settings.embedding_model)
        retriever = Retriever(index, settings.reference_date)
        toolbox = ToolBox(settings.dataset_dir, settings.tickets_file)
        if llm is None and build_llm_if_missing and settings.llm_configured:
            from .llm import build_llm
            llm = build_llm(settings)
        return cls(settings, llm, dataset, retriever, toolbox)

    # ------------------------------------------------------------------ planning
    def _fallback_plan(self, session: Session, message: str) -> PlannerOutput:
        oids = order_id_candidates(message)
        vals = verification_candidates(message) if session.awaiting == "verification" else []
        return PlannerOutput(
            language=guess_language(message), intent="other",
            order_id=oids[0] if len(oids) == 1 else None,
            verification_value=vals[0] if len(vals) == 1 else None,
            needs_order_details=bool(oids) or session.awaiting is not None,
            is_general_policy_question=not (bool(oids) or session.awaiting is not None),
            search_queries=[message], issue_summary=session.issue_summary or message[:200],
        )

    def _plan(self, session: Session, message: str, diag: dict) -> PlannerOutput:
        system = prompts.PLANNER_SYSTEM.format(today=self.settings.reference_date.isoformat())
        user = prompts.PLANNER_USER.format(state=session.summary_for_model(), history=session.history_text(8), message=message)
        last_error = ""
        for attempt in range(2):
            try:
                raw = self.llm.generate_json("planner", system, user + (f"\n\nYour previous output was invalid: {last_error}. Return valid JSON." if last_error else ""),
                                             PLANNER_SCHEMA, max_tokens=self.settings.planner_max_tokens, effort=self.settings.planner_effort)
                return PlannerOutput.model_validate(raw)
            except (ValueError, ValidationError) as exc:
                last_error = str(exc)[:300]
            except LLMError as exc:
                if exc.kind in ("auth", "missing_credentials", "model_not_found", "sdk_missing"):
                    raise
                last_error = f"{exc.kind}: {exc}"
                if not exc.retriable:
                    break
        diag["planner_fallback"] = last_error or "invalid output"
        return self._fallback_plan(session, message)

    # ------------------------------------------------------------------ tickets
    def _ticket(self, session: Session, plan: PlannerOutput, reason: str, checks: list[str],
                tool_calls: list[ToolCall], summary_hint: str = "") -> dict:
        key = (reason, session.current_order_id)
        for t in session.tickets:
            if (t["reason"], t["order_id"]) == key:
                return {"created": True, "ticket_id": t["ticket_id"], "reused": True}
        said = " | ".join(t["text"] for t in session.turns if t["role"] == "customer")[-600:]
        summary = (f"Issue: {summary_hint or plan.issue_summary or session.issue_summary or said[:200]}. "
                   f"Intent: {plan.intent}. Order ID: {session.current_order_id or 'not provided'}. "
                   f"Customer said: {said}. Checks done: {'; '.join(checks) or 'none'}. "
                   f"Reason for escalation: {REASON_TEXT.get(reason, reason)}.")
        result, call = self.toolbox.create_ticket(summary, session.current_order_id, reason, plan.intent, session.secrets)
        tool_calls.append(call)
        if result.get("created") is True:
            session.tickets.append({"ticket_id": result["ticket_id"], "reason": reason, "order_id": session.current_order_id})
        return result

    def _checks(self, session: Session) -> list[str]:
        st = session.verification_state()
        checks = {"verified": "identity verified with lookup_order", "locked": "identity verification failed twice; no order details shared",
                  "failed_once": "identity verification failed once", "awaiting_verification": "order lookup started, identity not yet verified",
                  "no_order": "no order ID provided", "not_started": "identity not verified"}
        out = [checks.get(st, st)]
        order = session.verified_order
        if order:
            out.append(f"order status {order.get('status')}, contacts_last_7_days={order.get('contacts_last_7_days')}")
        return out

    def _item_value(self, session: Session, plan: PlannerOutput, message: str) -> tuple[int | None, str]:
        mentioned = self.dataset.match_products(message, plan.product_mentions)
        skus = {p["sku"] for p in mentioned}
        order = session.verified_order
        if order:
            items = order.get("items") or []
            if order.get("damage_sku"):
                skus.add(order["damage_sku"])
            chosen = [i for i in items if i.get("sku") in skus] or (items if len(items) == 1 else [])
            if chosen:
                v = max(int(i.get("unit_price_inr") or 0) for i in chosen)
                return v, f"item value ₹{v} from the verified order"
        if mentioned:
            v = min(int(p["price_inr"]) for p in mentioned)
            return v, f"catalog price ₹{v} (products.csv)"
        return None, "item value unknown"

    def _mandatory_escalation(self, session: Session, plan: PlannerOutput, message: str, event: str | None) -> tuple[str | None, list[str]]:
        sig = set(plan.escalation_signals)
        checks = self._checks(session)
        for s, reason in (("human_requested", "customer_requested_human"), ("unauthorized_activity", "unauthorized_activity"),
                          ("legal_threat", "legal_complaint"), ("safety_risk", "safety_risk")):
            if s in sig:
                return reason, checks
        if "damaged_wrong_or_missing_item" in sig:
            value, note = self._item_value(session, plan, message)
            if value is not None and value > HIGH_VALUE_INR:
                return "high_value_damage", checks + [note + f" > ₹{HIGH_VALUE_INR}"]
        order = session.verified_order
        if order and ("refund_not_received" in sig or "refund" in (plan.intent or "")):
            for chk in refund_checks(order, self.refund_table, self.refund_section, self.calendar, self.settings.reference_date):
                if chk.get("overdue_more_than_3_business_days"):
                    return "overdue_refund", checks + [
                        f"{chk['refund']} refund initiated {chk['initiated_on']} via {chk['method']}; policy {chk['policy_business_days']} "
                        f"business days ({chk['source']}) -> latest {chk['expected_latest']}; {chk['business_days_past_latest']} business days past"]
        if order and event not in ("verified_now",):
            contacts = int(order.get("contacts_last_7_days") or 0)
            about_order = "repeat_contact" in sig or plan.needs_order_details
            if contacts >= 2 and about_order:
                return "repeat_contact", checks + [f"contacts_last_7_days={contacts} (>= 2)"]
        return None, checks

    # ------------------------------------------------------------------ evidence
    @staticmethod
    def _fmt_chunk(c) -> str:
        attrs = f'id="{c.section_id}" doc="{c.doc_id}" type="{c.doc_type}" status="{c.status}" authority="{AUTHORITY_LABEL.get(c.authority_rank)}" effective_from="{c.effective_from}"'
        if c.applicability_note:
            attrs += f' applies_only_to="{c.applicability_note}"'
        return f"<source {attrs}>\n{c.doc_title} — {c.section_title}\n{c.text}\n</source>"

    def _order_facts(self, session: Session) -> str:
        order = session.verified_order
        if not order:
            return "(none — the customer is not verified for any order in this conversation)"
        hidden = {"delivery_pincode", "customer_id", "found", "verified", "verification_required"}
        facts = {k: v for k, v in order.items() if k not in hidden and v not in (None, "", []) and k != "items"}
        items = []
        for it in order.get("items") or []:
            prod = self.dataset.products.get(it.get("sku"), {})
            items.append({"line_no": it.get("line_no"), "sku": it.get("sku"), "title": it.get("title"), "qty": it.get("qty"),
                          "unit_price_inr": it.get("unit_price_inr"), "return_category": it.get("return_category"),
                          "fulfilled_by": prod.get("fulfilled_by"), "assured_badge": prod.get("assured"),
                          "warranty_months": prod.get("warranty_months")})
        facts["items"] = items
        checks = refund_checks(order, self.refund_table, self.refund_section, self.calendar, self.settings.reference_date)
        if checks:
            facts["computed_refund_timeline"] = checks  # deterministic, from the refund-timeline table in the KB
        pin = self.dataset.pincodes.get(str(order.get("delivery_pincode") or ""))
        if pin:
            facts["delivery_area"] = {"city": pin["city"], "metro": pin["metro"], "serviceable": pin["serviceable"],
                                      "cod_available": pin["cod_available"], "express_available": pin["express_available"]}
        return f"Verified order {session.current_order_id} (source: lookup_order):\n" + json.dumps(facts, ensure_ascii=False, indent=1)

    # ------------------------------------------------------------------ responder tools
    def _tools(self, session: Session, plan: PlannerOutput, tool_calls: list[ToolCall], ticket_log: list[dict]) -> list[ToolSpec]:
        cal = self.calendar
        date_prop = {"type": "string", "description": "YYYY-MM-DD"}

        def ticket_handler(args: dict) -> dict:
            reason = args.get("reason") or "agent_judgement"
            if reason not in TICKET_REASONS:
                reason = "agent_judgement"
            res = self._ticket(session, plan, reason, self._checks(session), tool_calls, summary_hint=str(args.get("summary", ""))[:500])
            ticket_log.append({"reason": reason, **res})
            return {"created": res.get("created") is True, "ticket_id": res.get("ticket_id"), "error": res.get("error")}

        return [
            ToolSpec("business_day_window", "Earliest and latest dates N business days after a start date (Mon-Sat, skipping warehouse holidays). Use for refund/logistics timelines.",
                     {"type": "object", "properties": {"start_date": date_prop, "min_days": {"type": "integer"}, "max_days": {"type": "integer"}},
                      "required": ["start_date", "min_days", "max_days"], "additionalProperties": False},
                     lambda a: cal.tool_business_day_window(a["start_date"], a["min_days"], a["max_days"])),
            ToolSpec("add_calendar_days", "Date that is N calendar days after a start date (use for return windows from the delivery date).",
                     {"type": "object", "properties": {"start_date": date_prop, "days": {"type": "integer"}},
                      "required": ["start_date", "days"], "additionalProperties": False},
                     lambda a: cal.tool_add_calendar_days(a["start_date"], a["days"])),
            ToolSpec("business_days_between", "Number of business days after start_date up to and including end_date.",
                     {"type": "object", "properties": {"start_date": date_prop, "end_date": date_prop},
                      "required": ["start_date", "end_date"], "additionalProperties": False},
                     lambda a: cal.tool_business_days_between(a["start_date"], a["end_date"])),
            ToolSpec("create_ticket", "Hand the case over to a human colleague. Use only when a handover rule (ESC-1, DWM-3, DWM-5) applies or the customer asks for a human. Returns created=true on success.",
                     {"type": "object", "properties": {"summary": {"type": "string", "description": "issue, intent, order ID, what the customer said, what you checked, reason"},
                                                       "reason": {"type": "string", "enum": TICKET_REASONS}},
                      "required": ["summary", "reason"], "additionalProperties": False},
                     ticket_handler),
        ]

    # ------------------------------------------------------------------ main entry
    def respond(self, session: Session, message: str, channel: str | None = None) -> TurnResult:
        t0 = time.time()
        channel = channel or session.channel
        message = (message or "").strip()[:2000]
        turn_no = len([t for t in session.turns if t["role"] == "customer"]) + 1
        diag: dict = {"llm_provider": getattr(self.llm, "provider", None), "llm_model": getattr(self.llm, "model", None),
                      "llm_mock": bool(getattr(self.llm, "is_mock", False)), "retrieval_backend": self.retriever.index.backend_description,
                      "semantic_embeddings": self.retriever.index.semantic}
        if not message:
            reply = localized("empty", session.language)
            return self._finish(session, turn_no, reply, "clarify", "empty_input", channel, [], [], [], {}, [], diag, t0, add_customer=False)

        session.turns.append({"role": "customer", "text": message})
        if self.llm is None:
            return self._finish(session, turn_no, localized("llm_down", guess_language(message)), "error", "", channel, [], [], [],
                                {}, ["LLM not configured: set ANTHROPIC_API_KEY (see README)."], diag, t0)

        errors: list[str] = []
        tool_calls: list[ToolCall] = []
        actions: list[str] = []
        records_before = len(self.llm.records)
        try:
            plan = self._plan(session, message, diag)
        except LLMError as exc:
            return self._finish(session, turn_no, localized("llm_down", guess_language(message)), "error", "", channel, [], [], [],
                                {}, [f"LLM error ({exc.kind}): {exc}"], diag, t0)
        lang = plan.language or guess_language(message)
        session.language = lang
        session.last_intent = plan.intent
        if plan.issue_summary and plan.intent not in ("verification", "greeting_or_unspecified"):
            session.issue_summary = scrub(plan.issue_summary, session.secrets)

        # ---------------- validate extracted entities against the literal message
        oid_cands = order_id_candidates(message)
        ver_cands = verification_candidates(message)
        order_id = plan.order_id if plan.order_id and re.sub(r"\D", "", plan.order_id) in oid_cands else None
        if order_id:
            order_id = re.sub(r"\D", "", order_id)
        elif len(oid_cands) == 1 and (session.awaiting == "order_id" or plan.needs_order_details):
            order_id = oid_cands[0]
        vval = plan.verification_value if plan.verification_value and re.sub(r"\D", "", plan.verification_value) in ver_cands else None
        if vval:
            vval = re.sub(r"\D", "", vval)
        elif session.awaiting == "verification" and len(ver_cands) == 1 and len(message) < 80 and not oid_cands:
            vval = ver_cands[0]
        if vval and vval not in session.secrets:
            session.secrets.append(vval)

        if order_id and order_id != session.current_order_id:
            session.current_order_id = order_id
        if order_id and session.awaiting == "order_id":
            session.awaiting = None

        # ---------------- identity workflow (IDV-1..3) — enforced in code
        # The order workflow runs when the customer gives an order ID / proof value, or when the
        # request genuinely depends on their own order. General policy questions are answered
        # directly, even while a verification is pending.
        needs_order = (bool(order_id)
                       or (bool(vval) and session.awaiting in ("verification", "order_id"))
                       or (plan.needs_order_details and not plan.is_general_policy_question))
        event = None
        oid = session.current_order_id
        if needs_order:
            if not oid:
                if vval:
                    session.pending_verification_value = vval
                session.awaiting = "order_id"
                event = "ask_order_id"
                actions.append("ASK_ORDER_ID: the request may depend on the customer's own order, which is not identified yet. If the evidence already answers the question in general terms, give that answer first (decision=answer) and offer to check their order; if nothing can be said without the order, just ask for the order ID (decision=clarify).")
            elif oid in session.locked_orders:
                event = "locked"
                actions.append(f"VERIFICATION_LOCKED for order {oid}: two failed verification attempts. Do NOT share any order details; offer general policy help and suggest contacting support from the registered phone number.")
            elif oid not in session.verified:
                if oid not in session.lookups_started:
                    res, call = self.toolbox.lookup_order(oid)
                    tool_calls.append(call)
                    session.lookups_started.add(oid)
                    if "error" in res:
                        errors.append(f"lookup_order failed: {res['error']}")
                value = vval or session.pending_verification_value
                session.pending_verification_value = None
                if value:
                    res, call = self.toolbox.lookup_order(oid, value)
                    tool_calls.append(call)
                    if res.get("verified") is True and res.get("found") is True:
                        session.verified[oid] = res
                        session.awaiting = None
                        event = "verified_now"
                        actions.append(f"VERIFIED: lookup_order verified the customer for order {oid}. Order facts are now available; continue with the customer's pending request if they made one, otherwise confirm and ask how you can help.")
                    elif "error" in res:
                        event = "tool_error"
                        errors.append(f"lookup_order failed: {res['error']}")
                        actions.append("LOOKUP_ERROR: the order system is unavailable. Say you cannot check the order right now; offer general help or a human colleague.")
                    else:
                        fails = session.failed_attempts.get(oid, 0) + 1
                        session.failed_attempts[oid] = fails
                        if fails >= MAX_VERIFICATION_ATTEMPTS:
                            session.locked_orders.add(oid)
                            session.awaiting = None
                            event = "failed_final"
                            actions.append("VERIFICATION_FAILED_TWICE: the details did not match again. Do NOT share any order details. Politely say you can't share order details, offer general policy help, and suggest contacting support from the registered phone number.")
                        else:
                            session.awaiting = "verification"
                            event = "failed_retry"
                            actions.append("VERIFICATION_FAILED_ONCE: the details did not match. Do not reveal what is on file. Ask the customer to try again with the last 4 digits of the registered phone number or the delivery pincode (one retry allowed).")
                else:
                    session.awaiting = "verification"
                    event = "ask_verification"
                    actions.append(f"ASK_VERIFICATION: lookup_order({oid}) requires verification. Ask for ONE of: last 4 digits of the registered phone number OR the 6-digit delivery pincode. Do not share any order details yet.")

        # ---------------- mandatory escalations (ESC-1) — ticket BEFORE any claim
        esc_reason, checks = self._mandatory_escalation(session, plan, message, event)
        ticket_log: list[dict] = []
        if esc_reason:
            res = self._ticket(session, plan, esc_reason, checks, tool_calls)
            ticket_log.append({"reason": esc_reason, **res})
            if res.get("created") is True:
                actions.append(f"TICKET_CREATED: {res['ticket_id']} (reason: {REASON_TEXT.get(esc_reason, esc_reason)}). Tell the customer a human colleague will contact them; never promise an outcome, amount or time beyond policy.")
            else:
                actions.append(f"TICKET_FAILED: create_ticket did not succeed ({res.get('error')}). Do NOT claim a handover; say honestly that it could not be created.")
        if plan.requests_exception_or_compensation and not esc_reason:
            actions.append("The customer asks for compensation/approval/exception: check the policy; refuse politely if the policy does not allow it.")

        # ---------------- retrieval
        queries = [message] + plan.search_queries
        try:
            rres = self.retriever.retrieve(queries, k=self.settings.top_k, exclude_doc_ids=STANDING_DOCS,
                                           mentioned_dates=plan.mentioned_dates)
            hits = rres.hits
        except Exception as exc:  # retrieval failure: answer only from standing rules, flag it
            hits = []
            errors.append(f"retrieval failed: {type(exc).__name__}: {exc}")
        products = self.dataset.match_products(message, plan.product_mentions)
        if products:
            session.last_products = [p["title"] for p in products]
        evidence_blocks = [self._fmt_chunk(h.chunk) for h in hits]
        evidence_blocks += [f'<source id="products.csv:{p["sku"]}" type="catalog">\n{product_evidence_text(p)}\n</source>' for p in products]
        if not evidence_blocks:
            evidence_blocks = ["(no relevant documents retrieved)"]
        standing_text = "\n".join(self._fmt_chunk(c) for c in self.standing)

        # ---------------- responder
        user = prompts.RESPONDER_USER.format(
            standing=standing_text, evidence="\n".join(evidence_blocks), order_facts=self._order_facts(session),
            actions="\n".join(actions) or "(no system actions this turn)", state=session.summary_for_model(),
            history=session.history_text(10), message=message)
        system = prompts.RESPONDER_SYSTEM.format(
            today=f"{self.settings.reference_date.isoformat()} ({self.settings.reference_date.strftime('%A')})",
            channel_rules=prompts.VOICE_RULES if channel == "voice" else prompts.CHAT_RULES,
            reply_language=prompts.LANG_NAMES.get(lang, "the customer's language"))
        tools = self._tools(session, plan, tool_calls, ticket_log)
        out: ResponderOutput | None = None
        calc_trace: list[dict] = []
        last_err = ""
        for attempt in range(2):
            try:
                raw, trace = self.llm.run_with_tools("responder", system, user + (f"\n\nYour previous output was invalid ({last_err}). Return only the JSON object." if last_err else ""),
                                                     tools, RESPONDER_SCHEMA, max_tokens=self.settings.responder_max_tokens,
                                                     effort=self.settings.responder_effort)
                calc_trace += trace
                out = ResponderOutput.model_validate(raw)
                break
            except (ValueError, ValidationError) as exc:
                last_err = str(exc)[:300]
            except LLMError as exc:
                last_err = f"{exc.kind}: {exc}"
                if not exc.retriable:
                    break
        diag["llm_calls"] = [r.__dict__ for r in self.llm.records[records_before:]]
        diag["calculator_calls"] = [t for t in calc_trace if t["tool"] != "create_ticket"]
        if out is None:
            errors.append(f"responder failed: {last_err}")
            created = [t for t in ticket_log if t.get("created")]
            if created:  # a ticket exists: say so honestly even though generation failed
                reply = {"en": f"I've created ticket {created[0]['ticket_id']} and a human colleague will contact you.",
                         "hi": f"मैंने टिकट {created[0]['ticket_id']} बना दिया है, एक सहयोगी आपसे संपर्क करेंगे।",
                         "hi-en": f"Maine ticket {created[0]['ticket_id']} bana diya hai, ek colleague aapse contact karenge."}.get(lang)
                reply = reply or f"I've created ticket {created[0]['ticket_id']} and a human colleague will contact you."
                decision = "escalate"
            else:
                reply, decision = localized("llm_down", lang), "error"
            return self._finish(session, turn_no, reply, decision, plan.intent, channel, hits, products, tool_calls,
                                {"event": event}, errors, diag, t0, ticket_log=ticket_log, plan=plan)

        # ---------------- guardrails
        reply, decision = out.reply, out.decision
        created = [t for t in ticket_log if t.get("created")]
        if created:
            decision = "escalate"
        elif decision == "escalate":
            res = self._ticket(session, plan, "agent_judgement", self._checks(session), tool_calls)
            ticket_log.append({"reason": "agent_judgement", **res})
            if res.get("created") is not True:
                reply = localized("ticket_failed", lang)
                errors.append("agent decided to escalate but create_ticket failed")
        if not created and any(t.get("created") is False for t in ticket_log) and decision != "escalate":
            pass
        if event in ("ask_verification", "failed_retry") and decision not in ("escalate",):
            decision = "clarify"
        elif event == "failed_final" and decision not in ("escalate",):
            decision = "refuse"
        elif event == "verified_now" and decision == "clarify":
            decision = "answer"  # nothing is missing after a successful verification
        if session.assistant_turns == 0 and not AI_RE.search(reply):
            reply = f"{INTRO.get(lang, INTRO['en'])} {reply}"
        secrets_to_scrub = list(session.secrets) + [str(o.get("delivery_pincode") or "") for o in session.verified.values()]
        reply = scrub(reply, secrets_to_scrub)
        diag["dropped_unknown_section_ids"] = []
        available = {c.section_id for c in self.standing} | {h.chunk.section_id for h in hits} | {f"products.csv:{p['sku']}" for p in products}
        used = []
        for sid in out.used_section_ids:
            sid = sid.strip()
            if sid in available and sid not in used:
                used.append(sid)
            elif sid not in available:
                diag["dropped_unknown_section_ids"].append(sid)
        diag["answer_supported_by_documents"] = out.answer_supported_by_documents
        return self._finish(session, turn_no, reply, decision, plan.intent, channel, hits, products, tool_calls,
                            {"event": event}, errors, diag, t0, used=used, ticket_log=ticket_log, plan=plan)

    # ------------------------------------------------------------------ result
    def _finish(self, session: Session, turn_no: int, reply: str, decision: str, intent: str, channel: str, hits, products,
                tool_calls, verification_extra: dict, errors: list[str], diag: dict, t0: float, used: list[str] | None = None,
                ticket_log: list[dict] | None = None, plan: PlannerOutput | None = None, add_customer: bool = True) -> TurnResult:
        session.turns.append({"role": "assistant", "text": reply})
        session.assistant_turns += 1
        session.updated_at = time.time()
        used = used or []
        sources = []
        for c in self.standing:
            if c.section_id in used:
                sources.append({"section_id": c.section_id, "doc_id": c.doc_id, "file": c.file, "doc_title": c.doc_title,
                                "section_title": c.section_title, "doc_type": c.doc_type, "status": c.status,
                                "authority": AUTHORITY_LABEL.get(c.authority_rank), "via": "standing_rules", "score": None,
                                "used": True})
        for h in hits:
            d = h.to_dict()
            d["used"] = h.chunk.section_id in used
            sources.append(d)
        for p in products:
            sid = f"products.csv:{p['sku']}"
            sources.append({"section_id": sid, "doc_id": "products.csv", "file": "data/products.csv", "doc_title": p["title"],
                            "section_title": "catalog row", "doc_type": "catalog", "status": "current", "via": "catalog_match",
                            "score": None, "used": sid in used})
        tickets = [t for t in (ticket_log or []) if t.get("created")]
        escalation = {"required": bool(ticket_log), "created": bool(tickets),
                      "ticket_ids": [t.get("ticket_id") for t in tickets],
                      "reasons": [t.get("reason") for t in (ticket_log or [])],
                      "failed": [t.get("error") for t in (ticket_log or []) if not t.get("created")]}
        verification = {"order_id": session.current_order_id, "state": session.verification_state(),
                        "failures": session.failed_attempts.get(session.current_order_id or "", 0), **verification_extra}
        diag["latency_ms"] = int((time.time() - t0) * 1000)
        return TurnResult(session_id=session.session_id, turn=turn_no, reply=reply, decision=decision, intent=intent,
                          language=session.language, channel=channel, sources=sources, used_sections=used,
                          retrieved_sections=[h.chunk.section_id for h in hits] + [f"products.csv:{p['sku']}" for p in products],
                          tool_calls=tool_calls, verification=verification, escalation=escalation, errors=errors,
                          diagnostics=diag, planner=plan.model_dump() if plan else {})


class BaselineAgent:
    """Step 1: the same LLM with a ZentroMart support prompt, but no documents, data or tools."""

    def __init__(self, llm, settings: Settings):
        self.llm = llm
        self.settings = settings

    def respond(self, message: str) -> dict:
        t0 = time.time()
        try:
            raw = self.llm.generate_json("baseline", prompts.BASELINE_SYSTEM, f"Customer message:\n<<<{message}>>>",
                                         BASELINE_SCHEMA, max_tokens=self.settings.planner_max_tokens, effort=self.settings.planner_effort)
            out = BaselineOutput.model_validate(raw)
            return {"reply": out.reply, "decision": out.decision, "errors": [], "latency_ms": int((time.time() - t0) * 1000),
                    "mock": bool(getattr(self.llm, "is_mock", False))}
        except (LLMError, ValueError, ValidationError) as exc:
            return {"reply": "", "decision": "error", "errors": [f"{type(exc).__name__}: {exc}"],
                    "latency_ms": int((time.time() - t0) * 1000), "mock": bool(getattr(self.llm, "is_mock", False))}
