"""Structured-output schemas (JSON Schema for the API + pydantic validation before use)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

LANGS = ["en", "hi", "hi-en", "ta", "te", "kn", "ml", "mr", "bn", "gu", "pa"]
SIGNALS = ["human_requested", "unauthorized_activity", "legal_threat", "safety_risk",
           "damaged_wrong_or_missing_item", "repeat_contact", "refund_not_received"]
DECISIONS = ["answer", "clarify", "escalate", "refuse"]


def _strict(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


PLANNER_SCHEMA = _strict({
    "language": {"type": "string", "enum": LANGS,
                 "description": "Language of the CURRENT customer message: en, hi (Devanagari Hindi), hi-en (Hinglish in Latin script), or another Indian language code."},
    "intent": {"type": "string", "description": "Short snake_case intent, e.g. order_status, cancel_order, return_window, return_eligibility, refund_status, refund_timeline, damaged_item, warranty, delivery, payment, coupon, membership, troubleshooting, account_security, human_request, compensation_request, verification, greeting_or_unspecified, other."},
    "order_id": {"type": ["string", "null"], "description": "8-digit order ID typed in THIS message (digits only), else null. Never invent."},
    "verification_value": {"type": ["string", "null"], "description": "Digits the customer gives in THIS message as last 4 phone digits or 6-digit delivery pincode, else null. Never invent."},
    "needs_order_details": {"type": "boolean", "description": "True if a correct reply requires private data about the customer's specific order (status, items, dates, refund of THEIR order)."},
    "is_general_policy_question": {"type": "boolean", "description": "True if the question can be answered from company policy without the customer's order."},
    "search_queries": {"type": "array", "items": {"type": "string"},
                       "description": "1-3 standalone ENGLISH search queries for the company knowledge base, resolving pronouns/follow-ups from the conversation (e.g. 'refund timeline for UPI after return pickup')."},
    "product_mentions": {"type": "array", "items": {"type": "string"},
                         "description": "Product names or SKUs mentioned in this message or the follow-up's referent, normalised (e.g. 'ELE-0012', 'Kaafi Running Sneakers')."},
    "mentioned_dates": {"type": "array", "items": {"type": "string"},
                        "description": "Calendar dates referred to in the message, resolved to YYYY-MM-DD using today's date (e.g. 'parso' = 2 days ago)."},
    "escalation_signals": {"type": "array", "items": {"type": "string", "enum": SIGNALS},
                           "description": "Only signals clearly present in the customer's words."},
    "requests_exception_or_compensation": {"type": "boolean",
                                           "description": "Customer demands money, vouchers, approvals or exceptions the agent cannot grant."},
    "issue_summary": {"type": "string", "description": "One English sentence summarising the customer's issue so far. No phone digits or pincodes."},
})


class PlannerOutput(BaseModel):
    language: str = "en"
    intent: str = "other"
    order_id: Optional[str] = None
    verification_value: Optional[str] = None
    needs_order_details: bool = False
    is_general_policy_question: bool = True
    search_queries: list[str] = Field(default_factory=list)
    product_mentions: list[str] = Field(default_factory=list)
    mentioned_dates: list[str] = Field(default_factory=list)
    escalation_signals: list[str] = Field(default_factory=list)
    requests_exception_or_compensation: bool = False
    issue_summary: str = ""

    @field_validator("language")
    @classmethod
    def _lang(cls, v: str) -> str:
        return v if v in LANGS else "en"

    @field_validator("escalation_signals")
    @classmethod
    def _signals(cls, v: list[str]) -> list[str]:
        return [s for s in v if s in SIGNALS]

    @field_validator("search_queries", "product_mentions", "mentioned_dates")
    @classmethod
    def _cap(cls, v: list[str]) -> list[str]:
        return [s for s in v if isinstance(s, str) and s.strip()][:5]


RESPONDER_SCHEMA = _strict({
    "reply": {"type": "string", "description": "The exact customer-facing reply, in the required language."},
    "decision": {"type": "string", "enum": DECISIONS},
    "used_section_ids": {"type": "array", "items": {"type": "string"},
                         "description": "Section IDs (e.g. RET-2) and catalog IDs (products.csv:SKU) the reply relies on. Only IDs present in the evidence."},
    "answer_supported_by_documents": {"type": "boolean",
                                      "description": "False if the documents do not cover the question and you said you don't have that information."},
})


class ResponderOutput(BaseModel):
    reply: str
    decision: Literal["answer", "clarify", "escalate", "refuse"]
    used_section_ids: list[str] = Field(default_factory=list)
    answer_supported_by_documents: bool = True

    @field_validator("reply")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("reply is empty")
        return v.strip()


BASELINE_SCHEMA = _strict({
    "reply": {"type": "string"},
    "decision": {"type": "string", "enum": DECISIONS},
})


class BaselineOutput(BaseModel):
    reply: str
    decision: Literal["answer", "clarify", "escalate", "refuse"]
