"""Adapter around the SUPPLIED mock tools (dataset ``tools/tools.py``), used unchanged.

* The module is imported from the dataset folder, so the original identity gate in
  ``lookup_order`` is what protects order data.
* ``create_ticket`` writes to ``tools/tickets_created.csv`` by default; we point the module's
  ``_TICKETS`` path at ``runtime/tickets/`` so the source dataset is never modified.
* Every call is recorded with secrets redacted; summaries are scrubbed of verification
  values and phone-like numbers before they reach the tool.
"""
from __future__ import annotations

import importlib.util
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_MODULE_CACHE: dict[str, Any] = {}


class ToolsUnavailable(RuntimeError):
    pass


def load_supplied_tools(dataset_dir: Path, tickets_file: Path):
    """Import ``<dataset>/tools`` as an isolated package and return its module."""
    pkg_dir = dataset_dir / "tools"
    key = str(pkg_dir.resolve())
    with _LOCK:
        mod = _MODULE_CACHE.get(key)
        if mod is None:
            init = pkg_dir / "__init__.py"
            if not init.is_file():
                raise ToolsUnavailable(f"supplied tools package not found at {pkg_dir}")
            name = "zentromart_supplied_tools"
            spec = importlib.util.spec_from_file_location(name, init, submodule_search_locations=[str(pkg_dir)])
            if spec is None or spec.loader is None:
                raise ToolsUnavailable("could not load supplied tools")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            _MODULE_CACHE[key] = mod
        inner = sys.modules.get("zentromart_supplied_tools.tools")
        if inner is not None:
            tickets_file.parent.mkdir(parents=True, exist_ok=True)
            inner._TICKETS = str(tickets_file)  # redirect runtime ticket log out of the dataset
        return mod


PHONE_RE = re.compile(r"(\+?91[\s-]?)?\b\d{5}[\s-]?\d{5}\b")


def scrub(text: str, secrets: list[str]) -> str:
    """Remove verification values and phone-like numbers from free text."""
    out = text or ""
    for s in sorted({s for s in secrets if s}, key=len, reverse=True):
        out = re.sub(rf"(?<!\d){re.escape(s)}(?!\d)", "[redacted]", out)
    return PHONE_RE.sub("[redacted-phone]", out)


@dataclass
class ToolCall:
    name: str
    args: dict            # redacted, safe to show / log
    ok: bool
    outcome: str          # short human-readable outcome
    expected_match_key: dict = field(default_factory=dict)  # internal only, for evaluation

    def public(self) -> dict:
        return {"name": self.name, "args": self.args, "ok": self.ok, "outcome": self.outcome}


class ToolBox:
    """Executes the supplied tools and records each call (per conversation turn)."""

    def __init__(self, dataset_dir: Path, tickets_file: Path, module: Any = None):
        self.module = module or load_supplied_tools(dataset_dir, tickets_file)

    def lookup_order(self, order_id: str, verification_value: str | None = None) -> tuple[dict, ToolCall]:
        try:
            if verification_value is None:
                result = self.module.lookup_order(order_id)
            else:
                result = self.module.lookup_order(order_id, verification_value=verification_value)
        except Exception as exc:  # tool failure is reported, never hidden
            call = ToolCall("lookup_order", {"order_id": order_id, **({"verification_value": "[redacted]"} if verification_value else {})},
                            False, f"tool error: {type(exc).__name__}",
                            {"order_id": order_id, "verification_value": verification_value})
            return {"error": str(exc)}, call
        if verification_value is None:
            outcome = "verification required (no order details returned)"
            ok = bool(result.get("verification_required"))
        elif result.get("verified"):
            outcome, ok = "verified: order details unlocked", True
        else:
            outcome, ok = "verification failed (no order details returned)", True
        args = {"order_id": order_id}
        if verification_value is not None:
            args["verification_value"] = "[redacted]"
        return result, ToolCall("lookup_order", args, ok, outcome,
                                {"order_id": order_id, "verification_value": verification_value})

    def create_ticket(self, summary: str, order_id: str | None, reason: str, intent: str,
                      secrets: list[str]) -> tuple[dict, ToolCall]:
        clean = scrub(summary, secrets)
        try:
            result = self.module.create_ticket(summary=clean, order_id=order_id, reason=reason, intent=intent)
        except Exception as exc:
            result = {"created": False, "error": f"{type(exc).__name__}: {exc}"}
        created = result.get("created") is True
        outcome = f"created ticket {result.get('ticket_id')}" if created else f"NOT created: {result.get('error', 'unknown error')}"
        call = ToolCall("create_ticket", {"order_id": order_id, "reason": reason, "intent": intent, "summary": clean},
                        created, outcome, {"created": created})
        return result, call
