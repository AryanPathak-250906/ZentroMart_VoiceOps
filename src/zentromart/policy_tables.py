"""Facts read from policy TABLES in the knowledge base at start-up (no numbers hard-coded here).

Currently: the refund-timeline table (the current policy section whose table has a
"Payment method" column, i.e. REF-2). It lets code check ESC-1 rule 5 (refund more than 3
business days past its maximum timeline) deterministically for a verified order.
"""
from __future__ import annotations

import re
from datetime import date

# order-record refund method code -> words that identify its row in the policy table
METHOD_ROW = {
    "upi": (["upi"], []),
    "card": (["card"], []),
    "netbanking": (["net banking"], []),
    "wallet": (["wallet"], ["cash-on-delivery", "cash on delivery"]),
    "cod_bank": (["bank transfer"], []),
    "cod_wallet": (["wallet", "cash-on-delivery"], []),
}


def parse_refund_table(chunks) -> tuple[str | None, dict[str, tuple[int, int]]]:
    """Return (section_id, {method_code: (min_business_days, max_business_days)})."""
    for c in chunks:
        if c.doc_type != "policy" or c.status != "current" or "| payment method" not in c.text.lower():
            continue
        rows = {}
        for line in c.text.splitlines():
            cells = [x.strip() for x in line.strip().strip("|").split("|")]
            if len(cells) != 2 or cells[0].lower().startswith(("payment", "---")):
                continue
            label, timing = cells[0].lower(), cells[1].lower()
            m = re.search(r"(\d+)\s*to\s*(\d+)\s*business days", timing)
            span = (int(m.group(1)), int(m.group(2))) if m else ((0, 0) if "instant" in timing else None)
            if span is None:
                continue
            for code, (need, avoid) in METHOD_ROW.items():
                if all(w in label for w in need) and not any(w in label for w in avoid):
                    rows.setdefault(code, span)
        if rows:
            return c.section_id, rows
    return None, {}


def refund_checks(order: dict, table: dict[str, tuple[int, int]], section_id: str | None, calendar, today: date) -> list[dict]:
    out = []
    for kind in ("return", "cancellation", "rto"):
        init = order.get(f"{kind}_refund_initiated_on")
        method = order.get(f"{kind}_refund_method")
        if not init or not method:
            continue
        completed = order.get("return_refund_completed_on") if kind == "return" else None
        span = table.get(str(method))
        item = {"refund": kind, "initiated_on": str(init), "method": method, "completed_on": completed,
                "rrn": order.get(f"{kind}_rrn") or order.get("cancellation_rrn" if kind == "cancellation" else ""),
                "source": section_id}
        if span:
            start = date.fromisoformat(str(init))
            earliest = calendar.add_business_days(start, span[0])
            latest = calendar.add_business_days(start, span[1])
            past = calendar.business_days_between(latest, today) if today > latest else 0
            item.update({"policy_business_days": f"{span[0]} to {span[1]}", "expected_earliest": earliest.isoformat(),
                         "expected_latest": latest.isoformat(), "business_days_past_latest": past,
                         "overdue_more_than_3_business_days": (not completed) and past > 3})
        out.append(item)
    return out
