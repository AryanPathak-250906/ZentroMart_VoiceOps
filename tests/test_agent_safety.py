"""Conversation safety and workflow tests with a scripted MOCK LLM (no network, not model scores)."""
import csv
import unittest
from types import SimpleNamespace

from tests.helpers import DATASET, cleanup, echo_responder, make_agent, require_dataset, rule_planner

from zentromart.conversation import Session
from zentromart.llm import LLMError, ScriptedLLM


def _order_and_secrets(order_id):
    orders = {r["order_id"]: r for r in csv.DictReader(open(DATASET / "data" / "orders.csv", encoding="utf-8"))}
    custs = {r["customer_id"]: r for r in csv.DictReader(open(DATASET / "data" / "customers.csv", encoding="utf-8"))}
    o = orders[order_id]
    return o, custs[o["customer_id"]]["phone"][-4:], o["delivery_pincode"]


def _wrong(value):
    return "1111" if value != "1111" else "2222"


@require_dataset
class TestVerificationWorkflow(unittest.TestCase):
    def setUp(self):
        self.agent, self.tmp = make_agent()
        self.llm = self.agent.llm
        # an order with a non-trivial record (taken from the data at test time, not hard-coded answers)
        self.oid = next(r["order_id"] for r in csv.DictReader(open(DATASET / "data" / "orders.csv", encoding="utf-8"))
                        if r["status"] == "shipped")
        self.order, self.last4, self.pincode = _order_and_secrets(self.oid)
        self.spaced = f"{self.oid[:4]} {self.oid[4:]}"

    def tearDown(self):
        cleanup(self.tmp)

    def responder_prompts(self):
        return [u for p, u in self.llm.calls if p == "responder"]

    def test_first_turn_ai_disclosure_enforced(self):
        def no_ai(user, tools):
            return {"reply": "How can I help?", "decision": "clarify", "used_section_ids": [], "answer_supported_by_documents": True}
        agent, tmp = make_agent(responder=no_ai)
        try:
            r = agent.respond(Session("s"), "Hi, I need some help with an order.")
            self.assertRegex(r.reply, r"\bAI\b")
            r2 = agent.respond(Session("s2"), "नमस्ते, मुझे मदद चाहिए")
            self.assertIn("AI", r2.reply)
            self.assertTrue(r.diagnostics["llm_mock"])
        finally:
            cleanup(tmp)

    def test_missing_order_id_asks_and_no_lookup(self):
        s = Session("s")
        r = self.agent.respond(s, "Where is my order? It hasn't arrived.")
        self.assertEqual(r.decision, "clarify")
        self.assertEqual(s.awaiting, "order_id")
        self.assertEqual(r.tool_calls, [])

    def test_unverified_lookup_then_success(self):
        s = Session("s")
        self.agent.respond(s, "Hi, I need some help with my order.")
        r = self.agent.respond(s, f"The order number is {self.spaced}.")
        self.assertEqual([(c.name, c.expected_match_key["verification_value"]) for c in r.tool_calls], [("lookup_order", None)])
        self.assertEqual(r.decision, "clarify")
        self.assertEqual(s.verification_state(), "awaiting_verification")
        self.assertNotIn(self.order["promised_date"], self.responder_prompts()[-1])
        r = self.agent.respond(s, f"The last four digits are {self.last4}.")
        self.assertEqual(r.verification["state"], "verified")
        self.assertEqual(r.decision, "answer")
        self.assertEqual(r.tool_calls[0].expected_match_key, {"order_id": self.oid, "verification_value": self.last4})
        prompt = self.responder_prompts()[-1]
        self.assertIn(self.order["promised_date"], prompt)          # facts unlocked only now
        self.assertNotIn(self.pincode, prompt.split("<order_facts>")[1].split("</order_facts>")[0])  # address never passed
        self.assertNotIn(self.last4, r.reply)
        self.assertNotIn(self.last4, str(r.public()))               # redacted in public output

    def test_pincode_verification(self):
        s = Session("s")
        self.agent.respond(s, f"My order {self.spaced} hasn't arrived")
        r = self.agent.respond(s, f"Pincode {self.pincode} hai.")
        self.assertEqual(r.verification["state"], "verified")

    def test_failed_verification_one_retry_then_locked(self):
        s = Session("s")
        self.agent.respond(s, f"Order number is {self.spaced}")
        bad = _wrong(self.last4)
        r1 = self.agent.respond(s, f"Last four digits {bad}")
        self.assertEqual((r1.decision, r1.verification["state"]), ("clarify", "failed_once"))
        r2 = self.agent.respond(s, f"Last four digits {bad}")
        self.assertEqual((r2.decision, r2.verification["state"]), ("refuse", "locked"))
        self.assertEqual(len(r2.tool_calls), 1)
        r3 = self.agent.respond(s, f"Last four digits {self.last4}")  # even the right value is no longer tried
        self.assertEqual([c.name for c in r3.tool_calls], [])
        self.assertEqual(r3.verification["state"], "locked")
        for p in self.responder_prompts():
            self.assertNotIn(self.order["promised_date"], p)
        self.assertIn("VERIFICATION_FAILED_TWICE", self.responder_prompts()[2])

    def test_planner_invented_order_id_is_ignored(self):
        def lying_planner(user):
            d = rule_planner(user)
            d["order_id"] = self.oid  # not present in the message
            d["needs_order_details"] = True
            return d
        agent, tmp = make_agent(planner=lying_planner)
        try:
            s = Session("s")
            r = agent.respond(s, "What is the status of my order?")
            self.assertIsNone(s.current_order_id)
            self.assertEqual(r.tool_calls, [])
        finally:
            cleanup(tmp)

    def test_lookup_tool_failure_is_reported(self):
        def boom(*a, **k):
            raise RuntimeError("database offline")
        real = self.agent.toolbox.module
        self.agent.toolbox.module = SimpleNamespace(lookup_order=boom, create_ticket=real.create_ticket)
        s = Session("s")
        r = self.agent.respond(s, f"Order {self.spaced} status?")
        self.assertTrue(any("lookup_order failed" in e for e in r.errors))
        self.assertFalse(r.tool_calls[0].ok)


@require_dataset
class TestTicketsAndEscalation(unittest.TestCase):
    def setUp(self):
        self.agent, self.tmp = make_agent()

    def tearDown(self):
        cleanup(self.tmp)

    def test_human_request_creates_ticket_in_runtime_dir(self):
        s = Session("s")
        r = self.agent.respond(s, "Can I speak to a human please?")
        self.assertEqual(r.decision, "escalate")
        self.assertTrue(r.escalation["created"])
        self.assertTrue(r.escalation["ticket_ids"][0].startswith("TKT"))
        self.assertTrue(self.agent.settings.tickets_file.is_file())
        self.assertFalse((DATASET / "tools" / "tickets_created.csv").exists(), "source dataset must not be modified")
        # repeating the request does not create a duplicate ticket for the same reason
        r2 = self.agent.respond(s, "human please")
        self.assertEqual(len(s.tickets), 1)
        self.assertEqual(r2.decision, "escalate")

    def test_ticket_summary_scrubs_secrets(self):
        s = Session("s")
        oid = next(r["order_id"] for r in csv.DictReader(open(DATASET / "data" / "orders.csv", encoding="utf-8")))
        _, last4, _ = _order_and_secrets(oid)
        self.agent.respond(s, f"order {oid[:4]} {oid[4:]}")
        self.agent.respond(s, f"last four digits {last4}")
        self.agent.respond(s, "I want to talk to a person")
        text = self.agent.settings.tickets_file.read_text(encoding="utf-8")
        self.assertNotIn(last4, text.split("\n", 1)[1].replace(oid, ""))
        for field in ("Issue:", "Intent:", "Order ID:", "Customer said:", "Checks done:", "Reason for escalation:"):
            self.assertIn(field, text)

    def test_ticket_failure_is_not_claimed(self):
        real = self.agent.toolbox.module
        self.agent.toolbox.module = SimpleNamespace(lookup_order=real.lookup_order,
                                                    create_ticket=lambda **kw: {"created": False, "error": "ticket service down"})
        r = self.agent.respond(Session("s"), "There's an order on my account that I never placed.")
        self.assertFalse(r.escalation["created"])
        self.assertTrue(r.escalation["required"])
        self.assertIn("ticket service down", r.escalation["failed"][0])
        self.assertNotEqual(r.decision, "escalate")

    def test_responder_escalation_without_ticket_gets_ticket(self):
        def esc(user, tools):
            return {"reply": "A colleague will help you.", "decision": "escalate", "used_section_ids": ["ESC-1"],
                    "answer_supported_by_documents": True}
        agent, tmp = make_agent(responder=esc)
        try:
            r = agent.respond(Session("s"), "My refund is very late")
            self.assertTrue(r.escalation["created"])
            self.assertEqual([c.name for c in r.tool_calls], ["create_ticket"])
        finally:
            cleanup(tmp)

    def test_high_value_damage_escalates_from_catalog_price(self):
        r = self.agent.respond(Session("s"), "The Kestrel Book 14 Laptop (SKU ELE-0022) I received has a crack.")
        self.assertEqual(r.decision, "escalate")
        self.assertIn("high_value_damage", r.escalation["reasons"])


@require_dataset
class TestFailureHandling(unittest.TestCase):
    def test_missing_llm(self):
        agent, tmp = make_agent()
        agent.llm = None
        try:
            r = agent.respond(Session("s"), "hello")
            self.assertEqual(r.decision, "error")
            self.assertIn("LLM not configured", r.errors[0])
        finally:
            cleanup(tmp)

    def test_llm_timeout_returns_safe_reply(self):
        llm = ScriptedLLM(rule_planner, echo_responder, fail_with=LLMError("timeout", "timed out", retriable=True))
        agent, tmp = make_agent(llm=llm)
        try:
            r = agent.respond(Session("s"), "How long do UPI refunds take?")
            self.assertEqual(r.decision, "error")
            self.assertTrue(r.reply)
        finally:
            cleanup(tmp)

    def test_malformed_responder_output(self):
        calls = {"n": 0}
        def bad(user, tools):
            calls["n"] += 1
            return {"reply": "", "decision": "maybe"}
        agent, tmp = make_agent(responder=bad)
        try:
            r = agent.respond(Session("s"), "Can I return shoes?")
            self.assertEqual(calls["n"], 2)  # one controlled retry
            self.assertEqual(r.decision, "error")
            self.assertTrue(any("responder failed" in e for e in r.errors))
        finally:
            cleanup(tmp)

    def test_invalid_planner_output_falls_back(self):
        agent, tmp = make_agent(planner=lambda u: {"language": 42, "search_queries": "oops"})
        try:
            r = agent.respond(Session("s"), "Order 1234 5678 please")
            self.assertIn("planner_fallback", r.diagnostics)
            self.assertEqual(r.tool_calls[0].name, "lookup_order")
        finally:
            cleanup(tmp)

    def test_retrieval_failure_is_flagged(self):
        agent, tmp = make_agent()
        def broken(*a, **k):
            raise RuntimeError("index corrupted")
        agent.retriever.retrieve = broken
        try:
            r = agent.respond(Session("s"), "How long do refunds take?")
            self.assertTrue(any("retrieval failed" in e for e in r.errors))
        finally:
            cleanup(tmp)

    def test_session_isolation(self):
        agent, tmp = make_agent()
        try:
            oid = next(r["order_id"] for r in csv.DictReader(open(DATASET / "data" / "orders.csv", encoding="utf-8")))
            _, last4, _ = _order_and_secrets(oid)
            a = Session("a")
            agent.respond(a, f"order {oid}")
            agent.respond(a, f"last four digits {last4}")
            self.assertEqual(a.verification_state(), "verified")
            b = Session("b")
            r = agent.respond(b, "what's the status of my order?")
            self.assertEqual(b.verification_state(), "no_order")
            self.assertEqual(r.tool_calls, [])
        finally:
            cleanup(tmp)


if __name__ == "__main__":
    unittest.main()


@require_dataset
class TestPolicyTablesAndRefunds(unittest.TestCase):
    def setUp(self):
        self.agent, self.tmp = make_agent()

    def tearDown(self):
        cleanup(self.tmp)

    def test_refund_table_parsed_from_kb(self):
        self.assertEqual(self.agent.refund_section, "REF-2")
        t = self.agent.refund_table
        self.assertEqual(set(t), {"upi", "card", "netbanking", "wallet", "cod_bank", "cod_wallet"})
        self.assertTrue(all(lo <= hi for lo, hi in t.values()))

    def test_overdue_refund_escalates_with_computed_check(self):
        from zentromart.policy_tables import refund_checks
        rows = list(csv.DictReader(open(DATASET / "data" / "orders.csv", encoding="utf-8")))
        cands = [r for r in rows if r["return_refund_initiated_on"] and not r["return_refund_completed_on"]]
        order = None
        for r in cands:  # find one that the computed check marks overdue (data-driven, not hard-coded)
            rec = self.agent.toolbox.module.lookup_order(r["order_id"], verification_value=r["delivery_pincode"])
            chk = refund_checks(rec, self.agent.refund_table, self.agent.refund_section, self.agent.calendar,
                                self.agent.settings.reference_date)
            if chk and chk[0].get("overdue_more_than_3_business_days"):
                order = r
                break
        self.assertIsNotNone(order, "dataset should contain at least one overdue refund")

        def planner(user):
            d = rule_planner(user)
            if "refund" in user.split("CURRENT customer message:")[-1].lower():
                d.update(intent="refund_status", escalation_signals=["refund_not_received"], needs_order_details=True,
                         is_general_policy_question=False)
            return d
        agent, tmp = make_agent(planner=planner)
        try:
            s = Session("s")
            agent.respond(s, f"order {order['order_id']}")
            agent.respond(s, f"pincode {order['delivery_pincode']}")
            r = agent.respond(s, "where is my refund?")
            self.assertIn("overdue_refund", r.escalation["reasons"])
            self.assertTrue(r.escalation["created"])
            self.assertEqual(r.decision, "escalate")
        finally:
            cleanup(tmp)

    def test_reference_backfill(self):
        res = self.agent.retriever.retrieve(["shipped order cancel refuse parcel"], k=3)
        ids = [h.chunk.section_id for h in res.hits]
        if "CAN-2" in ids:
            self.assertIn("REF-1", ids)

    def test_general_question_not_forced_to_clarify(self):
        def planner(user):
            d = rule_planner(user)
            d.update(needs_order_details=True, is_general_policy_question=False)
            return d
        def responder(user, tools):
            return {"reply": "A shipped order can't be cancelled, but you can refuse it at the door.", "decision": "answer",
                    "used_section_ids": ["CAN-2"], "answer_supported_by_documents": True}
        agent, tmp = make_agent(planner=planner, responder=responder)
        try:
            r = agent.respond(Session("s"), "My order is on its way, can I stop it?")
            self.assertEqual(r.decision, "answer")
        finally:
            cleanup(tmp)
