"""Evaluation scoring semantics (must_include / must_not_include / tools)."""
import unittest

from tests.helpers import ROOT  # noqa: F401

from zentromart.evaluation.scoring import (check_must_include, check_must_not, check_tools, contains_fact, dates_in,
                                           parse_expected_tools)


class TestScoring(unittest.TestCase):
    def test_alternatives_and_groups(self):
        ok, missing = check_must_include("Refunds reach you instantly.", "instant / 2 hours")
        self.assertTrue(ok)
        ok, missing = check_must_include("It takes 3 days", "1 ; 3")
        self.assertFalse(ok)
        self.assertEqual(missing, ["1"])
        self.assertTrue(check_must_include("anything", "")[0])

    def test_dates_any_format(self):
        for text in ("by 27 October 2026", "by October 27, 2026", "on the 27th of October", "2026-10-27", "27 अक्टूबर 2026", "27/10/2026"):
            self.assertTrue(contains_fact(text, "2026-10-27"), text)
        self.assertFalse(contains_fact("by 28 October 2026", "27 October 2026"))
        self.assertIn(__import__("datetime").date(2026, 11, 13), dates_in("13 नवंबर 2026 तक"))

    def test_numbers_words_and_boundaries(self):
        self.assertTrue(contains_fact("up to three delivery attempts", "3"))
        self.assertTrue(contains_fact("सात दिन", "7"))
        self.assertFalse(contains_fact("on 17 October", "7"))
        self.assertFalse(contains_fact("it is free", "AI"))
        self.assertTrue(contains_fact("I'm ZentroMart's AI assistant", "AI"))
        self.assertTrue(contains_fact("Mai aapki ek hi coupon", "ek"))
        self.assertTrue(contains_fact("₹10,000 se zyada", "10000"))

    def test_must_not(self):
        ok, hits = check_must_not("Refunds take 5 to 7 days.", "5 to 7 days ; 5-7 days")
        self.assertFalse(ok)
        self.assertTrue(check_must_not("UPI refunds take 1 to 3 business days.", "5 to 7 days ; 5-7 days")[0])
        self.assertFalse(check_must_not("It shipped on October 9", "2026-10-09 ; 9 October 2026")[0])

    def test_tool_expectations(self):
        exp = parse_expected_tools("lookup_order(order_id=46637226, verification_value=4533)")
        self.assertEqual(exp, [{"name": "lookup_order", "args": {"order_id": "46637226", "verification_value": "4533"}}])
        act = [{"name": "lookup_order", "order_id": "46637226", "verification_value": "4533", "created": None}]
        self.assertTrue(check_tools(exp, act)["pass"])
        self.assertFalse(check_tools(parse_expected_tools("lookup_order(order_id=46637226)"), act)["pass"])
        self.assertTrue(check_tools(parse_expected_tools("create_ticket()"), [{"name": "create_ticket", "created": True}])["pass"])
        self.assertFalse(check_tools(parse_expected_tools("create_ticket()"), [{"name": "create_ticket", "created": False}])["pass"])
        res = check_tools([], act)
        self.assertFalse(res["pass"])
        self.assertEqual(res["unexpected"][0]["verification_value"], "[redacted]")


if __name__ == "__main__":
    unittest.main()
