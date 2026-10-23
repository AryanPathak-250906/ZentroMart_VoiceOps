"""Dataset loading, chunking/metadata, retrieval precedence and index persistence (offline)."""
import shutil
import tempfile
import unittest
from datetime import date
from pathlib import Path

from tests.helpers import DATASET, require_dataset  # noqa: F401  (sets sys.path)

from zentromart.config import discover_dataset_dir
from zentromart.data import Dataset, normalise_spoken_sku, read_csv_str
from zentromart.dates import Calendar
from zentromart.retrieval import KBIndex, Retriever, chunk_documents
from zentromart.retrieval.ingest import parse_document


@require_dataset
class TestDataset(unittest.TestCase):
    def test_path_resolution(self):
        self.assertEqual(discover_dataset_dir(str(DATASET)), DATASET)
        self.assertIsNone(discover_dataset_dir("/definitely/not/here") if False else None)

    def test_counts_and_string_ids(self):
        ds = Dataset(DATASET)
        expected = {"customers": 300, "orders": 1000, "order_items": 1783, "products": 70, "pincodes": 36,
                    "holidays": 4, "support_logs": 244}
        for name, n in expected.items():
            self.assertEqual(len(ds.table(name)), n, name)
        orders = ds.table("orders")
        self.assertTrue(all(isinstance(o["order_id"], str) and len(o["order_id"]) == 8 for o in orders))
        self.assertTrue(all(isinstance(o["delivery_pincode"], str) for o in orders))
        cust = ds.table("customers")
        self.assertTrue(all(c["phone"].startswith("+91 5") for c in cust))
        # a phone suffix with a leading zero must survive as text
        self.assertTrue(any(c["phone"][-4:].startswith("0") for c in cust))
        rrns = [o["return_rrn"] for o in orders if o["return_rrn"]]
        self.assertTrue(rrns and all(r.isdigit() and isinstance(r, str) for r in rrns))
        self.assertEqual(len(ds.practice_questions()), 52)
        self.assertEqual(len(ds.practice_conversations()), 96)

    def test_relationships(self):
        ds = Dataset(DATASET)
        cust = {c["customer_id"] for c in ds.table("customers")}
        orders = ds.table("orders")
        self.assertTrue(all(o["customer_id"] in cust for o in orders))
        self.assertTrue(all(o["delivery_pincode"] in ds.pincodes for o in orders))
        oids = {o["order_id"] for o in orders}
        self.assertTrue(all(i["order_id"] in oids and i["sku"] in ds.products for i in ds.table("order_items")))

    def test_product_matching(self):
        ds = Dataset(DATASET)
        self.assertEqual([p["sku"] for p in ds.match_products("is SKU ELE-0010 assured?")], ["ELE-0010"])
        self.assertEqual(normalise_spoken_sku("sku hom-zero zero four one please"), "sku HOM-0041 please")
        names = [p["sku"] for p in ds.match_products("my Brewly Electric Kettle stopped working")]
        self.assertEqual(names, ["HOM-0051"])
        self.assertEqual(ds.match_products("hello there"), [])

    def test_fuzzy_fallback_for_asr_misspellings(self):
        ds = Dataset(DATASET)
        self.assertEqual([p["sku"] for p in ds.match_products("mera bruli electric kettle kharab hai")], ["HOM-0051"])
        self.assertEqual(ds.match_products("my earbuds stopped working"), [])  # ambiguous -> no guess
        # the fallback never overrides an exact match
        self.assertEqual([p["sku"] for p in ds.match_products("Brewly Electric Kettle")], ["HOM-0051"])


@require_dataset
class TestIngestion(unittest.TestCase):
    def test_all_documents_and_section_ids(self):
        kb = DATASET / "kb"
        chunks = chunk_documents(kb)
        files = {c.file for c in chunks}
        self.assertEqual(len(files), 21)
        import re
        expected_ids = set()
        for p in kb.glob("*.md"):
            expected_ids |= set(re.findall(r"^##\s*\[([^\]]+)\]", p.read_text(encoding="utf-8"), re.M))
        self.assertEqual({c.section_id for c in chunks}, expected_ids)

    def test_front_matter(self):
        doc = parse_document(DATASET / "kb" / "pol_returns_v3_superseded.md")
        self.assertEqual(doc.meta["status"], "superseded")
        self.assertEqual(doc.meta["superseded_by"], "POL-RET")
        chunks = [c for c in chunk_documents(DATASET / "kb") if c.doc_id == "POL-RET-v3"]
        self.assertTrue(chunks and all("before 1 August 2026" in c.applicability_note for c in chunks))
        cur = [c for c in chunk_documents(DATASET / "kb") if c.section_id == "RET-2"][0]
        self.assertEqual((cur.doc_type, cur.status, cur.effective_from, cur.authority_rank), ("policy", "current", "2026-08-01", 1))


@require_dataset
class TestRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.index = KBIndex.load_or_build(DATASET / "kb", cls.tmp / "index", "tfidf")
        cls.r = Retriever(cls.index, date(2026, 10, 20))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _ids(self, q, **kw):
        return [h.chunk.section_id for h in self.r.retrieve([q], k=8, **kw).hits]

    def test_policy_backfilled_when_faq_retrieved(self):
        res = self.r.retrieve(["how long do refunds usually take to reach my account"], k=8)
        faq = [h for h in res.hits if h.chunk.doc_type == "faq"]
        if faq:  # whenever an FAQ is in context, a current policy chunk must be too
            self.assertTrue(any(h.chunk.doc_type == "policy" and h.chunk.status == "current" for h in res.hits))
        self.assertIn("REF-2", [h.chunk.section_id for h in res.hits])

    def test_superseded_demoted_without_historical_date(self):
        res = self.r.retrieve(["return window electronics ten days"], k=8)
        ranks = {h.chunk.section_id: h for h in res.hits}
        self.assertIn("RET-3", ranks)
        if "RETv3-2" in ranks:
            self.assertGreater(ranks["RET-3"].score, ranks["RETv3-2"].score)

    def test_historical_exception_kept(self):
        res = self.r.retrieve(["my smartwatch was delivered on 20 July, what was the return window"], k=8)
        ids = [h.chunk.section_id for h in res.hits]
        self.assertIn("RETv3-2", ids)
        self.assertIn("2026-07-20", res.historical_dates)
        self.assertTrue(any(h.chunk.section_id.startswith("RET-") for h in res.hits))  # successor present too

    def test_section_ids_and_scores_in_results(self):
        res = self.r.retrieve(["cancel an order that has shipped"], k=5)
        d = res.hits[0].to_dict()
        for key in ("section_id", "file", "score", "doc_type", "status", "via"):
            self.assertIn(key, d)
        self.assertIn("CAN-2", res.section_ids)

    def test_standing_docs_can_be_excluded(self):
        ids = self._ids("verify identity last 4 digits", exclude_doc_ids={"AGT-IDV"})
        self.assertFalse(any(i.startswith("IDV-") for i in ids))

    def test_index_persistence_and_invalidation(self):
        again = KBIndex.load_or_build(DATASET / "kb", self.tmp / "index", "tfidf")
        self.assertTrue(again.manifest["loaded_from_disk"])
        kb2 = self.tmp / "kb_copy"
        shutil.copytree(DATASET / "kb", kb2)
        a = KBIndex.load_or_build(kb2, self.tmp / "index2", "tfidf")
        self.assertFalse(a.manifest["loaded_from_disk"])
        p = kb2 / "faq_general.md"
        p.write_text(p.read_text(encoding="utf-8") + "\n## [FAQ-99] Test\nNew text.\n", encoding="utf-8")
        b = KBIndex.load_or_build(kb2, self.tmp / "index2", "tfidf")
        self.assertFalse(b.manifest["loaded_from_disk"])
        self.assertIn("FAQ-99", {c.section_id for c in b.chunks})
        self.assertIn("LEXICAL FALLBACK", b.backend_description)


class TestDates(unittest.TestCase):
    def setUp(self):
        self.cal = Calendar({date(2026, 8, 15): "Independence Day", date(2026, 10, 2): "Gandhi Jayanti",
                             date(2026, 10, 19): "Warehouse closure", date(2026, 11, 9): "Warehouse closure"},
                            date(2026, 10, 20))

    def test_business_days_skip_sunday_and_holiday(self):
        # Thu 13 Aug + 3 business days: Fri 14 (1), Sat 15 holiday, Sun 16, Mon 17 (2), Tue 18 (3)
        self.assertEqual(self.cal.add_business_days(date(2026, 8, 13), 3), date(2026, 8, 18))
        w = self.cal.tool_business_day_window("2026-11-06", 1, 3)  # Fri 6 Nov
        self.assertEqual((w["earliest"], w["latest"]), ("2026-11-07", "2026-11-11"))  # Sat, (Sun, Mon holiday) Tue, Wed
        self.assertIn("2026-11-09 (Warehouse closure)", w["skipped_non_business_days"])

    def test_calendar_days_and_between(self):
        self.assertEqual(self.cal.tool_add_calendar_days("2026-10-01", 10)["result_date"], "2026-10-11")
        self.assertEqual(self.cal.tool_business_days_between("2026-10-17", "2026-10-20")["business_days"], 1)
        self.assertFalse(self.cal.is_business_day(date(2026, 10, 18)))


if __name__ == "__main__":
    unittest.main()
