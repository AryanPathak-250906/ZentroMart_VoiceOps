"""Evaluation harness.

  retrieval  – retrieval-only metrics on the 52 questions (no LLM needed)
  step1      – Step 1 baseline: LLM + support prompt, NO documents (52 questions)
  step2      – Step 2/3 agent with RAG + tools (52 questions, fresh conversation each)
  multiturn  – 23 conversations / 96 turns, in turn order, state reset per conversation

Gold fields (reference_answer, must_include, expected_decision, gold_evidence, ...) are read
here only for scoring; they are never passed to the agent. Results are written incrementally
(JSONL) so an interrupted run can be resumed with --resume.
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from ..agent import STANDING_DOCS, BaselineAgent, SupportAgent
from ..conversation import Session
from ..data import Dataset
from ..llm import LLMError
from .scoring import (check_must_include, check_must_not, check_tools, contains_fact, dates_in, parse_expected_tools,
                      split_list)

STANDING_PREFIXES = ("KB-00", "STY-", "IDV-", "ESC-")


def _is_standing(sid: str) -> bool:
    return sid.startswith(STANDING_PREFIXES)


def _load_done(path: Path) -> dict[str, dict]:
    done = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("status") == "completed":
                    done[rec["key"]] = rec
    return done


def _append(path: Path, rec: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({(100.0 * n / d):.1f}%)" if d else "0/0 (n/a)"


# ---------------------------------------------------------------------------- retrieval only
def eval_retrieval(agent: SupportAgent, dataset: Dataset, use_asr: bool = False) -> dict:
    rows, k = [], agent.settings.top_k
    for q in dataset.practice_questions():
        text = q["asr_text"] if use_asr else q["question"]
        res = agent.retriever.retrieve([text], k=k, exclude_doc_ids=STANDING_DOCS)
        retrieved = res.section_ids
        products = [f"products.csv:{p['sku']}" for p in dataset.match_products(text)]
        gold = split_list(q["gold_evidence"])
        kb_gold = [g for g in gold if not g.startswith("products.csv:") and not _is_standing(g)]
        prod_gold = [g for g in gold if g.startswith("products.csv:")]
        hn = split_list(q["hard_negatives"])
        top_k_ids = [h.chunk.section_id for h in res.hits if h.via == "search"]
        rows.append({
            "id": q["id"], "language": q["language"], "text": text, "retrieved": retrieved, "products": products,
            "kb_gold": kb_gold, "kb_gold_hit": [g for g in kb_gold if g in retrieved],
            "first_gold_rank": next((i + 1 for i, s in enumerate(top_k_ids) if s in kb_gold), None),
            "product_gold": prod_gold, "product_gold_hit": [g for g in prod_gold if g in products],
            "standing_gold": [g for g in gold if _is_standing(g)],
            "hard_negatives": hn, "hard_negative_retrieved": [h for h in hn if h in retrieved],
        })
    with_gold = [r for r in rows if r["kb_gold"]]
    full = sum(1 for r in with_gold if len(r["kb_gold_hit"]) == len(r["kb_gold"]))
    any_hit = sum(1 for r in with_gold if r["kb_gold_hit"])
    sec_total = sum(len(r["kb_gold"]) for r in with_gold)
    sec_hit = sum(len(r["kb_gold_hit"]) for r in with_gold)
    mrr = sum(1.0 / r["first_gold_rank"] for r in with_gold if r["first_gold_rank"]) / len(with_gold) if with_gold else 0
    prod = [r for r in rows if r["product_gold"]]
    prod_hit = sum(1 for r in prod if len(r["product_gold_hit"]) == len(r["product_gold"]))
    hn_rows = [r for r in rows if r["hard_negatives"]]
    hn_hit = sum(1 for r in hn_rows if r["hard_negative_retrieved"])
    by_lang = defaultdict(lambda: [0, 0])
    for r in with_gold:
        by_lang[r["language"]][1] += 1
        by_lang[r["language"]][0] += int(len(r["kb_gold_hit"]) == len(r["kb_gold"]))
    return {
        "input": "asr_text" if use_asr else "question", "k": k, "backend": agent.retriever.index.backend_description,
        "questions": len(rows), "questions_with_kb_gold": len(with_gold),
        "all_kb_gold_retrieved": _pct(full, len(with_gold)), "any_kb_gold_retrieved": _pct(any_hit, len(with_gold)),
        "kb_gold_section_recall": _pct(sec_hit, sec_total), "mrr_first_gold_in_top_k": round(mrr, 3),
        "product_rows_resolved": _pct(prod_hit, len(prod)),
        "hard_negative_in_context": _pct(hn_hit, len(hn_rows)),
        "recall_by_language": {k_: _pct(v[0], v[1]) for k_, v in sorted(by_lang.items())},
        "note": "Retrieval-only (customer text as the only query; no LLM query rewriting). Standing sections "
                "(KB-00, STY, IDV, ESC) are always in the agent's context and excluded from this recall.",
        "rows": rows,
    }


# ---------------------------------------------------------------------------- single turn
def _score_single(q: dict, reply: str, decision: str, used: list[str] | None, retrieved: list[str] | None) -> dict:
    inc_ok, missing = check_must_include(reply, q["must_include"])
    not_ok, hits = check_must_not(reply, q["must_not_include"])
    acceptable = split_list(q["acceptable_decisions"]) or [q["expected_decision"]]
    dec_ok = decision in acceptable
    gold = split_list(q["gold_evidence"])
    out = {"decision_ok": dec_ok, "must_include_ok": inc_ok, "missing": missing, "must_not_ok": not_ok,
           "forbidden_found": hits, "content_ok": inc_ok and not_ok, "pass": dec_ok and inc_ok and not_ok,
           "answerable": q["answerable"] == "True"}
    if used is not None:
        out["gold_cited"] = [g for g in gold if g in used]
        out["gold_cited_all"] = all(g in used for g in gold) if gold else None
        out["gold_in_context"] = [g for g in gold if g in (retrieved or []) or _is_standing(g)]
        out["hard_negative_cited"] = [h for h in split_list(q["hard_negatives"]) if h in used]
    return out


def run_single(kind: str, agent: SupportAgent, dataset: Dataset, out_dir: Path, use_asr: bool = False,
               resume: bool = True, limit: int | None = None, log=print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{kind}{'_asr' if use_asr else ''}_results.jsonl"
    done = _load_done(path) if resume else {}
    if not resume and path.exists():
        path.unlink()
    baseline = BaselineAgent(agent.llm, agent.settings) if kind == "step1" else None
    questions = dataset.practice_questions()[: limit or None]
    for i, q in enumerate(questions, 1):
        if q["id"] in done:
            continue
        text = q["asr_text"] if use_asr else q["question"]
        t0 = time.time()
        if baseline:
            r = baseline.respond(text)
            reply, decision, errors, used, retrieved, extra = r["reply"], r["decision"], r["errors"], None, None, {}
        else:
            session = Session(session_id=f"eval-{q['id']}")
            turn = agent.respond(session, text)
            reply, decision, errors = turn.reply, turn.decision, turn.errors
            used, retrieved = turn.used_sections, turn.retrieved_sections
            extra = {"tool_calls": [c.public() for c in turn.tool_calls], "escalation": turn.escalation,
                     "calculator_calls": turn.diagnostics.get("calculator_calls", []),
                     "answer_supported_by_documents": turn.diagnostics.get("answer_supported_by_documents"),
                     "planner_fallback": turn.diagnostics.get("planner_fallback"),
                     "llm_calls": turn.diagnostics.get("llm_calls", [])}
        failed = decision == "error" or not reply
        rec = {"key": q["id"], "id": q["id"], "kind": kind, "input": "asr_text" if use_asr else "question",
               "language": q["language"], "intent": q["intent"], "question": text, "reply": reply, "decision": decision,
               "errors": errors, "used_sections": used, "retrieved_sections": retrieved,
               "seconds": round(time.time() - t0, 2), "mock": bool(getattr(agent.llm, "is_mock", False)),
               "model": getattr(agent.llm, "model", None), **extra,
               "status": "failed" if failed else "completed"}
        if not failed:
            rec["score"] = _score_single(q, reply, decision, used, retrieved)
        _append(path, rec)
        log(f"[{kind}{'/asr' if use_asr else ''} {i}/{len(questions)}] {q['id']} {rec['status']} "
            f"{'PASS' if rec.get('score', {}).get('pass') else 'fail'} ({rec['seconds']}s)")
        if failed and any("quota_exhausted" in e or "daily quota" in e for e in errors):
            log("Stopping: provider quota exhausted. Re-run the same command later with --resume.")
            break
    return summarize_single(path, dataset)


def summarize_single(path: Path, dataset: Dataset) -> dict:
    recs = list(_load_done(path).values())
    total = len(dataset.practice_questions())
    n = len(recs)
    s = [r["score"] for r in recs]
    unsupported = [r for r in recs if not r["score"]["answerable"]]
    res = {
        "file": path.name, "questions_total": total, "completed": n, "not_completed": total - n,
        "mock": any(r.get("mock") for r in recs), "model": recs[0].get("model") if recs else None,
        "full_pass": _pct(sum(x["pass"] for x in s), n),
        "decision_correct": _pct(sum(x["decision_ok"] for x in s), n),
        "must_include_satisfied": _pct(sum(x["must_include_ok"] for x in s), n),
        "no_forbidden_content": _pct(sum(x["must_not_ok"] for x in s), n),
        "content_correct": _pct(sum(x["content_ok"] for x in s), n),
        "unsupported_handled": _pct(sum(r["score"]["pass"] for r in unsupported), len(unsupported)),
    }
    cited = [x for x in s if x.get("gold_cited_all") is not None]
    if cited:
        res["gold_evidence_all_cited"] = _pct(sum(bool(x["gold_cited_all"]) for x in cited), len(cited))
        res["hard_negative_cited"] = _pct(sum(bool(x.get("hard_negative_cited")) for x in s), n)
    by_intent = defaultdict(lambda: [0, 0])
    for r in recs:
        by_intent[r["intent"]][1] += 1
        by_intent[r["intent"]][0] += int(r["score"]["pass"])
    res["pass_by_intent"] = {k: _pct(v[0], v[1]) for k, v in sorted(by_intent.items())}
    by_lang = defaultdict(lambda: [0, 0])
    for r in recs:
        by_lang[r["language"]][1] += 1
        by_lang[r["language"]][0] += int(r["score"]["pass"])
    res["pass_by_language"] = {k: _pct(v[0], v[1]) for k, v in sorted(by_lang.items())}
    return res


# ---------------------------------------------------------------------------- multi turn
def _private_values(dataset: Dataset, order_id: str) -> tuple[list[str], list[str]]:
    """(order-private values that must not appear before verification, secrets never to read out)."""
    orders = {o["order_id"]: o for o in dataset.table("orders")}
    custs = {c["customer_id"]: c for c in dataset.table("customers")}
    o = orders.get(order_id)
    if not o:
        return [], []
    c = custs.get(o["customer_id"], {})
    private = [o[f] for f in ("promised_date", "shipped_on", "delivered_on") if o.get(f)]
    secrets = [o["delivery_pincode"]] + ([c["phone"][-4:]] if c.get("phone") else [])
    return private, secrets


def run_multiturn(agent: SupportAgent, dataset: Dataset, out_dir: Path, resume: bool = True,
                  limit: int | None = None, log=print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "multiturn_results.jsonl"
    done = _load_done(path) if resume else {}
    if not resume and path.exists():
        path.unlink()
    convs = defaultdict(list)
    for row in dataset.practice_conversations():
        convs[row["conversation_id"]].append(row)
    ids = sorted(convs)[: limit or None]
    for ci, cid in enumerate(ids, 1):
        if cid in done:
            continue
        rows = sorted(convs[cid], key=lambda r: int(r["turn"]))
        session = Session(session_id=f"eval-{cid}")  # fresh state for every conversation
        private, secrets = _private_values(dataset, rows[0]["order_id"])
        turns, failed = [], False
        for row in rows:
            t0 = time.time()
            turn = agent.respond(session, row["customer_message"])
            if turn.decision == "error":
                failed = True
            internal = [{"name": c.name, "order_id": c.expected_match_key.get("order_id"),
                         "verification_value": c.expected_match_key.get("verification_value"),
                         "created": c.expected_match_key.get("created")} for c in turn.tool_calls]
            tools = check_tools(parse_expected_tools(row["expected_tool_calls"]), internal)
            inc_ok, missing = check_must_include(turn.reply, row["must_include"])
            not_ok, hits = check_must_not(turn.reply, row["must_not_include"])
            verified_now = turn.verification.get("state") == "verified"
            leaks = []
            if not verified_now:
                leaks += [v for v in private if contains_fact(turn.reply, v)]
            leaks += [s for s in secrets if s and s in turn.reply]
            gold = split_list(row["gold_evidence"])
            turns.append({
                "turn": int(row["turn"]), "message": row["customer_message"], "reply": turn.reply,
                "expected_decision": row["expected_decision"], "decision": turn.decision,
                "decision_ok": turn.decision == row["expected_decision"], "must_include_ok": inc_ok, "missing": missing,
                "must_not_ok": not_ok, "forbidden_found": hits, "tools_ok": tools["pass"], "tools_detail": tools,
                "tool_calls": [c.public() for c in turn.tool_calls], "privacy_ok": not leaks,
                "privacy_leaks": ["[order detail]" if l in private else "[verification secret]" for l in leaks],
                "verification": turn.verification, "escalation": turn.escalation,
                "used_sections": turn.used_sections, "retrieved_sections": turn.retrieved_sections,
                "gold_evidence": gold, "gold_cited": [g for g in gold if g in turn.used_sections],
                "errors": turn.errors, "seconds": round(time.time() - t0, 2),
                "pass": (turn.decision == row["expected_decision"]) and inc_ok and not_ok and tools["pass"] and not leaks,
            })
            if failed and any("quota" in e for e in turn.errors):
                break
        rec = {"key": cid, "conversation_id": cid, "scenario": rows[0]["scenario"], "language": rows[0]["language"],
               "turns": turns, "mock": bool(getattr(agent.llm, "is_mock", False)), "model": getattr(agent.llm, "model", None),
               "status": "failed" if failed or len(turns) != len(rows) else "completed"}
        _append(path, rec)
        log(f"[multiturn {ci}/{len(ids)}] {cid} {rec['status']} turns passed {sum(t['pass'] for t in turns)}/{len(rows)}")
        if failed and any("quota_exhausted" in e or "daily quota" in e for t in turns for e in t["errors"]):
            log("Stopping: provider quota exhausted. Re-run later with --resume.")
            break
    return summarize_multiturn(path, dataset)


def summarize_multiturn(path: Path, dataset: Dataset) -> dict:
    recs = list(_load_done(path).values())
    all_rows = dataset.practice_conversations()
    turns = [t for r in recs for t in r["turns"]]
    n = len(turns)
    by_dec = defaultdict(lambda: [0, 0])
    for t in turns:
        by_dec[t["expected_decision"]][1] += 1
        by_dec[t["expected_decision"]][0] += int(t["decision_ok"])
    tool_turns = [t for t in turns if t["tools_detail"]["missing"] or t["tool_calls"] or t["tools_detail"]["unexpected"]]
    followups = [t for r in recs for t in r["turns"] if t["turn"] >= 4]
    by_scen = defaultdict(lambda: [0, 0])
    for r in recs:
        by_scen[r["scenario"]][1] += 1
        by_scen[r["scenario"]][0] += int(all(t["pass"] for t in r["turns"]))
    return {
        "file": path.name, "conversations_total": len({r["conversation_id"] for r in all_rows}),
        "turns_total": len(all_rows), "conversations_completed": len(recs), "turns_completed": n,
        "mock": any(r.get("mock") for r in recs), "model": recs[0].get("model") if recs else None,
        "turn_full_pass": _pct(sum(t["pass"] for t in turns), n),
        "decision_correct": _pct(sum(t["decision_ok"] for t in turns), n),
        "decision_by_expected": {k: _pct(v[0], v[1]) for k, v in sorted(by_dec.items())},
        "must_include_satisfied": _pct(sum(t["must_include_ok"] for t in turns), n),
        "no_forbidden_content": _pct(sum(t["must_not_ok"] for t in turns), n),
        "tool_calls_correct": _pct(sum(t["tools_ok"] for t in turns), n),
        "turns_with_tool_activity_correct": _pct(sum(t["tools_ok"] for t in tool_turns), len(tool_turns)),
        "privacy_no_leak": _pct(sum(t["privacy_ok"] for t in turns), n),
        "first_turn_ai_disclosure": _pct(sum(t["must_include_ok"] for t in turns if t["turn"] == 1), sum(1 for t in turns if t["turn"] == 1)),
        "followup_turns_pass (turn>=4)": _pct(sum(t["pass"] for t in followups), len(followups)),
        "conversations_all_turns_pass": _pct(sum(all(t["pass"] for t in r["turns"]) for r in recs), len(recs)),
        "conversations_by_scenario": {k: _pct(v[0], v[1]) for k, v in sorted(by_scen.items())},
        "gold_evidence_cited_turns": _pct(sum(1 for t in turns if t["gold_evidence"] and set(t["gold_evidence"]) <= set(t["used_sections"])),
                                          sum(1 for t in turns if t["gold_evidence"])),
    }


def write_report(out_dir: Path, metrics: dict) -> Path:
    lines = ["# Evaluation results", "", f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    if any(isinstance(v, dict) and v.get("mock") for v in metrics.values()):
        lines += ["> **MOCK RUN — these numbers come from a scripted mock LLM and are NOT model scores.**", ""]
    for name, m in metrics.items():
        if not isinstance(m, dict):
            continue
        lines += [f"## {name}", ""]
        for k, v in m.items():
            if k == "rows":
                continue
            if isinstance(v, dict):
                lines.append(f"- **{k}**: " + ", ".join(f"{a}: {b}" for a, b in v.items()))
            else:
                lines.append(f"- **{k}**: {v}")
        lines.append("")
    path = out_dir / "EVAL_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
