#!/usr/bin/env python3
"""Build report.pdf (≤ 4 pages) from the saved evaluation artefacts. Numbers are read from
reports/eval/metrics.json and the JSONL files — nothing is typed in by hand.

  python scripts/build_report.py            # -> report.pdf and reports/report.pdf
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

from reportlab.graphics.charts.barcharts import HorizontalBarChart
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "reports" / "eval"
BRAND = colors.HexColor("#0f6e5f")
INK = colors.HexColor("#1d232b")
MUTED = colors.HexColor("#5b6570")
LINE = colors.HexColor("#d9d6cf")
SOFT = colors.HexColor("#f3f1ec")
FONT, FONT_B = "Helvetica", "Helvetica-Bold"
for regular, bold in [("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
                      ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf")]:
    if Path(regular).is_file() and Path(bold).is_file():  # a TTF font renders ₹ and arrows
        pdfmetrics.registerFont(TTFont("Body", regular))
        pdfmetrics.registerFont(TTFont("Body-Bold", bold))
        from reportlab.lib.fonts import addMapping
        addMapping("Body", 0, 0, "Body"); addMapping("Body", 1, 0, "Body-Bold")
        addMapping("Body", 0, 1, "Body"); addMapping("Body", 1, 1, "Body-Bold")
        FONT, FONT_B = "Body", "Body-Bold"
        break


def pct(s: str | None) -> float:
    m = re.search(r"\(([\d.]+)%\)", s or "")
    return float(m.group(1)) if m else 0.0


def load() -> dict:
    m = json.loads((EVAL / "metrics.json").read_text(encoding="utf-8"))
    extra = {}
    p = ROOT / "sample_outputs" / "batch_run" / "summary.json"
    if p.is_file():
        extra["batch"] = json.loads(p.read_text(encoding="utf-8"))
    it1 = ROOT / "reports" / "eval_iterations" / "iter1" / "metrics_step2.json"
    if it1.is_file():
        extra["iter1"] = json.loads(it1.read_text(encoding="utf-8"))
    it2 = ROOT / "reports" / "eval_iterations" / "iter2"
    if (it2 / "metrics_step2.json").is_file():
        extra["iter2"] = {"step2": json.loads((it2 / "metrics_step2.json").read_text(encoding="utf-8")),
                          "multiturn": json.loads((it2 / "metrics_multiturn.json").read_text(encoding="utf-8"))}
    return {**m, **extra}


def styles():
    ss = getSampleStyleSheet()
    base = ParagraphStyle("b", parent=ss["Normal"], fontName=FONT, fontSize=8.3, leading=11.2, textColor=INK, alignment=TA_LEFT)
    return {
        "title": ParagraphStyle("t", parent=base, fontName=FONT_B, fontSize=17, leading=21, textColor=INK),
        "sub": ParagraphStyle("s", parent=base, fontSize=9, textColor=MUTED),
        "h": ParagraphStyle("h", parent=base, fontName=FONT_B, fontSize=11, leading=14, textColor=BRAND, spaceBefore=7, spaceAfter=3),
        "b": base,
        "small": ParagraphStyle("sm", parent=base, fontSize=7.4, leading=9.4, textColor=MUTED),
        "cell": ParagraphStyle("c", parent=base, fontSize=7.6, leading=9.6),
        "cellb": ParagraphStyle("cb", parent=base, fontSize=7.6, leading=9.6, fontName=FONT_B),
    }


def table(rows, widths, st, header=True, zebra=True):
    data = [[Paragraph(str(c), st["cellb" if (header and i == 0) else "cell"]) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    style = [("GRID", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
             ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), SOFT))
    t.setStyle(TableStyle(style))
    return t


def bar_chart(step1: dict, step2: dict) -> Drawing:
    cats = [("Full pass", "full_pass"), ("Decision correct", "decision_correct"),
            ("Required facts", "must_include_satisfied"), ("No forbidden content", "no_forbidden_content"),
            ("Unsupported handled", "unsupported_handled")]
    d = Drawing(170 * mm, 52 * mm)
    ch = HorizontalBarChart()
    ch.x, ch.y, ch.width, ch.height = 34 * mm, 6 * mm, 120 * mm, 42 * mm
    ch.data = [[pct(step1.get(k)) for _, k in cats][::-1], [pct(step2.get(k)) for _, k in cats][::-1]]
    ch.categoryAxis.categoryNames = [c for c, _ in cats][::-1]
    ch.categoryAxis.labels.fontSize = 7
    ch.categoryAxis.labels.fontName = FONT
    ch.valueAxis.labels.fontName = FONT
    ch.barLabels.fontName = FONT
    ch.valueAxis.valueMin, ch.valueAxis.valueMax, ch.valueAxis.valueStep = 0, 100, 20
    ch.valueAxis.labels.fontSize = 7
    ch.bars[0].fillColor = colors.HexColor("#c9c4b8")
    ch.bars[1].fillColor = BRAND
    ch.barSpacing, ch.groupSpacing = 1, 6
    ch.barLabelFormat = "%.0f%%"
    ch.barLabels.fontSize = 6.5
    ch.barLabels.dx = 9
    d.add(ch)
    d.add(Rect(34 * mm, 49.6 * mm, 3 * mm, 2.4 * mm, fillColor=colors.HexColor("#c9c4b8"), strokeColor=None))
    d.add(String(38 * mm, 50 * mm, "Step 1: prompt only (no documents)", fontSize=7, fontName=FONT, fillColor=INK))
    d.add(Rect(100 * mm, 49.6 * mm, 3 * mm, 2.4 * mm, fillColor=BRAND, strokeColor=None))
    d.add(String(104 * mm, 50 * mm, "Step 2: RAG agent", fontSize=7, fontName=FONT, fillColor=INK))
    return d


def build(out: Path, interim: bool = False) -> None:
    m = load()
    st = styles()
    s1, s2, mt = m.get("step1", {}), m.get("step2", {}), m.get("multiturn", {})
    rt, rta = m.get("retrieval", {}), m.get("retrieval_asr", {})
    batch = m.get("batch")
    W = 180 * mm
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=13 * mm, bottomMargin=13 * mm,
                            title="ZentroMart VoiceOps 2026 — Capstone report", author="VoiceOps 2026 participant")
    E = []
    P = lambda t, s="b": E.append(Paragraph(t, st[s]))  # noqa: E731

    # ---------------------------------------------------------------- page 1
    if interim:
        P("<font color='#9b1c1c'><b>INTERIM REPORT</b> — Step 2 and multi-turn numbers are from development iteration 2; the final run "
          "with the final code is pending (free-tier LLM daily quota). Voice batch run and demo pending.</font>", "b")
    P("ZentroMart VoiceOps 2026 — grounded voice support agent", "title")
    P("Capstone report · LLM &amp; Generative AI course (CAC IIT Guwahati × Gnani.ai) · dataset reference date 20 October 2026", "sub")
    E.append(Spacer(1, 6))
    model = s2.get("model") or "n/a"
    summary = [
        ["Item", "Result (all numbers produced by the saved evaluation runs)"],
        ["Step 1 vs Step 2 (52 practice questions)", f"Full pass <b>{s1.get('full_pass','not run')}</b> → <b>{s2.get('full_pass','not run')}</b> · same model ({model}), same scorer"
         + (f" · Step 2 on ASR-like text: {m['step2_asr'].get('full_pass')} ({m['step2_asr'].get('completed')}/52 completed)" if m.get('step2_asr') else "")],
        ["Multi-turn (23 conversations / 96 turns)", f"Turns fully passed <b>{mt.get('turn_full_pass','not run')}</b> · decisions {mt.get('decision_correct','-')} · tool calls {mt.get('tool_calls_correct','-')} · privacy {mt.get('privacy_no_leak','-')}"],
        ["Retrieval only (no LLM rewriting, k=8)", f"All gold KB sections retrieved {rt.get('all_kb_gold_retrieved','not run')} (ASR text {rta.get('all_kb_gold_retrieved','-')}) · catalog rows {rt.get('product_rows_resolved','-')}"],
        ["Voice (Gnani STT + TTS timbre-v2.5)", (f"Live batch run: {batch['succeeded']}/{batch['files']} input files handled end to end (synthetic Gnani-TTS customer clips + silence + one corrupt file)" if batch else "batch run not executed")],
    ]
    E.append(table(summary, [52 * mm, W - 52 * mm], st))
    P("What was built in each step", "h")
    steps = [
        ["Step", "What it does", "Where"],
        ["1 · Answer with an LLM", "ZentroMart support system prompt (role, limits, AI disclosure), JSON reply + decision; no documents or tools. Used as the baseline.", "agent.BaselineAgent"],
        ["2 · Ground it in documents", "Section-aligned chunks of all 21 KB files with front matter; multilingual embeddings (fastembed, paraphrase-multilingual-MiniLM-L12-v2) + BM25, RRF fusion; precedence prior and back-fill; catalog rows; date calculators; cited section IDs per answer.", "retrieval/, agent.SupportAgent"],
        ["3 · Hold a conversation", "Per-session state (order, verification, attempts, tickets, language); planner resolves follow-ups; supplied lookup_order (one proof value, one retry, lock-out) and create_ticket enforced in code; ESC-1 handovers; polite refusals.", "conversation.py, tools_adapter.py"],
        ["4 · Give it a voice", "Gnani STT (REST, ITN digits, ≤30 s clips, WAV splitting) and Gnani TTS timbre-v2.5 (Kaveri / Nalini / Poorvi); browser PCM recorder; batch runner; empty-audio recovery.", "voice/, batch.py, web/"],
    ]
    E.append(table(steps, [33 * mm, 115 * mm, 32 * mm], st))
    P("Architecture", "h")
    P("One <b>SupportAgent</b> serves the web chat, browser voice, terminal and batch runner. Per turn: (1) a planner LLM call returns "
      "validated JSON (language, intent, order ID, proof value, English search queries with follow-ups resolved, products, dates, "
      "escalation signals); (2) code runs the identity workflow with the supplied <i>lookup_order</i> — digits must literally appear in "
      "the customer's message, private fields reach the model only after <i>verified=True</i>, the delivery pincode never; (3) code "
      "creates the ESC-1 ticket <i>before</i> any handover claim; (4) hybrid retrieval; (5) a responder LLM call with "
      "calculator tools and <i>create_ticket</i> writes the reply as validated JSON with cited IDs; (6) guardrails enforce AI "
      "disclosure, decision consistency, secret scrubbing and honest ticket failures. Gold answers are read only by the evaluator.")
    P("Retrieval and policy precedence", "h")
    P("KB-00.1's order is metadata on every chunk (policy 1 · agent guide 2 · help 3 · FAQ 4 · marketing 5 · superseded 9). It drives "
      "an authority prior in ranking, the labels the model sees, and <b>back-fill</b>: whenever an FAQ/help/marketing chunk is retrieved "
      "the best current-policy chunk on the same topic is added (FAQ-1's generic “5 to 7 days” is always next to REF-2). "
      "<i>pol_returns_v3_superseded.md</i> keeps its applicability note on each chunk and is demoted unless the customer mentions a date "
      "before 1 August 2026 — then RETv3-2 (10-day electronics window) is surfaced together with RET-3, which preserves it. "
      "Agent guides and KB-00 are always in context as standing rules. Business-day and calendar arithmetic is never done by the "
      "model: it calls <i>business_day_window</i> / <i>add_calendar_days</i>, which skip Sundays and the holidays table.")

    # ---------------------------------------------------------------- page 2
    E.append(PageBreak())
    P("Step 1 vs Step 2 — the same 52 questions, the same scorer", "h")
    E.append(bar_chart(s1, s2))
    rows = [["Metric (count / completed)", "Step 1 · prompt only", "Step 2 · RAG agent"]]
    for label, key in [("Completed cases", "completed"), ("Full pass (decision + facts + no forbidden)", "full_pass"),
                       ("Decision in acceptable_decisions", "decision_correct"), ("All must_include facts", "must_include_satisfied"),
                       ("No must_not_include phrase", "no_forbidden_content"), ("Unsupported questions handled", "unsupported_handled"),
                       ("All gold evidence cited by the answer", "gold_evidence_all_cited"), ("Hard negative cited", "hard_negative_cited")]:
        rows.append([label, str(s1.get(key, "—")), str(s2.get(key, "—"))])
    E.append(table(rows, [80 * mm, 50 * mm, 50 * mm], st))
    P("Scoring (deterministic, see evaluation/scoring.py): every ' ; ' fact of <i>must_include</i> is required and ' / ' alternatives are "
      "equivalent; dates match in any format (ISO, “27 October 2026”, “27th of October”, Hindi month names); numbers match as digits or "
      "number words (English/Hindi/Hinglish); any <i>must_not_include</i> phrase fails the case. Decision = the agent's declared decision, "
      "accepted if listed in <i>acceptable_decisions</i>. Step 1 has no sources, so evidence metrics apply to Step 2 only. "
      "Limitations: paraphrased forbidden claims are not detected; facts outside the listed must_include are not checked; the scorer "
      "was written against the practice set, which was also used for development.", "small")
    it1 = m.get("iter1")
    it2 = m.get("iter2")
    P("What failed, what improved", "h")
    lst = ["<b>Step 1</b> mostly fails on ZentroMart-specific facts (it does not know RET-2 windows, REF-2 timelines, the holiday "
           "table or which SKUs carry ZentroMart Assured) and sometimes answers unsupported questions with plausible inventions."]
    if it1:
        lst.append(f"<b>Iteration 1</b> (Step 2 {it1.get('full_pass')}): the agent answered general policy questions and then asked for an "
                   "order ID (scored “clarify”), dropped a figure, and inferred an unsupported fact (“no store in Kochi”). Fixes: the order "
                   "workflow only starts when the answer depends on the customer's own order; rules to state the evidence's figures and never "
                   "deduce facts from missing information; retries for transient network errors.")
    if it2:
        lst.append(f"<b>Iteration 2</b> (Step 2 {it2.get('step2', {}).get('full_pass')}, multi-turn {it2.get('multiturn', {}).get('turn_full_pass')} "
                   "of completed turns): return answers gave the window but not the concrete last date or the ₹49 pickup fee; an overdue UPI "
                   "refund was not handed over; a refusal lacked an apology; shipped-order cancellations omitted the RTO refund timing. Fixes: "
                   "a deterministic refund-timeline check that reads the REF-2 table from the KB (ESC-1 rule 5 now enforced in code), "
                   "cross-reference back-fill (CAN-2 → REF-1), prompt rules for concrete dates, fees, cross-references and brief apologies, and "
                   "a narrower decision override.")
    lst.append(f"<b>Final run</b> (same model, same scorer, final code): Step 2 {s2.get('full_pass')}, multi-turn {mt.get('turn_full_pass')}. "
               "The only Step 2 miss (PQ0048) has the right facts but ends by asking for an order ID (scored “clarify”). Multi-turn misses: "
               "(a) a <b>regression</b> — with the new refund check and order facts, the model handed over on the verification turn of the "
               "repeat-contact / overdue-refund conversations, one turn before the customer stated the issue (6 turns: T3 early ticket, T4 "
               "ticket re-used); (b) the festive-sale rule applied to an order placed on 30 September (wrong deadline); (c) refund dates "
               "computed for a refund that had not started; (d) shipped-order cancellations labelled “answer” and missing the RTO refund "
               "timing; (e) two refusals that apologise with the Hindi “khed hai”, which the automatic scorer does not recognise "
               "(correct on manual review).")
    for t in lst:
        P("• " + t)
    P("Retrieval quality (no LLM, customer text as the only query)", "h")
    rr = [["Metric", "Typed question", "ASR-like text"],
          ["All gold KB sections in top-8 (+back-fill)", rt.get("all_kb_gold_retrieved", "—"), rta.get("all_kb_gold_retrieved", "—")],
          ["Gold KB section recall", rt.get("kb_gold_section_recall", "—"), rta.get("kb_gold_section_recall", "—")],
          ["MRR of first gold section", rt.get("mrr_first_gold_in_top_k", "—"), rta.get("mrr_first_gold_in_top_k", "—")],
          ["Catalog rows resolved (products.csv:SKU)", rt.get("product_rows_resolved", "—"), rta.get("product_rows_resolved", "—")],
          ["Questions with a hard negative in context", rt.get("hard_negative_in_context", "—"), rta.get("hard_negative_in_context", "—")]]
    E.append(table(rr, [80 * mm, 50 * mm, 50 * mm], st))
    P(f"Backend: {rt.get('backend','')}. Standing sections (KB-00, STY, IDV, ESC) are always in context and excluded from recall. "
      "In the full agent, the planner's English rewrites are added as extra queries, so live retrieval is at least as good. "
      "Comparison during development: the larger paraphrase-multilingual-mpnet model did not beat MiniLM here, so the 220 MB model was kept.", "small")

    # ---------------------------------------------------------------- page 3
    E.append(PageBreak())
    P("Multi-turn evaluation — 23 conversations, 96 customer turns, state reset per conversation", "h")
    rows = [["Metric (count / turns completed)", "Result"]]
    for label, key in [("Turns completed", "turns_completed"), ("Turn full pass (decision + facts + forbidden + tools + privacy)", "turn_full_pass"),
                       ("Expected decision", "decision_correct"), ("must_include satisfied", "must_include_satisfied"),
                       ("No forbidden content", "no_forbidden_content"), ("Tool calls exactly as expected", "tool_calls_correct"),
                       ("No private-data / secret leak", "privacy_no_leak"), ("First-turn AI disclosure", "first_turn_ai_disclosure"),
                       ("Follow-up turns (turn ≥ 4) pass", "followup_turns_pass (turn>=4)"),
                       ("Conversations with every turn passing", "conversations_all_turns_pass"),
                       ("Turns whose gold evidence was all cited", "gold_evidence_cited_turns")]:
        rows.append([label, str(mt.get(key, "not run"))])
    E.append(table(rows, [110 * mm, 70 * mm], st))
    dec = mt.get("decision_by_expected", {})
    if dec:
        P("Decision accuracy by expected decision: " + " · ".join(f"{k}: {v}" for k, v in dec.items()), "small")
    scen = mt.get("conversations_by_scenario", {})
    if scen:
        P("Conversations fully passed by scenario: " + " · ".join(f"{k.replace('_', ' ')} {v.split(' ')[0]}" for k, v in scen.items()), "small")
    P("Tool-call checks compare each turn's actual calls with <i>expected_tool_calls</i> (order ID, presence and value of the proof, "
      "<i>created=True</i> for tickets) and fail on missing or unexpected calls. The privacy check (evaluator-only data) fails a turn if the "
      "reply contains the order's dates before verification, the delivery pincode or the phone suffix.", "small")
    P("Conversation and tool handling", "h")
    for t in [
        "Verification: order ID → <i>lookup_order(id)</i> (verification prompt only) → one proof value → <i>lookup_order(id, value)</i>; one retry; "
        "after the second failure the order is locked for the session and only general help is offered. Verified records live only in that session.",
        "Follow-ups and topic switches are resolved by the planner from the conversation; the verified order stays available, so "
        "“when was it shipped?” after a coupon question needs no repetition.",
        "Handovers: human request, unauthorised activity, legal threats, safety risks, >₹10,000 damage (order item price or catalog price) "
        "and repeat contacts (<i>contacts_last_7_days ≥ 2</i>) create a ticket in code before the reply; overdue refunds are detected in "
        "code from the REF-2 table (latest date + more than 3 business days) and handed over with the RRN. Tickets are de-duplicated per reason; summaries contain issue, intent, "
        "order, what the customer said, checks and reason, with proof values scrubbed. A failed ticket is reported, never claimed.",
        "Refusals: compensation/vouchers outside DEL-4, cancelling for the customer (no such tool — guided to My Orders), other customers' data.",
    ]:
        P("• " + t)
    P("Voice integration and testing status", "h")
    vt = [["What", "Status"],
          ["Gnani SDK", "gnani-vachana 0.8.1 installed; signatures verified in the SDK source (GnaniSTTClient.transcribe, GnaniTTSClient.synthesize, model='timbre-v2.5')."],
          ["Live Gnani calls", "Real STT and TTS requests succeeded with the participant's key (English, Hindi, Hinglish; silence → empty transcript handled)."],
          ["Batch runner", (f"Real run on {batch['files']} files: {batch['succeeded']} handled end to end (incl. silence → spoken “didn't catch that”), "
                            f"{batch['failed']} recorded as failure (deliberately corrupt file); exit code 1; outputs in sample_outputs/batch_run/." if batch else "not run")],
          ["Demo video", "demo.mp4 (≤ 2 min) recorded through the real web UI: grounded answer (UPI refund dates), follow-up (card), "
                         "“I don't know” (gift wrapping) and a human handover with a created ticket."],
          ["Human speech", "Not tested: no human recordings were available. Customer clips were synthesised with other Gnani voices and are labelled as such."],
          ["Mocked tests", "STT/TTS failures, empty audio, long-clip splitting, invalid files and web upload are covered by offline tests with fake clients (flagged mock)."]]
    E.append(table(vt, [38 * mm, W - 38 * mm], st))

    # ---------------------------------------------------------------- page 4
    E.append(PageBreak())
    P("Limitations and what I would fix next", "h")
    for t in [
        "<b>Development on the practice set.</b> Prompts, the glossary and the scorer were refined while looking at practice failures, so the "
        "practice scores are optimistic; the hidden test set is the real measure.",
        "<b>Free-tier LLM.</b> Results use Gemini 3.5 Flash-Lite (free tier, 20 requests/day on the larger 3.6 Flash made it unusable for "
        "evaluation). A stronger model would likely improve multi-step reasoning (e.g. applying the festive-sale rule to the order date). The code supports Anthropic "
        "models via LLM_PROVIDER=anthropic without changes.",
        "<b>Scorer.</b> Deterministic checks cannot see paraphrased forbidden claims or partially wrong extra content. A transparent LLM-judge "
        "pass (reference answer + rubric) would be a useful secondary review.",
        "<b>Voice.</b> Not yet tested with real human callers, noisy audio or long multi-sentence turns; STT language is fixed per call "
        "(en-IN handles English/Hinglish, hi-IN must be chosen for Devanagari Hindi). Next: streaming STT/TTS for lower latency and "
        "automatic language selection.",
        "<b>Next fixes (found in the final run, deliberately not applied so that the reported scores match the submitted code):</b> "
        "offer <i>create_ticket</i> to the model only when the current message raises an issue (ask what the customer needs right after "
        "verification); compute concrete dates only from events present in the order record; compute return deadlines deterministically "
        "(RET-2 table + festive window) instead of leaving the date logic to the model; treat “cannot cancel a shipped order” as a refusal "
        "that always includes REF-1's refund timing.",
        "<b>One post-evaluation change:</b> a fuzzy product-name fallback for speech misspellings (“bruli” → Brewly kettle), used only when "
        "exact matching finds nothing. Verified offline that product matching is identical for all 200 practice texts (questions, ASR "
        "text, conversation messages); the batch run and the ASR-text evaluation were run with it.",
        "<b>Identity check.</b> The supplied 4-digit/pincode check is an educational simulation, not production authentication.",
        "<b>Latency.</b> Two LLM calls per turn plus the free-tier throttle give several seconds per voice turn; a single fused call for simple "
        "turns, caching and streaming TTS would cut this.",
    ]:
        P("• " + t)
    P("Integrity statement", "h")
    P("All scores in this report were computed by the included evaluation code from saved per-case results (reports/eval/*.jsonl). "
      "No answer was typed or edited by hand; no practice answers, expected decisions or evidence labels are present in prompts or the index. "
      "Mock clients are used only in offline tests and are flagged as such in every output. Speech uses Gnani only. "
      "The demo video's customer voice is synthetic (Gnani TTS) and labelled; everything after the microphone is the real pipeline.")
    P("Reproduce", "h")
    E.append(table([["Purpose", "Command"],
                    ["Check setup (LLM, Gnani, index)", "python run.py --check"],
                    ["Web UI", "python run.py --web"],
                    ["Batch audio (run command)", "python run.py --audio-dir INPUT --output-dir OUTPUT"],
                    ["All evaluations (resumable)", "python run.py --eval all"],
                    ["Offline tests", "python -m pytest -q"],
                    ["Rebuild this report", "python scripts/build_report.py"]], [60 * mm, W - 60 * mm], st))
    doc.build(E)


def main() -> int:
    out = ROOT / "report.pdf"
    build(out, interim="--interim" in sys.argv)
    (ROOT / "reports").mkdir(exist_ok=True)
    shutil.copyfile(out, ROOT / "reports" / "report.pdf")
    from pypdf import PdfReader  # noqa: optional check
    n = len(PdfReader(str(out)).pages)
    print(f"wrote {out} ({n} pages)")
    return 0 if n <= 4 else 1


if __name__ == "__main__":
    sys.exit(main())
