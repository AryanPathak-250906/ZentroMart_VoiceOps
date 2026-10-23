# Architecture — ZentroMart VoiceOps 2026

One reusable agent (`SupportAgent`) powers the browser chat, the browser voice mode, the
terminal chat and the batch audio runner. Speech is handled **only** by Gnani
(`gnani-vachana` SDK); the language model is a configurable provider (Gemini by default,
Anthropic supported) and is never switched silently.

```
             ┌──────────────┐   WAV (browser PCM recorder)   ┌──────────────────────────┐
 Browser ───►│ Starlette    │──────────────────────────────►│ GnaniVoice.transcribe    │──► transcript
 text/voice  │ web/app.py   │◄── reply text + /api/audio ───│ GnaniVoice.synthesize    │◄── reply text
             └──────┬───────┘                               └──────────────────────────┘
 run.py --audio-dir │ batch.py (same voice_turn)
                    ▼
        ┌───────────────────────────── SupportAgent.respond(session, text) ─────────────────────────────┐
        │ 1 Planner LLM (JSON schema) ─ language, intent, order_id, verification value, English search    │
        │   queries (follow-ups resolved), products, dates, escalation signals  → pydantic validation     │
        │ 2 Identity workflow (code): order ID → lookup_order(id) → ask ONE proof → lookup_order(id, v)   │
        │   max 2 attempts, then lock; digits must literally appear in the customer's message             │
        │ 3 Mandatory escalations (code, ESC-1): human request, unauthorised activity, legal, safety,     │
        │   >₹10,000 damage, repeat contact → create_ticket BEFORE any claim                              │
        │ 4 Retrieval: BM25 (+glossary) ∪ multilingual embeddings, RRF fusion, authority prior,           │
        │   superseded demotion / historical applicability, policy + successor back-fill, catalog rows    │
        │ 5 Responder LLM: standing rules + evidence + verified order facts + turn actions;               │
        │   tools: business_day_window, add_calendar_days, business_days_between, create_ticket          │
        │ 6 Guardrails: AI disclosure, decision consistency, secret scrubbing, honest ticket failures,   │
        │   cited IDs filtered to retrieved evidence                                                      │
        └──────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                       ▼
                              TurnResult: reply · decision · used/retrieved sections + scores ·
                              tool calls (redacted) · verification · escalation · diagnostics
```

## Modules (`src/zentromart/`)

| Module | Responsibility |
|---|---|
| `config.py` | Settings from env / `.env`; dataset discovery; voice profiles; reference date 2026-10-20 |
| `data.py` | CSVs read as strings (stdlib `csv`), catalog matching (SKU, spoken SKU, product names) |
| `dates.py` | Business-day / calendar-day calculators (Mon–Sat minus `holidays.csv`) used as LLM tools |
| `retrieval/ingest.py` | Front matter, `[SECTION-ID]` sections → chunks; authority rank; superseded applicability note |
| `retrieval/embeddings.py` | fastembed multilingual model; labelled TF-IDF fallback |
| `retrieval/index.py` | BM25 + persisted vectors (`runtime/index`), fingerprint invalidation |
| `retrieval/retriever.py` | Hybrid RRF search, precedence prior, back-fills, historical dates |
| `retrieval/glossary.py` | Customer phrasing → document vocabulary (lexical channel only) |
| `llm.py` | `GeminiLLM` (google-genai) and `AnthropicLLM` (anthropic): JSON output, tool loop, retries, throttling; `ScriptedLLM` mock |
| `tools_adapter.py` | Imports the **supplied** `tools/tools.py` unchanged; redirects ticket log to `runtime/`; redaction |
| `conversation.py` | Per-session state (order, verification, attempts, tickets, language) and session store |
| `agent.py` | `SupportAgent` (Steps 2–3) and `BaselineAgent` (Step 1) |
| `voice/gnani.py`, `voice/pipeline.py` | Gnani STT/TTS adapter (injectable), WAV splitting, `voice_turn` |
| `batch.py` | Folder of audio → results + `summary.csv/json`, per-file failure isolation, exit codes |
| `web/app.py` + `web/static/` | Starlette API + plain HTML/CSS/JS UI |
| `evaluation/` | Scoring rules, Step 1 vs Step 2, multi-turn, retrieval-only metrics |

## Key design decisions

* **Security in code, language in the model.** The model extracts the order ID and the proof
  value, but code validates both against the literal message, calls `lookup_order`, counts
  attempts and decides what the responder may see. Private fields reach the responder only in
  `<order_facts>`, which exists only after `verified=True`; the delivery pincode is never passed.
* **Escalate before you claim.** Rules that do not need judgement are enforced in code and the
  ticket is created before the responder writes the reply. The responder can also call
  `create_ticket` (e.g. overdue refunds), and if it declares `escalate` without a ticket, code
  creates one; a failed ticket produces an honest reply.
* **Precedence as mechanics, not only prose.** Authority rank is metadata on every chunk, used
  for ranking, back-fill of the governing policy and the evidence labels shown to the model.
  `pol_returns_v3_superseded.md` keeps its applicability note on every chunk; it is demoted
  unless the customer mentions a date before 1 August 2026, in which case the old 10-day
  electronics rule is surfaced together with the current RET-3 that preserves it.
* **Deterministic arithmetic.** Deadlines and refund windows come from calculator tools, never
  from the model's mental arithmetic.
* **Gold data isolation.** `evaluation/` is the only package that reads `reference_answer`,
  `must_include`, `expected_*`, `gold_evidence`; none of it is in prompts or the index.
* **Honest mocks.** `ScriptedLLM`, fake Gnani clients and their outputs carry `mock=True`, shown in
  the UI chips, batch summaries and evaluation reports.

## Data flow of secrets
`.env` (git-ignored) → `Settings` → SDK clients on the server. The browser never receives a key.
Verification values supplied by the customer are kept in `Session.secrets` for scrubbing and
are shown as `[redacted]` in tool-call logs, ticket summaries and public outputs.
