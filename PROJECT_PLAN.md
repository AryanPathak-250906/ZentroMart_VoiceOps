# PROJECT_PLAN — VoiceOps 2026 · ZentroMart voice support agent

Written after reading the full capstone brief (14 pages), the dataset `README.md`, `DATA_DICTIONARY.md`,
`ZIP_SCHEMA.md`, `CHANGELOG.md`, all 21 KB documents, `tools/tools.py`, `validate_dataset.py`,
all 52 practice questions and all 23 practice conversations (96 turns).

## 1. Facts established during inspection

| Item | Observed |
|---|---|
| Original validator `python validate_dataset.py` | **PASS** (exit 0): 9 CSVs, 21 KB docs, keys, order math, evidence, verification gates |
| Counts | 300 customers · 1,000 orders · 1,783 items · 70 products · 36 pincodes · 4 holidays · 244 logs · 52 questions · 23 conversations / 96 turns |
| Audio in archive | none |
| Reference date | 2026-10-20 (a Tuesday) |
| Cloud sandbox network | `api.anthropic.com` reachable; **PyPI, Hugging Face, `api.vachana.ai`, `docs.gnani.ai` (curl) blocked** |
| Laptop sandbox (device VM) network | same restriction (only `api.anthropic.com`) |
| Packages available in sandbox | starlette, uvicorn, python-multipart, pydantic, numpy, reportlab, ffmpeg; **not** anthropic, fastapi, pytest, gnani-vachana, fastembed |
| Credentials | `GNANI_API_KEY` supplied by user (stored only in git-ignored `.env`); **no `ANTHROPIC_API_KEY`** |

## 2. Architecture (one agent, three front-ends)

```
customer text ─┐                     ┌──────────── SupportAgent.respond(session, text) ────────────┐
web chat ──────┤                     │ 1 planner LLM call (structured JSON, validated by pydantic)  │
web voice ─► Gnani STT ─┤──────────► │ 2 code-enforced identity workflow → supplied lookup_order    │
batch audio ─► Gnani STT ┘           │ 3 hybrid retrieval (BM25 + vectors, RRF) + precedence rules │
                                     │ 4 code-mandated escalations → supplied create_ticket         │
                                     │ 5 responder LLM call with calculator / ticket tools, JSON    │
                                     │ 6 guardrails: AI disclosure, leak redaction, claim checks    │
                                     └──────────────────────────────────────────────────────────────┘
                                                   │ TurnResult (reply + sources + tools + state)
                              reply text ─► Gnani TTS (timbre-v2.5) ─► reply.wav
```

* `src/zentromart/config.py` – settings from env / `.env`, dataset path discovery.
* `data.py` – CSV loading with `dtype=str` semantics (stdlib csv), product catalog matching.
* `dates.py` – business-day / calendar-day arithmetic using `holidays.csv`.
* `retrieval/` – front-matter parsing, section chunking, BM25, embeddings (fastembed multilingual
  model; clearly-labelled char-n-gram TF-IDF fallback when the model cannot be downloaded),
  persisted index with content hash invalidation, precedence-aware hybrid retriever.
* `llm.py` – Anthropic client (official SDK; raw-HTTPS transport only if the SDK is not
  installed, reported in diagnostics), timeouts, retries, model check, mock client for tests.
* `tools_adapter.py` – imports the *supplied* `tools/tools.py` unchanged, redirects the ticket
  log to `runtime/`, records every call, redacts secrets.
* `agent.py` / `conversation.py` / `prompts.py` / `schemas.py` – the support agent.
* `voice/gnani.py` – Gnani STT/TTS adapter (official `gnani-vachana` SDK, injectable clients).
* `batch.py` – folder-of-audio runner. `web/app.py` + `web/static/` – Starlette app.
* `evaluation/` – Step 1 vs Step 2 single-turn eval, multi-turn eval, retrieval-only eval,
  deterministic scoring rules, optional LLM judge.

Deviations from the suggested stack (justified by the environment): **Starlette** instead of
FastAPI (FastAPI is Starlette + extras and cannot be installed here; Starlette can be tested),
**unittest** instead of pytest (tests still run under pytest).

## 3. Stages
1. Inspect + validate (done). 2. Core data/retrieval/LLM. 3. Agent, verification, tickets,
sessions. 4. Evaluation harness. 5. Gnani adapter. 6. Web UI + batch CLI. 7. Tests, fixes.
8. Docs, `report.pdf`, `prepare_submission.py`, ZIP verification, copy to the user's Documents.

## 4. Test strategy
* Offline unit/integration tests with injected mock LLM / mock Gnani clients (always labelled
  `mock=True` in outputs): data typing, chunk/section retention, precedence, superseded handling,
  verification workflow (success, one retry, lock-out, no leakage), tickets, escalation, web
  routes, batch failure isolation, missing credentials, malformed model output.
* Retrieval-only evaluation on the 52 questions (needs no LLM) – real numbers.
* Live evaluation (Step 1 vs Step 2, multi-turn) – requires `ANTHROPIC_API_KEY`.
* Live voice – requires `GNANI_API_KEY` **and** network access to `api.vachana.ai` + PyPI.

## 5. Credential- / network-dependent tasks (pending until available)
| Task | Needs |
|---|---|
| Step 1 vs Step 2 scores, multi-turn scores | `ANTHROPIC_API_KEY` (+ `pip install anthropic` or the HTTPS transport) |
| Live STT/TTS, batch audio run, `demo.mp4` | `GNANI_API_KEY`, `pip install gnani-vachana`, network to `api.vachana.ai` |
| Neural embedding index | `pip install fastembed` + Hugging Face download (first run) |
