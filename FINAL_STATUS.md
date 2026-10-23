# FINAL_STATUS

Status as of 21 October 2026. Legend: **verified** = executed here and checked ·
**implemented, not live-tested** · **pending manual action**.

| # | Deliverable | Status | Evidence |
|---|---|---|---|
| 1 | Dataset validation (original script) | verified — PASS | TEST_REPORT §1 |
| 2 | Data layer (string IDs, joins, catalog) | verified | tests/test_data_retrieval.py |
| 3 | KB ingestion, hybrid RAG, precedence, superseded handling | verified | retrieval eval 43/45 questions with all gold sections; tests |
| 4 | LLM runtime — Gemini (`LLM_PROVIDER=gemini`, used for all results) | verified (live) | reports/eval |
| 5 | LLM runtime — Anthropic (`LLM_PROVIDER=anthropic`) | implemented, not live-tested | no API credits; adapter covered by mock tests |
| 6 | Conversation agent: verification, retry limit, tickets, escalations, sessions | verified (live + mock) | multi-turn 84/96 turns; 96/96 privacy; safety tests |
| 7 | Step 1 vs Step 2 evaluation (52 questions) | verified | Step 1 16/52 (30.8%) → Step 2 51/52 (98.1%); ASR text 47/52 (90.4%) |
| 8 | Multi-turn evaluation (23 conversations, 96 turns) | verified | 84/96 turns (87.5%), 14/23 conversations fully passed |
| 9 | Gnani STT/TTS adapter (official `gnani-vachana` 0.8.1) | verified (live) | TEST_REPORT §3; batch run; demo |
| 10 | Batch runner (`--audio-dir … --output-dir …`) | verified (live) | sample_outputs/batch_run: 6/7 handled, 1 deliberately corrupt file recorded as failure, exit 1 |
| 11 | Web UI (chat + voice + evidence panel) | verified | route tests; screenshots; demo recorded through it |
| 12 | `report.pdf` (≤ 4 pages) | verified — 4 pages | report.pdf, reports/report.pdf |
| 13 | `demo.mp4` (≤ 2 min) | verified — 1:54, genuine run; **customer voice synthetic** (labelled) | DEMO_STATUS.md |
| 14 | Submission ZIP (allowlist, secret scan, clean extract) | verified | `prepare_submission.py` output |
| 15 | Human-voice test of Gnani STT | **pending manual action** | record your own voice (demo_script.md) |
| 16 | Public GitHub repository | **pending manual action** | create it, push, paste URL into README |
| 17 | Key rotation | **pending manual action** | regenerate the Gnani and Gemini keys after grading (they were shared in chat) |

## What you (the participant) still need to do
1. Create the public GitHub repo, push this folder (the `.gitignore` already excludes `.env`,
   `.venv/`, `dataset/`, `runtime/`), and replace the URL placeholder in README.md.
2. Optional but recommended: record a demo with your own voice (`demo_script.md`) and replace
   `demo.mp4`; it shows Gnani STT on human speech.
3. Rebuild the ZIP after any change: `python prepare_submission.py --name <team_name>`.
4. Regenerate your API keys after grading.

## Known limitations (also in report.pdf)
* Scores were measured on the practice set that was also used for development.
* Free-tier model (Gemini 3.5 Flash-Lite, 500 requests/day); remaining multi-turn failures are
  listed in the report with proposed fixes that were deliberately not applied after the final run.
* The automatic scorer misses paraphrases (e.g. the Hindi apology "khed hai").
