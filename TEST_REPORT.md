# TEST_REPORT

Every entry below was actually executed in this project's environment (Linux, Python 3.13, venv
built from `requirements*.txt`). "Live" = real external API calls; "mock" = injected fake
clients (never presented as live results). Numbers are copied from the saved artefacts named.

## 1. Dataset validator (original, unmodified)
`cd dataset/Capstone_Project_Dataset_FTTT_2026 && python validate_dataset.py` → exit 0
```
PASS — all 9 CSVs, 21 KB documents, primary/foreign keys, order math,
       practice evidence, verification gates and company naming validated.
       1,000 orders | 1,783 items | 244 logs | 52 questions | 23 conversations
```
The source dataset was not modified; runtime tickets go to `runtime/tickets/` (a test asserts
that `tools/tickets_created.csv` is never created inside the dataset).

## 2. Offline automated tests (mock LLM, fake Gnani clients, TF-IDF backend)
`python -m pytest -q` → **61 passed**; `python -m unittest discover -s tests -t .` → **Ran 61 tests, OK**.

| File | Covers |
|---|---|
| `tests/test_data_retrieval.py` | dataset path resolution; row counts; fuzzy product fallback for ASR misspellings (never overrides exact matches, refuses ambiguous ones); IDs/pincodes/phones/RRNs stay strings (leading zeros); joins; SKU / spoken-SKU / product-name matching; all 21 docs + every `[SECTION-ID]` survive ingestion; front matter; superseded applicability note; FAQ → policy back-fill; superseded demoted without a historical date; historical exception kept with a pre-August date; section IDs/scores in results; standing-doc exclusion; index persistence and invalidation on KB change; business-day maths with Sundays + holidays |
| `tests/test_agent_safety.py` | first-turn AI disclosure (EN/HI); missing order ID → ask, no lookup; unverified lookup returns nothing private; success with phone suffix and with pincode; one retry then lock-out (even the correct value is not tried after lock); invented order IDs ignored; private facts only after verification; pincode never passed to the model; secrets redacted in public output; lookup tool failure reported; human request → ticket in `runtime/`; duplicate tickets avoided; ticket summary fields + secret scrubbing; failed ticket never claimed; model "escalate" without ticket → ticket created; high-value damage from catalog price; missing LLM; LLM timeout; malformed responder JSON (one controlled retry); invalid planner JSON → fallback; retrieval failure flagged; session isolation; refund table parsed from the KB; overdue refund escalated deterministically; cross-reference back-fill; general questions not forced to "clarify" |
| `tests/test_llm_voice_web.py` | missing keys; JSON parsing; Anthropic adapter (no sampling params, feature drop on HTTP 400, tool loop, error mapping); Gemini adapter (per-minute 429 retry, daily quota → stop); Gnani: not configured, ITN format, long WAV split, invalid/empty/missing audio, STT failure kind, TTS voice selection + same-provider fallback, WAV header fix; voice turn with empty transcript; batch run with one corrupt file (exit 1, no fake WAV), fatal errors (exit 2); web routes: health (no secrets), session create/reset/delete, chat, voice upload (+ 503 when Gnani missing, 400 when empty), audio serving + path traversal |
| `tests/test_scoring.py` | must_include groups/alternatives, dates in many formats incl. Hindi, number words, word boundaries, must_not, expected tool-call parsing/matching |

## 3. Live API checks
| Check | Result |
|---|---|
| Gnani TTS → STT round trip (`gnani-vachana` 0.8.1) | EN (Kaveri, en-IN): "hello my order number is 89782837 can i cancel it" (ITN digits); HI (Nalini, hi-IN) and Hinglish (Poorvi, `auto`) synthesised and transcribed; 2 s silence → empty transcript (`success: true`) |
| SDK facts verified in source | `GnaniSTTClient.transcribe(audio, language_code, format)`, `GnaniTTSClient.synthesize(text, voice, model, language, audio_config, output_file)`; SDK default `timbre-v2.0` (so the model is always passed); `language="hi-en"` is rejected locally by 0.8.1 → Hinglish uses voice Poorvi with `language="auto"` |
| `python run.py --check` | dataset found; 84 chunks / 21 files; fastembed multilingual MiniLM index; Gemini model available; Gnani TTS live OK |
| Gemini free tier | `gemini-3.6-flash` / `gemini-3.5-flash`: 20 requests/day (unusable for evaluation); `gemini-3.5-flash-lite`: 500 requests/day → used for all LLM results |
| Web UI | `/api/health` live; desktop (1280 px) and mobile (390 px, no horizontal scroll) screenshots checked |

## 4. Evaluations (live LLM unless stated) — `reports/eval/`
All with `gemini-3.5-flash-lite` (free tier), `LLM_PROVIDER=gemini`, final code unless marked as an iteration.

| Run | Result | Artefact |
|---|---|---|
| Retrieval only (no LLM), typed questions | all gold KB sections retrieved 43/45 (95.6%); section recall 58/60 (96.7%); catalog rows 13/13 | `metrics.json → retrieval` |
| Retrieval only, `asr_text` | 43/45 (95.6%); 58/60 (96.7%) | `metrics.json → retrieval_asr` |
| **Step 1** (prompt only), 52 questions | **16/52 (30.8%)** full pass; decisions 41/52; facts 19/52; unsupported 3/5 | `step1_results.jsonl` |
| Step 2 iteration 1 | 43/51 (84.3%) (1 case lost to a network reset) | `reports/eval_iterations/iter1/` |
| Step 2 iteration 2 | 48/52 (92.3%) | `reports/eval_iterations/iter2/` |
| **Step 2 final**, typed questions | **51/52 (98.1%)** full pass; decisions 51/52; facts 52/52; no forbidden 52/52; unsupported 5/5; all gold evidence cited 31/47 | `step2_results.jsonl` |
| **Step 2 final, `asr_text`** (voice-like input) | **47/52 (90.4%)**; decisions 49/52; facts 50/52; unsupported 4/5 | `step2_asr_results.jsonl` |
| Multi-turn iteration 2 | 86/92 turns (93.5%) on 22/23 conversations (daily quota hit) | `reports/eval_iterations/iter2/` |
| **Multi-turn final**, 23 conversations / 96 turns | **84/96 turns (87.5%)**; decisions 91/96; facts 90/96; no forbidden 96/96; tools 90/96; privacy 96/96; AI disclosure 23/23; 14/23 conversations fully passed | `multiturn_results.jsonl` |
| **Batch audio run** (live Gnani + LLM) | 7 input files → 6 handled end to end, 1 deliberately corrupt file recorded as `invalid_audio`; exit code 1 | `sample_outputs/batch_run/` |
| **Demo** (live web UI) | 4 spoken turns, 113.7 s, audio −18.9 dB mean | `demo.mp4`, DEMO_STATUS.md |

Failure analysis of the final runs (all failures, from the JSONL files):
* Step 2 typed: PQ0048 — correct facts, but ends by asking for an order ID → "clarify".
* Step 2 ASR: PQ0004 (stated "we do not offer midnight delivery" instead of "I don't know"),
  PQ0032 (omitted "12 months"), PQ0024/PQ0027/PQ0047 (answered, then asked a question → "clarify").
* Multi-turn: PC007/PC008/PC010 T3–T4 (ticket created on the verification turn, one turn early;
  regression introduced by the refund check + order facts), PC003 T4 (festive rule misapplied),
  PC004 T5 (dates computed for a refund that had not started), PC019/PC020 T4 ("answer" instead
  of "refuse", RTO refund timing missing), PC015/PC016 T4 (apology "khed hai" not recognised by
  the scorer — correct on manual review).

## 5. Change made after the final typed/multi-turn runs
A fuzzy product-name fallback (`Dataset._fuzzy_products`) for speech misspellings was added after
those runs, prompted by the live batch run ("bruli" for "Brewly"). It only runs when exact matching
finds nothing. Verified offline: product matching returns identical results for all 200 practice
texts (52 questions, 52 `asr_text`, 96 conversation messages). The batch run and the ASR-text
evaluation were executed with it.

## 6. Not tested
* Human speech (no recordings were available) — STT was tested on Gnani-TTS-generated clips and silence.
* The Anthropic provider against the live API (no API credits).
* Paraphrased forbidden claims (scorer limitation).
