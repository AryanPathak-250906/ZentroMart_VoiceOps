# ZentroMart VoiceOps 2026 — grounded voice support agent

Capstone project for **VoiceOps 2026** (LLM & Generative AI course, Consulting & Analytics Club,
IIT Guwahati × Gnani.ai). A customer-support agent for the fictional store ZentroMart that:

* answers from ZentroMart's policy documents with **hybrid RAG** (multilingual embeddings + BM25)
  and a **policy-precedence** engine (current policy > agent guide > help > FAQ > marketing;
  superseded documents only for the cases their successor preserves),
* holds **multi-turn conversations** in English, Hindi and Hinglish, verifies callers with the
  supplied `lookup_order` tool (one proof value, one retry), and hands over to humans with the
  supplied `create_ticket` tool,
* **listens and speaks with Gnani only** (`gnani-vachana` SDK: STT REST + TTS `timbre-v2.5`),
* ships a browser UI, a terminal chat, a **batch audio runner** and an **evaluation harness**
  (Step 1 prompt-only vs Step 2 RAG, plus all 23 practice conversations).

**Public GitHub repository:** `<<FILL IN: https://github.com/<your-user>/<repo>>>` (not created yet)

---

## 1. Requirements
* Python **3.10 – 3.13** (tested on 3.13 Linux; the code uses only cross-platform paths).
* Internet access to: `api.vachana.ai` (Gnani), the LLM API (`generativelanguage.googleapis.com`
  for Gemini or `api.anthropic.com`), and `huggingface.co` once, to download the embedding model.
* Optional: `ffmpeg` (only to accept `.webm/.opus/.mp4` audio in the batch runner).

## 2. Folder structure
```
VoiceOps2026_ZentroMart/
├── run.py                    # single entry point (web, batch, eval, check, chat)
├── prepare_submission.py     # builds + verifies the submission ZIP (allowlist)
├── requirements.txt          # runtime deps   · requirements-dev.txt (tests, report, demo)
├── requirements-lock.txt     # exact versions from the tested environment (Linux, Py 3.13)
├── .env.example              # copy to .env — no keys inside
├── src/zentromart/           # agent, retrieval, llm, tools adapter, voice, batch, web, evaluation
├── web/static/               # index.html, app.js, styles.css (browser UI)
├── tests/                    # offline unit/integration tests (mock LLM / mock Gnani)
├── scripts/                  # make_sample_audio.py, build_report.py, record_demo.py
├── reports/eval/             # metrics.json, EVAL_REPORT.md, per-case JSONL results
├── sample_outputs/           # real batch run outputs (audio in → transcript, reply, reply.wav, sources)
├── report.pdf                # ≤ 4-page report (also reports/report.pdf)
└── ARCHITECTURE.md · TEST_REPORT.md · FINAL_STATUS.md · DEMO_STATUS.md · demo_script.md · PROJECT_PLAN.md
```

## 3. Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
**Dataset (not included in the repo/ZIP):** extract `Capstone_Project_Dataset_FTTT_2026.zip` so
that this path exists: `dataset/Capstone_Project_Dataset_FTTT_2026/kb/`. Alternatively set
`ZENTROMART_DATASET_DIR` or pass `--dataset-dir`. The supplied files are never modified
(tickets are written to `runtime/tickets/`, not to the dataset's `tools/` folder).

Validate the dataset first (original validator, unchanged):
```bash
cd dataset/Capstone_Project_Dataset_FTTT_2026 && python validate_dataset.py && cd ../..
```

**Keys:** copy `.env.example` to `.env` and fill in:

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | `gemini` (default) or `anthropic` |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | Google AI Studio key; default model `gemini-3.5-flash-lite` (free-tier friendly) |
| `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` | Anthropic Console key (separate from a Claude chat subscription) |
| `GNANI_API_KEY` | Gnani key from app.gnani.ai/voice — one key for STT and TTS |

Then check everything (contacts the APIs once; `--offline` skips that):
```bash
python run.py --check
```
The first run downloads the embedding model (~220 MB) and builds `runtime/index/`; later runs
load it from disk and rebuild automatically when a KB file or the embedding setting changes.

## 4. Run
| Task | Command |
|---|---|
| Browser UI (text + voice) | `python run.py --web` → open http://127.0.0.1:8000 |
| **Batch audio (organisers' run command)** | `python run.py --audio-dir INPUT_AUDIO_FOLDER --output-dir OUTPUT_FOLDER` |
| One question | `python run.py --ask "How long do UPI refunds take?"` |
| Terminal conversation | `python run.py --chat` |
| All evaluations | `python run.py --eval all` |
| Single evaluation | `python run.py --eval retrieval` · `step1` · `step2` · `multiturn` (add `--asr` to use `asr_text`) |

Batch options: `--stt-language hi-IN` (default `en-IN`, which also handles Hinglish),
`--conversation` (treat the sorted files as one conversation). Supported audio: `.wav .mp3 .flac
.ogg .m4a .aac` (plus `.webm .opus .mp4` if ffmpeg is installed). WAV files longer than
`GNANI_STT_MAX_SECONDS` (30 s) are split. Exit code: `0` all files OK, `1` some failed, `2` fatal
configuration/input error.

Batch output:
```
OUTPUT_FOLDER/
├── summary.csv, summary.json         # one row per input file (status, transcript, reply, sources, tickets, errors)
└── results/<file-stem>/
    ├── transcript.txt                # Gnani STT output
    ├── reply.txt                     # agent reply
    ├── reply.wav                     # Gnani TTS audio (only written when synthesis succeeded)
    ├── sources.json                  # used + retrieved section IDs with scores
    └── result.json                   # full turn record (tool calls redacted, diagnostics)
```

Evaluation output (`reports/eval/`): `metrics.json`, `EVAL_REPORT.md`, `step1_results.jsonl`,
`step2_results.jsonl`, `multiturn_results.jsonl`, `retrieval_rows.json`. Runs are **resumable**:
re-running the same command skips completed cases (useful when a free-tier daily quota ends);
use `--fresh` to start over.

## 5. Tests
```bash
pip install -r requirements-dev.txt
python -m pytest -q            # or: python -m unittest discover -s tests -t .
```
Tests are offline: they use a scripted mock LLM and fake Gnani clients (all flagged `mock=True`)
and the TF-IDF retrieval backend, so they need the dataset but no keys or network.

## 6. How it works (short)
See `ARCHITECTURE.md`. Per turn: planner LLM (JSON) → code-enforced verification with
`lookup_order` → code-enforced escalations with `create_ticket` → hybrid retrieval with
precedence → responder LLM with date calculators → guardrails. The same `SupportAgent` serves
the web UI, voice turns, the batch runner and the evaluation.

## 7. Troubleshooting
| Symptom | Fix |
|---|---|
| `dataset not found` | extract the dataset into `./dataset/` or set `ZENTROMART_DATASET_DIR` |
| `LLM not configured` | set `LLM_PROVIDER` and the matching key in `.env` |
| Gemini `quota_exhausted` | free-tier daily quota used up; resume later (`--eval ... ` resumes) or set another `GEMINI_MODEL` |
| `Voice is not configured` | set `GNANI_API_KEY`; `pip install gnani-vachana` |
| Gnani 400 on TTS | keep `GNANI_TTS_MODEL=timbre-v2.5`; voices must exist for that model |
| Microphone blocked | the browser needs `http://127.0.0.1` / `localhost` (or HTTPS) and mic permission |
| Embedding download fails | set `EMBEDDING_BACKEND=tfidf` (clearly labelled lexical fallback) |

## 8. Results (practice set, `gemini-3.5-flash-lite`, see report.pdf)
| Evaluation | Result |
|---|---|
| Step 1 — prompt only (52 questions) | 16/52 (30.8%) full pass |
| Step 2 — RAG agent (52 questions) | 51/52 (98.1%) · on ASR-like text 47/52 (90.4%) |
| Multi-turn (23 conversations, 96 turns) | 84/96 turns (87.5%) · privacy 96/96 · AI disclosure 23/23 |
| Retrieval only (no LLM) | all gold sections retrieved for 43/45 questions |
| Live batch (Gnani STT → agent → Gnani TTS) | 6/7 files handled; 1 deliberately corrupt file recorded as failure |

## 9. Status, limitations and pending work
See `FINAL_STATUS.md` (what is verified vs. pending) and `TEST_REPORT.md` (what was run).
Key limitations: practice-set scores are measured on the data the system was developed against;
the automatic scorer cannot detect paraphrased forbidden claims; the identity check is the
capstone's educational simulation, not production authentication.
