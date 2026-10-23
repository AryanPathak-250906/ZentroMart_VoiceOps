# Demo script (≤ 2 minutes)

The brief asks for a ≤ 2-minute video: a question answered from the documents with a follow-up,
plus one handover or "I don't know". `demo.mp4` in this folder was produced automatically by
`scripts/record_demo.py` (real app, real Gnani STT/TTS, real LLM; customer voice synthesised with a
second Gnani voice — see DEMO_STATUS.md). To record **your own** voice version:

## Setup (1 minute before recording)
1. `python run.py --check` — all lines OK (LLM, Gnani TTS live).
2. `python run.py --web`, open http://127.0.0.1:8000 in Chrome/Edge, allow the microphone.
3. Start a screen recorder **with system audio** (Windows: Xbox Game Bar `Win+Alt+R`, or OBS).
4. Click **New conversation**. Keep "Speak: English / Hinglish" selected.

## Script (click **Speak**, talk, click **Stop**; wait for the spoken reply)
| # | You say | What should happen (point at it) |
|---|---|---|
| 1 | "Hi, my refund to UPI was started on the sixteenth of October 2026. By when should I get it?" | AI introduces itself; 1–3 business days → between 17 and 21 October 2026 (Sunday + 19 Oct holiday skipped). Panel: REF-2, KB-00.3, `business_day_window`. |
| 2 | "And how long would it take if I had paid by credit card?" | Follow-up understood without repeating the date: 5–7 business days → 23–26 October 2026. |
| 3 | "Do you offer gift wrapping for my orders?" | "I don't have that information" + offer of a colleague (not covered by the documents). |
| 4 | "Yes please, connect me to a human." | `create_ticket` → ticket ID shown; handover outcome banner in the panel. |

Optional extra (shows verification): "My order 2629 5015 hasn't arrived" → asks for last 4 digits
or pincode → give a **wrong** value once → it asks again without revealing anything.

## Tips
* Speak 1–2 seconds after clicking Speak; keep each turn under ~25 s (the recorder stops at 28 s).
* If a reply is slow, it is the free-tier rate limit (GEMINI_RPM); keep recording, or trim later.
* Trim the video to ≤ 2:00 and save as `demo.mp4` in the project root before running
  `python prepare_submission.py`.
