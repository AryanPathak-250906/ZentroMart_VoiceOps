# DEMO_STATUS

**`demo.mp4` exists (1 min 54 s, 1280×760, H.264 + AAC) and shows a genuine run of the project.**

How it was made (`scripts/record_demo.py`, 21 October 2026):
* The real web app (`python run.py --web`) was started; Chromium (Playwright) opened it and the
  screen was recorded.
* **Customer voice — synthetic and labelled on the title card.** No human recording was available,
  so the four customer utterances were generated with a *different* Gnani TTS voice (Shlok,
  `timbre-v2.5`). They were fed to the page's microphone (`getUserMedia`), so the app's own
  recorder, WAV encoder and upload path were used.
* Everything after the microphone is real: Gnani STT transcribed each clip, the agent (Gemini
  `gemini-3.5-flash-lite`, RAG, tools) answered, Gnani TTS (voice Kaveri) spoke the reply, and the
  evidence panel shows the sources, the `business_day_window` calculator and the created ticket.
* The audio track was assembled with ffmpeg from the same customer clips and the Gnani reply files
  that the browser played, placed at the times they played.

Conversation in the video:
1. "My refund to UPI was started on the sixteenth of October 2026. By when should I get it?" →
   between 17 and 21 October 2026 (Sunday and the 19 October holiday skipped) — REF-2, KB-00.3.
2. Follow-up: "And how long would it take if I had paid by credit card?" → 23 to 26 October 2026.
3. "Do you offer gift wrapping for my orders?" → "I don't have information … connect you to a colleague?"
4. "Yes please, connect me to a human." → `create_ticket` → ticket created, handover confirmed.

Known imperfection visible in the video: in turn 4 the spoken reply reads a shortened, re-grouped
form of the ticket ID (the full ID is shown in the UI).

What is still worth doing yourself: record the same script with **your own voice**
(`demo_script.md`) — the brief allows recording your own voice, and it would demonstrate Gnani STT
on human speech, which this video does not.
