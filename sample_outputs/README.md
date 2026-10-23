# sample_outputs

Real outputs of the project's run command, produced on 21 October 2026 with the final code:

```
python run.py --audio-dir sample_outputs/batch_input --output-dir sample_outputs/batch_run
```

* **Speech:** Gnani STT (REST, `en-IN`, ITN) and Gnani TTS (`timbre-v2.5`) — real API calls.
* **LLM:** Gemini `gemini-3.5-flash-lite` (free tier) — real API calls.
* **Inputs (`batch_input/`) are synthetic:** the `*_gnani_tts.wav` clips were generated with Gnani TTS
  voices from the texts in `scripts/make_sample_audio.py` (no human recordings were available);
  `customer_006_silence.wav` is 2 s of silence; `customer_007_corrupt.wav` is deliberately invalid.
* **Result:** 6/7 files handled end to end (5 answered/escalated with spoken replies, silence → "Sorry, I didn't
  catch that" with audio); the corrupt file is recorded as `invalid_audio` with no transcript or
  audio written. Process exit code was `1` (at least one file failed), as designed.

Per file (`batch_run/results/<name>/`): `transcript.txt` (Gnani STT), `reply.txt`, `reply.wav`
(Gnani TTS), `sources.json` (used + retrieved section IDs with scores), `result.json` (full turn
record; tool-call arguments with verification values redacted). `summary.csv` / `summary.json`
list every file with status, decision, sources, tickets and errors.

Note: the ticket created for the safety-risk clip (`customer_005`) was written to the local
runtime ticket log, which is not part of the submission.
