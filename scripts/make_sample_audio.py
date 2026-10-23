#!/usr/bin/env python3
"""Create SYNTHETIC customer test clips with Gnani TTS (no human recordings are supplied).

These clips exist only to exercise the batch runner end to end. They are machine speech from
Gnani's own TTS voices, clearly named *_gnani_tts.wav, and are NOT evidence of how Gnani STT
performs on human callers. Record your own voice for that (see demo_script.md).

  python scripts/make_sample_audio.py sample_outputs/batch_input
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zentromart.config import get_settings  # noqa: E402
from zentromart.voice.gnani import fix_wav_header  # noqa: E402

CLIPS = [  # (file stem, text, voice, language) — new phrasings, not copied from the practice set
    ("customer_001_en_cod_limit", "Hi, can I pay cash on delivery for an order of twelve thousand rupees?", "Shlok", "en-IN"),
    ("customer_002_hinglish_warranty", "Mere Brewly Electric Kettle ki warranty kitne mahine ki hai?", "Poorvi", "auto"),
    ("customer_003_hi_gift_card", "क्या मैं गिफ्ट कार्ड वापस कर सकती हूँ?", "Bhavna", "hi-IN"),
    ("customer_004_en_unsupported", "Do you offer gift wrapping for birthday orders?", "Trupti", "en-IN"),
    ("customer_005_en_safety", "My power bank is swelling up and smells like something is burning.", "Pranav", "en-IN"),
]


def main(out_dir: str) -> int:
    from gnani.tts import AudioConfig, GnaniTTSClient

    s = get_settings()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tts = GnaniTTSClient(api_key=s.gnani_api_key, timeout=60)
    for stem, text, voice, lang in CLIPS:
        audio = tts.synthesize(text, voice=voice, model=s.gnani_tts_model, language=lang,
                               audio_config=AudioConfig(sample_rate=16000))
        (out / f"{stem}_gnani_tts.wav").write_bytes(fix_wav_header(audio))
        print("wrote", stem)
    # 2 s of silence: exercises the "didn't catch that" path
    with wave.open(str(out / "customer_006_silence.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b"\x00\x00" * 32000)
    # an intentionally broken file: exercises per-file failure recording
    (out / "customer_007_corrupt.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEthis is not valid audio" * 20)
    (out / "README.md").write_text(
        "# Batch input clips\n\nSYNTHETIC test inputs. Files ending in `_gnani_tts.wav` were generated with Gnani TTS "
        "(timbre-v2.5) from the texts in `scripts/make_sample_audio.py`; they are machine speech, not recordings of people. "
        "`customer_006_silence.wav` is silence and `customer_007_corrupt.wav` is deliberately invalid, to show failure handling.\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "sample_outputs" / "batch_input")))
