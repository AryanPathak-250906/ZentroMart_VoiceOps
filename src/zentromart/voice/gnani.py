"""Gnani speech adapter — the ONLY speech-to-text / text-to-speech provider in this project.

Uses the official ``gnani-vachana`` SDK (verified against v0.8.1 source):
  * ``gnani.stt.GnaniSTTClient(api_key=..., timeout=...)``
      ``.transcribe(path, language_code="en-IN", format="transcribe")`` -> dict with "transcript"
      (REST; supported .wav .mp3 .flac .ogg .m4a .aac; docs state a 60 s / 30 s clip limit)
  * ``gnani.tts.GnaniTTSClient(api_key=..., timeout=...)``
      ``.synthesize(text, voice="Kaveri", model="timbre-v2.5", language="en-IN",
                    audio_config=AudioConfig(...), output_file=...)`` -> WAV bytes
The TTS model is always named explicitly (the SDK default timbre-v2.0 is no longer served).
Clients can be injected for offline tests; real clients are created lazily from GNANI_API_KEY.
"""
from __future__ import annotations

import io
import re
import struct
import tempfile
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
CONVERTIBLE_EXTENSIONS = {".webm", ".opus", ".mp4", ".mpeg", ".mpga", ".wma"}  # need ffmpeg


class VoiceError(RuntimeError):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind  # not_configured | sdk_missing | invalid_audio | stt_failed | tts_failed | too_long


@dataclass
class STTResult:
    transcript: str
    request_ids: list[str] = field(default_factory=list)
    segments: int = 1
    language_code: str = "en-IN"
    latency_ms: int = 0
    mock: bool = False


@dataclass
class TTSResult:
    audio: bytes
    voice: str
    language: str
    model: str
    latency_ms: int = 0
    mock: bool = False
    fallback_used: bool = False


def fix_wav_header(data: bytes) -> bytes:
    """Gnani streams WAV with placeholder sizes; rewrite RIFF/data sizes from the real length."""
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return data
    buf = bytearray(data)
    struct.pack_into("<I", buf, 4, len(buf) - 8)
    idx = buf.find(b"data", 12)
    if idx != -1 and idx + 8 <= len(buf):
        struct.pack_into("<I", buf, idx + 4, len(buf) - idx - 8)
    return bytes(buf)


def wav_duration_s(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except (wave.Error, EOFError, OSError):
        return None


def split_wav(path: Path, max_s: float, out_dir: Path) -> list[Path]:
    """Split a PCM WAV into consecutive chunks of at most ``max_s`` seconds (stdlib only)."""
    with wave.open(str(path), "rb") as w:
        params = w.getparams()
        frames_per_chunk = int(max_s * params.framerate)
        chunks, i = [], 0
        while True:
            frames = w.readframes(frames_per_chunk)
            if not frames:
                break
            out = out_dir / f"{path.stem}_part{i:02d}.wav"
            with wave.open(str(out), "wb") as o:
                o.setparams(params)
                o.writeframes(frames)
            chunks.append(out)
            i += 1
    return chunks


def convert_with_ffmpeg(src: Path, out_dir: Path) -> Path:
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise VoiceError("invalid_audio", f"{src.suffix} audio needs ffmpeg for conversion (not installed). Use WAV/MP3/FLAC/OGG/M4A/AAC.")
    out = out_dir / f"{src.stem}_16k.wav"
    proc = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", "16000", str(out)],
                          capture_output=True, text=True, timeout=120)
    if proc.returncode != 0 or not out.is_file():
        raise VoiceError("invalid_audio", f"ffmpeg could not convert {src.name}: {proc.stderr.strip()[:200]}")
    return out


def text_for_speech(text: str) -> str:
    """Remove markdown/symbols that TTS would read literally."""
    t = re.sub(r"[*_#`>|]", "", text)
    t = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


class GnaniVoice:
    def __init__(self, settings, stt_client=None, tts_client=None):
        self.settings = settings
        self._stt = stt_client
        self._tts = tts_client
        self.mock = bool(getattr(stt_client, "is_mock", False) or getattr(tts_client, "is_mock", False))

    # ------------------------------------------------------------------ availability
    def status(self) -> dict:
        sdk = True
        try:
            import gnani.stt  # noqa: F401
            import gnani.tts  # noqa: F401
        except ImportError:
            sdk = False
        configured = bool(self.settings.gnani_api_key) or self._stt is not None
        return {"provider": "gnani", "sdk_installed": sdk or self._stt is not None, "configured": configured,
                "tts_model": self.settings.gnani_tts_model, "stt_language_default": self.settings.gnani_stt_language,
                "mock": self.mock,
                "message": ("ready" if configured and (sdk or self._stt is not None)
                            else "GNANI_API_KEY is not set" if not configured else "pip install gnani-vachana")}

    def _clients(self):
        if self._stt is not None and self._tts is not None:
            return self._stt, self._tts
        if not self.settings.gnani_api_key:
            raise VoiceError("not_configured", "Voice is not configured: set GNANI_API_KEY in .env (see README).")
        try:
            from gnani.stt import GnaniSTTClient
            from gnani.tts import GnaniTTSClient
        except ImportError as exc:
            raise VoiceError("sdk_missing", "Gnani SDK not installed: pip install gnani-vachana") from exc
        self._stt = self._stt or GnaniSTTClient(api_key=self.settings.gnani_api_key, timeout=60)
        self._tts = self._tts or GnaniTTSClient(api_key=self.settings.gnani_api_key, timeout=60)
        return self._stt, self._tts

    # ------------------------------------------------------------------ STT
    def transcribe_file(self, path: Path, language_code: str | None = None) -> STTResult:
        path = Path(path)
        if not path.is_file():
            raise VoiceError("invalid_audio", f"audio file not found: {path.name}")
        if path.stat().st_size < 100:
            raise VoiceError("invalid_audio", f"audio file is empty or truncated: {path.name}")
        stt, _ = self._clients()
        lang = language_code or self.settings.gnani_stt_language
        fmt = self.settings.gnani_stt_format if lang in ("en-IN", "hi-IN") else "verbatim"
        t0 = time.time()
        with tempfile.TemporaryDirectory(prefix="zm_stt_") as tmp:
            tmpdir = Path(tmp)
            src = path
            ext = path.suffix.lower()
            if ext in CONVERTIBLE_EXTENSIONS:
                src = convert_with_ffmpeg(path, tmpdir)
            elif ext not in SUPPORTED_EXTENSIONS:
                raise VoiceError("invalid_audio", f"unsupported audio format '{ext}'")
            parts = [src]
            if src.suffix.lower() == ".wav":
                dur = wav_duration_s(src)
                if dur is None:
                    raise VoiceError("invalid_audio", f"{path.name} is not a readable WAV file")
                if dur > self.settings.gnani_stt_max_seconds:
                    parts = split_wav(src, self.settings.gnani_stt_max_seconds - 0.5, tmpdir)
            texts, ids = [], []
            for part in parts:
                try:
                    resp = stt.transcribe(str(part), language_code=lang, format=fmt)
                except VoiceError:
                    raise
                except Exception as exc:  # SDK APIError, InvalidAudioError, requests errors
                    raise VoiceError("stt_failed", f"Gnani STT failed: {type(exc).__name__}: {str(exc)[:200]}") from exc
                if isinstance(resp, dict) and resp.get("success") is False:
                    raise VoiceError("stt_failed", f"Gnani STT returned an error: {str(resp)[:200]}")
                texts.append(str((resp or {}).get("transcript") or "").strip())
                if resp.get("request_id"):
                    ids.append(resp["request_id"])
        return STTResult(" ".join(t for t in texts if t).strip(), ids, len(parts), lang, int((time.time() - t0) * 1000), self.mock)

    def transcribe_bytes(self, data: bytes, filename: str, language_code: str | None = None) -> STTResult:
        suffix = Path(filename).suffix.lower() or ".wav"
        with tempfile.TemporaryDirectory(prefix="zm_up_") as tmp:
            p = Path(tmp) / f"upload{suffix}"
            p.write_bytes(data)
            return self.transcribe_file(p, language_code)

    # ------------------------------------------------------------------ TTS
    def synthesize(self, text: str, reply_language: str = "en", output_file: Path | None = None) -> TTSResult:
        _, tts = self._clients()
        text = text_for_speech(text)
        if not text:
            raise VoiceError("tts_failed", "nothing to synthesise")
        voice, lang = self.settings.voice_profiles.get(reply_language, self.settings.voice_profiles["en"])
        attempts = [(voice, lang, False)]
        if (voice, lang) != tuple(self.settings.voice_profiles["en"]):
            attempts.append((*self.settings.voice_profiles["en"], True))  # same provider, English voice
        last_exc: Exception | None = None
        t0 = time.time()
        for v, l, is_fallback in attempts:
            try:
                kwargs = {"voice": v, "model": self.settings.gnani_tts_model, "language": l}
                try:
                    from gnani.tts import AudioConfig
                    kwargs["audio_config"] = AudioConfig(sample_rate=24000)
                except ImportError:
                    pass
                audio = tts.synthesize(text, **kwargs)
                if not audio or len(audio) < 100:
                    raise VoiceError("tts_failed", "Gnani TTS returned no audio")
                audio = fix_wav_header(bytes(audio))
                if output_file is not None:
                    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
                    Path(output_file).write_bytes(audio)
                return TTSResult(audio, v, l, self.settings.gnani_tts_model, int((time.time() - t0) * 1000), self.mock, is_fallback)
            except Exception as exc:
                last_exc = exc
        raise VoiceError("tts_failed", f"Gnani TTS failed: {type(last_exc).__name__}: {str(last_exc)[:200]}")


def wav_bytes_duration(data: bytes) -> float | None:
    try:
        with wave.open(io.BytesIO(fix_wav_header(data)), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except (wave.Error, EOFError):
        return None
