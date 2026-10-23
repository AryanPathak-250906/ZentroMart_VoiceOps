"""One spoken turn: Gnani STT -> SupportAgent -> Gnani TTS (the brief's voice_turn, hardened)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .gnani import GnaniVoice, VoiceError


@dataclass
class VoiceTurn:
    status: str                      # ok | empty_transcript | stt_failed | tts_failed | agent_error | invalid_audio | not_configured
    transcript: str = ""
    reply: str = ""
    audio: bytes | None = None
    turn: object | None = None       # agent TurnResult
    errors: list[str] = field(default_factory=list)
    stt: dict = field(default_factory=dict)
    tts: dict = field(default_factory=dict)


def voice_turn(agent, voice: GnaniVoice, session, audio_path: Path, language_code: str | None = None,
               reply_audio_path: Path | None = None) -> VoiceTurn:
    try:
        stt = voice.transcribe_file(Path(audio_path), language_code)
    except VoiceError as exc:
        return VoiceTurn(status=exc.kind if exc.kind in ("invalid_audio", "not_configured", "sdk_missing") else "stt_failed",
                         errors=[str(exc)])
    vt = VoiceTurn(status="ok", transcript=stt.transcript,
                   stt={"request_ids": stt.request_ids, "segments": stt.segments, "language_code": stt.language_code,
                        "latency_ms": stt.latency_ms, "mock": stt.mock})
    # Empty / unclear audio: ask the caller to repeat (the agent returns a localised prompt, no LLM call)
    turn = agent.respond(session, stt.transcript, channel="voice")
    vt.turn = turn
    vt.reply = turn.reply
    if not stt.transcript.strip():
        vt.status = "empty_transcript"
    elif turn.decision == "error":
        vt.status = "agent_error"
        vt.errors += turn.errors
    try:
        tts = voice.synthesize(turn.reply, turn.language, output_file=reply_audio_path)
        vt.audio = tts.audio
        vt.tts = {"voice": tts.voice, "language": tts.language, "model": tts.model, "latency_ms": tts.latency_ms,
                  "mock": tts.mock, "fallback_voice_used": tts.fallback_used}
    except VoiceError as exc:
        vt.errors.append(str(exc))
        if vt.status == "ok":
            vt.status = "tts_failed"
    return vt
