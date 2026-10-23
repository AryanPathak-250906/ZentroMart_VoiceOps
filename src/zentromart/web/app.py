"""Starlette web backend (same SupportAgent as the CLI and batch runner).

Routes
  GET  /                      chat UI (web/static/index.html)
  GET  /api/health            configuration status (no secrets)
  POST /api/session           start a fresh conversation -> {session_id}
  DELETE /api/session/{id}    end a conversation (state discarded)
  POST /api/chat              {session_id, message} -> TurnResult (public fields only)
  POST /api/voice             multipart: session_id, audio (WAV from the browser), language
                              -> transcript + TurnResult + audio_url of the Gnani reply
  GET  /api/audio/{name}      reply audio produced by Gnani TTS (kept for a limited time)
API keys stay on the server; the browser only records audio and plays returned audio.
"""
from __future__ import annotations

import re
import secrets
import tempfile
import threading
import time
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from ..config import PROJECT_ROOT
from ..conversation import SessionStore
from ..voice.gnani import VoiceError
from ..voice.pipeline import voice_turn

STATIC_DIR = PROJECT_ROOT / "web" / "static"
MAX_AUDIO_BYTES = 8 * 1024 * 1024
AUDIO_TTL_S = 1800


def create_app(agent, voice, store: SessionStore | None = None) -> Starlette:
    store = store or SessionStore()
    audio_dir = agent.settings.audio_dir
    audio_dir.mkdir(parents=True, exist_ok=True)
    agent_lock = threading.Lock()  # one turn at a time per process keeps tool/ticket logs consistent

    def _cleanup_audio() -> None:
        now = time.time()
        for p in audio_dir.glob("*.wav"):
            if now - p.stat().st_mtime > AUDIO_TTL_S:
                p.unlink(missing_ok=True)

    async def index(request: Request):
        return FileResponse(STATIC_DIR / "index.html")

    async def health(request: Request):
        s = agent.settings
        llm = agent.llm
        return JSONResponse({
            "llm": {"configured": llm is not None, "provider": getattr(llm, "provider", s.llm_provider),
                    "model": getattr(llm, "model", s.llm_model), "mock": bool(getattr(llm, "is_mock", False)),
                    "message": "ready" if llm is not None else f"LLM not configured: set the API key for LLM_PROVIDER={s.llm_provider} in .env"},
            "voice": voice.status(),
            "retrieval": {"backend": agent.retriever.index.backend_description, "chunks": len(agent.retriever.index.chunks),
                          "semantic": agent.retriever.index.semantic},
            "dataset": str(s.dataset_dir.name) if s.dataset_dir else None,
            "reference_date": s.reference_date.isoformat(),
        })

    async def new_session(request: Request):
        try:
            body = await request.json()
        except Exception:
            body = {}
        old = (body or {}).get("previous_session_id")
        if old:
            store.delete(str(old))
        s = store.create(channel="chat")
        return JSONResponse({"session_id": s.session_id})

    async def end_session(request: Request):
        return JSONResponse({"deleted": store.delete(request.path_params["sid"])})

    def _get_session(sid):
        s = store.get(str(sid or ""))
        if s is None:
            raise KeyError("unknown or expired session; start a new conversation")
        return s

    async def chat(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        message = str(body.get("message", ""))[:2000]
        try:
            session = _get_session(body.get("session_id"))
        except KeyError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        if not message.strip():
            return JSONResponse({"error": "empty message"}, status_code=400)
        import anyio
        def run():
            with agent_lock:
                return agent.respond(session, message, channel="chat")
        turn = await anyio.to_thread.run_sync(run)
        return JSONResponse({"turn": turn.public(), "state": session.public_state()},
                            status_code=200 if turn.decision != "error" else 503)

    async def voice_endpoint(request: Request):
        st = voice.status()
        if not st["configured"] or not st["sdk_installed"]:
            return JSONResponse({"error": f"Voice is not available: {st['message']}", "kind": "not_configured"}, status_code=503)
        form = await request.form()
        try:
            session = _get_session(form.get("session_id"))
        except KeyError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        upload = form.get("audio")
        if upload is None or not hasattr(upload, "read"):
            return JSONResponse({"error": "no audio uploaded"}, status_code=400)
        data = await upload.read()
        if not data or len(data) < 200:
            return JSONResponse({"error": "the recording is empty", "kind": "invalid_audio"}, status_code=400)
        if len(data) > MAX_AUDIO_BYTES:
            return JSONResponse({"error": "recording too large (max 8 MB)", "kind": "invalid_audio"}, status_code=413)
        lang = str(form.get("language") or agent.settings.gnani_stt_language)
        if lang not in ("en-IN", "hi-IN", "bn-IN", "gu-IN", "kn-IN", "ml-IN", "mr-IN", "pa-IN", "ta-IN", "te-IN"):
            lang = agent.settings.gnani_stt_language
        suffix = Path(getattr(upload, "filename", "") or "audio.wav").suffix.lower() or ".wav"
        if not re.fullmatch(r"\.[a-z0-9]{2,5}", suffix):
            suffix = ".wav"
        _cleanup_audio()
        name = f"{secrets.token_hex(8)}.wav"
        import anyio

        def run():
            with tempfile.TemporaryDirectory(prefix="zm_web_") as tmp:
                p = Path(tmp) / f"input{suffix}"
                p.write_bytes(data)
                with agent_lock:
                    return voice_turn(agent, voice, session, p, lang, reply_audio_path=audio_dir / name)

        vt = await anyio.to_thread.run_sync(run)
        payload = {"status": vt.status, "transcript": vt.transcript, "errors": vt.errors, "stt": vt.stt, "tts": vt.tts,
                   "turn": vt.turn.public() if vt.turn is not None else None, "state": session.public_state(),
                   "audio_url": f"/api/audio/{name}" if vt.audio else None}
        code = 200 if vt.status in ("ok", "empty_transcript", "tts_failed") else 502 if vt.status == "stt_failed" else 400
        return JSONResponse(payload, status_code=code)

    async def audio(request: Request):
        name = request.path_params["name"]
        if not re.fullmatch(r"[0-9a-f]{16}\.wav", name):
            return Response(status_code=404)
        p = audio_dir / name
        if not p.is_file():
            return Response(status_code=404)
        return FileResponse(p, media_type="audio/wav")

    routes = [
        Route("/", index),
        Route("/api/health", health),
        Route("/api/session", new_session, methods=["POST"]),
        Route("/api/session/{sid}", end_session, methods=["DELETE"]),
        Route("/api/chat", chat, methods=["POST"]),
        Route("/api/voice", voice_endpoint, methods=["POST"]),
        Route("/api/audio/{name}", audio),
        Mount("/static", app=StaticFiles(directory=str(STATIC_DIR)), name="static"),
    ]
    return Starlette(routes=routes)
