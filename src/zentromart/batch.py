"""Batch runner: a folder of customer audio -> Gnani STT -> agent -> Gnani TTS -> saved results.

Output layout (per input file, processed independently with a fresh conversation):
  OUTPUT/summary.csv, OUTPUT/summary.json
  OUTPUT/results/<stem>/transcript.txt, reply.txt, reply.wav, sources.json, result.json
Nothing is fabricated: if a step fails, the failure is recorded and no placeholder is written.
Exit codes: 0 all files ok · 1 at least one file failed · 2 fatal configuration/input error.
"""
from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path

from .conversation import Session
from .voice.gnani import CONVERTIBLE_EXTENSIONS, SUPPORTED_EXTENSIONS, VoiceError
from .voice.pipeline import voice_turn

AUDIO_EXTENSIONS = SUPPORTED_EXTENSIONS | CONVERTIBLE_EXTENSIONS


def discover_audio(folder: Path) -> list[Path]:
    files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS]
    return sorted(files, key=lambda p: str(p.relative_to(folder)).lower())


def _safe_name(rel: Path) -> str:
    stem = str(rel.with_suffix("")).replace("\\", "/").replace("/", "__")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem) or "audio"


def run_batch(agent, voice, audio_dir: Path, output_dir: Path, language_code: str | None = None,
              conversation: bool = False, log=print) -> int:
    audio_dir, output_dir = Path(audio_dir), Path(output_dir)
    if not audio_dir.is_dir():
        log(f"FATAL: audio folder not found: {audio_dir}")
        return 2
    status = voice.status()
    if not status["configured"] or not status["sdk_installed"]:
        log(f"FATAL: Gnani voice not available: {status['message']}")
        return 2
    if agent.llm is None:
        log("FATAL: LLM not configured (set LLM_PROVIDER and its API key in .env)")
        return 2
    files = discover_audio(audio_dir)
    if not files:
        log(f"FATAL: no supported audio files in {audio_dir} (extensions: {', '.join(sorted(AUDIO_EXTENSIONS))})")
        return 2
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    shared = Session(session_id="batch-conversation", channel="voice") if conversation else None
    used_names: set[str] = set()
    for n, path in enumerate(files, 1):
        rel = path.relative_to(audio_dir)
        name = _safe_name(rel)
        while name in used_names:
            name += "_dup"
        used_names.add(name)
        out = results_dir / name
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        session = shared or Session(session_id=f"batch-{name}", channel="voice")
        row = {"file": str(rel).replace("\\", "/"), "result_dir": f"results/{name}", "status": "", "transcript": "",
               "reply": "", "decision": "", "language": "", "used_sections": "", "retrieved_sections": "",
               "tool_calls": "", "ticket_ids": "", "audio_reply": "", "errors": "", "seconds": 0.0}
        try:
            vt = voice_turn(agent, voice, session, path, language_code, reply_audio_path=out / "reply.wav")
        except Exception as exc:  # never let one file stop the batch
            vt = None
            row.update(status="crashed", errors=f"{type(exc).__name__}: {exc}")
        if vt is not None:
            turn = vt.turn
            row.update(status=vt.status, transcript=vt.transcript, reply=vt.reply, errors=" | ".join(vt.errors))
            if vt.status not in ("invalid_audio", "not_configured", "sdk_missing", "stt_failed"):
                (out / "transcript.txt").write_text(vt.transcript + "\n", encoding="utf-8")
            if turn is not None:
                (out / "reply.txt").write_text(turn.reply + "\n", encoding="utf-8")
                pub = turn.public()
                row.update(decision=turn.decision, language=turn.language,
                           used_sections=" ; ".join(turn.used_sections),
                           retrieved_sections=" ; ".join(turn.retrieved_sections),
                           tool_calls=" ; ".join(f"{c['name']}:{c['outcome']}" for c in pub["tool_calls"]),
                           ticket_ids=" ; ".join(t for t in turn.escalation.get("ticket_ids", []) if t))
                (out / "sources.json").write_text(json.dumps({
                    "used_sections": turn.used_sections, "retrieved_sections": turn.retrieved_sections,
                    "sources": pub["sources"]}, ensure_ascii=False, indent=2), encoding="utf-8")
            if vt.audio:
                row["audio_reply"] = f"results/{name}/reply.wav"
            elif (out / "reply.wav").exists():
                (out / "reply.wav").unlink()  # never leave a partial/fake file
            result = {"file": row["file"], "status": vt.status, "transcript": vt.transcript, "stt": vt.stt, "tts": vt.tts,
                      "errors": vt.errors, "turn": turn.public() if turn is not None else None}
            (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        row["seconds"] = round(time.time() - t0, 2)
        rows.append(row)
        log(f"[{n}/{len(files)}] {row['file']}: {row['status']} ({row['seconds']}s)")
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ok = sum(1 for r in rows if r["status"] in ("ok", "empty_transcript"))
    summary = {"audio_dir": str(audio_dir), "files": len(rows), "succeeded": ok, "failed": len(rows) - ok,
               "stt_provider": "gnani", "tts_provider": "gnani", "tts_model": agent.settings.gnani_tts_model,
               "llm": f"{getattr(agent.llm, 'provider', '')}:{getattr(agent.llm, 'model', '')}",
               "llm_mock": bool(getattr(agent.llm, "is_mock", False)), "voice_mock": bool(voice.mock),
               "retrieval_backend": agent.retriever.index.backend_description,
               "conversation_mode": conversation, "results": rows}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Done: {ok}/{len(rows)} files succeeded. Summary: {output_dir / 'summary.csv'}")
    return 0 if ok == len(rows) else 1
