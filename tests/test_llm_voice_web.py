"""LLM adapters, Gnani adapter, batch runner and web routes — all with injected MOCK clients."""
import io
import json
import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tests.helpers import DATASET, cleanup, make_agent, require_dataset

from zentromart.batch import run_batch
from zentromart.config import get_settings
from zentromart.conversation import Session, SessionStore
from zentromart.llm import AnthropicLLM, GeminiLLM, LLMError, ToolSpec, parse_json_text
from zentromart.voice.gnani import GnaniVoice, VoiceError, fix_wav_header, split_wav, wav_duration_s
from zentromart.voice.pipeline import voice_turn


def make_wav(path: Path, seconds: float = 1.0, rate: int = 16000, silent: bool = False) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        frames = b"".join(struct.pack("<h", 0 if silent else int(8000 * math.sin(i / 8))) for i in range(int(seconds * rate)))
        w.writeframes(frames)
    return path


def wav_bytes(seconds=0.5) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b"\x00\x00" * int(16000 * seconds))
    return buf.getvalue()


class FakeSTT:
    is_mock = True

    def __init__(self, transcripts=None, fail=False):
        self.transcripts = list(transcripts or ["How long do UPI refunds take?"])
        self.fail = fail
        self.calls = []

    def transcribe(self, path, language_code="en-IN", format="verbatim"):
        self.calls.append((Path(path).name, language_code, format))
        if self.fail:
            raise RuntimeError("APIError 503 service unavailable")
        return {"success": True, "request_id": f"req_mock_{len(self.calls)}", "transcript": self.transcripts[min(len(self.calls) - 1, len(self.transcripts) - 1)]}


class FakeTTS:
    is_mock = True

    def __init__(self, fail_voices=()):
        self.fail_voices = set(fail_voices)
        self.calls = []

    def synthesize(self, text, voice=None, model=None, language=None, audio_config=None):
        self.calls.append((voice, model, language))
        if voice in self.fail_voices or "*" in self.fail_voices:
            raise RuntimeError("APIError 400")
        assert model == "timbre-v2.5", "TTS model must be named explicitly"
        return wav_bytes()


class FakeAnthropic:
    """Mimics anthropic.Anthropic().messages.create / models.retrieve."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.messages = SimpleNamespace(create=self._create)
        self.models = SimpleNamespace(retrieve=lambda m: SimpleNamespace(id=m, display_name="Fake"))

    def _create(self, **kw):
        self.requests.append(json.loads(json.dumps(kw, default=str)))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def text_resp(text):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], usage=SimpleNamespace(input_tokens=5, output_tokens=5))


def tool_resp(name, args, tid="t1"):
    return SimpleNamespace(content=[SimpleNamespace(type="tool_use", name=name, input=args, id=tid)], usage=SimpleNamespace(input_tokens=5, output_tokens=5))


class BadRequestError(Exception):
    pass


class TestLLMAdapters(unittest.TestCase):
    def test_missing_key(self):
        with self.assertRaises(LLMError) as ctx:
            AnthropicLLM("", "claude-sonnet-5-5")
        self.assertEqual(ctx.exception.kind, "missing_credentials")
        with self.assertRaises(LLMError):
            GeminiLLM("", "gemini-x")

    def test_parse_json_text(self):
        self.assertEqual(parse_json_text('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(parse_json_text('Sure: {"a": 2} done'), {"a": 2})
        with self.assertRaises(ValueError):
            parse_json_text("no json here")

    def test_anthropic_json_and_feature_drop(self):
        fake = FakeAnthropic([BadRequestError("output_config.effort: not supported"), text_resp('{"x": "y"}')])
        llm = AnthropicLLM("k", "m", client=fake)
        out = llm.generate_json("planner", "sys", "user", {"type": "object"}, effort="low")
        self.assertEqual(out, {"x": "y"})
        self.assertNotIn("temperature", fake.requests[-1])          # current models reject sampling params
        self.assertNotIn("effort", fake.requests[-1].get("output_config", {}))
        self.assertEqual(llm.records[-1].dropped_features, ["effort"])

    def test_anthropic_tool_loop(self):
        fake = FakeAnthropic([tool_resp("add", {"a": 1, "b": 2}), text_resp('{"reply": "3"}')])
        llm = AnthropicLLM("k", "m", client=fake)
        spec = ToolSpec("add", "add", {"type": "object"}, lambda a: {"sum": a["a"] + a["b"]})
        out, trace = llm.run_with_tools("responder", "s", "u", [spec], {"type": "object"})
        self.assertEqual(out, {"reply": "3"})
        self.assertEqual(trace[0]["output"], {"sum": 3})
        self.assertEqual(fake.requests[0]["tool_choice"], {"type": "auto"})

    def test_anthropic_errors_mapped(self):
        class APITimeoutError(Exception):
            pass
        llm = AnthropicLLM("k", "m", client=FakeAnthropic([APITimeoutError("slow")]))
        with self.assertRaises(LLMError) as ctx:
            llm.generate_json("p", "s", "u", {})
        self.assertEqual(ctx.exception.kind, "timeout")
        self.assertTrue(ctx.exception.retriable)

    def test_gemini_retry_on_rate_limit_and_daily_quota(self):
        class ClientError(Exception):
            def __init__(self, code, msg):
                super().__init__(msg); self.code = code
        ok = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text='{"ok": true}', thought=False)]))],
                             usage_metadata=None, function_calls=None, model_version="gemini-mock")
        seq = [ClientError(429, "RESOURCE_EXHAUSTED quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier' retryDelay': '1s"), ok]
        client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: (lambda x: (_ for _ in ()).throw(x) if isinstance(x, Exception) else x)(seq.pop(0))))
        llm = GeminiLLM("k", "gemini-mock", rpm=0, client=client)
        llm._types = SimpleNamespace(GenerateContentConfig=lambda **kw: SimpleNamespace(**kw), ThinkingConfig=lambda **kw: kw)
        with mock.patch("time.sleep"):
            self.assertEqual(llm.generate_json("p", "s", "u", {}), {"ok": True})
        daily = [ClientError(429, "RESOURCE_EXHAUSTED quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'")]
        client2 = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: (_ for _ in ()).throw(daily[0])))
        llm2 = GeminiLLM("k", "gemini-mock", rpm=0, client=client2)
        llm2._types = llm._types
        with self.assertRaises(LLMError) as ctx:
            llm2.generate_json("p", "s", "u", {})
        self.assertEqual(ctx.exception.kind, "quota_exhausted")


class TestGnaniAdapter(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.settings = get_settings(load_env=False)
        self.settings.gnani_api_key = ""

    def test_not_configured(self):
        v = GnaniVoice(self.settings)
        self.assertFalse(v.status()["configured"])
        with self.assertRaises(VoiceError) as ctx:
            v.transcribe_file(make_wav(self.tmp / "a.wav"))
        self.assertEqual(ctx.exception.kind, "not_configured")

    def test_transcribe_and_itn_format(self):
        stt = FakeSTT(["my order is 89782837"])
        v = GnaniVoice(self.settings, stt, FakeTTS())
        r = v.transcribe_file(make_wav(self.tmp / "a.wav"), "en-IN")
        self.assertEqual(r.transcript, "my order is 89782837")
        self.assertEqual(stt.calls[0][1:], ("en-IN", "transcribe"))
        self.assertTrue(r.mock)

    def test_long_audio_is_split(self):
        self.settings.gnani_stt_max_seconds = 4
        stt = FakeSTT(["part one", "part two", "part three"])
        v = GnaniVoice(self.settings, stt, FakeTTS())
        r = v.transcribe_file(make_wav(self.tmp / "long.wav", seconds=10), "en-IN")
        self.assertEqual(r.segments, 3)
        self.assertEqual(r.transcript, "part one part two part three")
        parts = split_wav(make_wav(self.tmp / "l2.wav", seconds=10), 3.5, self.tmp)
        self.assertTrue(all(wav_duration_s(p) <= 3.5 for p in parts))

    def test_invalid_audio(self):
        v = GnaniVoice(self.settings, FakeSTT(), FakeTTS())
        bad = self.tmp / "x.txt"; bad.write_text("hello" * 100)
        with self.assertRaises(VoiceError) as ctx:
            v.transcribe_file(bad)
        self.assertEqual(ctx.exception.kind, "invalid_audio")
        empty = self.tmp / "e.wav"; empty.write_bytes(b"")
        with self.assertRaises(VoiceError):
            v.transcribe_file(empty)
        with self.assertRaises(VoiceError):
            v.transcribe_file(self.tmp / "missing.wav")

    def test_stt_failure_is_distinct(self):
        v = GnaniVoice(self.settings, FakeSTT(fail=True), FakeTTS())
        with self.assertRaises(VoiceError) as ctx:
            v.transcribe_file(make_wav(self.tmp / "a.wav"))
        self.assertEqual(ctx.exception.kind, "stt_failed")

    def test_tts_voice_selection_and_fallback(self):
        tts = FakeTTS(fail_voices={"Poorvi"})
        v = GnaniVoice(self.settings, FakeSTT(), tts)
        r = v.synthesize("**Haan**, refund 3 din mein.", "hi-en", output_file=self.tmp / "r.wav")
        self.assertEqual(tts.calls[0], ("Poorvi", "timbre-v2.5", "auto"))
        self.assertTrue(r.fallback_used)
        self.assertEqual(tts.calls[1][0], "Kaveri")
        self.assertTrue((self.tmp / "r.wav").is_file())
        with self.assertRaises(VoiceError):
            GnaniVoice(self.settings, FakeSTT(), FakeTTS(fail_voices={"*"})).synthesize("hello", "en")

    def test_fix_wav_header(self):
        data = bytearray(wav_bytes(0.25))
        struct.pack_into("<I", data, 40, 0x7FFFFFFF)  # streaming placeholder size
        fixed = fix_wav_header(bytes(data))
        with wave.open(io.BytesIO(fixed)) as w:
            self.assertAlmostEqual(w.getnframes() / w.getframerate(), 0.25, places=2)


@require_dataset
class TestVoicePipelineAndBatch(unittest.TestCase):
    def setUp(self):
        self.agent, self.tmp = make_agent()
        self.agent.settings.gnani_api_key = ""

    def tearDown(self):
        cleanup(self.tmp)

    def test_empty_transcript_asks_to_repeat(self):
        v = GnaniVoice(self.agent.settings, FakeSTT([""]), FakeTTS())
        vt = voice_turn(self.agent, v, Session("s"), make_wav(self.tmp / "q.wav", silent=True), "en-IN")
        self.assertEqual(vt.status, "empty_transcript")
        self.assertIn("again", vt.reply.lower())
        self.assertTrue(vt.audio)

    def test_batch_records_individual_failures(self):
        audio_in = self.tmp / "in"; audio_in.mkdir()
        make_wav(audio_in / "customer_001.wav")
        make_wav(audio_in / "customer_002.wav")
        (audio_in / "customer_003.wav").write_bytes(b"RIFF....not really a wav file at all" * 10)
        (audio_in / "notes.txt").write_text("ignored")
        out = self.tmp / "out"
        v = GnaniVoice(self.agent.settings, FakeSTT(["How long do UPI refunds take?"]), FakeTTS())
        code = run_batch(self.agent, v, audio_in, out, log=lambda *a: None)
        self.assertEqual(code, 1)
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["files"], 3)
        self.assertEqual([r["status"] for r in summary["results"]], ["ok", "ok", "invalid_audio"])
        self.assertTrue(summary["llm_mock"] and summary["voice_mock"])
        for name in ("transcript.txt", "reply.txt", "reply.wav", "sources.json", "result.json"):
            self.assertTrue((out / "results" / "customer_001" / name).is_file(), name)
        self.assertFalse((out / "results" / "customer_003" / "reply.wav").exists())
        self.assertFalse((out / "results" / "customer_003" / "transcript.txt").exists())
        self.assertTrue((out / "summary.csv").is_file())

    def test_batch_fatal_errors(self):
        v = GnaniVoice(self.agent.settings, FakeSTT(), FakeTTS())
        self.assertEqual(run_batch(self.agent, v, self.tmp / "nope", self.tmp / "o", log=lambda *a: None), 2)
        (self.tmp / "empty").mkdir()
        self.assertEqual(run_batch(self.agent, v, self.tmp / "empty", self.tmp / "o", log=lambda *a: None), 2)
        unconfigured = GnaniVoice(self.agent.settings)
        (self.tmp / "one").mkdir(); make_wav(self.tmp / "one" / "a.wav")
        self.assertEqual(run_batch(self.agent, unconfigured, self.tmp / "one", self.tmp / "o", log=lambda *a: None), 2)


@require_dataset
class TestWebRoutes(unittest.TestCase):
    def setUp(self):
        from starlette.testclient import TestClient
        from zentromart.web import create_app
        self.agent, self.tmp = make_agent()
        self.agent.settings.gnani_api_key = ""
        self.voice = GnaniVoice(self.agent.settings, FakeSTT(["Can I cancel a shipped order?"]), FakeTTS())
        self.client = TestClient(create_app(self.agent, self.voice, SessionStore()))
        self.unconfigured = TestClient(create_app(self.agent, GnaniVoice(self.agent.settings), SessionStore()))

    def tearDown(self):
        cleanup(self.tmp)

    def test_health_and_index(self):
        h = self.client.get("/api/health").json()
        self.assertTrue(h["llm"]["mock"])
        self.assertTrue(h["voice"]["mock"])
        body = json.dumps(h)
        self.assertNotIn("vach_", body)
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)

    def test_chat_flow_and_session_reset(self):
        sid = self.client.post("/api/session").json()["session_id"]
        r = self.client.post("/api/chat", json={"session_id": sid, "message": "How long do refunds take?"})
        self.assertEqual(r.status_code, 200)
        turn = r.json()["turn"]
        for key in ("reply", "decision", "used_sections", "sources", "tool_calls", "verification", "escalation", "diagnostics"):
            self.assertIn(key, turn)
        new = self.client.post("/api/session", json={"previous_session_id": sid}).json()["session_id"]
        self.assertNotEqual(new, sid)
        self.assertEqual(self.client.post("/api/chat", json={"session_id": sid, "message": "hi"}).status_code, 404)
        self.assertEqual(self.client.post("/api/chat", json={"session_id": new, "message": ""}).status_code, 400)
        self.assertTrue(self.client.delete(f"/api/session/{new}").json()["deleted"])

    def test_voice_endpoint(self):
        sid = self.client.post("/api/session").json()["session_id"]
        r = self.client.post("/api/voice", data={"session_id": sid, "language": "en-IN"},
                             files={"audio": ("recording.wav", wav_bytes(1.0), "audio/wav")})
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertEqual(d["transcript"], "Can I cancel a shipped order?")
        self.assertTrue(d["audio_url"])
        self.assertEqual(self.client.get(d["audio_url"]).headers["content-type"], "audio/wav")
        self.assertEqual(self.client.get("/api/audio/../../.env").status_code, 404)
        sid2 = self.unconfigured.post("/api/session").json()["session_id"]
        r2 = self.unconfigured.post("/api/voice", data={"session_id": sid2}, files={"audio": ("r.wav", wav_bytes(), "audio/wav")})
        self.assertEqual(r2.status_code, 503)
        self.assertEqual(r2.json()["kind"], "not_configured")
        r3 = self.client.post("/api/voice", data={"session_id": sid}, files={"audio": ("r.wav", b"", "audio/wav")})
        self.assertEqual(r3.status_code, 400)


if __name__ == "__main__":
    unittest.main()
