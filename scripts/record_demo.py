#!/usr/bin/env python3
"""Record demo.mp4: a real spoken conversation through the real web UI.

What is real: the web app, Gnani STT on every customer clip, the agent (LLM + RAG + tools), Gnani
TTS replies, the evidence panel. What is synthetic and labelled on the title card: the
CUSTOMER's voice, generated with a different Gnani TTS voice because no human recording was
available. The browser's microphone is fed from those clips via getUserMedia, so the app's own
recorder, upload and playback paths are exercised.

Needs: the project venv (server), Playwright + Chromium (recorder) and ffmpeg (muxing).
  python scripts/record_demo.py --server-python .venv/bin/python --out demo.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TURNS = [  # new phrasings (not from the practice set)
    "Hi, my refund to UPI was started on the sixteenth of October 2026. By when should I get it?",
    "And how long would it take if I had paid by credit card?",
    "Do you offer gift wrapping for my orders?",
    "Yes please, connect me to a human.",
]
CUSTOMER_VOICE = "Shlok"

INIT_JS = """
window.__demo = {clip: null, ended: false, events: []};
navigator.mediaDevices.getUserMedia = async (c) => {
  const ctx = new AudioContext();
  const resp = await fetch(window.__demo.clip);
  const buf = await ctx.decodeAudioData(await resp.arrayBuffer());
  const src = ctx.createBufferSource(); src.buffer = buf;
  const dest = ctx.createMediaStreamDestination(); src.connect(dest);
  window.__demo.ended = false;
  src.onended = () => { window.__demo.ended = true; };
  src.start(); window.__demo.events.push({type: 'customer', t: Date.now(), dur: buf.duration});
  return dest.stream;
};
document.addEventListener('play', (e) => {
  if (e.target.tagName === 'AUDIO') {
    window.__demo.events.push({type: 'reply', t: Date.now(), src: e.target.src});
    e.target.addEventListener('ended', () => window.__demo.events.push({type: 'reply_end', t: Date.now()}), {once: true});
  }
}, true);
"""

TITLE = """<html><body style="margin:0;background:#0f6e5f;color:#fff;font-family:system-ui,Segoe UI,sans-serif;display:flex;
align-items:center;justify-content:center;height:100vh"><div style="max-width:900px;padding:40px">
<div style="font-size:16px;opacity:.85;letter-spacing:.08em">VOICEOPS 2026 · CAPSTONE DEMO</div>
<h1 style="font-size:44px;margin:10px 0 18px">ZentroMart AI voice support agent</h1>
<p style="font-size:20px;line-height:1.5">Live run of the project's web app. Speech-to-text and text-to-speech: <b>Gnani only</b>
(STT REST, TTS timbre-v2.5, voice Kaveri). LLM: <b>{model}</b>. Answers come from ZentroMart's documents (RAG).</p>
<p style="font-size:17px;line-height:1.5;background:rgba(0,0,0,.18);padding:12px 16px;border-radius:10px">
Note: the <b>customer's voice is synthetic</b> — generated with another Gnani TTS voice ({voice}) because no human
recording was available. Everything after the microphone is the real pipeline.</p></div></body></html>"""


def wait_http(url: str, timeout: float = 120) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"server did not start: {url}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-python", default=sys.executable)
    ap.add_argument("--out", default=str(ROOT / "demo.mp4"))
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    from zentromart.config import get_settings
    from gnani.tts import AudioConfig, GnaniTTSClient  # noqa: needs the venv on PYTHONPATH or same interpreter

    s = get_settings()
    work = ROOT / "runtime" / "demo"
    work.mkdir(parents=True, exist_ok=True)
    tts = GnaniTTSClient(api_key=s.gnani_api_key, timeout=60)
    clips = []
    for i, text in enumerate(TURNS, 1):
        p = work / f"customer_{i}.wav"
        if not p.exists():
            from zentromart.voice.gnani import fix_wav_header
            p.write_bytes(fix_wav_header(tts.synthesize(text, voice=CUSTOMER_VOICE, model=s.gnani_tts_model,
                                                        language="en-IN", audio_config=AudioConfig(sample_rate=24000))))
        clips.append(p)

    env = dict(os.environ, GEMINI_RPM=os.environ.get("DEMO_GEMINI_RPM", "20"))
    server = subprocess.Popen([args.server_python, str(ROOT / "run.py"), "--web", "--port", str(args.port)],
                              cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{args.port}"
    try:
        wait_http(base + "/api/health")
        health = json.loads(urllib.request.urlopen(base + "/api/health").read())
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            browser = pw.chromium.launch(args=["--autoplay-policy=no-user-gesture-required", "--use-fake-ui-for-media-stream"])
            ctx = browser.new_context(viewport={"width": 1280, "height": 760}, record_video_dir=str(work / "video"),
                                      record_video_size={"width": 1280, "height": 760}, permissions=["microphone"])
            ctx.add_init_script(INIT_JS)
            page = ctx.new_page()
            t_video0 = time.time() * 1000
            page.route("**/__demo/*.wav", lambda route: route.fulfill(path=str(work / route.request.url.rsplit("/", 1)[-1]),
                                                                    content_type="audio/wav"))
            page.set_content(TITLE.format(model=f"{health['llm']['provider']} {health['llm']['model']}", voice=CUSTOMER_VOICE))
            page.wait_for_timeout(4500)
            page.goto(base + "/")
            page.wait_for_selector("#chip-llm.ok", timeout=30000)
            page.wait_for_timeout(1500)
            for i, clip in enumerate(clips, 1):
                n_bot = page.locator(".msg.bot").count()
                page.evaluate(f"window.__demo.clip = '{base}/__demo/{clip.name}'; window.__demo.ended = false;")
                page.click("#mic")
                n_cust = i  # wait until THIS clip has started and finished playing into the mic
                page.wait_for_function(f"window.__demo.events.filter(e => e.type==='customer').length >= {n_cust} && window.__demo.ended === true", timeout=60000)
                page.wait_for_timeout(400)
                page.click("#mic")
                page.wait_for_function(f"document.querySelectorAll('.msg.bot').length > {n_bot}", timeout=120000)
                # wait for the reply audio to finish (or 25 s max)
                try:
                    page.wait_for_function("window.__demo.events.filter(e => e.type==='reply_end').length >= %d" % i, timeout=25000)
                except Exception:
                    pass
                page.wait_for_timeout(1200)
            page.locator(".msg.bot").last.click()
            page.wait_for_timeout(3000)
            events = page.evaluate("window.__demo.events")
            video_path = page.video.path()
            replies = page.evaluate("[...document.querySelectorAll('.msg.bot audio')].map(a => a.src)")
            ctx.close()
            browser.close()
        # download the reply audio actually played in the UI
        reply_files = []
        for j, url in enumerate(replies, 1):
            dest = work / f"reply_{j}.wav"
            dest.write_bytes(urllib.request.urlopen(url).read())
            reply_files.append(dest)
        (work / "events.json").write_text(json.dumps({"t_video0": t_video0, "events": events}, indent=1), encoding="utf-8")
    finally:
        server.terminate()
        try:
            server.wait(10)
        except Exception:
            server.kill()

    # ---- mux: video + customer clips + Gnani replies at the times they played
    cust_ev = [e for e in events if e["type"] == "customer"]
    reply_ev = [e for e in events if e["type"] == "reply"]
    inputs, filters, labels = ["-i", str(video_path)], [], []
    k = 1
    for clip, ev in zip(clips, cust_ev):
        inputs += ["-i", str(clip)]
        d = max(0, int(ev["t"] - t_video0))
        filters.append(f"[{k}:a]aresample=24000,adelay={d}|{d}[a{k}]"); labels.append(f"[a{k}]"); k += 1
    for rf, ev in zip(reply_files, reply_ev):
        inputs += ["-i", str(rf)]
        d = max(0, int(ev["t"] - t_video0))
        filters.append(f"[{k}:a]aresample=24000,adelay={d}|{d}[a{k}]"); labels.append(f"[a{k}]"); k += 1
    filters.append(f"{''.join(labels)}amix=inputs={len(labels)}:normalize=0[aout]")
    out = Path(args.out)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(filters), "-map", "0:v", "-map", "[aout]",
           "-c:v", "libx264", "-preset", "medium", "-crf", "26", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
           "-t", "119", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
                         capture_output=True, text=True).stdout.strip()
    print(f"wrote {out} ({float(dur):.1f} s); customer clips {len(cust_ev)}, Gnani replies {len(reply_ev)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
