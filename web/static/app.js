// ZentroMart support UI. The browser only records audio and plays audio returned by the server
// (which uses Gnani STT/TTS). No speech recognition or synthesis happens in the browser.
(() => {
  const $ = (id) => document.getElementById(id);
  const messages = $("messages"), input = $("input"), banner = $("banner");
  let sessionId = null, busy = false, health = null, turns = [];
  const MAX_RECORD_S = 28; // server STT limit is 30 s per clip

  function esc(s) { return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
  function showBanner(text, kind = "error") { banner.textContent = text; banner.className = "banner" + (kind === "info" ? " info" : ""); }
  function hideBanner() { banner.className = "banner hidden"; }
  function setBusy(b) { busy = b; $("send").disabled = b; $("mic").disabled = b && !recorder; }

  async function api(path, opts = {}) {
    const res = await fetch(path, opts);
    let data = null;
    try { data = await res.json(); } catch (e) { data = { error: `HTTP ${res.status}` }; }
    return { ok: res.ok, status: res.status, data };
  }

  async function loadHealth() {
    const r = await api("/api/health");
    health = r.data || {};
    const llm = health.llm || {}, voice = health.voice || {}, rag = health.retrieval || {};
    const chip = (id, text, cls) => { const el = $(id); el.textContent = text; el.className = "chip " + cls; };
    chip("chip-llm", llm.configured ? `LLM: ${llm.provider} · ${llm.model}${llm.mock ? " (MOCK)" : ""}` : "LLM: not configured", llm.configured ? (llm.mock ? "warn" : "ok") : "bad");
    chip("chip-voice", voice.configured && voice.sdk_installed ? `Voice: Gnani ${voice.tts_model}${voice.mock ? " (MOCK)" : ""}` : "Voice: not configured", voice.configured && voice.sdk_installed ? "ok" : "bad");
    chip("chip-rag", rag.semantic ? "Retrieval: hybrid (embeddings + BM25)" : "Retrieval: lexical fallback", rag.semantic ? "ok" : "warn");
    if (!llm.configured) showBanner(llm.message || "The language model is not configured on the server.");
    if (!(voice.configured && voice.sdk_installed)) $("mic").title = "Voice unavailable: " + (voice.message || "not configured");
  }

  async function newSession() {
    const r = await api("/api/session", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ previous_session_id: sessionId }) });
    sessionId = r.data.session_id; turns = [];
    messages.innerHTML = "";
    addSystem("New conversation started. Previous conversation details were discarded.");
    renderPanel(null, null);
  }

  function addSystem(text) { const d = document.createElement("div"); d.className = "msg system"; d.textContent = text; messages.appendChild(d); scroll(); }
  function scroll() { messages.scrollTop = messages.scrollHeight; }

  function addUser(text, viaVoice) {
    const d = document.createElement("div"); d.className = "msg user";
    d.innerHTML = `${esc(text)}<div class="meta">${viaVoice ? "voice · transcribed by Gnani STT" : "typed"}</div>`;
    messages.appendChild(d); scroll();
  }

  function addBot(turn, state, audioUrl) {
    const idx = turns.push({ turn, state }) - 1;
    const d = document.createElement("div"); d.className = "msg bot"; d.dataset.idx = idx;
    const esc_ = turn.escalation || {};
    const ticket = esc_.created && esc_.ticket_ids?.length ? ` · ticket ${esc(esc_.ticket_ids[0])}` : "";
    d.innerHTML = `${esc(turn.reply)}<div class="meta"><span class="badge ${esc(turn.decision)}">${esc(turn.decision)}</span>` +
      `<span>${(turn.used_sections || []).map(esc).join(", ") || "no sources"}</span><span>${ticket}</span></div>` +
      (audioUrl ? `<audio controls src="${esc(audioUrl)}"></audio>` : "");
    d.addEventListener("click", () => select(idx));
    messages.appendChild(d); select(idx); scroll();
    if (audioUrl) { const a = d.querySelector("audio"); a.play().catch(() => {}); }
  }

  function select(idx) {
    document.querySelectorAll(".msg.bot").forEach((el) => el.classList.toggle("selected", Number(el.dataset.idx) === idx));
    const t = turns[idx]; renderPanel(t.turn, t.state);
  }

  function renderPanel(turn, state) {
    $("panel-turn").textContent = turn ? `Turn ${turn.turn} · ${turn.channel} · ${turn.language}` : "No turn yet";
    const out = $("outcome");
    if (turn && (turn.escalation?.required || turn.decision === "refuse" || turn.decision === "error" || turn.diagnostics?.answer_supported_by_documents === false)) {
      const e = turn.escalation || {};
      let txt = "";
      if (e.created) txt = `Handed over to a human colleague — ticket ${e.ticket_ids.join(", ")} (${(e.reasons || []).join(", ")}).`;
      else if (e.required) txt = `Handover attempted but the ticket was NOT created: ${(e.failed || []).join("; ")}`;
      else if (turn.decision === "refuse") txt = "Request declined under policy.";
      else if (turn.decision === "error") txt = "Technical error: " + (turn.errors || []).join("; ");
      else txt = "Not covered by ZentroMart's documents — the assistant said it doesn't know and offered a colleague.";
      out.textContent = txt; out.className = "outcome";
    } else { out.className = "outcome hidden"; }
    const used = new Set(turn?.used_sections || []);
    $("used").innerHTML = turn && used.size ? [...used].map((s) => `<span class="sec used">${esc(s)}</span>`).join("") : "—";
    const srcs = (turn?.sources || []);
    $("retrieved").innerHTML = srcs.length ? srcs.map((s) => `<details><summary><span class="sec ${s.used ? "used" : ""}">${esc(s.section_id)}</span>` +
      `<span class="small">${esc(s.doc_type)}${s.status === "superseded" ? " · SUPERSEDED" : ""} · ${esc(s.via)}${s.score != null ? " · " + Number(s.score).toFixed(4) : ""}</span></summary>` +
      `<div class="small">${esc(s.file)} — ${esc(s.section_title)}<br>${esc(s.authority || "")}</div></details>`).join("") : "—";
    const tools = turn?.tool_calls || [];
    $("tools").innerHTML = tools.length ? tools.map((c) => `<div class="tool ${c.ok ? "ok" : "fail"}"><strong>${esc(c.name)}</strong>(${esc(Object.entries(c.args).filter(([k]) => k !== "summary").map(([k, v]) => `${k}=${v}`).join(", "))})<br><span class="small">${esc(c.outcome)}</span></div>`).join("")
      : (turn?.diagnostics?.calculator_calls?.length ? "" : "—");
    const calc = turn?.diagnostics?.calculator_calls || [];
    if (calc.length) $("tools").innerHTML += calc.map((c) => `<div class="tool ok"><strong>${esc(c.tool)}</strong>(${esc(JSON.stringify(c.input))})<br><span class="small">${esc(JSON.stringify(c.output).slice(0, 160))}</span></div>`).join("");
    const kv = $("state"); kv.innerHTML = "";
    if (state) {
      const rows = { Order: state.order_id || "—", Verification: state.verification, Failures: state.verification_failures,
        "Waiting for": state.awaiting || "—", Tickets: (state.tickets || []).map((t) => t.ticket_id).join(", ") || "—", Language: state.language };
      for (const [k, v] of Object.entries(rows)) kv.innerHTML += `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`;
    }
    $("diag").textContent = turn ? JSON.stringify({ intent: turn.intent, errors: turn.errors, diagnostics: turn.diagnostics }, null, 1) : "";
  }

  function typing(on) {
    let el = document.getElementById("typing");
    if (on && !el) { el = document.createElement("div"); el.id = "typing"; el.className = "typing"; el.textContent = "Assistant is thinking…"; messages.appendChild(el); scroll(); }
    if (!on && el) el.remove();
  }

  async function sendText(text) {
    if (!text.trim() || busy) return;
    hideBanner(); addUser(text, false); setBusy(true); typing(true);
    try {
      const r = await api("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: sessionId, message: text }) });
      typing(false);
      if (r.status === 404) { showBanner("Your session expired. Starting a new conversation."); await newSession(); return; }
      if (r.data.turn) addBot(r.data.turn, r.data.state, null);
      if (!r.ok) showBanner((r.data.turn?.errors || [r.data.error]).join("; ") || `Request failed (${r.status})`);
    } catch (e) { typing(false); showBanner("Network error: could not reach the server."); }
    finally { setBusy(false); input.focus(); }
  }

  // ---------------------------------------------------------------- WAV recorder (PCM16, 16 kHz mono)
  let recorder = null;
  async function startRecording() {
    if (!(health?.voice?.configured && health?.voice?.sdk_installed)) { showBanner("Voice is not configured on the server: " + (health?.voice?.message || "")); return; }
    if (!navigator.mediaDevices?.getUserMedia) { showBanner("This browser cannot record audio (needs HTTPS or localhost)."); return; }
    let stream;
    try { stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } }); }
    catch (e) { showBanner("Microphone permission was denied or no microphone is available."); return; }
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const source = ctx.createMediaStreamSource(stream);
    const proc = ctx.createScriptProcessor(4096, 1, 1);
    const chunks = [];
    proc.onaudioprocess = (e) => chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
    source.connect(proc); proc.connect(ctx.destination);
    const timer = setTimeout(() => stopRecording(), MAX_RECORD_S * 1000);
    recorder = { stream, ctx, source, proc, chunks, timer, started: Date.now() };
    $("mic").classList.add("recording"); $("mic-label").textContent = "Stop";
    showBanner(`Recording… speak now (max ${MAX_RECORD_S} s). Click Stop when done.`, "info");
  }

  function encodeWav(float32, inRate, outRate = 16000) {
    const ratio = inRate / outRate, n = Math.floor(float32.length / ratio);
    const buf = new ArrayBuffer(44 + n * 2), v = new DataView(buf);
    const w = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
    w(0, "RIFF"); v.setUint32(4, 36 + n * 2, true); w(8, "WAVE"); w(12, "fmt "); v.setUint32(16, 16, true);
    v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, outRate, true); v.setUint32(28, outRate * 2, true);
    v.setUint16(32, 2, true); v.setUint16(34, 16, true); w(36, "data"); v.setUint32(40, n * 2, true);
    for (let i = 0; i < n; i++) {
      const start = Math.floor(i * ratio), end = Math.min(Math.floor((i + 1) * ratio), float32.length);
      let sum = 0; for (let j = start; j < end; j++) sum += float32[j];
      const s = Math.max(-1, Math.min(1, sum / Math.max(1, end - start)));
      v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return new Blob([buf], { type: "audio/wav" });
  }

  async function stopRecording() {
    if (!recorder) return;
    const r = recorder; recorder = null; clearTimeout(r.timer);
    r.proc.disconnect(); r.source.disconnect(); r.stream.getTracks().forEach((t) => t.stop());
    const rate = r.ctx.sampleRate; await r.ctx.close();
    $("mic").classList.remove("recording"); $("mic-label").textContent = "Speak"; hideBanner();
    const total = r.chunks.reduce((a, c) => a + c.length, 0);
    if (total < rate * 0.3) { showBanner("The recording was too short. Please try again.", "info"); return; }
    const all = new Float32Array(total); let off = 0; for (const c of r.chunks) { all.set(c, off); off += c.length; }
    await sendVoice(encodeWav(all, rate));
  }

  async function sendVoice(blob) {
    setBusy(true); typing(true);
    const fd = new FormData(); fd.append("session_id", sessionId); fd.append("language", $("lang").value); fd.append("audio", blob, "recording.wav");
    try {
      const r = await api("/api/voice", { method: "POST", body: fd });
      typing(false);
      if (r.status === 404) { showBanner("Your session expired. Starting a new conversation."); await newSession(); return; }
      const d = r.data || {};
      if (d.transcript !== undefined && d.turn) addUser(d.transcript || "(nothing heard)", true);
      if (d.turn) addBot(d.turn, d.state, d.audio_url);
      if (d.status === "empty_transcript") showBanner("I couldn't hear anything clearly — please try again a bit closer to the microphone.", "info");
      else if (d.status === "tts_failed") showBanner("The reply could not be spoken (Gnani TTS error); the text is shown above. " + (d.errors || []).join("; "));
      else if (!r.ok) showBanner((d.errors && d.errors.length ? d.errors.join("; ") : d.error) || `Voice request failed (${r.status})`);
    } catch (e) { typing(false); showBanner("Network error while sending audio."); }
    finally { setBusy(false); }
  }

  // ---------------------------------------------------------------- wiring
  $("composer").addEventListener("submit", (e) => { e.preventDefault(); const t = input.value; input.value = ""; sendText(t); });
  input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("composer").requestSubmit(); } });
  input.addEventListener("input", () => { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 140) + "px"; });
  $("mic").addEventListener("click", () => (recorder ? stopRecording() : (!busy && startRecording())));
  $("new-chat").addEventListener("click", () => { if (!busy) newSession(); });
  loadHealth().then(newSession).catch(() => showBanner("Could not reach the server."));
})();
