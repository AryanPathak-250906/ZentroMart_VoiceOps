#!/usr/bin/env python3
"""ZentroMart VoiceOps 2026 — single entry point.

Examples
  python run.py --check                                   # verify dataset, index, LLM and Gnani setup
  python run.py --web                                     # browser UI at http://127.0.0.1:8000
  python run.py --audio-dir INPUT_AUDIO --output-dir OUT  # batch: audio -> Gnani STT -> agent -> Gnani TTS
  python run.py --eval all                                # retrieval + Step 1 + Step 2 + multi-turn
  python run.py --ask "How long do UPI refunds take?"     # one question, printed with sources
  python run.py --chat                                    # interactive terminal conversation
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def _agent(args, need_llm: bool = True):
    from zentromart.agent import SupportAgent
    from zentromart.config import get_settings

    settings = get_settings(dataset_dir=args.dataset_dir)
    if settings.dataset_dir is None:
        print("ERROR: dataset not found. Extract Capstone_Project_Dataset_FTTT_2026.zip into ./dataset/ "
              "or pass --dataset-dir / set ZENTROMART_DATASET_DIR.", file=sys.stderr)
        sys.exit(2)
    agent = SupportAgent.from_settings(settings)
    if need_llm and agent.llm is None:
        key = "GEMINI_API_KEY" if settings.llm_provider == "gemini" else "ANTHROPIC_API_KEY"
        print(f"ERROR: LLM not configured: LLM_PROVIDER={settings.llm_provider} needs {key} in .env", file=sys.stderr)
        sys.exit(2)
    return agent


def cmd_check(args) -> int:
    from zentromart.config import get_settings
    from zentromart.voice import GnaniVoice

    s = get_settings(dataset_dir=args.dataset_dir)
    ok = True
    print(f"Dataset           : {s.dataset_dir or 'NOT FOUND'}")
    if s.dataset_dir is None:
        return 2
    agent = _agent(args, need_llm=False)
    idx = agent.retriever.index
    print(f"KB index          : {idx.manifest['chunk_count']} chunks from {len(idx.manifest['kb_files'])} files; "
          f"{idx.backend_description}; {'loaded from disk' if idx.manifest.get('loaded_from_disk') else 'built now'}")
    for w in idx.manifest.get("warnings", []):
        print(f"  warning         : {w}")
    print(f"Reference date    : {s.reference_date}")
    print(f"LLM provider      : {s.llm_provider} / model {s.llm_model}")
    if agent.llm is None:
        print("LLM status        : NOT CONFIGURED (add the API key to .env)")
        ok = False
    elif args.offline:
        print("LLM status        : key present (not contacted: --offline)")
    else:
        from zentromart.llm import LLMError
        try:
            info = agent.llm.check_model()
            print(f"LLM status        : OK — model available ({info.get('id')})")
        except LLMError as exc:
            print(f"LLM status        : FAILED ({exc.kind}): {exc}")
            ok = False
    voice = GnaniVoice(s)
    st = voice.status()
    print(f"Gnani voice       : sdk_installed={st['sdk_installed']} configured={st['configured']} tts_model={st['tts_model']} ({st['message']})")
    if st["configured"] and st["sdk_installed"] and not args.offline:
        try:
            r = voice.synthesize("Hello from ZentroMart.", "en")
            print(f"Gnani TTS live    : OK ({len(r.audio)} bytes, voice {r.voice})")
        except Exception as exc:
            print(f"Gnani TTS live    : FAILED: {exc}")
            ok = False
    elif not (st["configured"] and st["sdk_installed"]):
        ok = False
    return 0 if ok else 1


def cmd_web(args) -> int:
    import uvicorn
    from zentromart.voice import GnaniVoice
    from zentromart.web import create_app

    agent = _agent(args, need_llm=False)
    if agent.llm is None:
        print("WARNING: LLM not configured — the UI will start but cannot answer until a key is set.")
    app = create_app(agent, GnaniVoice(agent.settings))
    print(f"ZentroMart support UI: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_batch(args) -> int:
    from zentromart.batch import run_batch
    from zentromart.voice import GnaniVoice

    agent = _agent(args, need_llm=False)
    return run_batch(agent, GnaniVoice(agent.settings), Path(args.audio_dir), Path(args.output_dir),
                     language_code=args.stt_language, conversation=args.conversation)


def cmd_eval(args) -> int:
    from zentromart.data import Dataset
    from zentromart.evaluation import eval_retrieval, run_multiturn, run_single, write_report

    which = args.eval
    agent = _agent(args, need_llm=which not in ("retrieval", "summary"))
    ds = Dataset(agent.settings.dataset_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    resume = not args.fresh

    def save() -> None:  # persist after every phase so an interrupted run keeps its summaries
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    if which == "summary":  # recompute summaries from the saved per-case JSONL files
        from zentromart.evaluation.runner import summarize_multiturn, summarize_single
        for name in ("step1", "step2", "step1_asr", "step2_asr"):
            p = out / f"{name}_results.jsonl"
            if p.is_file():
                metrics[name] = summarize_single(p, ds)
        if (out / "multiturn_results.jsonl").is_file():
            metrics["multiturn"] = summarize_multiturn(out / "multiturn_results.jsonl", ds)
    if which in ("retrieval", "all"):
        for asr in (False, True):
            res = eval_retrieval(agent, ds, use_asr=asr)
            name = "retrieval_asr" if asr else "retrieval"
            (out / f"{name}_rows.json").write_text(json.dumps(res.pop("rows"), ensure_ascii=False, indent=1), encoding="utf-8")
            metrics[name] = res
            print(f"{name}: all gold sections retrieved {res['all_kb_gold_retrieved']}; section recall {res['kb_gold_section_recall']}")
        save()
    if which in ("step1", "all"):
        metrics["step1" + ("_asr" if args.asr else "")] = run_single("step1", agent, ds, out, use_asr=args.asr, resume=resume, limit=args.limit)
        save()
    if which in ("step2", "all"):
        metrics["step2" + ("_asr" if args.asr else "")] = run_single("step2", agent, ds, out, use_asr=args.asr, resume=resume, limit=args.limit)
        save()
    if which in ("multiturn", "all"):
        metrics["multiturn"] = run_multiturn(agent, ds, out, resume=resume, limit=args.limit)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    rep = write_report(out, metrics)
    print(f"Metrics: {metrics_path}\nReport : {rep}")
    return 0


def cmd_ask(args) -> int:
    from zentromart.conversation import Session

    agent = _agent(args)
    turn = agent.respond(Session("cli"), args.ask)
    print(turn.reply)
    print(f"\n[decision={turn.decision} used={turn.used_sections} retrieved={turn.retrieved_sections} "
          f"tools={[c.public()['name'] for c in turn.tool_calls]} errors={turn.errors}]")
    return 0 if turn.decision != "error" else 1


def cmd_chat(args) -> int:
    from zentromart.conversation import Session

    agent = _agent(args)
    session = Session("terminal")
    print("ZentroMart AI support (type 'quit' to exit, 'new' for a fresh conversation)")
    while True:
        try:
            msg = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if msg.lower() in ("quit", "exit"):
            break
        if msg.lower() == "new":
            session = Session("terminal")
            print("(new conversation)")
            continue
        turn = agent.respond(session, msg)
        print(f"Assistant: {turn.reply}\n  [{turn.decision} · sources {', '.join(turn.used_sections) or '-'}"
              f"{' · ticket ' + turn.escalation['ticket_ids'][0] if turn.escalation.get('ticket_ids') else ''}]")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="ZentroMart VoiceOps 2026 support agent (Gnani voice + RAG).",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--web", action="store_true", help="start the browser interface")
    mode.add_argument("--audio-dir", metavar="DIR", help="batch-process a folder of customer audio files")
    mode.add_argument("--eval", choices=["retrieval", "step1", "step2", "multiturn", "all", "summary"],
                      help="run an evaluation ('summary' recomputes metrics from saved results)")
    mode.add_argument("--check", action="store_true", help="check dataset, index, LLM and Gnani configuration")
    mode.add_argument("--ask", metavar="TEXT", help="answer one question (new conversation)")
    mode.add_argument("--chat", action="store_true", help="interactive terminal conversation")
    p.add_argument("--output-dir", metavar="DIR", help="output folder for --audio-dir (required with it)")
    p.add_argument("--stt-language", default=None, help="Gnani STT language code for batch audio (default from .env, en-IN)")
    p.add_argument("--conversation", action="store_true", help="batch: treat the files (sorted) as ONE conversation")
    p.add_argument("--dataset-dir", default=None, help="path to Capstone_Project_Dataset_FTTT_2026 (default: auto-detect)")
    p.add_argument("--out", default=str(ROOT / "reports" / "eval"), help="evaluation output folder")
    p.add_argument("--asr", action="store_true", help="eval step1/step2 on the asr_text column instead of the typed question")
    p.add_argument("--limit", type=int, default=None, help="evaluate only the first N items (smoke tests)")
    p.add_argument("--fresh", action="store_true", help="discard previous results instead of resuming")
    p.add_argument("--offline", action="store_true", help="--check without contacting external APIs")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)
    if args.audio_dir and not args.output_dir:
        p.error("--audio-dir requires --output-dir")
    if args.check:
        return cmd_check(args)
    if args.web:
        return cmd_web(args)
    if args.audio_dir:
        return cmd_batch(args)
    if args.eval:
        return cmd_eval(args)
    if args.ask:
        return cmd_ask(args)
    return cmd_chat(args)


if __name__ == "__main__":
    sys.exit(main())
