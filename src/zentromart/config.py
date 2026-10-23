"""Central configuration.

Every setting comes from environment variables (optionally loaded from a local, git-ignored
``.env`` file). Nothing secret is ever hard-coded here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_FOLDER_NAME = "Capstone_Project_Dataset_FTTT_2026"


def load_dotenv_file(path: Path | None = None) -> None:
    """Load KEY=VALUE pairs from .env without overriding variables that are already set.

    Uses python-dotenv when installed, otherwise a tiny compatible parser (so the project
    still starts on a bare Python install).
    """
    path = path or PROJECT_ROOT / ".env"
    if not path.is_file():
        return
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(path, override=False)
        return
    except ImportError:
        pass
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if value[:1] in ('"', "'") and value[-1:] == value[:1]:
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()  # inline comment
        if key and key not in os.environ:
            os.environ[key] = value


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else value.strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def discover_dataset_dir(explicit: str | None = None) -> Path | None:
    """Find the extracted capstone dataset.

    Order: explicit argument → ZENTROMART_DATASET_DIR → common locations relative to the
    project. A directory qualifies only if it contains kb/, data/ and tools/tools.py.
    """
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env_dir = _env("ZENTROMART_DATASET_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates += [
        PROJECT_ROOT / "dataset" / DATASET_FOLDER_NAME,
        PROJECT_ROOT / DATASET_FOLDER_NAME,
        PROJECT_ROOT.parent / DATASET_FOLDER_NAME,
        PROJECT_ROOT / "dataset",
    ]
    for cand in candidates:
        cand = cand.expanduser().resolve()
        if (cand / "kb").is_dir() and (cand / "data").is_dir() and (cand / "tools" / "tools.py").is_file():
            return cand
        nested = cand / DATASET_FOLDER_NAME
        if (nested / "kb").is_dir() and (nested / "tools" / "tools.py").is_file():
            return nested.resolve()
    return None


@dataclass
class Settings:
    dataset_dir: Path | None
    runtime_dir: Path
    reference_date: date = date(2026, 10, 20)

    # LLM runtime: provider is an explicit setting (never switched silently)
    llm_provider: str = "anthropic"           # anthropic | gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_planner_model: str = ""            # optional lighter model for the planner step
    gemini_rpm: float = 12.0                  # client-side throttle for free-tier quotas
    gemini_thinking_budget: int = -1          # -1 = model default (older 2.x models)
    gemini_thinking_level: str = "low"        # Gemini 3.x: low | medium | high ("" = model default)
    llm_timeout_s: float = 90.0

    # Anthropic runtime
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5-5"
    anthropic_timeout_s: float = 60.0
    anthropic_max_retries: int = 2
    planner_effort: str = "low"               # sent as output_config.effort ("" = omit)
    responder_effort: str = "medium"
    planner_max_tokens: int = 3000
    responder_max_tokens: int = 4000

    # Retrieval
    embedding_backend: str = "auto"           # auto | fastembed | tfidf
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    top_k: int = 8

    # Gnani voice
    gnani_api_key: str = ""
    gnani_stt_language: str = "en-IN"
    gnani_stt_format: str = "transcribe"      # ITN: digits for order IDs, pincodes
    gnani_stt_max_seconds: float = 30.0       # docs list 60 s and 30 s; use the safer value
    gnani_tts_model: str = "timbre-v2.5"
    voice_profiles: dict = field(default_factory=dict)

    @property
    def index_dir(self) -> Path:
        return self.runtime_dir / "index"

    @property
    def tickets_file(self) -> Path:
        return self.runtime_dir / "tickets" / "tickets_created.csv"

    @property
    def audio_dir(self) -> Path:
        return self.runtime_dir / "audio"

    @property
    def llm_configured(self) -> bool:
        if self.llm_provider == "gemini":
            return bool(self.gemini_api_key)
        return bool(self.anthropic_api_key)

    @property
    def llm_model(self) -> str:
        return self.gemini_model if self.llm_provider == "gemini" else self.anthropic_model

    @property
    def gnani_configured(self) -> bool:
        return bool(self.gnani_api_key)


def get_settings(dataset_dir: str | None = None, load_env: bool = True) -> Settings:
    if load_env:
        load_dotenv_file()
    runtime = Path(_env("ZENTROMART_RUNTIME_DIR", str(PROJECT_ROOT / "runtime"))).expanduser()
    ref = _env("ZENTROMART_REFERENCE_DATE", "2026-10-20")
    try:
        reference_date = date.fromisoformat(ref)
    except ValueError:
        reference_date = date(2026, 10, 20)
    profiles = {
        # reply language -> (voice, Gnani TTS language code)
        "en": (_env("GNANI_TTS_VOICE_EN", "Kaveri"), _env("GNANI_TTS_LANG_EN", "en-IN")),
        "hi": (_env("GNANI_TTS_VOICE_HI", "Nalini"), _env("GNANI_TTS_LANG_HI", "hi-IN")),
        # gnani-vachana 0.8.1 rejects language="hi-en" locally; Poorvi (Hinglish) is used with "auto"
        "hi-en": (_env("GNANI_TTS_VOICE_HINGLISH", "Poorvi"), _env("GNANI_TTS_LANG_HINGLISH", "auto")),
        "ta": ("Asmita", "ta-IN"), "te": ("Suhana", "te-IN"), "kn": ("Saanvi", "kn-IN"), "ml": ("Reshma", "ml-IN"),
        "mr": ("Zahira", "mr-IN"), "bn": ("Kirra", "bn-IN"), "gu": ("Falak", "gu-IN"), "pa": ("Mehuli", "pa-IN"),
    }
    return Settings(
        dataset_dir=discover_dataset_dir(dataset_dir),
        runtime_dir=runtime,
        reference_date=reference_date,
        llm_provider=_env("LLM_PROVIDER", "anthropic").lower(),
        gemini_api_key=_env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY"),
        gemini_model=_env("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        gemini_planner_model=_env("GEMINI_PLANNER_MODEL", ""),
        gemini_rpm=_env_float("GEMINI_RPM", 12.0),
        gemini_thinking_budget=_env_int("GEMINI_THINKING_BUDGET", -1),
        gemini_thinking_level=_env("GEMINI_THINKING_LEVEL", "low").lower() if os.environ.get("GEMINI_THINKING_LEVEL", "low").strip() else "",
        llm_timeout_s=_env_float("LLM_TIMEOUT_S", 90.0),
        anthropic_api_key=_env("ANTHROPIC_API_KEY"),
        anthropic_model=_env("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
        anthropic_timeout_s=_env_float("ANTHROPIC_TIMEOUT_S", 60.0),
        anthropic_max_retries=_env_int("ANTHROPIC_MAX_RETRIES", 2),
        planner_effort=_env("ANTHROPIC_PLANNER_EFFORT", "low"),
        responder_effort=_env("ANTHROPIC_RESPONDER_EFFORT", "medium"),
        planner_max_tokens=_env_int("ANTHROPIC_PLANNER_MAX_TOKENS", 3000),
        responder_max_tokens=_env_int("ANTHROPIC_RESPONDER_MAX_TOKENS", 4000),
        embedding_backend=_env("EMBEDDING_BACKEND", "auto").lower(),
        embedding_model=_env("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        top_k=_env_int("RETRIEVAL_TOP_K", 8),
        gnani_api_key=_env("GNANI_API_KEY"),
        gnani_stt_language=_env("GNANI_STT_LANGUAGE", "en-IN"),
        gnani_stt_format=_env("GNANI_STT_FORMAT", "transcribe"),
        gnani_stt_max_seconds=_env_float("GNANI_STT_MAX_SECONDS", 30.0),
        gnani_tts_model=_env("GNANI_TTS_MODEL", "timbre-v2.5"),
        voice_profiles=profiles,
    )
