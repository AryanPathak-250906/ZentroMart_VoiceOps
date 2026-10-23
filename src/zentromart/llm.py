"""LLM runtime. Provider chosen explicitly with LLM_PROVIDER (anthropic | gemini).

Anthropic: Messages API through the official ``anthropic`` Python SDK.

* Credentials: ``ANTHROPIC_API_KEY``; model: ``ANTHROPIC_MODEL`` (default claude-sonnet-5-5).
* Timeouts + SDK retries (connection errors, 408/409/429/5xx); bounded output tokens.
* Current Claude models reject sampling parameters, forced ``tool_choice`` and prefill, so we
  use ``tool_choice=auto``, JSON structured outputs (``output_config.format``) and an
  explicit JSON-only instruction. If an older model rejects a feature with HTTP 400 we drop
  that feature once and retry, and record it in diagnostics. We never switch provider.
* ``ScriptedLLM`` is a deterministic mock for offline tests; every result produced through
  it is flagged ``mock=True``.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable


class LLMError(RuntimeError):
    def __init__(self, kind: str, message: str, retriable: bool = False):
        super().__init__(message)
        self.kind = kind
        self.retriable = retriable


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict], dict]

    def api_dict(self, strict: bool) -> dict:
        d = {"name": self.name, "description": self.description, "input_schema": self.input_schema}
        if strict:
            d["strict"] = True
        return d


@dataclass
class LLMCallRecord:
    purpose: str
    model: str
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    tool_rounds: int = 0
    dropped_features: list[str] = field(default_factory=list)
    mock: bool = False


def parse_json_text(text: str) -> dict:
    """Parse a JSON object from model text (tolerates ```json fences or stray prose)."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found in model output")
        value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("model output JSON is not an object")
    return value


class AnthropicLLM:
    provider = "anthropic"
    is_mock = False

    def __init__(self, api_key: str, model: str, timeout_s: float = 60.0, max_retries: int = 2, client: Any = None):
        if not api_key and client is None:
            raise LLMError("missing_credentials", "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")
        self.model = model
        if client is None:
            try:
                import anthropic  # type: ignore
            except ImportError as exc:
                raise LLMError("sdk_missing", "The 'anthropic' package is not installed: pip install -r requirements.txt") from exc
            client = anthropic.Anthropic(api_key=api_key, timeout=timeout_s, max_retries=max_retries)
        self._client = client
        self.records: list[LLMCallRecord] = []

    # ------------------------------------------------------------------ low level
    def _create(self, kwargs: dict, dropped: list[str]) -> Any:
        try:
            import anthropic  # type: ignore
        except ImportError:  # injected fake client in tests
            anthropic = None
        for _attempt in range(4):
            try:
                return self._client.messages.create(**kwargs)
            except Exception as exc:  # map SDK errors to LLMError
                name = type(exc).__name__
                msg = str(exc)
                if anthropic is not None and isinstance(exc, anthropic.BadRequestError) or name == "BadRequestError":
                    low = msg.lower()
                    if "effort" in low and "output_config" in kwargs and "effort" in kwargs["output_config"]:
                        kwargs["output_config"].pop("effort"); dropped.append("effort"); self._tidy(kwargs); continue
                    if ("format" in low or "json_schema" in low or "output_config" in low) and "output_config" in kwargs and "format" in kwargs["output_config"]:
                        kwargs["output_config"].pop("format"); dropped.append("json_format"); self._tidy(kwargs); continue
                    if "strict" in low and any(t.get("strict") for t in kwargs.get("tools", [])):
                        for t in kwargs["tools"]:
                            t.pop("strict", None)
                        dropped.append("strict_tools"); continue
                    raise LLMError("bad_request", msg) from exc
                if name in ("AuthenticationError", "PermissionDeniedError"):
                    raise LLMError("auth", f"Anthropic rejected the API key or permissions: {msg}") from exc
                if name == "NotFoundError":
                    raise LLMError("model_not_found", f"Model {self.model!r} not found or not available to this key: {msg}") from exc
                if name in ("APITimeoutError", "DeadlineExceededError"):
                    raise LLMError("timeout", f"Anthropic API timed out: {msg}", retriable=True) from exc
                if name in ("APIConnectionError",):
                    raise LLMError("network", f"Could not reach the Anthropic API: {msg}", retriable=True) from exc
                if name in ("RateLimitError", "OverloadedError", "InternalServerError", "ServiceUnavailableError"):
                    raise LLMError("unavailable", f"Anthropic API temporarily unavailable: {msg}", retriable=True) from exc
                raise LLMError("unknown", f"{name}: {msg}") from exc
        raise LLMError("bad_request", "request rejected after removing optional features")

    @staticmethod
    def _tidy(kwargs: dict) -> None:
        if "output_config" in kwargs and not kwargs["output_config"]:
            kwargs.pop("output_config")

    @staticmethod
    def _usage(resp: Any) -> tuple[int, int]:
        u = getattr(resp, "usage", None)
        return (getattr(u, "input_tokens", 0) or 0, getattr(u, "output_tokens", 0) or 0)

    @staticmethod
    def _text(resp: Any) -> str:
        return "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")

    def check_model(self) -> dict:
        """Verify that the configured model exists and is accessible with this key."""
        try:
            info = self._client.models.retrieve(self.model)
        except Exception as exc:
            name = type(exc).__name__
            kind = {"NotFoundError": "model_not_found", "AuthenticationError": "auth"}.get(name, "unknown")
            raise LLMError(kind, f"{name}: {exc}") from exc
        return {"id": getattr(info, "id", self.model), "display_name": getattr(info, "display_name", "")}

    # ------------------------------------------------------------------ high level
    def generate_json(self, purpose: str, system: str, user: str, schema: dict,
                      max_tokens: int = 2000, effort: str = "") -> dict:
        output_config: dict = {"format": {"type": "json_schema", "schema": schema}}
        if effort:
            output_config["effort"] = effort
        kwargs = {"model": self.model, "max_tokens": max_tokens, "system": system,
                  "messages": [{"role": "user", "content": user}], "output_config": output_config}
        dropped: list[str] = []
        t0 = time.time()
        resp = self._create(kwargs, dropped)
        tin, tout = self._usage(resp)
        self.records.append(LLMCallRecord(purpose, self.model, int((time.time() - t0) * 1000), tin, tout, 0, dropped))
        return parse_json_text(self._text(resp))

    def run_with_tools(self, purpose: str, system: str, user: str, tools: list[ToolSpec], final_schema: dict,
                       max_tokens: int = 3000, effort: str = "", max_rounds: int = 5) -> tuple[dict, list[dict]]:
        """Tool loop (tool_choice=auto). Returns (final JSON object, tool trace)."""
        by_name = {t.name: t for t in tools}
        messages: list[dict] = [{"role": "user", "content": user}]
        output_config: dict = {"format": {"type": "json_schema", "schema": final_schema}}
        if effort:
            output_config["effort"] = effort
        api_tools = [t.api_dict(strict=True) for t in tools]
        trace: list[dict] = []
        dropped: list[str] = []
        t0 = time.time()
        tin = tout = 0
        for round_no in range(max_rounds + 1):
            kwargs = {"model": self.model, "max_tokens": max_tokens, "system": system, "messages": messages,
                      "tools": api_tools, "tool_choice": {"type": "auto"}}
            if output_config:
                kwargs["output_config"] = output_config
            resp = self._create(kwargs, dropped)
            output_config = kwargs.get("output_config", {})  # keep any feature removal
            api_tools = kwargs["tools"]
            a, b = self._usage(resp)
            tin, tout = tin + a, tout + b
            uses = [blk for blk in resp.content if getattr(blk, "type", "") == "tool_use"]
            if not uses:
                self.records.append(LLMCallRecord(purpose, self.model, int((time.time() - t0) * 1000), tin, tout, round_no, dropped))
                return parse_json_text(self._text(resp)), trace
            if round_no == max_rounds:
                break
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for use in uses:
                spec = by_name.get(use.name)
                if spec is None:
                    out, is_error = {"error": f"unknown tool {use.name}"}, True
                else:
                    try:
                        out, is_error = spec.handler(dict(use.input or {})), False
                    except Exception as exc:  # tool errors are reported back to the model
                        out, is_error = {"error": f"{type(exc).__name__}: {exc}"}, True
                trace.append({"tool": use.name, "input": dict(use.input or {}), "output": out, "error": is_error})
                results.append({"type": "tool_result", "tool_use_id": use.id,
                                "content": json.dumps(out, ensure_ascii=False), "is_error": is_error})
            messages.append({"role": "user", "content": results})
        self.records.append(LLMCallRecord(purpose, self.model, int((time.time() - t0) * 1000), tin, tout, max_rounds, dropped))
        raise LLMError("tool_loop", f"model did not finish within {max_rounds} tool rounds")


class ScriptedLLM:
    """Offline mock. ``planner`` and ``responder`` are callables returning dicts.

    ``responder`` receives (user_prompt, tools) and may call tool handlers itself to mimic
    tool use; it returns the final JSON object. All outputs are explicitly mock outputs.
    """

    provider = "mock"
    is_mock = True

    def __init__(self, planner: Callable[[str], dict], responder: Callable[[str, list[ToolSpec]], dict],
                 model: str = "mock-llm", fail_with: LLMError | None = None):
        self.model = model
        self._planner = planner
        self._responder = responder
        self.fail_with = fail_with
        self.records: list[LLMCallRecord] = []
        self.calls: list[tuple[str, str]] = []

    def generate_json(self, purpose: str, system: str, user: str, schema: dict, max_tokens: int = 0, effort: str = "") -> dict:
        self.calls.append((purpose, user))
        if self.fail_with:
            raise self.fail_with
        self.records.append(LLMCallRecord(purpose, self.model, 0, mock=True))
        return self._planner(user)

    def run_with_tools(self, purpose: str, system: str, user: str, tools: list[ToolSpec], final_schema: dict,
                       max_tokens: int = 0, effort: str = "", max_rounds: int = 5) -> tuple[dict, list[dict]]:
        self.calls.append((purpose, user))
        if self.fail_with:
            raise self.fail_with
        trace: list[dict] = []
        wrapped = []
        for t in tools:
            def handler(args, _t=t):
                out = _t.handler(args)
                trace.append({"tool": _t.name, "input": args, "output": out, "error": False})
                return out
            wrapped.append(ToolSpec(t.name, t.description, t.input_schema, handler))
        self.records.append(LLMCallRecord(purpose, self.model, 0, mock=True))
        return self._responder(user, wrapped), trace


def _strip_key(schema, key: str):
    if isinstance(schema, dict):
        return {k: _strip_key(v, key) for k, v in schema.items() if k != key}
    if isinstance(schema, list):
        return [_strip_key(v, key) for v in schema]
    return schema


class GeminiLLM:
    """Google Gemini through the official ``google-genai`` SDK (same interface as AnthropicLLM).

    Free-tier friendly: a client-side requests-per-minute throttle and retry with back-off on
    HTTP 429 / 5xx (honouring the server's retryDelay when given). Final answers of the tool
    loop are requested as a JSON object in text and validated by pydantic downstream.
    """

    provider = "gemini"
    is_mock = False

    def __init__(self, api_key: str, model: str, timeout_s: float = 90.0, rpm: float = 8.0,
                 thinking_budget: int = -1, max_retries: int = 6, client: Any = None, thinking_level: str = "",
                 planner_model: str = ""):
        if not api_key and client is None:
            raise LLMError("missing_credentials", "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
        self.model = model
        self.rpm = rpm
        self.thinking_budget = thinking_budget
        self.thinking_level = thinking_level
        self.planner_model = planner_model or model
        self.quota_events: list[dict] = []
        self.max_retries = max_retries
        self._last_call = 0.0
        import threading
        self._lock = threading.Lock()
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except ImportError as exc:
            if client is None:
                raise LLMError("sdk_missing", "The 'google-genai' package is not installed: pip install -r requirements.txt") from exc
            types = None
        self._types = types
        if client is None:
            client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))
        self._client = client
        self.records: list[LLMCallRecord] = []

    # ------------------------------------------------------------------ plumbing
    def _throttle(self) -> None:
        if self.rpm <= 0:
            return
        with self._lock:
            gap = 60.0 / self.rpm
            wait = self._last_call + gap - time.time()
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()

    @staticmethod
    def _retry_delay(exc: Exception, attempt: int) -> float:
        text = str(exc)
        m = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", text)
        if m:
            return min(float(m.group(1)) + 1.0, 90.0)
        return min(5.0 * (2 ** attempt), 60.0)

    def _config(self, **kw):
        t = self._types
        if self.thinking_level:
            kw["thinking_config"] = t.ThinkingConfig(thinking_level=self.thinking_level)
        elif self.thinking_budget >= 0:
            kw["thinking_config"] = t.ThinkingConfig(thinking_budget=self.thinking_budget)
        return t.GenerateContentConfig(**kw)

    def _generate(self, contents, config, dropped: list[str], model: str | None = None):
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                return self._client.models.generate_content(model=model or self.model, contents=contents, config=config)
            except Exception as exc:
                code = getattr(exc, "code", None)
                name = type(exc).__name__
                msg = str(exc)
                if code == 429 or "RESOURCE_EXHAUSTED" in msg:
                    quota_ids = re.findall(r"quotaId['\"]?\s*:\s*['\"]([^'\"]+)", msg)
                    self.quota_events.append({"model": model or self.model, "quota_ids": quota_ids, "at": time.strftime("%H:%M:%S")})
                    if quota_ids and all("PerDay" in q for q in quota_ids):
                        raise LLMError("quota_exhausted", f"Gemini daily quota exhausted ({', '.join(quota_ids)})", retriable=False) from exc
                    if attempt < self.max_retries:
                        time.sleep(self._retry_delay(exc, attempt)); continue
                    raise LLMError("unavailable", f"Gemini rate limit ({', '.join(quota_ids) or 'unknown quota'})", retriable=True) from exc
                if (code and 500 <= int(code) < 600) or name in ("ServerError",):
                    if attempt < self.max_retries:
                        time.sleep(min(3.0 * (attempt + 1), 20.0)); continue
                    raise LLMError("unavailable", f"Gemini server error: {msg[:300]}", retriable=True) from exc
                if code in (401, 403) or "API_KEY_INVALID" in msg or "PERMISSION_DENIED" in msg:
                    raise LLMError("auth", f"Gemini rejected the API key: {msg[:300]}") from exc
                if code == 404:
                    raise LLMError("model_not_found", f"Gemini model {self.model!r} not found: {msg[:300]}") from exc
                if code == 400:
                    # optional features first: thinking budget, JSON schema, additionalProperties
                    if getattr(config, "thinking_config", None) is not None and "thinking" in msg.lower():
                        config.thinking_config = None; dropped.append("thinking_budget"); continue
                    if getattr(config, "response_json_schema", None) is not None:
                        config.response_json_schema = None; dropped.append("json_schema"); continue
                    if config.tools and "additional" in msg.lower():
                        for tool in config.tools:
                            for fd in tool.function_declarations or []:
                                fd.parameters_json_schema = _strip_key(fd.parameters_json_schema, "additionalProperties")
                        dropped.append("additionalProperties"); continue
                    raise LLMError("bad_request", msg[:500]) from exc
                if name in ("TimeoutException", "ReadTimeout", "ConnectTimeout", "WriteTimeout", "PoolTimeout") or "timed out" in msg.lower():
                    if attempt < self.max_retries:
                        time.sleep(min(2.0 * (attempt + 1), 10.0)); continue
                    raise LLMError("timeout", f"Gemini API timed out: {msg[:300]}", retriable=True) from exc
                if name in ("ConnectError", "ConnectionError", "ReadError", "WriteError", "RemoteProtocolError",
                            "ProtocolError", "ConnectionResetError") or "connection reset" in msg.lower():
                    if attempt < self.max_retries:
                        time.sleep(min(2.0 * (attempt + 1), 10.0)); continue
                    raise LLMError("network", f"Could not reach the Gemini API: {msg[:300]}", retriable=True) from exc
                raise LLMError("unknown", f"{name}: {msg[:300]}") from exc
        raise LLMError("unavailable", "Gemini request failed after retries", retriable=True)

    @staticmethod
    def _usage(resp) -> tuple[int, int]:
        u = getattr(resp, "usage_metadata", None)
        return (getattr(u, "prompt_token_count", 0) or 0,
                (getattr(u, "candidates_token_count", 0) or 0) + (getattr(u, "thoughts_token_count", 0) or 0))

    @staticmethod
    def _text(resp) -> str:
        try:
            parts = resp.candidates[0].content.parts or []
        except (AttributeError, IndexError, TypeError):
            return getattr(resp, "text", "") or ""
        return "".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False))

    def check_model(self) -> dict:
        try:
            info = self._client.models.get(model=self.model)
        except Exception as exc:
            raise LLMError("model_not_found" if getattr(exc, "code", None) == 404 else "unknown", str(exc)[:300]) from exc
        return {"id": getattr(info, "name", self.model), "display_name": getattr(info, "display_name", "")}

    # ------------------------------------------------------------------ high level
    def generate_json(self, purpose: str, system: str, user: str, schema: dict, max_tokens: int = 2000, effort: str = "") -> dict:
        dropped: list[str] = []
        config = self._config(system_instruction=system, response_mime_type="application/json",
                              response_json_schema=schema, max_output_tokens=max(max_tokens, 2048), temperature=0.0)
        t0 = time.time()
        model = self.planner_model if purpose == "planner" else self.model
        resp = self._generate(user, config, dropped, model=model)
        tin, tout = self._usage(resp)
        self.records.append(LLMCallRecord(purpose, getattr(resp, "model_version", None) or model,
                                          int((time.time() - t0) * 1000), tin, tout, 0, dropped))
        return parse_json_text(self._text(resp))

    def run_with_tools(self, purpose: str, system: str, user: str, tools: list[ToolSpec], final_schema: dict,
                       max_tokens: int = 3000, effort: str = "", max_rounds: int = 5) -> tuple[dict, list[dict]]:
        t = self._types
        by_name = {spec.name: spec for spec in tools}
        decls = [t.FunctionDeclaration(name=spec.name, description=spec.description, parameters_json_schema=spec.input_schema)
                 for spec in tools]
        sys_text = (system + "\n\nWhen you have finished any tool calls, answer with ONLY a JSON object matching this schema "
                    "(no prose, no code fences):\n" + json.dumps(final_schema, ensure_ascii=False))
        config = self._config(system_instruction=sys_text, tools=[t.Tool(function_declarations=decls)],
                              automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True),
                              max_output_tokens=max(max_tokens, 3000), temperature=0.0)
        contents = [t.Content(role="user", parts=[t.Part.from_text(text=user)])]
        trace: list[dict] = []
        dropped: list[str] = []
        t0 = time.time()
        tin = tout = 0
        for round_no in range(max_rounds + 1):
            resp = self._generate(contents, config, dropped)
            a, b = self._usage(resp)
            tin, tout = tin + a, tout + b
            calls = list(getattr(resp, "function_calls", None) or [])
            if not calls:
                self.records.append(LLMCallRecord(purpose, self.model, int((time.time() - t0) * 1000), tin, tout, round_no, dropped))
                return parse_json_text(self._text(resp)), trace
            if round_no == max_rounds:
                break
            contents.append(resp.candidates[0].content)  # keeps thought signatures
            parts = []
            for fc in calls:
                spec = by_name.get(fc.name)
                args = dict(fc.args or {})
                if spec is None:
                    out, err = {"error": f"unknown tool {fc.name}"}, True
                else:
                    try:
                        out, err = spec.handler(args), False
                    except Exception as exc:
                        out, err = {"error": f"{type(exc).__name__}: {exc}"}, True
                trace.append({"tool": fc.name, "input": args, "output": out, "error": err})
                parts.append(t.Part.from_function_response(name=fc.name, response={"result": out}))
            contents.append(t.Content(role="user", parts=parts))
        self.records.append(LLMCallRecord(purpose, self.model, int((time.time() - t0) * 1000), tin, tout, max_rounds, dropped))
        raise LLMError("tool_loop", f"model did not finish within {max_rounds} tool rounds")


def build_llm(settings):
    """Create the configured provider's client. The provider is never switched automatically."""
    if settings.llm_provider == "gemini":
        return GeminiLLM(settings.gemini_api_key, settings.gemini_model, timeout_s=settings.llm_timeout_s,
                         rpm=settings.gemini_rpm, thinking_budget=settings.gemini_thinking_budget,
                         thinking_level=settings.gemini_thinking_level, planner_model=settings.gemini_planner_model)
    if settings.llm_provider == "anthropic":
        return AnthropicLLM(settings.anthropic_api_key, settings.anthropic_model,
                            timeout_s=settings.anthropic_timeout_s, max_retries=settings.anthropic_max_retries)
    raise LLMError("bad_config", f"Unknown LLM_PROVIDER {settings.llm_provider!r} (use 'anthropic' or 'gemini')")
