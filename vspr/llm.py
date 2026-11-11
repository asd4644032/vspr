"""Thin wrapper around Ollama for structured (JSON-schema) chat calls.

Every language-model call in VSPR goes through :meth:`OllamaLLM.chat_json`, which
returns the parsed JSON object plus the model's thinking text (DeepSeek-R1
``<think>`` content), so the pipeline never has to post-process free text.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


@dataclass
class LLMResult:
    data: dict
    thinking: str = ""
    raw: str = ""


class LLMClient(Protocol):
    def chat_json(
        self,
        model: str,
        messages: list[dict],
        schema: dict,
        options: dict | None = None,
    ) -> LLMResult: ...

    def unload(self, model: str) -> None: ...


def split_thinking(text: str) -> tuple[str, str]:
    """Separate ``<think>...</think>`` blocks from the answer text."""
    thinking = "\n".join(m.strip() for m in _THINK_RE.findall(text))
    answer = _THINK_RE.sub("", text)
    # Unclosed think tag (generation cut off inside the reasoning).
    if "<think>" in answer.lower():
        head, _, tail = answer.partition("<think>")
        thinking = (thinking + "\n" + tail).strip()
        answer = head
    return thinking.strip(), answer.strip()


def extract_json(text: str) -> dict:
    """Parse the first JSON object in ``text`` (tolerates code fences / prose around it)."""
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise ValueError(f"No JSON object found in model output: {text[:200]!r}")


def _validate(data: dict, schema: dict) -> None:
    """Minimal check: required keys present (Ollama already constrains the format)."""
    missing = [k for k in schema.get("required", []) if k not in data]
    if missing:
        raise ValueError(f"Missing keys {missing} in {list(data)}")


def supports_thinking(model: str) -> bool:
    name = model.lower()
    return any(tag in name for tag in ("deepseek-r1", "qwq", "qwen3"))


@dataclass
class OllamaLLM:
    host: str = "http://127.0.0.1:11434"
    retries: int = 3
    keep_alive: str | int = "10m"
    think: str | bool = "auto"  # "auto" -> enabled for reasoning models only
    _client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        import ollama  # imported lazily so tests run without the package

        self._client = ollama.Client(host=self.host)

    def _think_flag(self, model: str) -> bool | None:
        if self.think == "auto":
            return True if supports_thinking(model) else None
        return bool(self.think)

    def chat_json(self, model, messages, schema, options=None) -> LLMResult:
        think = self._think_flag(model)
        msgs = list(messages)
        last_err: Exception | None = None
        for attempt in range(1, self.retries + 1):
            kwargs = dict(model=model, messages=msgs, format=schema,
                          options=options or {}, keep_alive=self.keep_alive)
            if think is not None:
                kwargs["think"] = think
            resp = self._client.chat(**kwargs)
            message = resp["message"] if isinstance(resp, dict) else resp.message
            content = (message.get("content") if isinstance(message, dict) else message.content) or ""
            thinking = (message.get("thinking") if isinstance(message, dict)
                        else getattr(message, "thinking", None)) or ""
            tag_thinking, content = split_thinking(content)
            thinking = (thinking or tag_thinking).strip()
            try:
                data = extract_json(content)
                _validate(data, schema)
                return LLMResult(data=data, thinking=thinking, raw=content)
            except ValueError as err:
                last_err = err
                log.warning("Invalid JSON from %s (attempt %d/%d): %s", model, attempt, self.retries, err)
                msgs = list(messages) + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": "Your reply was not valid JSON for the requested schema. "
                                                "Reply again with only the JSON object."},
                ]
        raise RuntimeError(f"{model} failed to return valid JSON after {self.retries} attempts: {last_err}")

    def describe(self, models: list[str]) -> dict:
        """Ollama server version and the digest of each model, saved with every run.

        Tags such as ``llama3.2-vision`` are re-pointed when re-pulled, so the digest is
        the only way to know later which checkpoint produced a run.
        """
        info: dict = {"host": self.host}
        try:
            import urllib.request

            with urllib.request.urlopen(self.host.rstrip("/") + "/api/version", timeout=10) as r:
                info["ollama_version"] = json.loads(r.read()).get("version")
        except Exception as err:  # noqa: BLE001
            info["ollama_version"] = f"unavailable: {err}"
        try:
            listed = self._client.list()
            rows = listed["models"] if isinstance(listed, dict) else listed.models
            table = {}
            for m in rows:
                get = m.get if isinstance(m, dict) else lambda k, m=m: getattr(m, k, None)
                name = get("model") or get("name")
                details = get("details")
                if details is not None and not isinstance(details, dict):
                    details = details.model_dump() if hasattr(details, "model_dump") else vars(details)
                table[name] = {"digest": get("digest"), "size": get("size"), "details": details}
            for model in models:
                key = model if model in table else f"{model}:latest"
                info[model] = table.get(key, "not found in `ollama list`")
        except Exception as err:  # noqa: BLE001
            info["error"] = repr(err)
        return info

    def unload(self, model: str) -> None:
        """Release the model from GPU memory (before the diffusion model runs)."""
        try:
            self._client.generate(model=model, prompt="", keep_alive=0)
        except Exception as err:  # noqa: BLE001 - best effort
            log.debug("Could not unload %s: %s", model, err)
