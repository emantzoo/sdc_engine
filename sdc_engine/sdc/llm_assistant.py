"""
Cerebras LLM backend wrapper for SDC pipeline.

Thin HTTP wrapper around the Cerebras chat completions API.
All network I/O isolated here. On any failure, logs warning and returns None.
"""

import json
import logging
import os
import re
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.cerebras.ai/v1/chat/completions"
_DEFAULT_MODEL = "qwen-3-235b-a22b-instruct-2507"
_DEFAULT_TIMEOUT = 60
_VALIDATION_TIMEOUT = 5

# Gemini backend
_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
_GEMINI_MODEL = "gemini-2.5-flash"


def _repair_json(text: str) -> str:
    """Fix common LLM JSON formatting mistakes before parsing.

    Handles:
    - Trailing commas before } or ]
    - Single-line // comments
    - Unescaped newlines inside string values
    - Smart quotes → regular quotes
    """
    # Replace smart quotes
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")

    # Remove single-line comments (outside strings)
    # Simple approach: remove lines that are just comments
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith('//'):
            continue
        # Remove inline comments (after a value, not inside a string)
        # Only if the // is not inside quotes
        cleaned.append(line)
    text = '\n'.join(cleaned)

    # Remove trailing commas before } or ]
    text = re.sub(r',\s*([}\]])', r'\1', text)

    return text


def _safe_json_loads(text: str) -> Any:
    """Try json.loads with progressively more aggressive repairs.

    1. Direct parse
    2. After _repair_json
    3. After stripping control characters from string values
    """
    # Attempt 1: direct
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: basic repair
    repaired = _repair_json(text)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Attempt 3: aggressive — replace unescaped control chars in strings,
    # remove JS-style single quotes as keys, fix truncated JSON
    aggressive = repaired
    # Replace any unescaped control characters (tabs, newlines inside strings)
    aggressive = re.sub(r'(?<=": ")(.*?)(?=")', lambda m: m.group(0).replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t'), aggressive)

    # Try to fix truncated JSON by closing open brackets
    try:
        return json.loads(aggressive)
    except json.JSONDecodeError:
        pass

    # Attempt 4: line-by-line — find the offending line and remove it
    lines = repaired.split('\n')
    for attempt in range(min(5, len(lines))):
        try:
            return json.loads('\n'.join(lines))
        except json.JSONDecodeError as e:
            # Find the problematic line and remove it
            bad_line = e.lineno - 1 if e.lineno else -1
            if 0 <= bad_line < len(lines):
                # Try removing the bad line
                lines.pop(bad_line)
            else:
                break

    # All attempts failed — raise original error
    return json.loads(text)


def _extract_json(text: str) -> Optional[str]:
    """Extract JSON content from text that may include markdown fences or preamble.

    Handles:
    - Bare JSON arrays or objects
    - ```json ... ``` fenced blocks
    - Leading/trailing text around JSON
    """
    if not text or not text.strip():
        return None

    # Strip markdown fences first
    fenced = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    # Find the first [ or { and match to the last ] or }
    # Try whichever appears first in the text; if tied, prefer {
    bracket_pos = text.find('[')
    brace_pos = text.find('{')
    if bracket_pos == -1 and brace_pos == -1:
        return None
    if brace_pos != -1 and (bracket_pos == -1 or brace_pos <= bracket_pos):
        search_order = [('{', '}'), ('[', ']')]
    else:
        search_order = [('[', ']'), ('{', '}')]

    for start_char, end_char in search_order:
        start_idx = text.find(start_char)
        if start_idx == -1:
            continue
        # Find matching closing bracket by counting nesting
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start_idx, len(text)):
            c = text[i]
            if escape_next:
                escape_next = False
                continue
            if c == '\\' and in_string:
                escape_next = True
                continue
            if c == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == start_char:
                depth += 1
            elif c == end_char:
                depth -= 1
                if depth == 0:
                    return text[start_idx:i + 1]

    return None


class CerebrasAssistant:
    """Wrapper for the Cerebras cloud API (Qwen 235B).

    All methods return None on failure — the caller should fall back
    to the rule-based engine when this happens.
    """

    def __init__(self, api_key: Optional[str] = None):
        self._base_url = _DEFAULT_BASE_URL
        self._model = _DEFAULT_MODEL
        self._timeout = _DEFAULT_TIMEOUT
        self._available = False
        self._backend = "cerebras"  # or "gemini"

        # Resolve API key: param → CEREBRAS env → GEMINI env
        self._api_key = api_key or os.environ.get("CEREBRAS_API_KEY")
        if not self._api_key:
            gemini_key = os.environ.get("GEMINI_API_KEY")
            if gemini_key:
                self._api_key = gemini_key
                self._backend = "gemini"
                logger.info("Using Gemini backend (GEMINI_API_KEY found)")
            else:
                logger.debug("No LLM API key found — LLM features disabled")
                return

        # Validate with a lightweight test call
        try:
            result = self._call_api(
                "You are a test assistant.",
                "Respond with exactly: OK",
                timeout=_VALIDATION_TIMEOUT,
            )
            if result and "OK" in result.upper():
                self._available = True
                logger.info("%s API validated — LLM features enabled", self._backend.title())
            else:
                logger.warning("%s API validation returned unexpected response — LLM disabled", self._backend.title())
        except Exception as e:
            logger.warning("%s API validation failed: %s — LLM disabled", self._backend.title(), e)

    def is_available(self) -> bool:
        """Check if the LLM backend is available and validated."""
        return self._available

    def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
        timeout: Optional[int] = None,
    ) -> Optional[str]:
        """Send a chat completion request to Cerebras or Gemini API.

        Returns the raw content string, or None on failure.
        """
        if self._backend == "gemini":
            return self._call_gemini(system_prompt, user_prompt, timeout)
        return self._call_cerebras(system_prompt, user_prompt, timeout)

    def _call_gemini(
        self,
        system_prompt: str,
        user_prompt: str,
        timeout: Optional[int] = None,
    ) -> Optional[str]:
        """Send a generateContent request to the Gemini API."""
        try:
            url = f"{_GEMINI_BASE_URL}/{_GEMINI_MODEL}:generateContent?key={self._api_key}"

            payload = json.dumps({
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": user_prompt}],
                    },
                ],
                "systemInstruction": {
                    "parts": [{"text": system_prompt}],
                },
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 16384,
                },
            }).encode("utf-8")

            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "SDCEngine/1.0",
                },
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=timeout or self._timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))

            # Gemini response: candidates[0].content.parts[0].text
            content = body["candidates"][0]["content"]["parts"][0]["text"]

            # Log token usage if available
            usage = body.get("usageMetadata", {})
            if usage:
                logger.debug(
                    "Gemini API: prompt=%d, completion=%d tokens",
                    usage.get("promptTokenCount", 0),
                    usage.get("candidatesTokenCount", 0),
                )

            return content

        except urllib.error.HTTPError as e:
            body_text = ""
            try:
                body_text = e.read().decode("utf-8", errors="replace")[:500]
            except (OSError, AttributeError):
                pass
            logger.warning("Gemini API HTTP error %d: %s — %s", e.code, e.reason, body_text)
            return None
        except urllib.error.URLError as e:
            logger.warning("Gemini API connection error: %s", e.reason)
            return None
        except (KeyError, IndexError) as e:
            logger.warning("Gemini API unexpected response format: %s", e)
            return None
        except Exception as e:
            logger.warning("Gemini API call failed: %s", e)
            return None

    def _call_cerebras(
        self,
        system_prompt: str,
        user_prompt: str,
        timeout: Optional[int] = None,
    ) -> Optional[str]:
        """Send a chat completion request to Cerebras API."""
        try:
            payload = json.dumps({
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 4096,
            }).encode("utf-8")

            req = urllib.request.Request(
                self._base_url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                    "User-Agent": "SDCEngine/1.0",
                },
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=timeout or self._timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))

            content = body["choices"][0]["message"]["content"]

            # Log token usage if available
            usage = body.get("usage", {})
            if usage:
                logger.debug(
                    "Cerebras API: %d tokens (prompt=%d, completion=%d)",
                    usage.get("total_tokens", 0),
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                )

            return content

        except urllib.error.HTTPError as e:
            logger.warning("Cerebras API HTTP error %d: %s", e.code, e.reason)
            return None
        except urllib.error.URLError as e:
            logger.warning("Cerebras API connection error: %s", e.reason)
            return None
        except Exception as e:
            logger.warning("Cerebras API call failed: %s", e)
            return None

    def classify_columns(
        self,
        user_prompt: str,
        system_prompt: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """Call the LLM for column classification.

        Args:
            user_prompt: Pre-formatted prompt with column metadata.
            system_prompt: The classification system prompt.

        Returns:
            Parsed JSON array of classification results, or None on failure.
        """
        try:
            raw = self._call_api(system_prompt, user_prompt)
            if raw is None:
                return None

            json_str = _extract_json(raw)
            if json_str is None:
                logger.warning("Could not extract JSON from classify response")
                return None

            result = _safe_json_loads(json_str)
            # Handle Gemini sometimes wrapping the array in an object
            if isinstance(result, dict):
                # Case 1: dict with a list value (e.g. {"columns": [...]})
                for v in result.values():
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        result = v
                        break
                else:
                    # Case 2: single classification object (truncated response)
                    if "name" in result:
                        logger.warning("classify_columns: got single object (truncated?), wrapping in list")
                        result = [result]
                    else:
                        logger.warning("classify_columns: expected JSON array, got dict with keys %s",
                                       list(result.keys())[:5])
                        return None
            if not isinstance(result, list):
                logger.warning("classify_columns: expected JSON array, got %s", type(result).__name__)
                return None

            return result

        except json.JSONDecodeError as e:
            logger.warning("classify_columns: JSON parse error: %s", e)
            return None
        except Exception as e:
            logger.warning("classify_columns failed: %s", e)
            return None

    def select_method(
        self,
        user_prompt: str,
        system_prompt: str,
    ) -> Optional[Dict[str, Any]]:
        """Call the LLM for method selection.

        Args:
            user_prompt: Pre-formatted prompt with dataset profile.
            system_prompt: The method selection system prompt.

        Returns:
            Parsed JSON object with method recommendation, or None on failure.
        """
        try:
            raw = self._call_api(system_prompt, user_prompt)
            if raw is None:
                return None

            json_str = _extract_json(raw)
            if json_str is None:
                logger.warning("Could not extract JSON from method selection response")
                return None

            result = _safe_json_loads(json_str)
            # Handle LLM wrapping the object in an array
            if isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
                result = result[0]
            if not isinstance(result, dict):
                logger.warning("select_method: expected JSON object, got %s", type(result).__name__)
                return None

            return result

        except json.JSONDecodeError as e:
            logger.warning("select_method: JSON parse error: %s", e)
            return None
        except Exception as e:
            logger.warning("select_method failed: %s", e)
            return None


# ---------------------------------------------------------------------------
# Module-level lazy singleton — avoids re-validation on every call
# ---------------------------------------------------------------------------
_instance: Optional[CerebrasAssistant] = None


def get_assistant(api_key: Optional[str] = None) -> CerebrasAssistant:
    """Return a cached CerebrasAssistant instance.

    Re-creates if the previous instance is unavailable or api_key changed.
    """
    global _instance
    if _instance is not None and _instance.is_available():
        # If caller provides an explicit key that differs, re-create
        if api_key and api_key != _instance._api_key:
            _instance = CerebrasAssistant(api_key=api_key)
        return _instance
    _instance = CerebrasAssistant(api_key=api_key)
    return _instance
