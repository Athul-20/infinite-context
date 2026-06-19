"""
Cloud API client for OpenAI, Anthropic, and OpenAI-compatible endpoints.

Handles payload formatting, header construction, and SSE streaming for
providers like Groq, Together, etc.
"""

import json
import httpx
from typing import Dict, Generator, List

__all__ = ["CloudClient"]

# Shared timeout: 30 s connect, 120 s read (large prompts can be slow).
_DEFAULT_TIMEOUT = httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=10.0)


class CloudClient:
    """HTTP client for cloud LLM providers."""

    def __init__(self, api_key: str, model_id: str, base_url: str = None):
        """
        Args:
            api_key: Bearer / x-api-key token.
            model_id: Model identifier (e.g. ``claude-3-5-sonnet-20240620``).
            base_url: Custom endpoint URL.  When set, the OpenAI chat-
                      completions format is assumed.
        """
        self.api_key = api_key
        self.model_id = model_id

        # Determine endpoint and provider format.
        if base_url:
            self.base_url = base_url
            self.provider = "openai"
        elif "claude" in self.model_id.lower():
            self.base_url = "https://api.anthropic.com/v1/messages"
            self.provider = "anthropic"
        else:
            self.base_url = "https://api.openai.com/v1/chat/completions"
            self.provider = "openai"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _prepare_headers(self) -> Dict[str, str]:
        if self.provider == "anthropic":
            return {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _prepare_payload(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
    ) -> dict:
        if self.provider == "anthropic":
            # Anthropic splits the system prompt out of the messages array.
            system_msg = next(
                (m["content"] for m in messages if m["role"] == "system"), ""
            )
            user_msgs = [m for m in messages if m["role"] != "system"]
            return {
                "model": self.model_id,
                "system": system_msg,
                "messages": user_msgs,
                "max_tokens": 4096,
                "stream": stream,
            }
        return {
            "model": self.model_id,
            "messages": messages,
            "stream": stream,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_request(self, messages: List[Dict[str, str]]) -> str:
        """Send a non-streaming chat completion request."""
        payload = self._prepare_payload(messages, stream=False)
        headers = self._prepare_headers()

        with httpx.Client(timeout=_DEFAULT_TIMEOUT) as client:
            response = client.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        if self.provider == "anthropic":
            return data["content"][0]["text"]
        return data["choices"][0]["message"]["content"]

    def stream_request(
        self, messages: List[Dict[str, str]]
    ) -> Generator[str, None, None]:
        """Yield text deltas from an SSE streaming response."""
        payload = self._prepare_payload(messages, stream=True)
        headers = self._prepare_headers()

        with httpx.Client(timeout=_DEFAULT_TIMEOUT) as client:
            with client.stream(
                "POST", self.base_url, json=payload, headers=headers
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        return
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        # Malformed chunk — skip gracefully.
                        continue
