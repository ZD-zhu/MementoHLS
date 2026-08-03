from __future__ import annotations

import hashlib
import json
import time
from threading import BoundedSemaphore
from typing import Any

import requests

from .models import LLMCallResult


_ALLOWED_ROLES = {"system", "user", "assistant"}


def _prompt_sha256(messages: list[dict[str, str]]) -> str:
    canonical = json.dumps(
        messages,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class VLLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float = 600.0,
        max_retries: int = 3,
        max_concurrency: int = 4,
        api_key: str = "EMPTY",
        context_limit: int = 8192,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.api_key = api_key
        self.context_limit = context_limit
        self.deployment_epoch: str | None = None
        self._semaphore = BoundedSemaphore(max_concurrency)

    def deployment_identity(self) -> dict[str, Any]:
        models_url = self.base_url + "/models"
        models = requests.get(models_url, timeout=10)
        models.raise_for_status()
        payload = models.json()
        selected = next(
            (
                item
                for item in payload.get("data", [])
                if item.get("id") == self.model
            ),
            None,
        )
        if selected is None:
            raise RuntimeError(f"Model {self.model!r} is not served")
        metrics_url = self.base_url.removesuffix("/v1") + "/metrics"
        metrics = requests.get(metrics_url, timeout=10)
        metrics.raise_for_status()
        process_start: float | None = None
        for line in metrics.text.splitlines():
            if line.startswith("process_start_time_seconds "):
                process_start = float(line.split(None, 1)[1])
                break
        if process_start is None:
            raise RuntimeError(
                "vLLM metrics lacks process_start_time_seconds deployment identity"
            )
        return {
            "prometheus_process_start_time_seconds": process_start,
            "model_root": selected.get("root"),
            "model_max_len": selected.get("max_model_len"),
            "deployment_epoch": self.deployment_epoch,
        }

    def health(self) -> dict[str, Any]:
        health_url = self.base_url.removesuffix("/v1") + "/health"
        models_url = self.base_url + "/models"
        health = requests.get(health_url, timeout=10)
        health.raise_for_status()
        models = requests.get(models_url, timeout=10)
        models.raise_for_status()
        payload = models.json()
        available = [item.get("id") for item in payload.get("data", [])]
        if self.model not in available:
            raise RuntimeError(
                f"Model {self.model!r} is not served; available={available}"
            )
        selected_model = next(
            item for item in payload.get("data", []) if item.get("id") == self.model
        )
        smoke = self.chat(
            [{"role": "user", "content": "Reply with exactly: HLS_SMOKE_OK"}],
            temperature=0.0,
            max_tokens=16,
            seed=0,
        )
        if smoke.text.strip() != "HLS_SMOKE_OK":
            raise RuntimeError(f"Fixed vLLM smoke test failed: {smoke.text!r}")
        return {
            "health_status": health.status_code,
            "models": available,
            "model_listing_created_at_response": selected_model.get("created"),
            "prometheus_process_start_time_seconds": self.deployment_identity()[
                "prometheus_process_start_time_seconds"
            ],
            "model_root": selected_model.get("root"),
            "model_max_len": selected_model.get("max_model_len"),
            "smoke_response": smoke.text,
            "smoke_elapsed_seconds": smoke.elapsed_seconds,
            "deployment_epoch": self.deployment_epoch,
        }

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        seed: int,
        *,
        minimum_completion_tokens: int = 1,
        user_prompt_variants: list[str] | None = None,
    ) -> LLMCallResult:
        if not 1 <= minimum_completion_tokens <= max_tokens:
            raise ValueError(
                "minimum_completion_tokens must be in [1, max_tokens]"
            )
        roles = [str(message.get("role") or "") for message in messages]
        if not roles or any(role not in _ALLOWED_ROLES for role in roles):
            raise ValueError(f"LLM message role is not allowlisted: {roles}")
        candidates: list[list[dict[str, str]]] = [
            [dict(message) for message in messages]
        ]
        for prompt in user_prompt_variants or []:
            variant = [dict(message) for message in messages]
            user_indices = [
                index
                for index, message in enumerate(variant)
                if message.get("role") == "user"
            ]
            if not user_indices:
                raise ValueError("Prompt compaction requires a user message")
            variant[user_indices[-1]]["content"] = prompt
            candidates.append(variant)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": candidates[0],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "stream": False,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        retryable = {408, 429, 500, 502, 503, 504}
        started = time.monotonic()
        last_error: Exception | None = None
        preflight_tokens: int | None = None
        variant_token_counts: list[int] = []
        selected_variant_index = 0
        selected_messages = candidates[0]
        attempt_history: list[dict[str, Any]] = []
        with self._semaphore:
            for attempt in range(1, self.max_retries + 2):
                attempt_row: dict[str, Any] = {
                    "attempt": attempt,
                    "seed": seed,
                    "temperature": temperature,
                    "deployment_epoch": self.deployment_epoch,
                }
                try:
                    if preflight_tokens is None:
                        variant_token_counts = []
                        tokenize_statuses: list[int] = []
                        selected = None
                        for index, candidate in enumerate(candidates):
                            token_response = requests.post(
                                self.base_url.removesuffix("/v1") + "/tokenize",
                                json={"model": self.model, "messages": candidate},
                                headers=headers,
                                timeout=min(self.timeout_seconds, 60.0),
                            )
                            tokenize_statuses.append(token_response.status_code)
                            if token_response.status_code in retryable:
                                raise RuntimeError(
                                    "retryable vLLM tokenize HTTP "
                                    f"{token_response.status_code}"
                                )
                            if token_response.status_code >= 400:
                                raise RuntimeError(
                                    f"vLLM tokenize HTTP {token_response.status_code}: "
                                    f"{token_response.text[:1000]}"
                                )
                            count = int(token_response.json()["count"])
                            variant_token_counts.append(count)
                            if count + minimum_completion_tokens + 1 <= self.context_limit:
                                selected = (index, candidate, count)
                                break
                        attempt_row["tokenize_http_statuses"] = tokenize_statuses
                        if selected is None:
                            raise RuntimeError(
                                "No deterministic prompt variant leaves the required "
                                f"{minimum_completion_tokens} completion tokens; "
                                f"counts={variant_token_counts}, context={self.context_limit}"
                            )
                        selected_variant_index, selected_messages, preflight_tokens = selected
                        available_tokens = self.context_limit - preflight_tokens - 1
                        payload["messages"] = selected_messages
                        payload["max_tokens"] = min(max_tokens, available_tokens)
                        if payload["max_tokens"] < minimum_completion_tokens:
                            raise RuntimeError(
                                "Internal prompt-budget error: reserved completion lost"
                            )
                    attempt_row.update(
                        {
                            "selected_prompt_variant_index": selected_variant_index,
                            "prompt_sha256": _prompt_sha256(selected_messages),
                            "preflight_prompt_tokens": preflight_tokens,
                            "effective_max_tokens": payload["max_tokens"],
                        }
                    )
                    response = requests.post(
                        self.base_url + "/chat/completions",
                        json=payload,
                        headers=headers,
                        timeout=self.timeout_seconds,
                    )
                    attempt_row["chat_http_status"] = response.status_code
                    if response.status_code in retryable and attempt <= self.max_retries:
                        attempt_row["status"] = "retryable_http"
                        attempt_history.append(attempt_row)
                        time.sleep(min(2 ** (attempt - 1), 8))
                        continue
                    if response.status_code >= 400:
                        raise RuntimeError(
                            f"vLLM chat HTTP {response.status_code}: "
                            f"{response.text[:1000]}"
                        )
                    data = response.json()
                    choices = data.get("choices") or []
                    if not choices:
                        raise RuntimeError("vLLM returned no choices")
                    text = choices[0].get("message", {}).get("content")
                    if not isinstance(text, str) or not text:
                        raise RuntimeError("vLLM returned empty assistant content")
                    finish_reason = choices[0].get("finish_reason")
                    selected_prompt_sha = _prompt_sha256(selected_messages)
                    data["_kernelmem_request_budget"] = {
                        "context_limit": self.context_limit,
                        "preflight_prompt_tokens": preflight_tokens,
                        "prompt_variant_token_counts": variant_token_counts,
                        "selected_prompt_variant_index": selected_variant_index,
                        "minimum_completion_tokens": minimum_completion_tokens,
                        "requested_max_tokens": max_tokens,
                        "effective_max_tokens": payload["max_tokens"],
                    }
                    data["_kernelmem_request_provenance"] = {
                        "model": self.model,
                        "seed": seed,
                        "temperature": temperature,
                        "message_roles": [
                            str(message.get("role")) for message in selected_messages
                        ],
                        "prompt_sha256": selected_prompt_sha,
                        "deployment_epoch": self.deployment_epoch,
                    }
                    usage = data.get("usage") or {}
                    effective_user_prompt = next(
                        (
                            message.get("content")
                            for message in reversed(selected_messages)
                            if message.get("role") == "user"
                        ),
                        None,
                    )
                    attempt_row["status"] = "success"
                    attempt_row["finish_reason"] = finish_reason
                    attempt_history.append(attempt_row)
                    return LLMCallResult(
                        text=text,
                        raw_response=data,
                        prompt_tokens=usage.get("prompt_tokens"),
                        completion_tokens=usage.get("completion_tokens"),
                        total_tokens=usage.get("total_tokens"),
                        elapsed_seconds=time.monotonic() - started,
                        attempts=attempt,
                        prompt_variant_index=selected_variant_index,
                        effective_user_prompt=effective_user_prompt,
                        request_seed=seed,
                        request_temperature=temperature,
                        requested_max_tokens=max_tokens,
                        effective_max_tokens=int(payload["max_tokens"]),
                        finish_reason=(
                            str(finish_reason) if finish_reason is not None else None
                        ),
                        message_roles=[
                            str(message.get("role")) for message in selected_messages
                        ],
                        prompt_sha256=selected_prompt_sha,
                        attempt_history=attempt_history,
                        deployment_epoch=self.deployment_epoch,
                    )
                except (requests.RequestException, ValueError, RuntimeError) as exc:
                    last_error = exc
                    attempt_row["status"] = "error"
                    attempt_row["error"] = f"{type(exc).__name__}: {exc}"[:1200]
                    attempt_history.append(attempt_row)
                    preflight_tokens = None
                    if attempt > self.max_retries:
                        break
                    time.sleep(min(2 ** (attempt - 1), 8))
        raise RuntimeError(f"vLLM call failed after retries: {last_error}")
