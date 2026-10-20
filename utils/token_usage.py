import threading
from copy import deepcopy
from typing import Any, Dict, Iterable, Optional

from langchain_core.callbacks import BaseCallbackHandler


USAGE_FIELDS = (
    "calls",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cache_hit_tokens",
    "cache_miss_tokens",
    "reasoning_tokens",
)


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def empty_usage() -> Dict[str, Any]:
    usage = {field: 0 for field in USAGE_FIELDS}
    usage["usage_complete"] = False
    return usage


def _normalize_usage(
    token_usage: Dict[str, Any],
    usage_metadata: Dict[str, Any],
    calls: int = 1,
) -> Dict[str, Any]:
    if not token_usage and not usage_metadata:
        usage = empty_usage()
        usage["calls"] = calls
        return usage

    prompt_tokens = _as_int(
        token_usage.get("prompt_tokens", usage_metadata.get("input_tokens"))
    )
    completion_tokens = _as_int(
        token_usage.get("completion_tokens", usage_metadata.get("output_tokens"))
    )
    total_tokens = _as_int(
        token_usage.get(
            "total_tokens",
            usage_metadata.get("total_tokens", prompt_tokens + completion_tokens),
        )
    )

    cache_hit_tokens = token_usage.get("prompt_cache_hit_tokens")
    if cache_hit_tokens is None:
        cache_hit_tokens = _as_dict(token_usage.get("prompt_tokens_details")).get(
            "cached_tokens"
        )
    if cache_hit_tokens is None:
        cache_hit_tokens = _as_dict(usage_metadata.get("input_token_details")).get(
            "cache_read"
        )
    cache_hit_tokens = _as_int(cache_hit_tokens)

    cache_miss_tokens = token_usage.get("prompt_cache_miss_tokens")
    if cache_miss_tokens is None:
        cache_miss_tokens = max(prompt_tokens - cache_hit_tokens, 0)
    cache_miss_tokens = _as_int(cache_miss_tokens)

    reasoning_tokens = _as_dict(
        token_usage.get("completion_tokens_details")
    ).get("reasoning_tokens")
    if reasoning_tokens is None:
        reasoning_tokens = _as_dict(
            usage_metadata.get("output_token_details")
        ).get("reasoning")
    reasoning_tokens = _as_int(reasoning_tokens)

    prompt_present = "prompt_tokens" in token_usage or "input_tokens" in usage_metadata
    completion_present = (
        "completion_tokens" in token_usage or "output_tokens" in usage_metadata
    )
    total_present = "total_tokens" in token_usage or "total_tokens" in usage_metadata

    return {
        "calls": calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cache_hit_tokens": cache_hit_tokens,
        "cache_miss_tokens": cache_miss_tokens,
        "reasoning_tokens": reasoning_tokens,
        "usage_complete": prompt_present and completion_present and total_present,
    }


def extract_usage(response: Any) -> Dict[str, Any]:
    metadata = _as_dict(getattr(response, "response_metadata", None))
    token_usage = _as_dict(metadata.get("token_usage"))
    usage_metadata = _as_dict(getattr(response, "usage_metadata", None))
    return _normalize_usage(token_usage, usage_metadata)


def extract_usage_from_llm_result(result: Any) -> Dict[str, Any]:
    llm_output = _as_dict(getattr(result, "llm_output", None))
    token_usage = _as_dict(llm_output.get("token_usage"))
    model = str(
        llm_output.get("model_name") or llm_output.get("model") or ""
    )

    usage_metadata: Dict[str, Any] = {}
    for generation_list in getattr(result, "generations", None) or []:
        for generation in generation_list:
            message = getattr(generation, "message", None)
            if message is not None:
                metadata = _as_dict(getattr(message, "response_metadata", None))
                if not token_usage:
                    token_usage = _as_dict(metadata.get("token_usage"))
                if not usage_metadata:
                    usage_metadata = _as_dict(
                        getattr(message, "usage_metadata", None)
                    )
                model = model or str(
                    metadata.get("model_name") or metadata.get("model") or ""
                )
            elif not token_usage:
                generation_info = _as_dict(
                    getattr(generation, "generation_info", None)
                )
                token_usage = _as_dict(generation_info.get("token_usage"))

    usage = _normalize_usage(token_usage, usage_metadata)
    usage["model"] = model or "unknown"
    return usage


def add_usage(target: Dict[str, Any], usage: Dict[str, Any]) -> None:
    previous_calls = _as_int(target.get("calls"))
    for field in USAGE_FIELDS:
        target[field] = _as_int(target.get(field)) + _as_int(usage.get(field))
    if previous_calls == 0:
        target["usage_complete"] = bool(usage.get("usage_complete"))
    else:
        target["usage_complete"] = bool(target.get("usage_complete")) and bool(
            usage.get("usage_complete")
        )


def aggregate_usages(usages: Iterable[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    total = empty_usage()
    by_stage: Dict[str, Dict[str, Any]] = {}

    for usage in usages:
        if not isinstance(usage, dict):
            continue
        add_usage(total, usage)
        for stage, stage_usage in _as_dict(usage.get("by_stage")).items():
            stage_bucket = by_stage.setdefault(stage, empty_usage())
            add_usage(stage_bucket, _as_dict(stage_usage))

    total["by_stage"] = by_stage
    return total


class TokenUsageCollector:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._total = empty_usage()
        self._by_stage: Dict[str, Dict[str, Any]] = {}

    def add(self, response: Any, stage: str) -> Dict[str, Any]:
        usage = extract_usage(response)
        add_usage(self._total, usage)
        stage_bucket = self._by_stage.setdefault(stage, empty_usage())
        add_usage(stage_bucket, usage)
        return usage

    def snapshot(self) -> Dict[str, Any]:
        result = deepcopy(self._total)
        result["by_stage"] = deepcopy(self._by_stage)
        return result


class EvalTokenUsageCallback(BaseCallbackHandler):
    def __init__(self, include_details: bool = True):
        super().__init__()
        self.include_details = include_details
        self._lock = threading.Lock()
        self._calls = []

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        try:
            usage = extract_usage_from_llm_result(response)
        except Exception:
            usage = empty_usage()
            usage["calls"] = 1
            usage["model"] = "unknown"
        usage["run_id"] = str(kwargs.get("run_id", ""))
        with self._lock:
            self._calls.append(usage)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            calls = [deepcopy(call) for call in self._calls]

        summary = aggregate_usages(calls)
        summary.pop("by_stage", None)
        by_model: Dict[str, Dict[str, Any]] = {}
        for call in calls:
            model = str(call.get("model") or "unknown")
            model_bucket = by_model.setdefault(model, empty_usage())
            add_usage(model_bucket, call)

        summary["by_model"] = by_model
        if self.include_details:
            summary["calls_detail"] = calls
        return summary
