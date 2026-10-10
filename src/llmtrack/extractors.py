"""Extract usage and model metadata from provider response objects and dictionaries."""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

UsageTuple = Tuple[Optional[str], Optional[int], Optional[int]]
ExtractorCallable = Callable[[Any], UsageTuple]


def extract_anthropic(resp: Any) -> UsageTuple:
    """Extract model, input_tokens, output_tokens from an Anthropic response or dict."""
    model: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    if isinstance(resp, dict):
        model = resp.get("model")
        usage = resp.get("usage")
        if isinstance(usage, dict):
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
    else:
        model = getattr(resp, "model", None)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            input_tokens = getattr(usage, "input_tokens", None)
            output_tokens = getattr(usage, "output_tokens", None)

    return model, input_tokens, output_tokens


def extract_openai(resp: Any) -> UsageTuple:
    """Extract model, input_tokens, output_tokens from an OpenAI response or dict."""
    model: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    if isinstance(resp, dict):
        model = resp.get("model")
        usage = resp.get("usage")
        if isinstance(usage, dict):
            input_tokens = usage.get("prompt_tokens")
            output_tokens = usage.get("completion_tokens")
    else:
        model = getattr(resp, "model", None)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            input_tokens = getattr(usage, "prompt_tokens", None)
            output_tokens = getattr(usage, "completion_tokens", None)

    return model, input_tokens, output_tokens


def extract_generic(resp: Any) -> UsageTuple:
    """Attempt generic extraction by checking known attribute or dictionary key conventions."""
    if resp is None:
        return None, None, None

    model: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    if isinstance(resp, dict):
        model = resp.get("model")
        usage = resp.get("usage") or resp.get("token_usage")
        if isinstance(usage, dict):
            input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
            output_tokens = usage.get("output_tokens") or usage.get("completion_tokens")
        else:
            input_tokens = resp.get("input_tokens") or resp.get("prompt_tokens")
            output_tokens = resp.get("output_tokens") or resp.get("completion_tokens")
    else:
        model = getattr(resp, "model", None)
        usage = getattr(resp, "usage", None) or getattr(resp, "token_usage", None)
        if usage is not None:
            if isinstance(usage, dict):
                input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
                output_tokens = usage.get("output_tokens") or usage.get("completion_tokens")
            else:
                input_tokens = getattr(usage, "input_tokens", None) or getattr(
                    usage, "prompt_tokens", None
                )
                output_tokens = getattr(usage, "output_tokens", None) or getattr(
                    usage, "completion_tokens", None
                )
        else:
            input_tokens = getattr(resp, "input_tokens", None) or getattr(
                resp, "prompt_tokens", None
            )
            output_tokens = getattr(resp, "output_tokens", None) or getattr(
                resp, "completion_tokens", None
            )

    return model, input_tokens, output_tokens


def extract_gemini(resp: Any) -> UsageTuple:
    """Extract model, input_tokens, output_tokens from a Google Gemini response or dict."""
    model: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    if isinstance(resp, dict):
        model = resp.get("model") or resp.get("modelVersion") or resp.get("model_version")
        usage = resp.get("usage_metadata") or resp.get("usageMetadata") or resp.get("usage")
        if isinstance(usage, dict):
            input_tokens = usage.get("prompt_token_count") or usage.get("promptTokenCount")
            output_tokens = usage.get("candidates_token_count") or usage.get("candidatesTokenCount")
    else:
        model = getattr(resp, "model", None) or getattr(resp, "model_version", None)
        usage = (
            getattr(resp, "usage_metadata", None)
            or getattr(resp, "usageMetadata", None)
            or getattr(resp, "usage", None)
        )
        if usage is not None:
            if isinstance(usage, dict):
                input_tokens = usage.get("prompt_token_count") or usage.get("promptTokenCount")
                output_tokens = usage.get("candidates_token_count") or usage.get("candidatesTokenCount")
            else:
                input_tokens = getattr(usage, "prompt_token_count", None) or getattr(
                    usage, "promptTokenCount", None
                )
                output_tokens = getattr(usage, "candidates_token_count", None) or getattr(
                    usage, "candidatesTokenCount", None
                )

    return model, input_tokens, output_tokens


REGISTRY: dict[str, ExtractorCallable] = {
    "anthropic": extract_anthropic,
    "openai": extract_openai,
    "deepseek": extract_openai,
    "gemini": extract_gemini,
    "google": extract_gemini,
    "generic": extract_generic,
}


def extract_usage(
    resp: Any,
    provider: Optional[str] = None,
    custom_extractor: Optional[ExtractorCallable] = None,
) -> UsageTuple:
    """Extract (model, input_tokens, output_tokens) from response using provider registry.

    Falls back cleanly to None if attributes are missing.
    """
    if custom_extractor is not None:
        try:
            return custom_extractor(resp)
        except Exception:
            return None, None, None

    if provider:
        p_lower = provider.lower()
        if p_lower in REGISTRY:
            res = REGISTRY[p_lower](resp)
            if res[0] is not None or res[1] is not None:
                return res

    # Auto-detect check
    # Check type name or module
    type_name = type(resp).__name__.lower()
    mod_name = getattr(type(resp), "__module__", "").lower()

    if "anthropic" in mod_name or "message" in type_name:
        res = extract_anthropic(resp)
        if res[0] is not None or res[1] is not None:
            return res

    if "openai" in mod_name or "completion" in type_name:
        res = extract_openai(resp)
        if res[0] is not None or res[1] is not None:
            return res

    if "gemini" in mod_name or "google" in mod_name or "generative" in mod_name:
        res = extract_gemini(resp)
        if res[0] is not None or res[1] is not None:
            return res

    # Fallback to generic inspection
    return extract_generic(resp)
