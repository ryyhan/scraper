"""
Token Usage Utilities
---------------------
Shared helpers that normalise provider-native SDK usage objects into the
unified ``ProviderTokenUsage`` model, insulating service code from each
SDK's differing field names.

Mapping reference
~~~~~~~~~~~~~~~~~
OpenAI (``response.usage``):
  * ``input_tokens``           → ProviderTokenUsage.input_tokens
  * ``output_tokens``          → ProviderTokenUsage.output_tokens
  * ``total_tokens``           → ProviderTokenUsage.total_tokens

Gemini (``response.usage_metadata``):
  * ``prompt_token_count``     → ProviderTokenUsage.input_tokens
  * ``candidates_token_count`` → ProviderTokenUsage.output_tokens
  * ``total_token_count``      → ProviderTokenUsage.total_tokens

All ``getattr`` calls default to 0 and the result is guarded with
``or 0`` to handle ``None`` values returned when a model does not report
certain counters (e.g. some Gemini model versions omit
``candidates_token_count`` when the response is cached).
"""

from __future__ import annotations

from typing import Any

from app.models import ProviderTokenUsage


def openai_usage(usage: Any) -> ProviderTokenUsage:
    """
    Convert an OpenAI ``response.usage`` object to ``ProviderTokenUsage``.

    Safe to call with ``None`` — returns a zero-filled instance.
    """
    if usage is None:
        return ProviderTokenUsage()
    return ProviderTokenUsage(
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        total_tokens=getattr(usage, "total_tokens", 0) or 0,
    )


def gemini_usage(usage_metadata: Any) -> ProviderTokenUsage:
    """
    Convert a Gemini ``response.usage_metadata`` object to ``ProviderTokenUsage``.

    Safe to call with ``None`` — returns a zero-filled instance.
    """
    if usage_metadata is None:
        return ProviderTokenUsage()
    return ProviderTokenUsage(
        input_tokens=getattr(usage_metadata, "prompt_token_count", 0) or 0,
        output_tokens=getattr(usage_metadata, "candidates_token_count", 0) or 0,
        total_tokens=getattr(usage_metadata, "total_token_count", 0) or 0,
    )
