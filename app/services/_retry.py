"""
Shared Retry Helpers
--------------------
Centralised ``tenacity`` retry decorator factories for every LLM provider
used in this project (Gemini, OpenAI, Groq).

Dual-layer retry strategy
~~~~~~~~~~~~~~~~~~~~~~~~~
This module forms the **outer layer** of a two-layer retry system:

  Layer 1 — SDK-native retries (inner):
      The OpenAI and Groq SDKs are instantiated with ``max_retries=2``.
      On a 429 or 5xx response the SDK will:
        a) Read the ``Retry-After`` header from the API response and wait
           exactly as long as the server requests before retrying — far
           more precise than any fixed exponential guess.
        b) Apply its own jittered backoff if no ``Retry-After`` header is
           present.
      This layer handles the common, expected quota-exhaustion case with
      maximum fidelity to the API's own signals.

  Layer 2 — tenacity retry (outer, this module):
      Wraps each SDK call site.  Catches exceptions that survive Layer 1
      (i.e., errors the SDK exhausted its own retries on, or exception
      types the SDK does not retry at all — timeouts, connection resets,
      Google ``DeadlineExceeded``, etc.) and applies a second round of
      exponential backoff with jitter.

Combined capacity:
      3 (tenacity) × 3 (1 initial + 2 SDK retries) = **9 max API calls**
      per logical request.  This is deliberately bounded to prevent runaway
      retry storms while still tolerating sustained quota windows.

Note on Gemini:
      ``google-genai`` supports native retry via ``types.HttpRetryOptions``
      passed to ``genai.Client(http_options=...)``.  All Gemini clients in
      this project are configured with ``HttpRetryOptions(attempts=2)`` so
      the SDK reads ``Retry-After`` headers on 429s — the same inner-layer
      behaviour as OpenAI and Groq.  Tenacity remains the outer-layer safety
      net for ``DeadlineExceeded``, ``ServiceUnavailable``, and any other
      errors the SDK layer does not absorb.

Design decisions
~~~~~~~~~~~~~~~~
* **One module, one source of truth** — all retry parameters (attempts, wait
  bounds, exception types) live here so they can be adjusted globally without
  hunting across service files.
* **Exponential backoff with full jitter** — combines ``wait_exponential``
  with ``wait_random_exponential`` semantics to avoid the thundering-herd
  problem when many concurrent tasks hit a rate-limit window simultaneously.
* **Provider-specific exception types** — each factory only catches errors
  that are actually retryable for that provider; everything else surfaces
  immediately.
* **``reraise=True``** — after all attempts are exhausted the *original*
  exception propagates so callers receive meaningful SDK errors rather than a
  generic tenacity ``RetryError``.
* **Structured logging** — ``before_sleep_log`` writes a WARNING (not ERROR)
  on every wait so operators can distinguish retrying from failing.

Usage::

    from app.services._retry import retry_gemini, retry_openai, retry_groq

    @retry_gemini()
    def _my_gemini_call():
        return client.models.generate_content(...)

    @retry_openai()
    def _my_openai_call():
        return client.responses.create(...)
"""

from __future__ import annotations

import logging as _logging

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

# ---------------------------------------------------------------------------
# Module-level logger (stdlib — tenacity's before_sleep_log requires it)
# ---------------------------------------------------------------------------

_log = _logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retry parameters
# ---------------------------------------------------------------------------

# Number of total attempts for the outer tenacity layer (1 initial + N-1 retries).
# With SDK max_retries=2 (inner layer), total max API calls = 3 × 3 = 9.
_ATTEMPTS: int = 3

# Exponential backoff base parameters (seconds).
# wait_exponential_jitter:  initial=I, max=M  → each attempt waits
#   min(I * 2**n + U(0, jitter), max)  where n = attempt index.
# Practical delays: ~3 s, ~6 s, ~12 s (capped at 60 s).
_INITIAL_WAIT: float = 3.0   # first backoff duration (seconds)
_MAX_WAIT: float = 60.0      # ceiling for any single wait
_JITTER: float = 2.0         # random jitter added to each wait


# ---------------------------------------------------------------------------
# Provider: Gemini  (google-genai SDK)
# ---------------------------------------------------------------------------

def retry_gemini() -> "retry":
    """
    Return a ``tenacity`` retry decorator configured for the Gemini SDK.

    Retries on:
      * ``google.api_core.exceptions.ResourceExhausted`` — HTTP 429 quota exceeded
      * ``google.api_core.exceptions.ServiceUnavailable``  — HTTP 503 transient error
      * ``google.api_core.exceptions.InternalServerError`` — HTTP 500 transient error
      * ``google.api_core.exceptions.DeadlineExceeded``   — request timeout

    Falls back to catching all ``Exception`` if the google-api-core package
    is not installed (defensive import).
    """
    try:
        from google.api_core.exceptions import (
            DeadlineExceeded,
            InternalServerError,
            ResourceExhausted,
            ServiceUnavailable,
        )
        exc_types: tuple = (
            ResourceExhausted,
            ServiceUnavailable,
            InternalServerError,
            DeadlineExceeded,
        )
    except ImportError:
        exc_types = (Exception,)

    return retry(
        stop=stop_after_attempt(_ATTEMPTS),
        wait=wait_exponential_jitter(initial=_INITIAL_WAIT, max=_MAX_WAIT, jitter=_JITTER),
        retry=retry_if_exception_type(exc_types),
        before_sleep=before_sleep_log(_log, _logging.WARNING),
        reraise=True,
    )


# ---------------------------------------------------------------------------
# Provider: OpenAI  (openai SDK)
# ---------------------------------------------------------------------------

def retry_openai() -> "retry":
    """
    Return a ``tenacity`` retry decorator configured for the OpenAI SDK.

    Retries on:
      * ``openai.RateLimitError``  — HTTP 429 rate limit exceeded
      * ``openai.APIStatusError``  — HTTP 5xx server-side errors
      * ``openai.APITimeoutError`` — request timed out
      * ``openai.APIConnectionError`` — transient network failure

    Falls back to catching all ``Exception`` if the openai package is not
    installed (defensive import).
    """
    try:
        from openai import (
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            RateLimitError,
        )
        exc_types = (RateLimitError, APIStatusError, APITimeoutError, APIConnectionError)
    except ImportError:
        exc_types = (Exception,)

    return retry(
        stop=stop_after_attempt(_ATTEMPTS),
        wait=wait_exponential_jitter(initial=_INITIAL_WAIT, max=_MAX_WAIT, jitter=_JITTER),
        retry=retry_if_exception_type(exc_types),
        before_sleep=before_sleep_log(_log, _logging.WARNING),
        reraise=True,
    )


# ---------------------------------------------------------------------------
# Provider: Groq  (groq SDK)
# ---------------------------------------------------------------------------

def retry_groq() -> "retry":
    """
    Return a ``tenacity`` retry decorator configured for the Groq SDK.

    Retries on:
      * ``groq.RateLimitError``  — HTTP 429 rate limit exceeded
      * ``groq.APIStatusError``  — HTTP 5xx server-side errors

    Falls back to catching all ``Exception`` if the groq package is not
    installed (defensive import).
    """
    try:
        from groq import APIStatusError, RateLimitError
        exc_types = (RateLimitError, APIStatusError)
    except ImportError:
        exc_types = (Exception,)

    return retry(
        stop=stop_after_attempt(_ATTEMPTS),
        wait=wait_exponential_jitter(initial=_INITIAL_WAIT, max=_MAX_WAIT, jitter=_JITTER),
        retry=retry_if_exception_type(exc_types),
        before_sleep=before_sleep_log(_log, _logging.WARNING),
        reraise=True,
    )
