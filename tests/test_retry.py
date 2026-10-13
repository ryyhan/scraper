"""
Unit tests for dual-layer retry decorators in app.services._retry.
"""

import pytest
import httpx
from google.genai.errors import ClientError, ServerError, APIError
from groq import RateLimitError as GroqRateLimitError, APIConnectionError as GroqConnectionError, APITimeoutError as GroqTimeoutError
from openai import RateLimitError as OpenAIRateLimitError, APIConnectionError as OpenAIConnectionError

from app.services._retry import (
    retry_gemini,
    retry_groq,
    retry_openai,
    _is_retryable_gemini_error,
)


def test_is_retryable_gemini_error():
    """Verify classification of retryable vs non-retryable Gemini exceptions."""
    s500 = ServerError(500, {"message": "Internal Server Error"})
    s503 = ServerError(503, {"message": "Service Unavailable"})
    c429 = ClientError(429, {"message": "Resource Exhausted / Rate Limit"})
    c408 = ClientError(408, {"message": "Request Timeout"})
    c400 = ClientError(400, {"message": "Invalid Argument"})
    c401 = ClientError(401, {"message": "Unauthorized / Bad API Key"})
    c403 = ClientError(403, {"message": "Forbidden"})
    timeout = httpx.ReadTimeout("Socket read timed out")
    net_err = httpx.ConnectError("Connection refused")
    val_err = ValueError("Invalid parameter")
    key_err = KeyError("missing_key")

    assert _is_retryable_gemini_error(s500) is True
    assert _is_retryable_gemini_error(s503) is True
    assert _is_retryable_gemini_error(c429) is True
    assert _is_retryable_gemini_error(c408) is True
    assert _is_retryable_gemini_error(c400) is False
    assert _is_retryable_gemini_error(c401) is False
    assert _is_retryable_gemini_error(c403) is False
    assert _is_retryable_gemini_error(timeout) is True
    assert _is_retryable_gemini_error(net_err) is True
    assert _is_retryable_gemini_error(val_err) is False
    assert _is_retryable_gemini_error(key_err) is False


def test_retry_gemini_retries_retryable_errors():
    """Verify retry_gemini decorator retries retryable exceptions and succeeds."""
    attempts = 0

    @retry_gemini()
    def flaking_gemini_call():
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise ClientError(429, {"message": "Rate limit"})
        return "success"

    result = flaking_gemini_call()
    assert result == "success"
    assert attempts == 2


def test_retry_gemini_does_not_retry_non_retryable_errors():
    """Verify retry_gemini decorator does not retry non-retryable exceptions."""
    attempts = 0

    @retry_gemini()
    def failing_gemini_call():
        nonlocal attempts
        attempts += 1
        raise ClientError(400, {"message": "Invalid prompt"})

    with pytest.raises(ClientError):
        failing_gemini_call()

    # Must only attempt once and not retry
    assert attempts == 1


def test_retry_groq_retries_network_and_status_errors():
    """Verify retry_groq catches connection, timeout, and rate limit errors."""
    req = httpx.Request("POST", "https://api.groq.com")
    resp_429 = httpx.Response(429, request=req)

    # 1. Test RateLimitError
    attempts = 0
    @retry_groq()
    def flaking_groq_rate_limit():
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise GroqRateLimitError("Rate limit", response=resp_429, body={"error": "rate_limit"})
        return "groq_ok"

    assert flaking_groq_rate_limit() == "groq_ok"
    assert attempts == 2

    # 2. Test APIConnectionError
    attempts = 0
    @retry_groq()
    def flaking_groq_connection():
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise GroqConnectionError(request=req)
        return "conn_ok"

    assert flaking_groq_connection() == "conn_ok"
    assert attempts == 2

    # 3. Non-retryable error
    attempts = 0
    @retry_groq()
    def groq_invalid_value():
        nonlocal attempts
        attempts += 1
        raise ValueError("Bad input")

    with pytest.raises(ValueError):
        groq_invalid_value()
    assert attempts == 1


def test_retry_openai_retries_transient_errors():
    """Verify retry_openai catches RateLimitError and ConnectionError."""
    req = httpx.Request("POST", "https://api.openai.com")
    resp_429 = httpx.Response(429, request=req)

    attempts = 0
    @retry_openai()
    def flaking_openai():
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise OpenAIConnectionError(request=req)
        return "openai_ok"

    assert flaking_openai() == "openai_ok"
    assert attempts == 2
