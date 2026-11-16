"""
Unit tests for sliding-window rate limiting and memory eviction in app.api.rate_limit.
"""

import asyncio
import time
from collections import deque
import pytest
from fastapi import HTTPException

from app.api.rate_limit import (
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
    _check_rate_limit,
    _store,
    _store_lock,
    purge_expired_clients,
    reset_rate_limit_store,
)


@pytest.mark.asyncio
async def test_rate_limit_allows_under_quota():
    """Verify that requests within quota succeed without raising."""
    client_ip = "192.168.1.100"
    for _ in range(RATE_LIMIT_REQUESTS):
        await _check_rate_limit(client_ip)

    async with _store_lock:
        assert client_ip in _store
        lock, timestamps = _store[client_ip]
        assert len(timestamps) == RATE_LIMIT_REQUESTS


@pytest.mark.asyncio
async def test_rate_limit_blocks_exceeding_quota():
    """Verify that exceeding quota raises HTTP 429 with Retry-After header."""
    client_ip = "192.168.1.101"
    for _ in range(RATE_LIMIT_REQUESTS):
        await _check_rate_limit(client_ip)

    with pytest.raises(HTTPException) as exc_info:
        await _check_rate_limit(client_ip)

    assert exc_info.value.status_code == 429
    assert "Rate limit exceeded" in exc_info.value.detail
    assert "Retry-After" in exc_info.value.headers


@pytest.mark.asyncio
async def test_purge_expired_clients_evicts_idle_ips():
    """Verify purge_expired_clients deletes idle and empty entries from memory."""
    await reset_rate_limit_store()
    now = time.monotonic()

    async with _store_lock:
        # IP 1: active request (recent)
        _store["10.0.0.1"] = (asyncio.Lock(), deque([now - 5]))
        # IP 2: idle request (expired past 60s)
        _store["10.0.0.2"] = (asyncio.Lock(), deque([now - 120]))
        # IP 3: empty deque
        _store["10.0.0.3"] = (asyncio.Lock(), deque())

    evicted = await purge_expired_clients(max_idle_seconds=60.0)
    assert evicted == 2

    async with _store_lock:
        assert "10.0.0.1" in _store
        assert "10.0.0.2" not in _store
        assert "10.0.0.3" not in _store


@pytest.mark.asyncio
async def test_purge_expired_clients_respects_max_capacity(monkeypatch):
    """Verify capacity ceiling drops oldest entries when store exceeds MAX_STORE_SIZE."""
    await reset_rate_limit_store()
    now = time.monotonic()

    import app.api.rate_limit as rl_module
    monkeypatch.setattr(rl_module, "MAX_STORE_SIZE", 3)

    async with _store_lock:
        # Insert 5 active IPs (none are expired past idle cutoff)
        _store["ip1"] = (asyncio.Lock(), deque([now - 40]))  # oldest
        _store["ip2"] = (asyncio.Lock(), deque([now - 30]))
        _store["ip3"] = (asyncio.Lock(), deque([now - 20]))
        _store["ip4"] = (asyncio.Lock(), deque([now - 10]))
        _store["ip5"] = (asyncio.Lock(), deque([now - 1]))   # newest

    evicted = await purge_expired_clients(max_idle_seconds=100.0)
    assert evicted == 2

    async with _store_lock:
        assert len(_store) == 3
        # The oldest two should be evicted
        assert "ip1" not in _store
        assert "ip2" not in _store
        assert "ip3" in _store
        assert "ip4" in _store
        assert "ip5" in _store


@pytest.mark.asyncio
async def test_reset_rate_limit_store():
    """Verify reset_rate_limit_store clears all internal state."""
    async with _store_lock:
        _store["some.ip"] = (asyncio.Lock(), deque([time.monotonic()]))

    await reset_rate_limit_store()

    async with _store_lock:
        assert len(_store) == 0


@pytest.mark.asyncio
async def test_task_rate_limit_untrusted_proxy_ignores_forwarded_for(monkeypatch):
    """Verify spoofed X-Forwarded-For is ignored when direct client is not a trusted proxy."""
    from unittest.mock import MagicMock
    from app.api.rate_limit import task_rate_limit
    from app.core.config import settings

    monkeypatch.setattr(settings, "TRUSTED_PROXIES", "127.0.0.1,::1")
    await reset_rate_limit_store()

    # Direct client IP is an untrusted public IP
    req = MagicMock()
    req.client.host = "203.0.113.50"
    req.headers = {"X-Forwarded-For": "1.2.3.4, 5.6.7.8"}

    await task_rate_limit(req)

    async with _store_lock:
        # Rate limit store should track the direct IP, NOT the spoofed header IP
        assert "203.0.113.50" in _store
        assert "1.2.3.4" not in _store


@pytest.mark.asyncio
async def test_task_rate_limit_trusted_proxy_honors_forwarded_for(monkeypatch):
    """Verify X-Forwarded-For is honored when direct client is a trusted proxy."""
    from unittest.mock import MagicMock
    from app.api.rate_limit import task_rate_limit
    from app.core.config import settings

    monkeypatch.setattr(settings, "TRUSTED_PROXIES", "127.0.0.1,::1")
    await reset_rate_limit_store()

    # Direct client is a local reverse proxy
    req = MagicMock()
    req.client.host = "127.0.0.1"
    req.headers = {"X-Forwarded-For": "198.51.100.99, 10.0.0.1"}

    await task_rate_limit(req)

    async with _store_lock:
        # Rate limit store should track the forwarded client IP
        assert "198.51.100.99" in _store
        assert "127.0.0.1" not in _store
