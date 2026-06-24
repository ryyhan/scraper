"""
Sliding-Window Rate Limiter
---------------------------
An in-process, asyncio-safe sliding-window rate limiter implemented as a
reusable FastAPI dependency.

Design decisions
~~~~~~~~~~~~~~~~
* **Dependency, not middleware** — Applied per-route via ``Depends()``.
  This means only the endpoints we deliberately protect are affected; health
  checks, GET polls, and webhook mock endpoints are left untouched.

* **Sliding window** — More accurate than a fixed "reset every N seconds"
  token-bucket because it counts requests in the *trailing* window rather
  than resetting a counter at a hard boundary. This prevents the burst
  that would otherwise occur right after a window resets.

* **asyncio.Lock per client** — Each IP gets its own lock so concurrent
  requests from *different* IPs never contend with each other. The global
  ``_store`` dict is only written under ``_store_lock`` to avoid a TOCTOU
  race when a new IP is first seen.

* **Single-process** — Suitable for a single uvicorn worker. If you scale
  to multiple workers (gunicorn -w N), replace the in-memory store with a
  Redis ZSET + Lua script for atomic sliding-window semantics across
  processes.

Usage::

    from app.api.rate_limit import task_rate_limit

    @router.post("/my-endpoint/")
    async def my_endpoint(..., _rl: None = Depends(task_rate_limit)):
        ...
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Deque, Dict, Tuple

from fastapi import Depends, HTTPException, Request, status


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Maximum number of task-creation requests allowed per IP within the window.
RATE_LIMIT_REQUESTS: int = 10

#: Duration of the sliding window in seconds.
RATE_LIMIT_WINDOW_SECONDS: float = 60.0


# ---------------------------------------------------------------------------
# Internal state  (module-level singletons, lives for the process lifetime)
# ---------------------------------------------------------------------------

# Maps client IP → (asyncio.Lock, deque[timestamp])
# The deque holds the wall-clock timestamps (float) of recent requests that
# fall inside the current sliding window.
_store: Dict[str, Tuple[asyncio.Lock, Deque[float]]] = {}

# Guards mutations to _store itself (not to individual client deques, which
# are protected by the per-client lock).
_store_lock: asyncio.Lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

async def _get_or_create_client_state(ip: str) -> Tuple[asyncio.Lock, Deque[float]]:
    """Return (lock, timestamps) for *ip*, creating the entry if absent."""
    if ip not in _store:
        async with _store_lock:
            # Double-checked locking: another coroutine may have inserted
            # the entry while we waited for _store_lock.
            if ip not in _store:
                _store[ip] = (asyncio.Lock(), deque())
    return _store[ip]


async def _check_rate_limit(ip: str) -> None:
    """
    Enforce the sliding-window limit for *ip*.

    Raises ``HTTP 429`` if the client has exceeded ``RATE_LIMIT_REQUESTS``
    within the last ``RATE_LIMIT_WINDOW_SECONDS`` seconds.
    """
    lock, timestamps = await _get_or_create_client_state(ip)

    async with lock:
        now = time.monotonic()
        window_start = now - RATE_LIMIT_WINDOW_SECONDS

        # Evict timestamps that have fallen outside the sliding window.
        while timestamps and timestamps[0] < window_start:
            timestamps.popleft()

        if len(timestamps) >= RATE_LIMIT_REQUESTS:
            # Calculate how long until the oldest request exits the window.
            retry_after = int(timestamps[0] - window_start) + 1
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Rate limit exceeded: {RATE_LIMIT_REQUESTS} requests "
                    f"per {int(RATE_LIMIT_WINDOW_SECONDS)}s per IP. "
                    f"Retry after {retry_after}s."
                ),
                headers={"Retry-After": str(retry_after)},
            )

        timestamps.append(now)


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

async def task_rate_limit(request: Request) -> None:
    """
    FastAPI dependency that applies the sliding-window rate limit.

    Resolves the client IP from ``X-Forwarded-For`` (set by a reverse proxy
    such as nginx or a cloud load balancer) with a fallback to the direct
    connection IP.  The first address in ``X-Forwarded-For`` is used because
    that is the original client IP in a standard proxy chain::

        X-Forwarded-For: <client>, <proxy1>, <proxy2>

    If you are not behind a trusted reverse proxy, remove the
    ``X-Forwarded-For`` branch to prevent IP spoofing.
    """
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the leftmost address (original client).
        client_ip = forwarded_for.split(",")[0].strip()
    else:
        client_ip = request.client.host if request.client else "unknown"

    await _check_rate_limit(client_ip)
