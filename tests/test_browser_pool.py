"""
Unit tests for persistent Playwright BrowserPool in app.services.scraper.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from app.services.scraper import BrowserPool, ScraperService


def test_browser_pool_singleton():
    """Verify BrowserPool provides a single process-wide instance."""
    pool1 = BrowserPool.get_instance()
    pool2 = BrowserPool.get_instance()
    assert pool1 is pool2


@pytest.mark.asyncio
async def test_browser_pool_close_idempotent():
    """Verify close() can be called safely even when browser is not running."""
    pool = BrowserPool()
    assert pool.is_running is False
    # Should not raise any errors
    await pool.close()
    assert pool.is_running is False


@pytest.mark.asyncio
async def test_scraper_service_uses_injected_browser():
    """Verify ScraperService can accept an existing browser instance."""
    mock_browser = MagicMock()
    scraper = ScraperService(browser=mock_browser)
    assert scraper.browser is mock_browser

    async with scraper as s:
        assert s.browser is mock_browser


@pytest.mark.asyncio
async def test_scraper_service_acquires_from_pool(monkeypatch):
    """Verify ScraperService acquires browser from BrowserPool when none provided."""
    mock_browser = MagicMock()
    mock_get_browser = AsyncMock(return_value=mock_browser)

    pool = BrowserPool.get_instance()
    monkeypatch.setattr(pool, "get_browser", mock_get_browser)

    scraper = ScraperService()
    async with scraper as s:
        assert s.browser is mock_browser
    mock_get_browser.assert_awaited_once()
