"""
Integration tests for FastAPI endpoints in app.api.routes.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from app.models import CompanyMatchResult, ScrapeResult, ContactInfo, TaskRecord


@pytest.mark.asyncio
async def test_docs_endpoint(client):
    """Verify OpenAPI documentation is reachable."""
    response = await client.get("/docs")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_webhook_mock(client):
    """Verify /webhook-mock receives and confirms payload."""
    payload = {
        "status": "SUCCESS",
        "message": "Scrape complete",
        "result": {
            "poe_name": "Acme Corp",
            "official_site": "https://acme.com",
            "poe_info": {
                "Phone": ["(555) 000-1111"],
                "Fax": [],
                "Email": ["info@acme.com"],
                "Address": ["100 Main St"],
                "DeptContacts": None,
            },
        },
    }
    response = await client.post("/webhook-mock", json=payload)
    assert response.status_code == 200
    assert response.json() == {"received": True}


@pytest.mark.asyncio
async def test_match_companies_validation(client):
    """Verify /match-companies/ rejects requests missing required fields."""
    # Missing company_b
    response = await client.post("/match-companies/", json={"company_a": "Acme"})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_match_companies_success(client):
    """Verify /match-companies/ returns structured CompanyMatchResult."""
    mock_result = CompanyMatchResult(
        company_a="IBM",
        company_b="International Business Machines",
        is_same_company=True,
        confidence=0.98,
        reasoning="IBM is the widely recognized acronym for International Business Machines.",
        provider="groq",
        deep_search_used=False,
        processing_time_seconds=0.35,
    )

    with patch(
        "app.services.CompanyMatcherService.match",
        new=AsyncMock(return_value=mock_result),
    ):
        response = await client.post(
            "/match-companies/",
            json={
                "company_a": "IBM",
                "company_b": "International Business Machines",
                "provider": "groq",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_same_company"] is True
        assert data["confidence"] == 0.98
        assert data["provider"] == "groq"


@pytest.mark.asyncio
async def test_extract_pdf_rejects_non_pdf(client):
    """Verify /extract-pdf/ rejects non-PDF file uploads with 400 Bad Request."""
    files = {"file": ("test.txt", b"This is plain text, not a PDF", "text/plain")}
    response = await client.post("/extract-pdf/", files=files, data={"provider": "gemini"})
    assert response.status_code == 400
    assert "Invalid file extension" in response.json()["detail"] or "PDF" in response.json()["detail"]


@pytest.mark.asyncio
async def test_task_status_not_found(client):
    """Verify querying non-existent task returns 404."""
    response = await client.get("/google-search/non-existent-uuid")
    assert response.status_code == 404
    assert response.json()["detail"] == "Task not found"


@pytest.mark.asyncio
async def test_failed_tasks_empty_list(client):
    """Verify /google-search/failed/ returns empty list when no failed tasks exist."""
    response = await client.get("/google-search/failed/")
    assert response.status_code == 200
    assert response.json() == {"failed_tasks": []}


@pytest.mark.asyncio
async def test_scraping_task_timeout_status_failure(db_session):
    """Verify that a scraping task timing out with no contacts is marked as FAILURE."""
    from app.api.routes import process_scraping_task
    from app.models import SearchRequest, TaskRecord
    from unittest.mock import patch

    task_id = "test-timeout-no-contacts"
    db_session.add(TaskRecord(task_id=task_id, status="IN_PROGRESS"))
    db_session.commit()

    req = SearchRequest(poe_name="Timeout Corp", timeout=10)

    with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
        await process_scraping_task(task_id, req, webhook_url=None)

    db_session.expire_all()
    task = db_session.get(TaskRecord, task_id)
    assert task is not None
    assert task.status == "FAILURE"
    assert "timeout limit" in task.message.lower()
