"""
Combined Search Service
-----------------------
Executes both OpenAI and Gemini search pipelines concurrently and aggregates their results.
"""

import asyncio
from loguru import logger
from typing import Optional

from app.models import (
    CombinedSearchRequest,
    CombinedSearchResult,
    CombinedCompanyInfo,
    OpenAISearchRequest,
    OpenAISearchResult,
    GeminiSearchRequest,
    GeminiSearchResult,
)
from app.services.openai_search import OpenAISearchService
from app.services.gemini_search import GeminiSearchService


class CombinedSearchService:
    """
    Service that delegates to both OpenAI and Gemini search services concurrently,
    then combines the results into a single comprehensive response.
    """

    def __init__(self) -> None:
        self.openai_service = OpenAISearchService()
        self.gemini_service = GeminiSearchService()

    async def search(self, request: CombinedSearchRequest) -> CombinedSearchResult:
        company_name = request.company_name
        logger.info(f"[CombinedSearchService] Starting combined research for: {company_name!r}")

        openai_req = OpenAISearchRequest(**request.model_dump())
        gemini_req = GeminiSearchRequest(**request.model_dump())

        # Execute both concurrently in threads to prevent blocking
        openai_task = asyncio.to_thread(
            self.openai_service.structured_llm_call, openai_req, OpenAISearchResult
        )
        gemini_task = asyncio.to_thread(
            self.gemini_service.structured_llm_call, gemini_req, GeminiSearchResult
        )

        openai_result: Optional[OpenAISearchResult] = None
        gemini_result: Optional[GeminiSearchResult] = None

        # Await tasks and handle exceptions individually
        results = await asyncio.gather(openai_task, gemini_task, return_exceptions=True)

        if isinstance(results[0], Exception):
            logger.error(f"[CombinedSearchService] OpenAI search failed: {results[0]}")
        else:
            openai_result = results[0]

        if isinstance(results[1], Exception):
            logger.error(f"[CombinedSearchService] Gemini search failed: {results[1]}")
        else:
            gemini_result = results[1]

        if not openai_result and not gemini_result:
            raise RuntimeError("Both OpenAI and Gemini search pipelines failed.")

        # Aggregate the results
        phones = set()
        faxes = set()
        emails = set()
        addresses = set()
        official_site = ""

        # Process OpenAI Result
        if openai_result:
            if openai_result.official_site:
                official_site = openai_result.official_site
            
            for p in openai_result.company_info.phones:
                if p: phones.add(p)
            for f in openai_result.company_info.faxes:
                if f: faxes.add(f)
            for e in openai_result.company_info.emails:
                if e: emails.add(e.lower())
            for a in openai_result.company_info.addresses:
                if a: addresses.add(a)

        # Process Gemini Result
        if gemini_result:
            if not official_site and gemini_result.official_site:
                official_site = gemini_result.official_site
            
            for p in gemini_result.company_info.phones:
                if p: phones.add(p)
            for f in gemini_result.company_info.faxes:
                if f: faxes.add(f)
            for e in gemini_result.company_info.emails:
                if e: emails.add(e.lower())
            for a in gemini_result.company_info.addresses:
                if a: addresses.add(a)

        company_info = CombinedCompanyInfo(
            phones=sorted(list(phones)),
            faxes=sorted(list(faxes)),
            emails=sorted(list(emails)),
            addresses=sorted(list(addresses)),
        )

        logger.info(f"[CombinedSearchService] Completed combined research for: {company_name!r}")
        return CombinedSearchResult(
            company_name=company_name,
            official_site=official_site,
            company_info=company_info,
            openai_result=openai_result,
            gemini_result=gemini_result,
        )
