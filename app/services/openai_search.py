"""
OpenAI Search Service
---------------------
Wraps the two-step OpenAI workflow (web research → structured extraction) as a
reusable, dependency-injectable service.  The core logic is preserved exactly as
authored in the original openai_search.py module.
"""

import json
import re
from typing import Type, TypeVar

from openai import OpenAI
from loguru import logger
from pydantic import BaseModel

from app.core.config import settings

T = TypeVar("T", bound=BaseModel)


class OpenAISearchService:
    """
    Thin service wrapper around the two-step OpenAI contact-research workflow.

    Step 1 – "Gatherer": performs a live web_search tool call to collect raw
             contact information for the queried company.
    Step 2 – "Filter & Formatter": distils the raw research into a single,
             validated Pydantic model instance.
    """

    def __init__(self) -> None:
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            logger.warning("OPENAI_API_KEY is not configured – OpenAI calls will fail.")
        self._client = OpenAI(api_key=api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def structured_llm_call(self, request: BaseModel, model_class: Type[T]) -> T:
        """
        Execute the two-step research + extraction pipeline for the target
        and return a validated instance of *model_class*.

        Raises:
            openai.OpenAIError: on unrecoverable API errors.
            pydantic.ValidationError: if the LLM output cannot be coerced into
                                      the target schema.
        """
        company_name = getattr(request, "company_name", str(request))
        country = getattr(request, "country", None)
        zip_code = getattr(request, "zip_code", None)
        url = getattr(request, "url", None)
        max_limit = getattr(request, "max_limit", None)

        logger.info(f"[OpenAISearchService] Starting research for: {company_name!r}")

        context_parts = [f"Company: {company_name}"]
        if country: context_parts.append(f"Country: {country}")
        if zip_code: context_parts.append(f"Zip Code: {zip_code}")
        if url: context_parts.append(f"URL: {url}")
        
        target_context = " | ".join(context_parts)

        # ── Step 1: Deep Research (The "Gatherer") ────────────────────────
        research_prompt = (
            f"Deep research task: Find contact information for the following target:\n{target_context}\n\n"
            "GOALS:\n"
            "1. Prioritize Human Resources (HR) contact info.\n"
            "2. If HR is unavailable, find General/Corporate info.\n"
            "3. Look for a Fax number.\n"
            "4. Find ALL available contact details (phones, emails, faxes, addresses).\n\n"
            "CRITICAL INSTRUCTION FOR EMAILS:\n"
            "If you see '[email protected]' on the company's website, it means their emails are hidden by Cloudflare.\n"
            "If this happens, DO NOT return '[email protected]'. Instead, you MUST run a new search query to find the company's email on OTHER public sources such as:\n"
            "- Press releases (PR Newswire, Business Wire)\n"
            "- LinkedIn, Apollo, or ZoomInfo summaries\n"
            "- Public PDF documents or SEC filings\n"
            "- Official social media pages"
        )

        research_response = self._client.responses.create(
            model="gpt-4o-mini",
            tools=[{"type": "web_search"}],
            input=research_prompt,
        )
        raw_research = research_response.output_text
        logger.debug(f"[OpenAISearchService] Raw research length: {len(raw_research)} chars")

        # ── Step 2: Extraction (The "Filter & Formatter") ─────────────────
        extraction_prompt = (
            f"Based on this research:\n{raw_research}\n\n"
            "TASK: Extract the info into JSON.\n"
            "STRICT RULES:\n"
            "1. Extract ALL valid emails, phone numbers, fax numbers, and physical addresses you can find into arrays.\n"
            "2. For addresses, format each distinct location into a single fully readable string (e.g., '123 Main St, City, State 12345').\n"
            f"3. The response MUST be a VALID JSON object matching exactly this schema:\n{json.dumps(model_class.model_json_schema(), indent=2)}\n"
            f"4. IMPORTANT: DO NOT return the schema definition itself. Return the ACTUAL extracted data values.\n"
            f"5. Make sure to fill in the 'company_name' key with the target company: {company_name}.\n"
            "6. EXCLUDE PLACEHOLDERS: Do NOT extract dummy/example emails (e.g., 'email@...', 'example@...', 'name@...', 'abc@...'). Only extract real, verified contact emails."
        )

        final_response = self._client.responses.create(
            model="gpt-4o-mini",
            input=extraction_prompt,
            text={"format": {"type": "json_object"}},
        )

        # ── Parse + clean + validate ───────────────────────────────────────
        data = json.loads(final_response.output_text)
        cleaned_data = self._clean_dict(data)
        result = model_class.model_validate(cleaned_data)
        
        if max_limit is not None and max_limit > 0 and hasattr(result, "company_info"):
            result.company_info.phones = result.company_info.phones[:max_limit]
            result.company_info.faxes = result.company_info.faxes[:max_limit]
            result.company_info.emails = result.company_info.emails[:max_limit]
            result.company_info.addresses = result.company_info.addresses[:max_limit]

        logger.info(f"[OpenAISearchService] Extraction complete for: {company_name!r}")
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_val(value: object) -> object:
        """Strip residual markdown citation artefacts from string values."""
        if isinstance(value, str):
            return re.sub(r"\s*\(\[.*?\]\(.*?\)\)", "", value).strip()
        return value

    def _clean_dict(self, data: dict) -> dict:
        """Recursively clean all string values inside *data*."""
        cleaned: dict = {}
        for k, v in data.items():
            if isinstance(v, dict):
                cleaned[k] = {nk: self._clean_val(nv) for nk, nv in v.items()}
            else:
                cleaned[k] = self._clean_val(v)
        return cleaned
