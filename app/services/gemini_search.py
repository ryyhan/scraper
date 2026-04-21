"""
Gemini Search Service
---------------------
Implements the same two-step contact-research pipeline as ``OpenAISearchService``
but drives Google's ``google-genai`` SDK (Gemini 2.0+) instead of OpenAI's.

Pipeline
~~~~~~~~
Step 1 – "Gatherer"
    A ``generate_content`` call with the ``google_search`` grounding tool enabled.
    Gemini performs live web searches and returns a rich, cited research summary.

Step 2 – "Extractor"
    A second ``generate_content`` call (no tools) with ``response_mime_type`` set
    to ``"application/json"`` and ``response_schema`` pointing at the target
    Pydantic model.  The SDK parses the JSON directly into the model via the
    ``.parsed`` attribute, eliminating manual ``json.loads`` + ``model_validate``
    plumbing.

Design decisions
~~~~~~~~~~~~~~~~
* Two-call separation mirrors the OpenAI implementation and keeps research
  and extraction concerns cleanly decoupled.
* The public ``structured_llm_call`` signature is intentionally identical to
  ``OpenAISearchService`` to allow future polymorphic dispatch.
* API key absence is logged as a warning (not raised) to be consistent with the
  OpenAI service contract.
"""

from __future__ import annotations

import re
from typing import Type, TypeVar

from google import genai
from google.genai import types as genai_types
from loguru import logger
from pydantic import BaseModel

from app.core.config import settings

T = TypeVar("T", bound=BaseModel)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_GEMINI_MODEL: str = "gemini-2.5-flash-lite"
"""Default model used for both research and extraction calls.

Gemini 2.0 Flash supports Google Search grounding, structured JSON output, and
is fast/cost-efficient.  Swap to ``gemini-1.5-pro`` or ``gemini-2.5-pro`` for
higher accuracy at greater latency/cost — no other code changes required.
"""


class GeminiSearchService:
    """
    Thin service wrapper around the two-step Gemini contact-research workflow.

    Step 1 – "Gatherer": triggers ``google_search`` grounding to collect raw,
             cited contact information for the queried company from the live web.
    Step 2 – "Extractor": distils the raw research into a single, validated
             Pydantic model instance via native JSON schema output.
    """

    def __init__(self) -> None:
        api_key = settings.GEMINI_API_KEY
        if not api_key:
            logger.warning(
                "GEMINI_API_KEY is not configured – Gemini calls will fail."
            )
        # Defer the ValueError from the SDK until an actual API call is made,
        # so that the service can be safely instantiated without a key (e.g.
        # during import-time checks and unit tests without live credentials).
        self._client: genai.Client | None = (
            genai.Client(api_key=api_key) if api_key else None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _require_client(self) -> genai.Client:
        """Return the configured client, or raise clearly if no API key was set."""
        if self._client is None:
            raise RuntimeError(
                "GeminiSearchService is not configured: GEMINI_API_KEY is missing. "
                "Set it in your .env file and restart the server."
            )
        return self._client

    def structured_llm_call(self, request: BaseModel, model_class: Type[T]) -> T:
        """
        Execute the two-step research + extraction pipeline for *request* and
        return a validated instance of *model_class*.

        Args:
            request:     A Pydantic model instance carrying at minimum a
                         ``company_name`` attribute.  Optional context fields
                         ``country``, ``zip_code``, and ``url`` are consumed
                         when present.
            model_class: The Pydantic model class that defines the extraction
                         schema.  Must be a concrete subclass of
                         ``pydantic.BaseModel``.

        Returns:
            A validated *model_class* instance populated with extracted data.

        Raises:
            google.genai.errors.APIError: on unrecoverable Gemini API errors.
            pydantic.ValidationError:     if the model response cannot be
                                          coerced into *model_class*
                                          (should be rare given schema forcing).
        """
        company_name: str = getattr(request, "company_name", str(request))
        country: str | None = getattr(request, "country", None)
        zip_code: str | None = getattr(request, "zip_code", None)
        url: str | None = getattr(request, "url", None)

        logger.info(f"[GeminiSearchService] Starting research for: {company_name!r}")

        target_context = self._build_context(company_name, country, zip_code, url)

        # ── Step 1: Deep Research (The "Gatherer") ────────────────────────────
        research_text = self._gather(target_context, company_name)

        # ── Step 2: Extraction (The "Extractor") ──────────────────────────────
        result: T = self._extract(research_text, company_name, model_class)

        logger.info(f"[GeminiSearchService] Extraction complete for: {company_name!r}")
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context(
        company_name: str,
        country: str | None,
        zip_code: str | None,
        url: str | None,
    ) -> str:
        """Assemble a concise, pipe-separated context string from available fields."""
        parts = [f"Company: {company_name}"]
        if country:
            parts.append(f"Country: {country}")
        if zip_code:
            parts.append(f"Zip Code: {zip_code}")
        if url:
            parts.append(f"URL: {url}")
        return " | ".join(parts)

    def _gather(self, target_context: str, company_name: str) -> str:
        """
        Step 1 — Run a grounded web-search via Gemini and return the raw
        research text.

        The ``google_search`` tool is passed in the request config so Gemini
        can issue real-time search queries and ground its response in live web
        data.  The returned text is a rich markdown summary with inline source
        citations.
        """
        research_prompt = (
            f"Deep research task: Find contact information for the following target:\n"
            f"{target_context}\n\n"
            "GOALS:\n"
            "1. Prioritize Human Resources (HR) contact info.\n"
            "2. If HR is unavailable, find General/Corporate contact info.\n"
            "3. Look specifically for a Fax number.\n"
            "4. Find ALL available contact details: phones, emails, faxes, addresses.\n"
            "5. Provide the official website URL if found.\n"
            "Be thorough and search multiple sources."
        )

        grounding_tool = genai_types.Tool(
            google_search=genai_types.GoogleSearch()
        )
        research_config = genai_types.GenerateContentConfig(
            tools=[grounding_tool],
        )

        logger.debug(f"[GeminiSearchService] Step 1: issuing grounded search for {company_name!r}")
        client = self._require_client()
        response = client.models.generate_content(
            model=_GEMINI_MODEL,
            contents=research_prompt,
            config=research_config,
        )
        raw_text: str = response.text or ""
        logger.debug(
            f"[GeminiSearchService] Step 1: received {len(raw_text)} chars of research"
        )
        return raw_text

    def _extract(
        self,
        research_text: str,
        company_name: str,
        model_class: Type[T],
    ) -> T:
        """
        Step 2 — Distil the raw research into a structured Pydantic model.

        Uses Gemini's native structured-output capability: the response is
        constrained to a JSON object that strictly matches *model_class*'s
        schema.  The SDK auto-parses it into the model via ``.parsed``.
        """
        extraction_prompt = (
            f"Based on the following research about '{company_name}':\n\n"
            f"{research_text}\n\n"
            "TASK: Extract the contact information into the required JSON schema.\n"
            "STRICT RULES:\n"
            "1. Extract ALL valid emails, phone numbers, fax numbers, and physical "
            "addresses you can find into their respective arrays.\n"
            "2. Format each distinct address as a single, fully readable string "
            "(e.g., '123 Main St, City, State 12345, Country').\n"
            f"3. Fill the 'company_name' key with the exact target name: {company_name}.\n"
            "4. Fill 'official_site' with the primary official website URL if found, "
            "otherwise leave it as an empty string.\n"
            "5. Return ONLY the extracted data values — do NOT return the schema itself.\n"
            "6. If a field has no data, use an empty array [] or empty string \"\"."
        )

        extraction_config = genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=model_class,
        )

        logger.debug(
            f"[GeminiSearchService] Step 2: issuing structured extraction for {company_name!r}"
        )
        client = self._require_client()
        response = client.models.generate_content(
            model=_GEMINI_MODEL,
            contents=extraction_prompt,
            config=extraction_config,
        )

        # `response.parsed` is automatically populated by the SDK when
        # `response_schema` is a Pydantic model class.
        if response.parsed is not None:
            logger.debug("[GeminiSearchService] Step 2: using SDK-parsed Pydantic object")
            return response.parsed  # type: ignore[return-value]

        # Fallback: the SDK may return raw JSON text on some model/version
        # combinations.  Parse manually to guarantee a validated result.
        logger.warning(
            "[GeminiSearchService] Step 2: .parsed was None — falling back to "
            "manual JSON parsing"
        )
        import json

        raw_json_text = self._strip_markdown_fences(response.text or "{}")
        data = json.loads(raw_json_text)
        return model_class.model_validate(data)

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """
        Remove Markdown code fences (```json ... ```) that occasionally wrap
        the model's JSON output in the fallback path.
        """
        return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
