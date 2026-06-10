"""
Background Check PDF Parser Service
-------------------------------------
Two-stage pipeline that extracts structured fields from background check /
screening reports (HireRight, Sterling, Checkr, Accurate, etc.).

Stage 1 – Raw Extraction
    Delegates to ``PdfExtractorService`` to obtain the full document text via
    the selected vision LLM (Gemini Files API or OpenAI gpt-4o-mini vision).
    This stage is provider-aware and handles all PDF rendering/upload logic.

Stage 2 – Structured Field Extraction
    Runs a focused, text-only LLM call on the raw text from Stage 1 to pull
    exactly the 7 target fields.  Using text for Stage 2 (instead of
    re-uploading the PDF) is cheaper, faster, and lets us write a tight,
    field-specific prompt without vision overhead.

    Gemini path  – ``generate_content`` with ``response_mime_type="application/json"``
                   and ``response_schema=BgCheckFields`` for native structured output.
    OpenAI path  – ``chat.completions.create`` with ``response_format=json_object``
                   and ``model_validate`` to hydrate the Pydantic model.

Fields extracted
~~~~~~~~~~~~~~~~
  file_number    – Case/order/reference ID on the report
  employee_name  – Full name of the subject / applicant
  date_of_birth  – Subject's DOB, normalised to YYYY-MM-DD
  requested_by   – Requester name or organisation
  employer_name  – Employer / client company named on the report
  position       – Job title or position applied for
  report_date    – Report generation / order date, normalised to YYYY-MM-DD
  status         – Overall report status (Clear, Consider, Adverse Action, …)

Design decisions
~~~~~~~~~~~~~~~~
* Two-stage separation keeps concerns clean: Stage 1 owns PDF reading,
  Stage 2 owns domain-specific parsing.
* Both stages use the same ``provider`` so every call goes to the same
  vendor — avoids mixing credentials and latency profiles.
* All missing fields return ``""`` rather than ``null`` for consistent
  downstream handling.
* ``tenacity`` retry with exponential backoff guards Stage 2 API calls.
* ``loguru`` is used for structured, level-appropriate logging throughout.
"""

from __future__ import annotations

import json
import re
import logging as _logging
from typing import Literal

from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from app.core.config import settings
from app.services.pdf_extractor import PdfExtractorService

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GEMINI_MODEL = settings.GEMINI_MODEL
_OPENAI_MODEL = settings.OPENAI_MODEL

# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

_FIELD_EXTRACTION_PROMPT = """\
You are a specialist data-extraction agent for background check / employee
screening reports. These are issued by vendors such as HireRight, Sterling,
Checkr, Accurate, First Advantage, and similar providers.

DOCUMENT TEXT:
{text}

TASK:
Extract the following fields from the document text above.
Return a single, flat JSON object with EXACTLY these 7 keys.

FIELD DEFINITIONS AND LABEL SYNONYMS
(look for any of the listed synonyms in the document):

1. "file_number"
   – The unique case, order, or reference identifier for this report.
   – Look for: "File No", "File Number", "Case ID", "Case Number",
     "Reference Number", "Reference #", "Order ID", "Order Number",
     "Report #", "Report Number", "Tracking Number", "Request ID".

2. "employee_name"
   – The full name of the subject/applicant/employee being screened.
   – Look for: "Applicant", "Subject", "Candidate", "Employee",
     "Employee Name", "Applicant Name", "Subject Name", "Name of Applicant".

3. "date_of_birth"
   – The subject's date of birth.
   – Look for: "DOB", "Date of Birth", "Birth Date", "Date of Birth (DOB)".
   – NORMALISE to ISO 8601 format: YYYY-MM-DD.
     Examples: "November 15, 1985" → "1985-11-15", "11/15/1985" → "1985-11-15".

4. "requested_by"
   – The person or organisation who ordered / requested this report.
   – Look for: "Requested By", "Ordered By", "Client", "Client Name",
     "Requestor", "Employer/Client", "Account Name", "Organization".

5. "employer_name"
   – The employer or company named on the report (may differ from requestor).
   – Look for: "Employer", "Company", "Employer Name", "Company Name",
     "Organization", "Hiring Company", "Client Company".

6. "position"
   – The job title or position the applicant is being considered for.
   – Look for: "Position", "Job Title", "Position Applied For",
     "Role", "Title", "Position Title", "Job Position".

7. "report_date"
   – The date the report was generated, completed, or ordered.
   – Look for: "Report Date", "Date", "Order Date", "Ordered Date",
     "Completed Date", "Date Completed", "Date Ordered", "Ordered On",
     "Generated On", "Processed Date".
   – NORMALISE to ISO 8601 format: YYYY-MM-DD.

8. "status"
   – The overall result or adjudication status of the background check.
   – Look for: "Status", "Overall Status", "Result", "Summary Status",
     "Adjudication", "Overall Result", "Final Status", "Report Status".
   – Return the value exactly as it appears (e.g., "Clear", "Consider",
     "Adverse Action", "Complete", "Meets Standards", "Review", "Pending").

VISUAL SELECTION NOTATION (produced by the OCR stage)
──────────────────────────────────────────────────────
The raw text above may contain inline tags that capture visual marks made on
the form (ticks, crosses, circles, scribbles, filled bubbles, etc.).

  [SELECTED]     – the preceding option was visually marked / chosen
  [NOT SELECTED] – the preceding option was present but left blank
  [UNCLEAR]      – the mark was ambiguous; do not infer intent

When resolving any field whose value is determined by one of these marks:
• Use the label immediately BEFORE [SELECTED] as the field value.
• Ignore all labels that are followed by [NOT SELECTED] or [UNCLEAR].
• If multiple options carry [SELECTED] and the field is normally single-value,
  return a comma-separated list of the selected labels.
• If every option carries [UNCLEAR] or no [SELECTED] exists, return "".

Example — employment status field in the raw text:
  "Employment Status: Full-time [NOT SELECTED]  Part-time [SELECTED]  Contract [NOT SELECTED]"
  → status field value = "Part-time"

RULES:
- Return ONLY the JSON object — no explanation, no markdown fences.
- If a field cannot be found anywhere in the document, return "" for that key.
- Do NOT invent or infer values. Only extract what is explicitly present.
- Dates MUST be in YYYY-MM-DD format or "".
- All other values should be plain strings.

REQUIRED OUTPUT FORMAT (example):
{{
  "file_number": "BGC-2024-00123",
  "employee_name": "John Michael Doe",
  "date_of_birth": "1985-04-22",
  "requested_by": "Acme Corp HR Department",
  "employer_name": "Acme Corporation",
  "position": "Senior Software Engineer",
  "report_date": "2024-11-15",
  "status": "Clear"
}}
"""


# ---------------------------------------------------------------------------
# Retry helpers (reuse same strategy as pdf_extractor.py)
# ---------------------------------------------------------------------------

def _retry_gemini():
    """tenacity retry decorator for Gemini Stage-2 calls."""
    try:
        from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
        exc_types: tuple = (ResourceExhausted, ServiceUnavailable)
    except ImportError:
        exc_types = (Exception,)

    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type(exc_types),
        before_sleep=before_sleep_log(_logging.getLogger(__name__), _logging.WARNING),
        reraise=True,
    )


def _retry_openai():
    """tenacity retry decorator for OpenAI Stage-2 calls."""
    try:
        from openai import RateLimitError, APIStatusError
        exc_types: tuple = (RateLimitError, APIStatusError)
    except ImportError:
        exc_types = (Exception,)

    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type(exc_types),
        before_sleep=before_sleep_log(_logging.getLogger(__name__), _logging.WARNING),
        reraise=True,
    )


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class BgCheckParserService:
    """
    Orchestrates the two-stage background-check PDF parsing pipeline.

    Stage 1 → ``PdfExtractorService.extract()``   (raw text via vision LLM)
    Stage 2 → ``_extract_fields_gemini / _openai`` (structured JSON output)

    Usage::

        service = BgCheckParserService()
        result = service.parse(pdf_bytes, "report.pdf", "gemini")
        # result is a dict matching BgCheckFields
    """

    # ── Public API ──────────────────────────────────────────────────────────

    def parse(
        self,
        pdf_bytes: bytes,
        filename: str,
        provider: Literal["gemini", "openai"],
    ) -> dict:
        """
        Run the full two-stage pipeline and return a plain dict matching
        ``BgCheckFields``.

        Args:
            pdf_bytes: Raw PDF bytes (already validated by the caller).
            filename:  Original filename (for logging).
            provider:  ``"gemini"`` or ``"openai"``.

        Returns:
            Dict with keys: file_number, employee_name, date_of_birth,
            requested_by, employer_name, position, report_date, status.

        Raises:
            RuntimeError: If a required API key is not configured.
            Exception:    Propagated from the LLM SDK on unrecoverable errors.
        """
        logger.info(
            f"[BgCheckParser] Stage 1 starting: file={filename!r}, "
            f"provider={provider!r}"
        )

        # ── Stage 1: raw text extraction ────────────────────────────────────
        extractor = PdfExtractorService()
        raw_text, page_count = extractor.extract(pdf_bytes, filename, provider)

        logger.info(
            f"[BgCheckParser] Stage 1 complete: ~{page_count} pages, "
            f"{len(raw_text)} chars extracted"
        )

        # ── Stage 2: structured field extraction ────────────────────────────
        logger.info(f"[BgCheckParser] Stage 2 starting: provider={provider!r}")

        if provider == "gemini":
            fields = self._extract_fields_gemini(raw_text)
        else:
            fields = self._extract_fields_openai(raw_text)

        logger.info(
            f"[BgCheckParser] Stage 2 complete: "
            f"employee={fields.get('employee_name', '')!r}, "
            f"status={fields.get('status', '')!r}"
        )

        return fields

    # ── Stage 2 — Gemini path ───────────────────────────────────────────────

    def _require_gemini_client(self):
        """Return a configured genai.Client or raise clearly."""
        from google import genai

        api_key = settings.GEMINI_API_KEY
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. "
                "Set it in your .env file and restart the server."
            )
        return genai.Client(api_key=api_key)

    def _extract_fields_gemini(self, raw_text: str) -> dict:
        """
        Use Gemini's native structured JSON output to extract the 7 fields.

        Mirrors the pattern used in ``GeminiSearchService._extract()``:
        ``response_mime_type="application/json"`` + ``response_schema`` forces
        Gemini to return a valid JSON object that matches the Pydantic schema.
        """
        from google.genai import types as genai_types

        # Import here to avoid circular imports at module load time
        from app.models import BgCheckFields

        client = self._require_gemini_client()
        prompt = _FIELD_EXTRACTION_PROMPT.format(text=raw_text)

        @_retry_gemini()
        def _call() -> dict:
            response = client.models.generate_content(
                model=_GEMINI_MODEL,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=BgCheckFields,
                    temperature=0.0,
                ),
            )

            # `response.parsed` is populated when response_schema is a Pydantic class
            if response.parsed is not None:
                logger.debug("[BgCheckParser][Gemini] Using SDK-parsed Pydantic object")
                return response.parsed.model_dump()

            # Fallback: SDK returned raw JSON text on some model/version combos
            logger.warning(
                "[BgCheckParser][Gemini] .parsed was None — falling back to "
                "manual JSON parsing"
            )
            raw_json = _strip_markdown_fences(response.text or "{}")
            data = json.loads(raw_json)
            return BgCheckFields.model_validate(data).model_dump()

        return _call()

    # ── Stage 2 — OpenAI path ───────────────────────────────────────────────

    def _require_openai_client(self):
        """Return a configured OpenAI client or raise clearly."""
        from openai import OpenAI

        api_key = settings.OPENAI_API_KEY
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured. "
                "Set it in your .env file and restart the server."
            )
        return OpenAI(api_key=api_key)

    def _extract_fields_openai(self, raw_text: str) -> dict:
        """
        Use OpenAI's json_object response format to extract the 7 fields.

        Mirrors the pattern in ``LLMService.extract_contact_info()``.
        """
        from app.models import BgCheckFields

        client = self._require_openai_client()
        prompt = _FIELD_EXTRACTION_PROMPT.format(text=raw_text)

        @_retry_openai()
        def _call() -> dict:
            response = client.chat.completions.create(
                model=_OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise data-extraction assistant "
                            "specialising in background check reports. "
                            "Output valid JSON only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            raw_json = response.choices[0].message.content or "{}"
            data = json.loads(raw_json)
            return BgCheckFields.model_validate(data).model_dump()

        return _call()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _strip_markdown_fences(text: str) -> str:
    """Remove ```json … ``` fences that occasionally wrap Gemini's output."""
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
