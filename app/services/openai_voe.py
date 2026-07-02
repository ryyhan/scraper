"""
OpenAI VOE Verification Service
--------------------------------
Verifies whether a named individual holds the stated job title at the given
company using a two-step OpenAI Responses API pipeline:

    Step 1 – "Investigator"
        A ``responses.create`` call with the ``web_search`` tool enabled.
        OpenAI performs live web searches across LinkedIn, company directories,
        press releases, news articles, and public records to gather evidence
        for or against the employment claim.

    Step 2 – "Analyst"
        A second ``responses.create`` call (no tools) with
        ``text={"format": {"type": "json_object"}}`` and a strict prompt
        that maps the raw evidence onto a ``VoeVerificationResult`` schema.

Score rubric (baked into the prompt — mirrors the Gemini VOE service exactly
so that scores from both providers are directly comparable):
    9–10  VERIFIED      – Direct confirmation found.
    6–8   LIKELY        – Indirect or partial evidence.
    3–5   UNVERIFIED    – Person and company exist but no clear link found.
    0–2   CONTRADICTED  – Evidence contradicts claim or person has no footprint.

Design decisions
~~~~~~~~~~~~~~~~
* Mirrors ``VoeVerificationService`` (Gemini) structure method-for-method so
  both implementations are composable and testable the same way.
* Uses the same ``_SCORE_RUBRIC`` constant wording so scores are calibrated
  identically across providers — this matters when ``provider="both"`` and
  the caller compares the two confidence scores.
* ``max_retries=2`` on the OpenAI client activates the SDK's inner retry layer
  (reads Retry-After headers on 429/5xx) before tenacity's outer layer fires.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from openai import OpenAI
from loguru import logger

from app.core.config import settings
from app.models import VoeRequest, VoeVerificationResult
from app.services._retry import retry_openai

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_OPENAI_MODEL: str = settings.OPENAI_MODEL
"""OpenAI model used for both research and structured-extraction calls.
Override via OPENAI_MODEL in your .env file (e.g. OPENAI_MODEL=gpt-4o).
"""

# "high" context pulls the deepest page content available, surfacing
# details buried in footers and PDFs that "medium" (the default) misses.
_WEB_SEARCH_TOOL: dict = {
    "type": "web_search",
    "search_context_size": "high",
}

_SCORE_RUBRIC: str = """\
Score rubric (apply strictly):
  9–10  VERIFIED      – Direct confirmation found: LinkedIn profile, official
                        company bio, press release, or news article explicitly
                        names the person in the stated role at this company.
  6–8   LIKELY        – Indirect or partial evidence supports the claim (e.g.
                        person and company are connected but the exact title or
                        current status is ambiguous).
  3–5   UNVERIFIED    – Company and person both exist publicly but no clear
                        link between them was found across searched sources.
  0–2   CONTRADICTED  – Evidence actively contradicts the claim (person left
                        the company, holds a different title, or their public
                        profile lists a different employer), OR the person
                        leaves no public footprint at all.\
"""


class OpenAIVoeService:
    """
    Verifies employment claims using OpenAI-powered web research.

    Usage::

        service = OpenAIVoeService()
        result: VoeVerificationResult = service.verify(request)
    """

    def __init__(self) -> None:
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            logger.warning(
                "OPENAI_API_KEY is not configured – OpenAI VOE verification calls will fail."
            )
        # max_retries=2: SDK inner-layer reads Retry-After headers on 429/5xx
        # before our tenacity outer-layer (in _retry.py) takes over.
        self._client: OpenAI | None = (
            OpenAI(api_key=api_key, max_retries=2) if api_key else None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(self, request: VoeRequest) -> VoeVerificationResult:
        """
        Run the two-step employment-verification pipeline for *request*.

        Args:
            request: A :class:`VoeRequest` carrying the person's name,
                     job title, company, and optional location context.

        Returns:
            A validated :class:`VoeVerificationResult` with a calibrated
            confidence score, verdict label, evidence summary, and sources.

        Raises:
            RuntimeError:          if OPENAI_API_KEY is not set.
            openai.OpenAIError:    on unrecoverable OpenAI API errors.
            pydantic.ValidationError: if the model response cannot be
                                      coerced into the result schema.
        """
        client = self._require_client()

        subject_context = self._build_context(request)
        logger.info(
            f"[OpenAIVoeService] Starting verification for "
            f"{request.full_name!r} @ {request.company!r}"
        )

        # ── Step 1: Web Research (The "Investigator") ──────────────────────
        raw_evidence = self._investigate(client, subject_context, request)

        # ── Step 2: Structured Scoring (The "Analyst") ─────────────────────
        result = self._analyse(client, raw_evidence, request)

        logger.info(
            f"[OpenAIVoeService] Verification complete for {request.full_name!r}: "
            f"score={result.confidence_score} verdict={result.verdict!r}"
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_client(self) -> OpenAI:
        """Return the configured OpenAI client or raise clearly if no key is set."""
        if self._client is None:
            raise RuntimeError(
                "OpenAIVoeService is not configured: OPENAI_API_KEY is missing. "
                "Set it in your .env file and restart the server."
            )
        return self._client

    @staticmethod
    def _build_context(request: VoeRequest) -> str:
        """Build a concise, pipe-separated context string from the request fields."""
        parts = [
            f"Person: {request.full_name}",
            f"Job Title: {request.job_title}",
            f"Company: {request.company}",
        ]
        if request.city:
            parts.append(f"City: {request.city}")
        if request.zip_code:
            parts.append(f"Zip Code: {request.zip_code}")
        if request.country:
            parts.append(f"Country: {request.country}")
        return " | ".join(parts)

    def _investigate(
        self,
        client: OpenAI,
        subject_context: str,
        request: VoeRequest,
    ) -> str:
        """
        Step 1 — Grounded web research via OpenAI web_search tool.

        Issues live web searches to collect evidence about whether
        *request.full_name* holds *request.job_title* at *request.company*.
        Returns a rich, cited research summary.
        """
        prompt = (
            f"Employment verification research task:\n"
            f"{subject_context}\n\n"
            "YOUR GOAL: Determine whether this person currently holds (or has\n"
            "recently held) the stated job title at the stated company.\n\n"
            "RESEARCH INSTRUCTIONS:\n"
            "1. Search LinkedIn, the company's official website, press releases,\n"
            "   news articles, SEC filings, and professional directories.\n"
            "2. Look for the person's name and title appearing together in any\n"
            "   credible public source.\n"
            "3. Note the EXACT source URL or publication name for each piece of\n"
            "   evidence you find.\n"
            "4. If you find contradictory information (e.g. person left the\n"
            "   company, holds a different title, or works elsewhere), document\n"
            "   it explicitly — contradictions are equally important evidence.\n"
            "5. If the person has a common name, narrow your search using the\n"
            "   company name and location context provided above.\n"
            "6. Be thorough. Search at least 3–5 distinct sources before\n"
            "   concluding that no evidence exists.\n\n"
            "Summarise ALL evidence found, including source URLs."
        )

        logger.debug(
            f"[OpenAIVoeService] Step 1: grounded search for "
            f"{request.full_name!r} @ {request.company!r}"
        )

        @retry_openai()
        def _call():
            return client.responses.create(
                model=_OPENAI_MODEL,
                tools=[_WEB_SEARCH_TOOL],
                input=prompt,
            )

        response = _call()
        raw_text: str = response.output_text or ""
        logger.debug(
            f"[OpenAIVoeService] Step 1: received {len(raw_text)} chars of evidence"
        )
        return raw_text

    def _analyse(
        self,
        client: OpenAI,
        raw_evidence: str,
        request: VoeRequest,
    ) -> VoeVerificationResult:
        """
        Step 2 — Structured scoring.

        Passes the raw evidence to the model with a strict JSON schema and a
        calibrated scoring rubric. Returns a validated
        :class:`VoeVerificationResult` instance.
        """
        schema = json.dumps(VoeVerificationResult.model_json_schema(), indent=2)
        prompt = (
            f"Employment verification analysis task:\n\n"
            f"SUBJECT:\n"
            f"  Name:      {request.full_name}\n"
            f"  Job Title: {request.job_title}\n"
            f"  Company:   {request.company}\n\n"
            f"EVIDENCE GATHERED:\n"
            f"{raw_evidence}\n\n"
            f"SCORING RUBRIC:\n"
            f"{_SCORE_RUBRIC}\n\n"
            "TASK: Based solely on the evidence above, produce a structured\n"
            f"JSON result matching EXACTLY this schema:\n{schema}\n\n"
            "STRICT RULES:\n"
            "1. `confidence_score` MUST be a float between 0.0 and 10.0,\n"
            "   calibrated against the rubric above. Do NOT default to 5.0.\n"
            "2. `verdict` MUST be exactly one of: VERIFIED, LIKELY, UNVERIFIED,\n"
            "   CONTRADICTED — chosen consistently with `confidence_score`.\n"
            "3. `evidence_summary` MUST be 2–3 sentences explaining what you\n"
            "   found (or did not find) and why you assigned that score.\n"
            "4. `sources_found` MUST list every URL or named publication you\n"
            "   cited above. If none were found, return an empty array.\n"
            f"5. Fill `full_name` with: {request.full_name}\n"
            f"6. Fill `company` with: {request.company}\n"
            f"7. Fill `job_title` with: {request.job_title}\n"
            "8. Return ONLY the extracted data values — do NOT return the schema."
        )

        logger.debug(
            f"[OpenAIVoeService] Step 2: structured scoring for {request.full_name!r}"
        )

        @retry_openai()
        def _call():
            return client.responses.create(
                model=_OPENAI_MODEL,
                input=prompt,
                text={"format": {"type": "json_object"}},
            )

        response = _call()

        raw_json = self._strip_markdown_fences(response.output_text or "")

        # Guard: if the model returned nothing or only whitespace, return a
        # deterministic UNVERIFIED result rather than letting model_validate({})
        # raise an opaque pydantic.ValidationError citing missing required fields.
        if not raw_json:
            logger.warning(
                f"[OpenAIVoeService] Step 2 returned empty output for "
                f"{request.full_name!r} — returning UNVERIFIED default."
            )
            return VoeVerificationResult(
                full_name=request.full_name,
                company=request.company,
                job_title=request.job_title,
                confidence_score=0.0,
                verdict="UNVERIFIED",
                evidence_summary=(
                    "The structured scoring step returned an empty response. "
                    "Evidence gathered in Step 1 could not be scored. "
                    "Manual verification is recommended."
                ),
                sources_found=[],
            )

        data = json.loads(raw_json)
        return VoeVerificationResult.model_validate(data)

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Remove Markdown code fences that occasionally wrap the model's JSON output.

        Note: re.MULTILINE is intentionally NOT used here.  Without it, ``^``
        and ``$`` anchor to the very start and end of the full string, which is
        the correct behaviour — we only want to strip the outermost fence pair,
        not every line that happens to begin or end with triple backticks.
        """
        return re.sub(r"^```(?:json)?\s*\n?|\n?\s*```$", "", text.strip())
