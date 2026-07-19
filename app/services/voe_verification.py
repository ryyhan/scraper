"""
VOE Verification Service
------------------------
Verifies whether a named individual holds the stated job title at the given
company by issuing a two-step Gemini pipeline:

    Step 1 – "Investigator"
        A ``generate_content`` call with the ``google_search`` grounding tool
        enabled.  Gemini performs live web searches across LinkedIn, company
        directories, press releases, news articles, and public records to
        gather evidence for or against the employment claim.

    Step 2 – "Analyst"
        A second ``generate_content`` call (no tools) with
        ``response_mime_type="application/json"`` and ``response_schema``
        pointing at ``VoeVerificationResult``.  The model scores its own
        findings against a well-defined rubric and returns a structured
        Pydantic object.

Score rubric (baked into the prompt for calibration):
    9–10  Directly confirmed  – LinkedIn profile/company bio/HR record shows
          the person in the exact role at that company.
    6–8   Likely              – Indirect or partial evidence (e.g. mentioned
          in a press release or a related article, but title not explicit).
    3–5   Inconclusive        – Company and person both exist publicly but no
          clear link between them was found.
    0–2   Contradicted/None   – Evidence actively contradicts the claim, or
          no trace of the person exists at all.

Design decisions
~~~~~~~~~~~~~~~~
* Mirrors the ``GeminiSearchService`` two-step pattern exactly so this
  service is composable and testable the same way.
* ``VoeVerificationResult`` is the *only* schema the extractor can return,
  keeping the route handler thin and type-safe.
* The model constant is defined at module level for easy swapping.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from google import genai
from google.genai import types as genai_types
from loguru import logger

from app.core.config import settings
from app.models import VoeRequest, VoeVerificationResult, TokenUsage, ProviderTokenUsage
from app.services._retry import retry_gemini
from app.services._token_utils import gemini_usage

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_GEMINI_MODEL: str = settings.GEMINI_MODEL
"""Gemini model used for both research and structured-extraction calls.

Override via GEMINI_MODEL in your .env file (e.g. GEMINI_MODEL=gemini-2.5-pro).
Defaults to the value in Settings.GEMINI_MODEL (currently "gemini-2.5-flash-lite").
"""

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


class VoeVerificationService:
    """
    Verifies employment claims using Gemini-powered web research.

    Usage::

        service = VoeVerificationService()
        result: VoeVerificationResult = service.verify(request)
    """

    def __init__(self) -> None:
        api_key = settings.GEMINI_API_KEY
        if not api_key:
            logger.warning(
                "GEMINI_API_KEY is not configured – VOE verification calls will fail."
            )
        # Dual-layer retry (inner): SDK retries 2x reading Retry-After headers
        # before tenacity's outer-layer takes over.
        # HttpOptions.timeout is in MILLISECONDS.
        from google.genai import types as _genai_types
        self._client: genai.Client | None = (
            genai.Client(
                api_key=api_key,
                http_options=_genai_types.HttpOptions(
                    timeout=60_000,
                    retry_options=_genai_types.HttpRetryOptions(attempts=2),
                ),
            )
            if api_key else None
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
            RuntimeError:                if GEMINI_API_KEY is not set.
            google.genai.errors.APIError: on unrecoverable Gemini API errors.
            pydantic.ValidationError:    if the model response cannot be
                                         coerced into the result schema.
        """
        client = self._require_client()

        subject_context = self._build_context(request)
        logger.info(
            f"[VoeVerificationService] Starting verification for "
            f"{request.full_name!r} @ {request.company!r}"
        )

        # ── Step 1: Web Research (The "Investigator") ──────────────────────
        raw_evidence, investigate_usage = self._investigate(client, subject_context, request)

        # Guard: if grounding returned nothing (safety filter, blocked content,
        # or a transient API issue), skip Step 2 to avoid a wasted LLM call.
        # Return a deterministic UNVERIFIED result so the caller always gets a
        # well-formed response rather than an opaque error.
        if not raw_evidence.strip():
            logger.warning(
                f"[VoeVerificationService] Step 1 returned empty evidence for "
                f"{request.full_name!r} @ {request.company!r} — "
                "likely a safety filter or blocked content. Returning UNVERIFIED."
            )
            result = VoeVerificationResult(
                full_name=request.full_name,
                company=request.company,
                job_title=request.job_title,
                confidence_score=0.0,
                verdict="UNVERIFIED",
                evidence_summary=(
                    "No evidence could be gathered from web sources. "
                    "The search may have been blocked or returned no results. "
                    "Manual verification is recommended."
                ),
                sources_found=[],
            )
            result.token_usage = TokenUsage(
                openai=None,
                gemini=investigate_usage,
                grand_total=investigate_usage,
            )
            return result

        # ── Step 2: Structured Scoring (The "Analyst") ────────────────────
        result, analyse_usage = self._analyse(client, raw_evidence, request)

        # ── Attach aggregated token usage ─────────────────────────────────
        total_gemini_usage = investigate_usage + analyse_usage
        result.token_usage = TokenUsage(
            openai=None,
            gemini=total_gemini_usage,
            grand_total=total_gemini_usage,
        )

        logger.info(
            f"[VoeVerificationService] Verification complete for {request.full_name!r}: "
            f"score={result.confidence_score} verdict={result.verdict!r} | "
            f"tokens: input={total_gemini_usage.input_tokens}, "
            f"output={total_gemini_usage.output_tokens}, "
            f"total={total_gemini_usage.total_tokens}"
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_client(self) -> genai.Client:
        """Return the configured Gemini client or raise clearly."""
        if self._client is None:
            raise RuntimeError(
                "VoeVerificationService is not configured: GEMINI_API_KEY is missing. "
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
        if request.state:
            parts.append(f"State: {request.state}")
        if request.zip_code:
            parts.append(f"Zip Code: {request.zip_code}")
        if request.country:
            parts.append(f"Country: {request.country}")
        return " | ".join(parts)

    def _investigate(
        self,
        client: genai.Client,
        subject_context: str,
        request: VoeRequest,
    ) -> tuple[str, ProviderTokenUsage]:
        """
        Step 1 — Grounded web research.

        Issues a live Google Search via Gemini to collect evidence about
        whether *request.full_name* holds *request.job_title* at
        *request.company*.  Returns a rich, cited research summary.

        Returns a tuple of (research_text, token_usage_for_this_step).
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

        grounding_tool = genai_types.Tool(
            google_search=genai_types.GoogleSearch()
        )
        config = genai_types.GenerateContentConfig(tools=[grounding_tool])

        logger.debug(
            f"[VoeVerificationService] Step 1: grounded search for "
            f"{request.full_name!r} @ {request.company!r}"
        )

        @retry_gemini()
        def _call() -> "genai.types.GenerateContentResponse":  # type: ignore[name-defined]
            return client.models.generate_content(
                model=_GEMINI_MODEL,
                contents=prompt,
                config=config,
            )

        response = _call()
        raw_text: str = response.text or ""
        step_usage = gemini_usage(response.usage_metadata)
        logger.debug(
            f"[VoeVerificationService] Step 1: received {len(raw_text)} chars of evidence "
            f"(tokens={step_usage.total_tokens})"
        )
        return raw_text, step_usage

    def _analyse(
        self,
        client: genai.Client,
        raw_evidence: str,
        request: VoeRequest,
    ) -> tuple[VoeVerificationResult, ProviderTokenUsage]:
        """
        Step 2 — Structured scoring.

        Passes the raw evidence to Gemini with a strict JSON schema and a
        calibrated scoring rubric.  Returns a validated
        :class:`VoeVerificationResult` instance.

        Returns a tuple of (result, token_usage_for_this_step).
        """
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
            "JSON result following the required schema.\n\n"
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
            "8. Return ONLY the extracted values — do NOT return the schema."
        )

        config = genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=VoeVerificationResult,
        )

        logger.debug(
            f"[VoeVerificationService] Step 2: structured scoring for {request.full_name!r}"
        )

        @retry_gemini()
        def _call() -> "genai.types.GenerateContentResponse":  # type: ignore[name-defined]
            return client.models.generate_content(
                model=_GEMINI_MODEL,
                contents=prompt,
                config=config,
            )

        response = _call()
        step_usage = gemini_usage(response.usage_metadata)

        # Prefer SDK-parsed Pydantic object (zero boilerplate)
        if response.parsed is not None:
            logger.debug("[VoeVerificationService] Step 2: using SDK-parsed Pydantic object")
            return response.parsed, step_usage  # type: ignore[return-value]

        # Fallback: manual JSON parse (handles edge-case model versions)
        logger.warning(
            "[VoeVerificationService] Step 2: .parsed was None — falling back to "
            "manual JSON parsing"
        )
        raw_json = self._strip_markdown_fences(response.text or "{}")
        data = json.loads(raw_json)
        return VoeVerificationResult.model_validate(data), step_usage

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Remove Markdown code fences that occasionally wrap the model's JSON output.

        Note: re.MULTILINE is intentionally NOT used here.  Without it, ``^``
        and ``$`` anchor to the very start and end of the full string, which is
        the correct behaviour — we only want to strip the outermost fence pair,
        not every line that happens to begin or end with triple backticks.
        """
        return re.sub(r"^```(?:json)?\s*\n?|\n?\s*```$", "", text.strip())
