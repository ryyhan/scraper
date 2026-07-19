"""
Gemini Find POE Service
-----------------------
Discovers the most likely employer(s) of a named individual using a two-step
Gemini pipeline:

    Step 1 – "Investigator"
        A ``generate_content`` call with the ``google_search`` grounding tool
        enabled.  Gemini performs live web searches across LinkedIn, company
        websites, press releases, news articles, and professional directories
        to gather evidence about where the named person works.

    Step 2 – "Analyst"
        A second ``generate_content`` call (no tools) with
        ``response_mime_type="application/json"`` and ``response_schema``
        pointing at a structured result.  The model scores each employer
        candidate against a well-defined rubric and returns a ranked list.

Confidence score rubric (baked into the prompt):
    9–10  CONFIRMED   – Direct confirmation found (LinkedIn, company bio,
                        press release, news article) naming this person at
                        this employer in a specific role.
    6–8   LIKELY      – Strong indirect evidence (multiple corroborating
                        sources, or one authoritative source without a
                        current role title).
    3–5   POSSIBLE    – Weak or ambiguous connection found — possible former
                        employer, or a single non-authoritative mention.
    0–2   UNLIKELY    – Only tenuous or coincidental mentions found; the
                        person is most probably NOT at this employer.

Design decisions
~~~~~~~~~~~~~~~~
* Mirrors ``VoeVerificationService`` structure method-for-method so both
  implementations are composable and testable the same way.
* Returns a *ranked* list of ``CompanyCandidate`` objects so callers can
  inspect all discovered employers, not just the top hit.
* ``best_match`` is computed from the returned candidates list (the one
  with the highest ``confidence_score``) to ensure internal consistency.
"""

from __future__ import annotations

import json
from typing import Optional

from google import genai
from google.genai import types as genai_types
from loguru import logger

from app.core.config import settings
from app.models import (
    FindPoeRequest,
    FindPoeResult,
    CompanyCandidate,
    TokenUsage,
    ProviderTokenUsage,
)
from app.services._retry import retry_gemini
from app.services._token_utils import gemini_usage

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_GEMINI_MODEL: str = settings.GEMINI_MODEL
"""Gemini model used for both research and structured-extraction calls.

Override via GEMINI_MODEL in your .env file (e.g. GEMINI_MODEL=gemini-2.5-pro).
"""

_SCORE_RUBRIC: str = """\
Confidence score rubric (apply strictly):
  9–10  CONFIRMED – Direct confirmation found: LinkedIn profile, official
                    company bio, press release, or news article explicitly
                    names this person as a current or very recent employee
                    at this employer.
  6–8   LIKELY    – Strong indirect evidence supports the connection (e.g.
                    the person is mentioned alongside the company in multiple
                    credible sources, but their exact current role or tenure
                    is ambiguous).
  3–5   POSSIBLE  – Weak or ambiguous evidence — a single non-authoritative
                    mention, a former/past employment connection, or a name
                    collision that cannot be ruled out.
  0–2   UNLIKELY  – Only tenuous or coincidental mentions; this employer is
                    almost certainly not correct for this individual.\
"""

# ---------------------------------------------------------------------------
# Schema fed to Step 2 — the analyst produces a list of candidates.
# We define it as a plain dict so Gemini's response_schema can accept it
# without needing a wrapping Pydantic model that references a list.
# ---------------------------------------------------------------------------

_CANDIDATES_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "company_name":       {"type": "string"},
                    "confidence_score":   {"type": "number"},
                    "evidence_summary":   {"type": "string"},
                    "sources_found":      {"type": "array", "items": {"type": "string"}},
                },
                "required": ["company_name", "confidence_score",
                             "evidence_summary", "sources_found"],
            },
        },
    },
    "required": ["candidates"],
}


class GeminiFindPoeService:
    """
    Discovers likely employers of a named individual using Gemini-powered web research.

    Usage::

        service = GeminiFindPoeService()
        result: FindPoeResult = service.find(request)
    """

    def __init__(self) -> None:
        api_key = settings.GEMINI_API_KEY
        if not api_key:
            logger.warning(
                "GEMINI_API_KEY is not configured – Gemini FindPOE calls will fail."
            )
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

    def find(self, request: FindPoeRequest) -> FindPoeResult:
        """
        Run the two-step employer-discovery pipeline for *request*.

        Args:
            request: A :class:`FindPoeRequest` carrying the person's name
                     and optional job title / location context.

        Returns:
            A :class:`FindPoeResult` with a ranked ``candidates`` list and
            the ``best_match`` candidate (highest confidence score).

        Raises:
            RuntimeError:                if GEMINI_API_KEY is not set.
            google.genai.errors.APIError: on unrecoverable Gemini API errors.
            pydantic.ValidationError:    if the model response cannot be
                                         coerced into the result schema.
        """
        client = self._require_client()

        subject_context = self._build_context(request)
        logger.info(
            f"[GeminiFindPoeService] Starting employer discovery for "
            f"{request.full_name!r}"
        )

        # ── Step 1: Web Research (The "Investigator") ──────────────────────
        raw_evidence, investigate_usage = self._investigate(client, subject_context, request)

        # Guard: if grounding returned nothing (safety filter, blocked content,
        # or a transient API issue) skip Step 2 and return an empty result.
        if not raw_evidence.strip():
            logger.warning(
                f"[GeminiFindPoeService] Step 1 returned empty evidence for "
                f"{request.full_name!r} — likely a safety filter. Returning empty result."
            )
            result = FindPoeResult(
                full_name=request.full_name,
                job_title=request.job_title,
                best_match=None,
                candidates=[],
            )
            result.token_usage = TokenUsage(
                openai=None,
                gemini=investigate_usage,
                grand_total=investigate_usage,
            )
            return result

        # ── Step 2: Structured Extraction (The "Analyst") ──────────────────
        candidates, analyse_usage = self._analyse(client, raw_evidence, request)

        # Rank by confidence descending; surface the top hit as best_match.
        candidates.sort(key=lambda c: c.confidence_score, reverse=True)
        best_match = candidates[0] if candidates else None

        # ── Attach aggregated token usage ─────────────────────────────────
        total_gemini_usage = investigate_usage + analyse_usage
        result = FindPoeResult(
            full_name=request.full_name,
            job_title=request.job_title,
            best_match=best_match,
            candidates=candidates,
            token_usage=TokenUsage(
                openai=None,
                gemini=total_gemini_usage,
                grand_total=total_gemini_usage,
            ),
        )

        logger.info(
            f"[GeminiFindPoeService] Discovery complete for {request.full_name!r}: "
            f"{len(candidates)} candidate(s), "
            f"best={best_match.company_name!r} (score={best_match.confidence_score}) "
            if best_match else
            f"[GeminiFindPoeService] Discovery complete for {request.full_name!r}: "
            f"no candidates found | "
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
                "GeminiFindPoeService is not configured: GEMINI_API_KEY is missing. "
                "Set it in your .env file and restart the server."
            )
        return self._client

    @staticmethod
    def _build_context(request: FindPoeRequest) -> str:
        """Build a concise, pipe-separated context string from the request fields."""
        parts = [f"Person: {request.full_name}"]
        if request.job_title:
            parts.append(f"Job Title: {request.job_title}")
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
        request: FindPoeRequest,
    ) -> tuple[str, ProviderTokenUsage]:
        """
        Step 1 — Grounded web research.

        Issues live Google Searches via Gemini to collect evidence about
        where *request.full_name* currently works.
        Returns a tuple of (research_text, token_usage_for_this_step).
        """
        job_title_hint = (
            f"Their known or suspected job title is: {request.job_title}\n"
            if request.job_title else ""
        )
        prompt = (
            f"Employer discovery research task:\n"
            f"{subject_context}\n\n"
            f"{job_title_hint}"
            "YOUR GOAL: Find the current employer (company/organisation) of this person.\n\n"
            "RESEARCH INSTRUCTIONS:\n"
            "1. Search LinkedIn, the company's official website, press releases,\n"
            "   news articles, SEC filings, and professional directories.\n"
            "2. Look for the person's name and any employer/company appearing together\n"
            "   in a credible public source — pay close attention to current role vs.\n"
            "   past roles.\n"
            "3. Note the EXACT source URL or publication name for each piece of\n"
            "   evidence you find.\n"
            "4. If you find MULTIPLE possible employers (e.g. a common name,\n"
            "   or the person has recently changed jobs), document ALL of them\n"
            "   with separate evidence — do NOT discard weaker candidates.\n"
            "5. If you find contradictory information (e.g. one source says\n"
            "   Company A, another says Company B), document the contradiction\n"
            "   explicitly.\n"
            "6. Use the location context above to disambiguate common names.\n"
            "7. Be thorough. Search at least 3–5 distinct sources before\n"
            "   concluding that no evidence exists.\n\n"
            "Summarise ALL evidence found, including source URLs."
        )

        grounding_tool = genai_types.Tool(
            google_search=genai_types.GoogleSearch()
        )
        config = genai_types.GenerateContentConfig(tools=[grounding_tool])

        logger.debug(
            f"[GeminiFindPoeService] Step 1: grounded search for {request.full_name!r}"
        )

        @retry_gemini()
        def _call() -> "genai_types.GenerateContentResponse":  # type: ignore[name-defined]
            return client.models.generate_content(
                model=_GEMINI_MODEL,
                contents=prompt,
                config=config,
            )

        response = _call()
        raw_text: str = response.text or ""
        step_usage = gemini_usage(response.usage_metadata)
        logger.debug(
            f"[GeminiFindPoeService] Step 1: received {len(raw_text)} chars of evidence "
            f"(tokens={step_usage.total_tokens})"
        )
        return raw_text, step_usage

    def _analyse(
        self,
        client: genai.Client,
        raw_evidence: str,
        request: FindPoeRequest,
    ) -> tuple[list[CompanyCandidate], ProviderTokenUsage]:
        """
        Step 2 — Structured extraction.

        Passes the raw evidence to Gemini with a strict JSON schema and a
        calibrated scoring rubric.  Returns a list of validated
        :class:`CompanyCandidate` objects plus token usage for this step.
        """
        prompt = (
            f"Employer discovery analysis task:\n\n"
            f"SUBJECT:\n"
            f"  Name:      {request.full_name}\n"
        )
        if request.job_title:
            prompt += f"  Job Title: {request.job_title}\n"
        if request.city or request.state or request.zip_code or request.country:
            location_parts = [
                p for p in [
                    request.city, request.state,
                    request.zip_code, request.country,
                ] if p
            ]
            prompt += f"  Location:  {', '.join(location_parts)}\n"

        prompt += (
            f"\nEVIDENCE GATHERED:\n"
            f"{raw_evidence}\n\n"
            f"SCORING RUBRIC:\n"
            f"{_SCORE_RUBRIC}\n\n"
            "TASK: Based solely on the evidence above, produce a structured JSON\n"
            "object listing all employer candidates you found.\n\n"
            "STRICT RULES:\n"
            "1. List EVERY distinct employer that appeared in the evidence — do NOT\n"
            "   discard lower-confidence candidates.\n"
            "2. `confidence_score` MUST be a float between 0.0 and 10.0, calibrated\n"
            "   against the rubric above.\n"
            "3. `evidence_summary` MUST be 2–3 sentences explaining what was found\n"
            "   and why you assigned that score.\n"
            "4. `sources_found` MUST list every URL or named publication you cited\n"
            "   for this employer. Use an empty array if none were found.\n"
            "5. `company_name` MUST be the employer's official name as it appears in\n"
            "   the sources — do NOT invent or paraphrase company names.\n"
            "6. If NO employer could be found at all, return an empty candidates array.\n"
            "7. Return ONLY the extracted data values — do NOT include the schema itself.\n"
            "8. Sort candidates by confidence_score DESCENDING (highest first)."
        )

        config = genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_CANDIDATES_SCHEMA,
        )

        logger.debug(
            f"[GeminiFindPoeService] Step 2: structured extraction for {request.full_name!r}"
        )

        @retry_gemini()
        def _call() -> "genai_types.GenerateContentResponse":  # type: ignore[name-defined]
            return client.models.generate_content(
                model=_GEMINI_MODEL,
                contents=prompt,
                config=config,
            )

        response = _call()
        step_usage = gemini_usage(response.usage_metadata)

        # Parse the candidates list from the response
        raw_text = response.text or "{}"
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            logger.warning(
                f"[GeminiFindPoeService] Step 2: JSON parse failed for "
                f"{request.full_name!r} — returning empty candidates."
            )
            return [], step_usage

        raw_candidates = data.get("candidates", [])
        candidates: list[CompanyCandidate] = []
        for item in raw_candidates:
            try:
                candidates.append(CompanyCandidate.model_validate(item))
            except Exception as exc:
                logger.debug(
                    f"[GeminiFindPoeService] Dropping invalid candidate {item!r}: {exc}"
                )

        return candidates, step_usage
