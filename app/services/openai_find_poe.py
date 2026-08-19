"""
OpenAI Find POE Service
-----------------------
Discovers the most likely employer(s) of a named individual using a two-step
OpenAI Responses API pipeline:

    Step 1 – "Investigator"
        A ``responses.create`` call with the ``web_search`` tool enabled
        (``search_context_size="high"``).  OpenAI performs live web searches
        across LinkedIn, company directories, press releases, news articles,
        and public records to gather evidence about where the person works.

    Step 2 – "Analyst"
        A second ``responses.create`` call (no tools) with
        ``text={"format": {"type": "json_object"}}`` and a strict prompt
        that maps the raw evidence onto a ranked list of ``CompanyCandidate``
        objects.

Confidence score rubric (mirrors GeminiFindPoeService exactly so scores are
directly comparable when ``provider="both"``):
    9–10  CONFIRMED – Direct confirmation found.
    6–8   LIKELY    – Strong indirect evidence.
    3–5   POSSIBLE  – Weak or ambiguous connection.
    0–2   UNLIKELY  – Only tenuous/coincidental mentions.

Design decisions
~~~~~~~~~~~~~~~~
* Mirrors ``GeminiFindPoeService`` method-for-method so both implementations
  are composable and testable the same way.
* Uses the same ``_SCORE_RUBRIC`` constant so scores are calibrated
  identically across providers — this matters for ``provider="both"``
  comparisons.
* ``max_retries=2`` on the OpenAI client activates the SDK's inner retry
  layer (reads Retry-After headers on 429/5xx) before tenacity's outer
  layer fires.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from openai import OpenAI
from loguru import logger

from app.core.config import settings
from app.models import (
    FindPoeRequest,
    FindPoeResult,
    CompanyCandidate,
    TokenUsage,
    ProviderTokenUsage,
)
from app.services._retry import retry_openai
from app.services._token_utils import openai_usage

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_OPENAI_MODEL: str = settings.OPENAI_MODEL
"""OpenAI model used for both research and structured-extraction calls.
Override via OPENAI_MODEL in your .env file.
"""

_WEB_SEARCH_TOOL: dict = {
    "type": "web_search",
    "search_context_size": "high",
}

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


class OpenAIFindPoeService:
    """
    Discovers likely employers of a named individual using OpenAI-powered web research.

    Usage::

        service = OpenAIFindPoeService()
        result: FindPoeResult = service.find(request)
    """

    def __init__(self) -> None:
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            logger.warning(
                "OPENAI_API_KEY is not configured – OpenAI FindPOE calls will fail."
            )
        # max_retries=2: SDK inner-layer reads Retry-After headers on 429/5xx
        # before our tenacity outer-layer (in _retry.py) takes over.
        self._client: OpenAI | None = (
            OpenAI(api_key=api_key, max_retries=2) if api_key else None
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
            RuntimeError:          if OPENAI_API_KEY is not set.
            openai.OpenAIError:    on unrecoverable OpenAI API errors.
            pydantic.ValidationError: if the model response cannot be
                                      coerced into the result schema.
        """
        client = self._require_client()

        subject_context = self._build_context(request)
        logger.info(
            f"[OpenAIFindPoeService] Starting employer discovery for "
            f"{request.full_name!r}"
        )

        # ── Step 1: Web Research (The "Investigator") ──────────────────────
        raw_evidence, investigate_usage = self._investigate(client, subject_context, request)

        # Guard: if the search returned nothing, skip Step 2 and return empty.
        if not raw_evidence.strip():
            logger.warning(
                f"[OpenAIFindPoeService] Step 1 returned empty evidence for "
                f"{request.full_name!r} — returning empty result."
            )
            result = FindPoeResult(
                full_name=request.full_name,
                job_title=request.job_title,
                best_match=None,
                candidates=[],
            )
            result.token_usage = TokenUsage(
                openai=investigate_usage,
                gemini=None,
                grand_total=investigate_usage,
            )
            return result

        # ── Step 2: Structured Extraction (The "Analyst") ──────────────────
        candidates, analyse_usage = self._analyse(client, raw_evidence, request)

        # Rank by confidence descending; surface the top hit as best_match.
        candidates.sort(key=lambda c: c.confidence_score, reverse=True)
        best_match = candidates[0] if candidates else None

        # ── Attach aggregated token usage ─────────────────────────────────
        total_openai_usage = investigate_usage + analyse_usage
        result = FindPoeResult(
            full_name=request.full_name,
            job_title=request.job_title,
            best_match=best_match,
            candidates=candidates,
            token_usage=TokenUsage(
                openai=total_openai_usage,
                gemini=None,
                grand_total=total_openai_usage,
            ),
        )

        logger.info(
            f"[OpenAIFindPoeService] Discovery complete for {request.full_name!r}: "
            f"{len(candidates)} candidate(s), "
            f"best={best_match.company_name!r} (score={best_match.confidence_score}) | "
            f"tokens: input={total_openai_usage.input_tokens}, "
            f"output={total_openai_usage.output_tokens}, "
            f"total={total_openai_usage.total_tokens}"
            if best_match else
            f"[OpenAIFindPoeService] Discovery complete for {request.full_name!r}: "
            f"no candidates found | "
            f"tokens: input={total_openai_usage.input_tokens}, "
            f"output={total_openai_usage.output_tokens}, "
            f"total={total_openai_usage.total_tokens}"
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_client(self) -> OpenAI:
        """Return the configured OpenAI client or raise clearly if no key is set."""
        if self._client is None:
            raise RuntimeError(
                "OpenAIFindPoeService is not configured: OPENAI_API_KEY is missing. "
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
        client: OpenAI,
        subject_context: str,
        request: FindPoeRequest,
    ) -> tuple[str, ProviderTokenUsage]:
        """
        Step 1 — Grounded web research via OpenAI web_search tool.

        Issues live web searches to collect evidence about where
        *request.full_name* currently works.
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
            "1. Search LinkedIn, company websites, press releases, news articles,\n"
            "   SEC filings, and professional directories.\n"
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

        logger.debug(
            f"[OpenAIFindPoeService] Step 1: grounded search for {request.full_name!r}"
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
        step_usage = openai_usage(response.usage)
        logger.debug(
            f"[OpenAIFindPoeService] Step 1: received {len(raw_text)} chars of evidence "
            f"(tokens={step_usage.total_tokens})"
        )
        return raw_text, step_usage

    def _analyse(
        self,
        client: OpenAI,
        raw_evidence: str,
        request: FindPoeRequest,
    ) -> tuple[list[CompanyCandidate], ProviderTokenUsage]:
        """
        Step 2 — Structured extraction.

        Passes the raw evidence to the model with a strict JSON schema and a
        calibrated scoring rubric.  Returns a list of validated
        :class:`CompanyCandidate` objects plus token usage for this step.
        """
        location_parts = [
            p for p in [
                request.city, request.state,
                request.zip_code, request.country,
            ] if p
        ]
        location_line = (
            f"  Location:  {', '.join(location_parts)}\n" if location_parts else ""
        )

        schema = json.dumps({
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "company_name":     {"type": "string"},
                            "confidence_score": {"type": "number"},
                            "evidence_summary": {"type": "string"},
                            "sources_found":    {"type": "array", "items": {"type": "string"}},
                        },
                        "required": [
                            "company_name", "confidence_score",
                            "evidence_summary", "sources_found",
                        ],
                    },
                },
            },
            "required": ["candidates"],
        }, indent=2)

        prompt = (
            f"Employer discovery analysis task:\n\n"
            f"SUBJECT:\n"
            f"  Name:      {request.full_name}\n"
        )
        if request.job_title:
            prompt += f"  Job Title: {request.job_title}\n"
        prompt += location_line
        prompt += (
            f"\nEVIDENCE GATHERED:\n"
            f"{raw_evidence}\n\n"
            f"SCORING RUBRIC:\n"
            f"{_SCORE_RUBRIC}\n\n"
            "TASK: Based solely on the evidence above, produce a structured JSON\n"
            f"object matching EXACTLY this schema:\n{schema}\n\n"
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
            "7. Return ONLY the extracted data values — do NOT return the schema itself.\n"
            "8. Sort candidates by confidence_score DESCENDING (highest first)."
        )

        logger.debug(
            f"[OpenAIFindPoeService] Step 2: structured extraction for {request.full_name!r}"
        )

        @retry_openai()
        def _call():
            return client.responses.create(
                model=_OPENAI_MODEL,
                input=prompt,
                text={"format": {"type": "json_object"}},
            )

        response = _call()
        step_usage = openai_usage(response.usage)

        raw_json = self._strip_markdown_fences(response.output_text or "")

        if not raw_json:
            logger.warning(
                f"[OpenAIFindPoeService] Step 2 returned empty output for "
                f"{request.full_name!r} — returning empty candidates."
            )
            return [], step_usage

        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            logger.warning(
                f"[OpenAIFindPoeService] Step 2: JSON parse failed for "
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
                    f"[OpenAIFindPoeService] Dropping invalid candidate {item!r}: {exc}"
                )

        return candidates, step_usage

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Remove Markdown code fences that occasionally wrap the model's JSON output.

        Note: re.MULTILINE is intentionally NOT used here.  Without it, ``^``
        and ``$`` anchor to the very start and end of the full string, which is
        the correct behaviour — we only want to strip the outermost fence pair,
        not every line that happens to begin or end with triple backticks.
        """
        return re.sub(r"^```(?:json)?\s*\n?|\n?\s*```$", "", text.strip())
