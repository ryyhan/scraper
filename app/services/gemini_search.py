"""
Gemini Search Service
---------------------
Implements the same three-step contact-research pipeline as ``OpenAISearchService``
but drives Google's ``google-genai`` SDK (Gemini 2.0+) instead of OpenAI's.

Pipeline
~~~~~~~~
Step 1 – "Gatherer" (parallel specialized sub-gatherers via ThreadPoolExecutor):
    Three concurrent ``generate_content`` calls, each with the ``google_search``
    grounding tool and a hyper-focused prompt targeting different source types:

      Sub-gatherer 1 – HR / General / Admin          (queries a–i)
      Sub-gatherer 2 – Directories / PDFs / Schema   (queries j–p)
      Sub-gatherer 3 – Social / Press / Databases    (queries q–w)

    Each sub-gatherer uses ``DynamicRetrievalConfig(MODE_ALWAYS)`` to force
    grounding on every call regardless of the model's internal confidence score,
    and harvests ``grounding_chunks`` metadata (source URLs + page titles) which
    is appended to the raw research text for the Extractor to see.

    When a known company URL is provided the HR/General sub-gatherer is
    instructed to crawl key subpages of the official site as its first action —
    this is entirely LLM-driven via the grounding tool; no Python HTTP calls.

Step 2 – "Extractor":
    A ``generate_content`` call (no tools) with ``response_mime_type`` set to
    ``"application/json"`` and ``response_schema`` pointing at the target Pydantic
    model.  The SDK parses the JSON directly into the model via ``.parsed``.

Step 3 – "Verifier":
    A third ``generate_content`` call (no tools) that cross-checks every extracted
    contact value against the raw research text from Step 1.  Phone and fax values
    are compared after digit-only normalisation (strips dashes, spaces, parentheses)
    so formatting differences do not cause false removals.

Design decisions
~~~~~~~~~~~~~~~~
* Parallel sub-gatherers replace the single monolithic gather call to maximise
  total search breadth without increasing per-call latency — all three run
  concurrently and complete roughly within the time of the slowest one.
* ``DynamicRetrievalConfig(MODE_ALWAYS, dynamic_threshold=0.0)`` prevents Gemini
  from skipping grounding on queries it deems "not search-worthy".
* Grounding chunk metadata enriches the research text so the Extractor has
  access to the exact URLs Gemini visited and per-page snippets.
* The public ``structured_llm_call`` signature is intentionally identical to
  ``OpenAISearchService`` to allow future polymorphic dispatch.
"""

from __future__ import annotations

import re
import concurrent.futures
from typing import Type, TypeVar

from google import genai
from google.genai import types as genai_types
from loguru import logger
from pydantic import BaseModel

from app.core.config import settings
from app.models import ContactTag, HintAddress, TokenUsage, ProviderTokenUsage
from app.services._retry import retry_gemini
from app.services._token_utils import gemini_usage

T = TypeVar("T", bound=BaseModel)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_GEMINI_MODEL: str = settings.GEMINI_MODEL
"""Gemini model used for all research, extraction, and verification calls.
Override via GEMINI_MODEL in your .env file (e.g. GEMINI_MODEL=gemini-2.5-pro).
"""

# Dynamically built from the enum — always in sync, no manual maintenance.
_ALLOWED_TAGS: str = ", ".join(f"'{t.value}'" for t in ContactTag)


class GeminiSearchService:
    """
    Service wrapper around the three-step Gemini contact-research workflow.

    Step 1 – "Gatherer"  : runs three parallel grounded sub-gatherers
             (HR/General, Directories/PDFs, Social/Press) each with
             ``DynamicRetrievalConfig(MODE_ALWAYS)`` and grounding chunk
             harvesting for maximum data surface area.
    Step 2 – "Extractor" : distils the combined research into a validated Pydantic
             model instance via native JSON schema output.
    Step 3 – "Verifier"  : cross-checks each extracted contact value against the
             raw research text; phone/fax values are digit-normalised before
             comparison to prevent false removals due to formatting differences.
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
        # Dual-layer retry (inner): SDK retries 2x reading Retry-After headers
        # before tenacity's outer-layer takes over.
        # HttpOptions.timeout is in MILLISECONDS; 120 s is generous for grounded searches.
        from google.genai import types as _genai_types
        self._client: genai.Client | None = (
            genai.Client(
                api_key=api_key,
                http_options=_genai_types.HttpOptions(
                    timeout=120_000,
                    retry_options=_genai_types.HttpRetryOptions(attempts=2),
                ),
            )
            if api_key else None
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
        Execute the three-step research + extraction + verification pipeline
        for *request* and return a validated instance of *model_class*.

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
        max_limit: int | None = getattr(request, "max_limit", None)
        phone: str | None = getattr(request, "phone", None)
        fax: str | None = getattr(request, "fax", None)
        address: HintAddress | None = getattr(request, "address", None)

        # Sanitize string-typed fields: treat any value that is clearly a
        # Swagger UI placeholder or non-meaningful input as None so it is
        # never injected into LLM prompts.
        # zip_code must contain at least one digit to be a real postal code.
        if zip_code and not any(c.isdigit() for c in zip_code):
            zip_code = None
        # url must begin with a valid scheme; 'string', 'null', bare hostnames, etc. are rejected.
        if url and not url.strip().startswith(("http://", "https://")):
            url = None

        logger.info(f"[GeminiSearchService] Starting research for: {company_name!r}")

        target_context = self._build_context(company_name, country, zip_code, url)
        hint_block = self._build_hint_block(company_name, country, zip_code, phone, fax, address)

        # ── Step 1: Parallel Deep Research (The "Gatherer") ────────────────────
        research_text, gather_usage = self._gather(target_context, company_name, url, hint_block)

        # Build entity scope note from ALL populated anchors for Steps 2 and 3.
        _clean_phone = phone if (phone and any(c.isdigit() for c in phone)) else None
        _clean_fax   = fax   if (fax   and any(c.isdigit() for c in fax))   else None
        _anchor_parts: list[str] = []
        if country:       _anchor_parts.append(f"Country: {country}")
        if zip_code:      _anchor_parts.append(f"Zip Code: {zip_code}")
        if _clean_phone:  _anchor_parts.append(f"Phone: {_clean_phone}")
        if _clean_fax:    _anchor_parts.append(f"Fax: {_clean_fax}")
        if address:
            _addr_str = address.to_prompt_string()
            if _addr_str: _anchor_parts.append(f"Address: {_addr_str}")

        entity_scope_note = ""
        if _anchor_parts:
            _anchor_summary = " | ".join(_anchor_parts)
            entity_scope_note = (
                f"\n\u26a0\ufe0f  ENTITY SCOPE CONSTRAINT — IMPORTANT:\n"
                f"The research may contain data from MULTIPLE organizations named '{company_name}'.\n"
                f"You are ONLY processing the specific entity that matches ALL of the following known anchors:\n"
                f"  {_anchor_summary}\n"
                f"EXCLUDE any contact that clearly belongs to a DIFFERENT '{company_name}' organization.\n"
                f"When the entity is ambiguous, default to KEEP.\n"
            )

        # ── Step 2: Extraction (The "Extractor") ────────────────────────────
        result, extract_usage = self._extract(research_text, company_name, model_class, entity_scope_note)

        # ── Step 3: Verification (The "Verifier") ───────────────────────────
        result, verify_usage = self._verify(research_text, result, company_name, model_class, entity_scope_note)

        if max_limit is not None and max_limit > 0 and hasattr(result, "company_info"):
            result.company_info.phones = result.company_info.phones[:max_limit]
            result.company_info.faxes = result.company_info.faxes[:max_limit]
            result.company_info.emails = result.company_info.emails[:max_limit]
            result.company_info.addresses = result.company_info.addresses[:max_limit]

        # ── Attach aggregated token usage ──────────────────────────────────────
        total_gemini_usage = gather_usage + extract_usage + verify_usage
        result.token_usage = TokenUsage(
            openai=None,
            gemini=total_gemini_usage,
            grand_total=total_gemini_usage,
        )
        logger.info(
            f"[GeminiSearchService] Extraction complete for: {company_name!r} | "
            f"tokens: input={total_gemini_usage.input_tokens}, "
            f"output={total_gemini_usage.output_tokens}, "
            f"total={total_gemini_usage.total_tokens}"
        )
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
        """Assemble a concise, pipe-separated identity context string from available fields."""
        parts = [f"Company: {company_name}"]
        if country:
            parts.append(f"Country: {country}")
        if zip_code:
            parts.append(f"Zip Code: {zip_code}")
        if url:
            parts.append(f"URL: {url}")
        return " | ".join(parts)

    @staticmethod
    def _build_hint_block(
        company_name: str,
        country: str | None,
        zip_code: str | None,
        phone: str | None,
        fax: str | None,
        address: HintAddress | None,
    ) -> str:
        """
        Compose the entity-filter block injected at the top of Sub-gatherer 1.
        ALL populated fields are included as identity anchors.  The most specific
        available anchor drives an active STEP 1 / STEP 2 discovery sequence
        (identical strategy to OpenAISearchService._build_hint_block).
        Returns empty string when no meaningful hints are present.
        """
        clean_phone: str | None = phone if (phone and any(c.isdigit() for c in phone)) else None
        clean_fax:   str | None = fax   if (fax   and any(c.isdigit() for c in fax))   else None

        identity_lines: list[str] = []
        if country:      identity_lines.append(f"  Country      : {country}")
        if zip_code:     identity_lines.append(f"  Zip Code     : {zip_code}")
        if clean_phone:  identity_lines.append(f"  Known phone  : {clean_phone}")
        if clean_fax:    identity_lines.append(f"  Known fax    : {clean_fax}")
        if address:
            addr_str = address.to_prompt_string()
            if addr_str:
                identity_lines.append(f"  Known address: {addr_str}")

        if not identity_lines:
            return ""

        country_note = f" in {country}" if country else ""

        if clean_phone:
            step1_lines = (
                f"STEP 1 (REQUIRED FIRST ACTION — do this before anything else):\n"
                f"  Search for the known phone number to pinpoint the EXACT entity:\n"
                f"  \u2192 Search: \"{clean_phone}\"\n"
                f"  \u2192 The organization that owns this phone number is your ONLY research target.\n"
                f"  \u2192 Note its exact legal name and city/state before proceeding.\n\n"
            )
        elif address and address.city and address.state:
            step1_lines = (
                f"STEP 1 (REQUIRED FIRST ACTION — do this before anything else):\n"
                f"  Search for the company using its known city and state:\n"
                f"  \u2192 Search: \"{company_name}\" \"{address.city}\" \"{address.state}\"\n"
                f"  \u2192 Use the result to confirm the exact entity at this location.\n"
                f"  \u2192 Note its exact legal name before proceeding.\n\n"
            )
        elif zip_code:
            step1_lines = (
                f"STEP 1 (REQUIRED FIRST ACTION — do this before anything else):\n"
                f"  Search for the company using its known zip code:\n"
                f"  \u2192 Search: \"{company_name}\" \"{zip_code}\"\n"
                f"  \u2192 Use the result to identify the exact entity in this zip code area.\n"
                f"  \u2192 Note its exact legal name before proceeding.\n\n"
            )
        else:
            step1_lines = None

        if step1_lines:
            return (
                f"\n\u26a0\ufe0f  MANDATORY ENTITY IDENTIFICATION — FOLLOW THESE STEPS IN ORDER:\n\n"
                + step1_lines
                + f"STEP 2: Use ONLY the entity identified in Step 1 for ALL subsequent research.\n"
                f"  \u2192 Search for additional contacts ONLY for that specific organization.\n"
                f"  \u2192 Do NOT visit websites of any other '{company_name}' organization.\n\n"
                "All known identity anchors (ALL must match the entity you research):\n"
                + "\n".join(identity_lines)
                + f"\n\nCRITICAL: There may be many organizations named '{company_name}'{country_note}. "
                f"You MUST ONLY research the specific one confirmed by the anchors above. "
                f"ALL other '{company_name}' organizations are completely irrelevant — "
                f"do NOT visit their websites or extract any of their contacts.\n"
            )
        else:
            return (
                f"\n\u26a0\ufe0f  MANDATORY ENTITY FILTER — READ THIS BEFORE SEARCHING:\n"
                f"You MUST ONLY research the specific '{company_name}' entity whose identity "
                f"matches ALL of the following anchors.\n\n"
                "Target identity anchors:\n"
                + "\n".join(identity_lines)
                + f"\n\nCRITICAL RULE: If your search returns results for a company with the same "
                f"name{country_note} in a DIFFERENT country, region, or industry:\n"
                "  1. IGNORE those results entirely.\n"
                "  2. Do NOT extract ANY contact information from those entities.\n"
                "  3. ONLY process sources that clearly belong to the target entity above.\n"
            )

    def _gather(
        self,
        target_context: str,
        company_name: str,
        url: str | None = None,
        hint_block: str = "",
    ) -> tuple[str, ProviderTokenUsage]:
        """
        Launch three parallel grounded sub-gatherers via ``ThreadPoolExecutor``
        and merge their results into a single combined research text.

        Sub-gatherers run concurrently so total latency ≈ slowest sub-gatherer,
        not the sum of all three.  Each failure is caught individually and logged
        as a warning; the merged output from surviving sub-gatherers is returned.

        Returns a tuple of (combined_research_text, accumulated_token_usage)
        so the caller can aggregate usage across all pipeline steps.
        """
        logger.debug(
            f"[GeminiSearchService] Launching 3 parallel sub-gatherers for {company_name!r}"
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    self._gather_hr_general, target_context, company_name, url, hint_block
                ): "HR/General",
                executor.submit(
                    self._gather_directories_pdfs, target_context, company_name
                ): "Directories/PDFs",
                executor.submit(
                    self._gather_social_press, target_context, company_name
                ): "Social/Press",
            }

            sections: list[str] = []
            accumulated_usage = ProviderTokenUsage()
            for future, label in futures.items():
                try:
                    text, sub_usage = future.result(timeout=180)
                    if text:
                        sections.append(
                            f"=== GEMINI SUB-SEARCH: {label} ===\n{text}"
                        )
                        logger.debug(
                            f"[GeminiSearchService] Sub-gatherer '{label}' "
                            f"returned {len(text)} chars for {company_name!r} "
                            f"(tokens={sub_usage.total_tokens})"
                        )
                    accumulated_usage = accumulated_usage + sub_usage
                except Exception as exc:
                    logger.warning(
                        f"[GeminiSearchService] Sub-gatherer '{label}' failed "
                        f"for {company_name!r}: {exc}"
                    )

        combined = "\n\n".join(sections)
        logger.debug(
            f"[GeminiSearchService] Parallel gather complete for {company_name!r}: "
            f"{len(combined)} total chars from {len(sections)}/3 sub-gatherers | "
            f"gather_tokens={accumulated_usage.total_tokens}"
        )
        return combined, accumulated_usage

    def _run_grounded_search(self, prompt: str, label: str, company_name: str) -> tuple[str, ProviderTokenUsage]:
        """
        Execute a single grounded Gemini search call with retry protection.

        Uses ``Tool(google_search=GoogleSearch())`` — the only grounding
        mechanism supported by ``gemini-2.5-flash-lite``.  The model always
        performs live web searches when this tool is present; no additional
        ``DynamicRetrievalConfig`` is required or accepted by this model.

        After the call, ``grounding_chunks`` metadata (source URLs + page
        titles) is extracted from the candidate and appended to the response
        text so the Extractor can see exactly which pages were visited.

        Transient errors (429 quota, 503 service unavailable, 500 server error,
        deadline exceeded) are automatically retried with exponential backoff
        via the shared ``retry_gemini()`` decorator from ``_retry.py``.

        Returns a tuple of (response_text, token_usage) so the caller can
        accumulate usage across multiple sub-gatherer calls.
        """
        grounding_tool = genai_types.Tool(
            google_search=genai_types.GoogleSearch()
        )
        research_config = genai_types.GenerateContentConfig(tools=[grounding_tool])

        client = self._require_client()

        @retry_gemini()
        def _call() -> genai.types.GenerateContentResponse:  # type: ignore[name-defined]
            return client.models.generate_content(
                model=_GEMINI_MODEL,
                contents=prompt,
                config=research_config,
            )

        response = _call()
        raw_text: str = response.text or ""
        step_usage = gemini_usage(response.usage_metadata)

        # Harvest grounding chunk metadata — source URLs + titles that Gemini
        # actually visited.  Appending these gives the Extractor more surface
        # area and lets it verify which sources were consulted.
        grounding_sources: list[str] = []
        if response.candidates:
            candidate = response.candidates[0]
            if candidate.grounding_metadata:
                for chunk in (candidate.grounding_metadata.grounding_chunks or []):
                    if chunk.web and chunk.web.uri:
                        title = chunk.web.title or "Unknown"
                        grounding_sources.append(f"  - {chunk.web.uri} ({title})")

        if grounding_sources:
            source_block = "\n".join(grounding_sources)
            raw_text += f"\n\nSOURCES VISITED BY GEMINI [{label}]:\n{source_block}"

        return raw_text, step_usage

    def _gather_hr_general(
        self,
        target_context: str,
        company_name: str,
        url: str | None,
        hint_block: str = "",
    ) -> tuple[str, ProviderTokenUsage]:
        """
        Sub-gatherer 1 — Broad HR / General / Admin / Business Directories.

        Covers queries a–i: direct HR searches, payroll, admin, careers,
        BBB listings, and general business directory sites.  When *url* is
        provided, the model is instructed to crawl specific subpages of the
        official site as its first action (LLM-driven, no Python scraping).
        When *hint_block* is non-empty, the disambiguation hints are injected
        immediately after the target context so the model anchors on the
        correct entity before searching.
        """
        url_instruction = ""
        if url:
            url_instruction = (
                f"\nPRIORITY: The official website is confirmed to be: {url}\n"
                "MANDATORY FIRST STEP: Before running any broad searches, visit the "
                "official website and specifically check ALL of the following subpages "
                "(handled by the grounding tool — no manual scraping):\n"
                f"  - {url}/contact        |  {url}/contact-us\n"
                f"  - {url}/about          |  {url}/about-us\n"
                f"  - {url}/careers        |  {url}/jobs\n"
                f"  - {url}/hr             |  {url}/human-resources\n"
                f"  - {url}/staff          |  {url}/team   |  {url}/our-people\n"
                f"  - {url}/locations      |  {url}/directory\n"
                "Extract ALL contact information visible on each page before proceeding "
                "to the searches below.\n"
            )

        prompt = (
            f"Deep research task: Find contact information for the following target:\n{target_context}\n"
            f"{hint_block}"
            f"{url_instruction}\n"
            "SEARCH STRATEGY — run ALL of the following targeted searches in order:\n"
            f"  a) '{company_name} HR department phone email fax'\n"
            f"  b) '{company_name} human resources contact information'\n"
            f"  c) '{company_name} payroll department email phone'\n"
            f"  d) '{company_name} careers department contact'\n"
            f"  e) '{company_name} personnel finance secretary labor relations contact'\n"
            f"  f) '{company_name} administrative assistant admin contact'\n"
            f"  g) '{company_name} contact us fax number address'\n"
            f"  h) site:bbb.org '{company_name}'\n"
            f"  i) '{company_name}' site:manta.com OR site:yellowpages.com\n"
            "Do NOT stop after the first query. Execute all of the above and consolidate results.\n\n"
            "GOALS:\n"
            "1. First priority: Find Human Resource (HR) contact info — phone, email, fax.\n"
            "2. Also find contact info for ALL of the following departments when available: "
            "Payroll, Admin (administrative assistants), Careers, "
            "Personnel, Finance, Secretary, Labor Relations.\n"
            "3. If department-specific contacts are unavailable, fall back to any General/Corporate contact info.\n"
            "4. Specifically look for a Fax number on every source visited.\n"
            "5. Collect ALL available contact details: phones, emails, faxes, and physical addresses.\n"
            "6. Provide the official website URL if found.\n"
            "7. Note the context (page section or surrounding label) for every contact found.\n"
            "Be thorough and search multiple sources.\n\n"
            "SOURCES TO CHECK (in priority order):\n"
            "  1. Official company website — check /contact, /about-us, /careers, /hr, /staff, /team pages.\n"
            "  2. LinkedIn company page — look for HR staff contact details.\n"
            "  3. ZoomInfo / Apollo / RocketReach — search for HR or payroll department listings.\n"
            "  4. Better Business Bureau (BBB) listing — often has direct phone/fax.\n"
            "  5. Press releases (PR Newswire, Business Wire) — emails often appear in press contact sections.\n"
            "  6. Indeed / Glassdoor job postings — HR contacts frequently listed.\n"
            "  7. Google Maps business profile — phone, fax, and address.\n"
            "  8. State business registry or SEC filings — for official address and executive contacts.\n\n"
            "CRITICAL INSTRUCTION FOR EMAILS:\n"
            "If you see '[email protected]' on the company's website, their emails are hidden by Cloudflare.\n"
            "DO NOT return '[email protected]'. Instead, run new searches on the alternative sources listed above.\n"
            "- Press releases (PR Newswire, Business Wire)\n"
            "- LinkedIn, Apollo, or ZoomInfo summaries\n"
            "- Public PDF documents or SEC filings\n"
            "- Official social media pages"
        )

        logger.debug(
            f"[GeminiSearchService] Sub-gatherer HR/General starting for {company_name!r}"
        )
        return self._run_grounded_search(prompt, "HR/General", company_name)

    def _gather_directories_pdfs(
        self,
        target_context: str,
        company_name: str,
    ) -> tuple[str, ProviderTokenUsage]:
        """
        Sub-gatherer 2 — Business Directories, PDF Documents, Schema Markup.

        Covers queries j–p: filetype:pdf searches, BBB fax lookups, business
        directory sites (YellowPages, Manta), inurl: patterns, and schema.org
        telephone/fax markup searches.  Fax numbers are especially likely to
        appear in BBB listings and PDF annual reports / press releases.
        """
        prompt = (
            f"Research task: Find phone numbers, fax numbers, emails, and addresses for:\n{target_context}\n\n"
            "Focus EXCLUSIVELY on these high-value source types: PDF documents, "
            "business directories, BBB listings, and schema-markup phone tags.\n\n"
            "SEARCH STRATEGY — run ALL of the following searches:\n"
            f"  j) '{company_name}' filetype:pdf HR contact\n"
            f"  k) '{company_name}' filetype:pdf payroll department contact\n"
            f"  l) '{company_name}' inurl:contact OR inurl:staff OR inurl:team\n"
            f"  m) '{company_name}' \"tel:\" OR \"fax:\" contact\n"
            f"  n) '{company_name} fax number' site:bbb.org\n"
            f"  o) '{company_name}' site:yellowpages.com OR site:manta.com phone fax\n"
            f"  p) '{company_name}' annual report OR corporate directory contact phone\n"
            "Do NOT stop after the first query. Execute all of the above.\n\n"
            "GOALS:\n"
            "1. Find fax numbers — these are especially likely to appear in BBB listings and PDF documents.\n"
            "2. Find phone numbers for HR, Payroll, and Admin departments.\n"
            "3. Find email addresses in PDF press releases or corporate documents.\n"
            "4. Find physical addresses from business directory listings.\n"
            "5. Extract ALL contact details visible in any PDF documents you can access.\n\n"
            "Report ALL contact information found with its source context."
        )

        logger.debug(
            f"[GeminiSearchService] Sub-gatherer Directories/PDFs starting for {company_name!r}"
        )
        return self._run_grounded_search(prompt, "Directories/PDFs", company_name)

    def _gather_social_press(
        self,
        target_context: str,
        company_name: str,
    ) -> tuple[str, ProviderTokenUsage]:
        """
        Sub-gatherer 3 — LinkedIn, Social Media, Press Releases, Contact Databases.

        Covers queries q–w: LinkedIn HR/payroll searches, press release contact
        emails, Glassdoor/Indeed HR contacts, ZoomInfo/Apollo database lookups,
        and SEC filing contact references.  Press release emails are especially
        valuable as they are real, verified, and often HR or PR contacts.
        """
        prompt = (
            f"Research task: Find phone numbers, fax numbers, emails, and addresses for:\n{target_context}\n\n"
            "Focus EXCLUSIVELY on professional networks, press releases, and contact databases.\n\n"
            "SEARCH STRATEGY — run ALL of the following searches:\n"
            f"  q) '{company_name}' HR email site:linkedin.com\n"
            f"  r) '{company_name}' payroll contact site:linkedin.com\n"
            f"  s) '{company_name}' 'for more information contact' press release email\n"
            f"  t) '{company_name}' site:glassdoor.com OR site:indeed.com HR contact email\n"
            f"  u) '{company_name}' site:zoominfo.com OR site:apollo.io HR phone email\n"
            f"  v) '{company_name}' news press release 'contact:' OR 'contact us at'\n"
            f"  w) '{company_name}' SEC filing 'human resources' contact phone\n"
            "Do NOT stop after the first query. Execute all of the above.\n\n"
            "GOALS:\n"
            "1. Find HR or Payroll email addresses from LinkedIn or professional profile summaries.\n"
            "2. Find press release contact emails (these are typically real, verified HR or PR addresses).\n"
            "3. Find executive or department contact info from ZoomInfo / Apollo summaries.\n"
            "4. Find contact info mentioned in SEC filings or investor relations documents.\n\n"
            "Report ALL contact information found with its source context.\n\n"
            "CRITICAL: If you see '[email protected]' on any website, DO NOT return it. "
            "Only return real, verified email addresses."
        )

        logger.debug(
            f"[GeminiSearchService] Sub-gatherer Social/Press starting for {company_name!r}"
        )
        return self._run_grounded_search(prompt, "Social/Press", company_name)

    def _extract(
        self,
        research_text: str,
        company_name: str,
        model_class: Type[T],
        entity_anchor_note: str = "",
    ) -> tuple[T, ProviderTokenUsage]:
        """
        Step 2 — Distil the raw research into a structured Pydantic model.

        When *entity_anchor_note* is non-empty (i.e. caller supplied phone/
        address hints) it is prepended to the extraction prompt so the LLM
        only extracts contacts from the anchor-matched entity.

        Returns a tuple of (extracted_result, token_usage_for_this_step).
        """
        extraction_prompt = (
            f"Based on the following research about '{company_name}':\n\n"
            f"{research_text}\n\n"
            f"{entity_anchor_note}\n"
            "TASK: Extract the contact information into the required JSON schema.\n"
            "STRICT RULES:\n"
            "1. Extract ALL valid emails, phone numbers, fax numbers, and physical "
            "addresses you can find into their respective arrays.\n"
            f"   - For each extracted contact, assign a 'tag' from the following allowed values ONLY: {_ALLOWED_TAGS}.\n"
            "   - Pick the tag that best describes the department or purpose of the contact.\n"
            "   - If it is an administrative assistant, tag it as 'Admin'.\n"
            "   - Use 'context' to optionally provide the webpage section or text where it was found (e.g., 'Found on Careers page').\n"
            "2. For addresses, populate the structured fields (address1, address2, city, state, zip, country, countryCode).\n"
            f"3. Fill the 'company_name' key with the exact target name: {company_name}.\n"
            "4. Fill 'official_company_name' with the company's LEGAL/OFFICIAL registered name "
            "as it appears on the official website, government records, SEC filings, or BBB listing "
            f"(e.g. 'Troopr Inc.' not 'Troopr', or 'DEMOULAS MARKET BASKET INC.' not 'Market Basket'). "
            f"If you cannot determine the legal name with confidence, copy the value of 'company_name' ({company_name}).\n"
            "5. Fill 'official_site' with the company's PRIMARY OFFICIAL website URL — "
            "the domain the company itself owns and operates (e.g. 'https://acme.com'). "
            "STRICT EXCLUSIONS — never use any of the following as official_site:\n"
            "   - Third-party directory or listing pages: BBB (bbb.org), Yelp, LinkedIn, "
            "Glassdoor, Indeed, ZoomInfo, Manta, YellowPages, Wikipedia, or any similar site.\n"
            "   - Provider-internal redirect URLs such as those containing "
            "'vertexaisearch.cloud.google.com' — these are grounding infrastructure URLs, "
            "not the company's website.\n"
            "   - Any URL with tracking query parameters (e.g. utm_source=openai) — return "
            "only the clean base URL without such parameters.\n"
            "If no company-owned official URL was found in the research, leave this as an empty string.\n"
            "6. Return ONLY the extracted data values — do NOT return the schema itself.\n"
            "7. If a field has no data, use an empty array [] or empty string \"\".\n"
            "8. EXCLUDE PLACEHOLDERS AND PLATFORM ACCESS EMAILS:\n"
            "   - Do NOT extract dummy/example emails (e.g., 'email@...', 'example@...', 'name@...', 'abc@...').\n"
            "   - Do NOT extract emails that appear as instructions for logging into or accessing a third-party "
            "software platform, LMS, or tool (e.g., 'use training@vendor.com to access the training portal', "
            "'log in at support@softwareplatform.com'). These are platform credentials, not HR contacts.\n"
            "   - DO keep emails from outsourced HR, payroll, or benefits providers if they are explicitly "
            "presented as a contact address for reaching that function on behalf of the target company "
            "(e.g., a PEO, staffing agency, or parent company managing payroll for this entity).\n"
            "   Only extract real, verified contact emails used to reach a person or department."
        )

        extraction_config = genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=model_class,
        )

        logger.debug(
            f"[GeminiSearchService] Step 2: issuing structured extraction for {company_name!r}"
        )
        client = self._require_client()

        @retry_gemini()
        def _call() -> genai.types.GenerateContentResponse:  # type: ignore[name-defined]
            return client.models.generate_content(
                model=_GEMINI_MODEL,
                contents=extraction_prompt,
                config=extraction_config,
            )

        response = _call()
        step_usage = gemini_usage(response.usage_metadata)

        # `response.parsed` is automatically populated by the SDK when
        # `response_schema` is a Pydantic model class.
        if response.parsed is not None:
            logger.debug("[GeminiSearchService] Step 2: using SDK-parsed Pydantic object")
            parsed = response.parsed
            # Guarantee official_company_name is always populated on the parsed object.
            if hasattr(parsed, "official_company_name") and not parsed.official_company_name:
                parsed.official_company_name = company_name
            return parsed, step_usage  # type: ignore[return-value]

        # Fallback: the SDK may return raw JSON text on some model/version
        # combinations.  Parse manually to guarantee a validated result.
        logger.warning(
            "[GeminiSearchService] Step 2: .parsed was None — falling back to "
            "manual JSON parsing"
        )
        import json

        raw_json_text = self._strip_markdown_fences(response.text or "{}")
        data = json.loads(raw_json_text)
        result = model_class.model_validate(data)

        # Guarantee official_company_name is always populated on the manually-parsed result.
        if hasattr(result, "official_company_name") and not result.official_company_name:
            result.official_company_name = company_name

        return result, step_usage

    def _verify(
        self,
        research_text: str,
        extracted: T,
        company_name: str,
        model_class: Type[T],
        entity_filter_note: str = "",
    ) -> tuple[T, ProviderTokenUsage]:
        """
        Step 3 — Verify extracted contacts against the raw research text.

        Each contact's ``value`` field is cross-checked against *research_text*.
        Phone and fax values are compared after digit-only normalisation (strips
        dashes, spaces, parentheses, dots) so formatting differences between the
        extracted value and research text do not cause valid contacts to be
        incorrectly removed.  Contacts that still cannot be found are removed,
        preventing the LLM from inventing plausible-looking but fictitious
        contact details (e.g. ``hr@company.com`` inferred from the domain).

        Returns a tuple of (verified_result, token_usage_for_this_step).
        """
        import json

        extracted_json = extracted.model_dump_json(indent=2)

        # entity_filter_note is built upstream from ALL populated anchors and
        # is injected as the first rule in the verifier prompt.
        entity_filter = (
            entity_filter_note.replace(
                "\u26a0\ufe0f  ENTITY SCOPE CONSTRAINT — IMPORTANT:",
                "  ENTITY SCOPE CONSTRAINT:",
            ).strip()
            + "\n"
        ) if entity_filter_note.strip() else ""

        verification_prompt = (
            f"You are a precise fact-checker for contact information about '{company_name}'.\n\n"
            "You will be given:\n"
            "  1. RAW RESEARCH TEXT — text retrieved from live web searches.\n"
            "  2. EXTRACTED CONTACTS — a JSON object of contacts pulled from that research.\n\n"
            "YOUR TASK: Review each contact and decide whether to KEEP or REMOVE it.\n\n"
            "PRE-FILTER (apply FIRST, before all other rules):\n"
            "  ALWAYS REMOVE any email whose value is '[email protected]' or contains "
            "'cloudflare' in its domain. These are Cloudflare email-obfuscation placeholders, "
            "never real contact addresses.\n"
            + entity_filter
            + "\n"
            "PHONE AND FAX NUMBER MATCHING — CRITICAL RULE:\n"
            "  When checking whether a phone or fax number appears in the research text, "
            "NORMALIZE both the extracted value AND the research text by stripping ALL "
            "non-digit characters (spaces, dashes, parentheses, dots, plus signs, etc.) "
            "before comparing.\n"
            "  A match on digits alone is SUFFICIENT to KEEP the contact.\n"
            "  EXAMPLE: '(555) 867-5309' MATCHES '555-867-5309', '555.867.5309', and '5558675309'.\n"
            "  Do NOT remove a phone or fax number solely because its formatting differs from "
            "what appears in the research text.\n\n"
            "KEEP a contact if ANY of these conditions are true:\n"
            "  a) Its exact value (or digit-normalized value for phones/faxes, or full address "
            "text for addresses) appears anywhere in the research text.\n"
            "  b) The research text explicitly attributes it to the company "
            "(e.g. 'contact us at ...', 'call us on ...').\n"
            "  c) It is a GENERIC email (prefix is one of: info, contact, mail, hello, "
            "support, enquiries, enquiry, general, office) AND its domain matches "
            "the company's official website domain found in the research text. "
            "These are legitimate catch-all addresses commonly used by companies.\n\n"
            "REMOVE a contact if ALL of these conditions are true:\n"
            "  a) Its exact value (or digit-normalized value for phones/faxes) does NOT appear "
            "anywhere in the research text.\n"
            "  b) It is NOT a generic email as defined above (i.e. it has a specific departmental "
            "prefix that was never explicitly found on any source page).\n\n"
            "- Do NOT add any new contacts not already in EXTRACTED CONTACTS.\n"
            "- Preserve 'tag', 'context', 'company_name', 'official_company_name', and 'official_site' fields unchanged.\n\n"
            f"RAW RESEARCH TEXT:\n{research_text}\n\n"
            f"EXTRACTED CONTACTS:\n{extracted_json}\n\n"
            "Return the filtered result in exactly the same JSON schema."
        )

        verification_config = genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=model_class,
        )

        logger.debug(
            f"[GeminiSearchService] Step 3: verifying extracted contacts for {company_name!r}"
        )
        client = self._require_client()

        @retry_gemini()
        def _call() -> genai.types.GenerateContentResponse:  # type: ignore[name-defined]
            return client.models.generate_content(
                model=_GEMINI_MODEL,
                contents=verification_prompt,
                config=verification_config,
            )

        response = _call()
        step_usage = gemini_usage(response.usage_metadata)

        if response.parsed is not None:
            verified = response.parsed  # type: ignore[assignment]
            logger.debug(
                "[GeminiSearchService] Step 3: verification complete via SDK-parsed object"
            )
            return verified, step_usage  # type: ignore[return-value]

        # Fallback to manual parse if SDK doesn't auto-parse
        logger.warning(
            "[GeminiSearchService] Step 3: .parsed was None — falling back to manual JSON parsing"
        )
        raw_json_text = self._strip_markdown_fences(response.text or "{}")
        data = json.loads(raw_json_text)
        return model_class.model_validate(data), step_usage

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """
        Remove Markdown code fences (```json ... ```) that occasionally wrap
        the model's JSON output in the fallback path.
        """
        return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
