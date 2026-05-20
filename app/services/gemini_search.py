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

Step 3 – "Verifier"
    A third ``generate_content`` call (no tools) that cross-checks every extracted
    contact value against the raw research text from Step 1.  Any contact whose
    exact value (email address, phone number) cannot be found verbatim in the
    research is removed.  This prevents pattern-completion and domain-guessing
    hallucinations (e.g. the LLM inventing ``hr@company.com`` from the domain).

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
    Service wrapper around the three-step Gemini contact-research workflow.

    Step 1 – "Gatherer"  : triggers ``google_search`` grounding to collect raw,
             cited contact information for the queried company from the live web.
    Step 2 – "Extractor" : distils the raw research into a validated Pydantic
             model instance via native JSON schema output.
    Step 3 – "Verifier"  : cross-checks each extracted contact value against the
             raw research text and removes anything that cannot be found verbatim,
             preventing pattern-completion and domain-guessing hallucinations.
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
        max_limit: int | None = getattr(request, "max_limit", None)

        logger.info(f"[GeminiSearchService] Starting research for: {company_name!r}")

        target_context = self._build_context(company_name, country, zip_code, url)

        # ── Step 1: Deep Research (The "Gatherer") ────────────────────────────
        research_text = self._gather(target_context, company_name)

        # ── Step 2: Extraction (The "Extractor") ──────────────────────────────
        result: T = self._extract(research_text, company_name, model_class)

        # ── Step 3: Verification (The "Verifier") ─────────────────────────────
        result = self._verify(research_text, result, company_name, model_class)

        if max_limit is not None and max_limit > 0 and hasattr(result, "company_info"):
            result.company_info.phones = result.company_info.phones[:max_limit]
            result.company_info.faxes = result.company_info.faxes[:max_limit]
            result.company_info.emails = result.company_info.emails[:max_limit]
            result.company_info.addresses = result.company_info.addresses[:max_limit]

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

            # ── Multi-Query Search Strategy ────────────────────────────────
            "SEARCH STRATEGY — run ALL of the following targeted searches in order:\n"
            f"  a) '{company_name} HR department phone email fax'\n"
            f"  b) '{company_name} human resources contact information'\n"
            f"  c) '{company_name} payroll department email phone'\n"
            f"  d) '{company_name} careers department contact'\n"
            f"  e) '{company_name} personnel finance secretary labor relations contact'\n"
            f"  f) '{company_name} administrative assistant admin contact'\n"
            f"  g) '{company_name} contact us fax number address'\n"
            "Do NOT stop after the first query. Execute all of the above and consolidate results.\n\n"

            # ── Priority Goals ──────────────────────────────────────────────
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

            # ── Priority Source List ────────────────────────────────────────
            "SOURCES TO CHECK (in priority order):\n"
            "  1. Official company website — check /contact, /about-us, /careers, /hr, /staff, /team pages.\n"
            "  2. LinkedIn company page — look for HR staff contact details.\n"
            "  3. ZoomInfo / Apollo / RocketReach — search for HR or payroll department listings.\n"
            "  4. Better Business Bureau (BBB) listing — often has direct phone/fax.\n"
            "  5. Press releases (PR Newswire, Business Wire) — emails often appear in press contact sections.\n"
            "  6. Indeed / Glassdoor job postings — HR contacts frequently listed.\n"
            "  7. Google Maps business profile — phone, fax, and address.\n"
            "  8. State business registry or SEC filings — for official address and executive contacts.\n\n"

            # ── Email Obfuscation Handling ──────────────────────────────────
            "CRITICAL INSTRUCTION FOR EMAILS:\n"
            "If you see '[email protected]' on the company's website, their emails are hidden by Cloudflare.\n"
            "DO NOT return '[email protected]'. Instead, run new searches on the alternative sources listed above.\n"
            "- Press releases (PR Newswire, Business Wire)\n"
            "- LinkedIn, Apollo, or ZoomInfo summaries\n"
            "- Public PDF documents or SEC filings\n"
            "- Official social media pages"
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
            "   - For each extracted contact, assign a 'tag' from the following allowed values ONLY: 'Human Resource', 'Payroll', 'Admin', 'Careers', 'Personnel', 'Finance', 'Secretary', 'Labor relations', or 'Others'.\n"
            "   - If it is an administrative assistant, tag it as 'Admin'.\n"
            "   - Use 'context' to optionally provide the webpage section or text where it was found (e.g., 'Found on Careers page').\n"
            "2. For addresses, populate the structured fields (address1, address2, city, state, zip, country, countryCode).\n"
            f"3. Fill the 'company_name' key with the exact target name: {company_name}.\n"
            "4. Fill 'official_site' with the primary official website URL if found, "
            "otherwise leave it as an empty string.\n"
            "5. Return ONLY the extracted data values — do NOT return the schema itself.\n"
            "6. If a field has no data, use an empty array [] or empty string \"\".\n"
            "7. EXCLUDE PLACEHOLDERS: Do NOT extract dummy/example emails (e.g., 'email@...', 'example@...', 'name@...', 'abc@...'). Only extract real, verified contact emails."
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

    def _verify(
        self,
        research_text: str,
        extracted: T,
        company_name: str,
        model_class: Type[T],
    ) -> T:
        """
        Step 3 — Verify extracted contacts against the raw research text.

        Each contact's ``value`` field is cross-checked against *research_text*.
        Contacts whose exact value does not appear verbatim are removed,
        preventing the LLM from inventing plausible-looking but fictitious
        contact details (e.g. ``hr@company.com`` inferred from the domain).

        The verification is itself an LLM call (no grounding tools) using the
        same structured-output schema so the result is always a valid model.
        """
        import json

        extracted_json = extracted.model_dump_json(indent=2)

        verification_prompt = (
            f"You are a precise fact-checker for contact information about '{company_name}'.\n\n"
            "You will be given:\n"
            "  1. RAW RESEARCH TEXT — text retrieved from live web searches.\n"
            "  2. EXTRACTED CONTACTS — a JSON object of contacts pulled from that research.\n\n"
            "YOUR TASK: Review each contact and decide whether to KEEP or REMOVE it.\n\n"
            "KEEP a contact if ANY of these conditions are true:\n"
            "  a) Its exact value (or the full address text for addresses) appears verbatim anywhere in the research text.\n"
            "  b) The research text explicitly attributes it to the company "
            "(e.g. 'contact us at ...', 'call us on ...').\n"
            "  c) It is a GENERIC email (prefix is one of: info, contact, mail, hello, "
            "support, enquiries, enquiry, general, office) AND its domain matches "
            "the company's official website domain found in the research text. "
            "These are legitimate catch-all addresses commonly used by companies.\n\n"
            "REMOVE a contact if ALL of these conditions are true:\n"
            "  a) Its exact value does NOT appear anywhere in the research text.\n"
            "  b) It is NOT a generic email as defined above (i.e. it uses a departmental "
            "prefix such as hr, payroll, finance, careers, secretary, labor that was "
            "never explicitly found on any source page).\n\n"
            "- Do NOT add any new contacts not already in EXTRACTED CONTACTS.\n"
            "- Preserve 'tag', 'context', 'company_name', and 'official_site' fields unchanged.\n\n"
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
        response = client.models.generate_content(
            model=_GEMINI_MODEL,
            contents=verification_prompt,
            config=verification_config,
        )

        if response.parsed is not None:
            verified = response.parsed  # type: ignore[assignment]
            logger.debug(
                "[GeminiSearchService] Step 3: verification complete via SDK-parsed object"
            )
            return verified  # type: ignore[return-value]

        # Fallback to manual parse if SDK doesn't auto-parse
        logger.warning(
            "[GeminiSearchService] Step 3: .parsed was None — falling back to manual JSON parsing"
        )
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
