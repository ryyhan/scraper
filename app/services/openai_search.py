"""
OpenAI Search Service
---------------------
Wraps the three-step OpenAI workflow (web research → structured extraction →
verification) as a reusable, dependency-injectable service.

Pipeline
~~~~~~~~
Step 1 – "Gatherer" (multi-turn, chained via ``previous_response_id``):
    Three sequential Responses API calls, each chained to the last, covering
    progressively deeper source types:

      Turn 1 – Broad HR / General / Directories  (queries a–i)
      Turn 2 – PDFs / BBB / Schema Markup        (queries j–o)
      Turn 3 – LinkedIn / Social / Press          (queries p–t)

    Every turn uses ``max_num_results=25`` and ``search_context_size="high"``
    to maximise raw data retrieved per query (vs. the bare default of 5 results
    at medium context).  When a known company URL is supplied the model is
    explicitly instructed to crawl key subpages (``/contact``, ``/hr``,
    ``/careers``, etc.) before doing any broad searches — this is entirely
    LLM-driven via the built-in web_search tool; no Python HTTP calls are made.

Step 2 – "Extractor":
    Distils the combined multi-turn research text into a validated Pydantic
    model instance.

Step 3 – "Verifier":
    Cross-checks each extracted contact against the raw research text.  Phone
    and fax values are compared using digit-only normalisation (strips dashes,
    spaces, parentheses) so formatting differences no longer cause valid numbers
    to be incorrectly removed.
"""

import json
import re
from typing import Any, Type, TypeVar

from openai import OpenAI
from loguru import logger
from pydantic import BaseModel

from app.core.config import settings
from app.models import ContactTag
from app.services._retry import retry_openai

T = TypeVar("T", bound=BaseModel)

# Dynamically built from the enum — always in sync, no manual maintenance.
_ALLOWED_TAGS: str = ", ".join(f"'{t.value}'" for t in ContactTag)

_OPENAI_MODEL: str = settings.OPENAI_MODEL
"""OpenAI model used for all research, extraction, and verification calls.
Override via OPENAI_MODEL in your .env file (e.g. OPENAI_MODEL=gpt-4o).
"""

# Shared web_search tool config — "high" context pulls the deepest page content
# available, surfacing phone/fax numbers buried in footers and PDFs that
# "medium" (the default) would miss.  Note: max_num_results is not supported
# in this version of the OpenAI Responses API; search_context_size is the
# primary lever for improving data depth per query.
_WEB_SEARCH_TOOL: dict = {
    "type": "web_search",
    "search_context_size": "high",
}


class OpenAISearchService:
    """
    Service wrapper around the three-step OpenAI contact-research workflow.

    Step 1 – "Gatherer"  : performs three chained live web_search calls
             (multi-turn via ``previous_response_id``) covering HR/general
             sources, PDFs/directories, and social/press channels.
    Step 2 – "Extractor" : distils the raw research into a validated Pydantic
             model instance.
    Step 3 – "Verifier"  : cross-checks each extracted contact value against the
             raw research text and removes anything that cannot be found verbatim,
             preventing pattern-completion and domain-guessing hallucinations.
             Phone/fax values are digit-normalised before comparison so that
             formatting differences do not cause false removals.
    """

    def __init__(self) -> None:
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            logger.warning("OPENAI_API_KEY is not configured – OpenAI calls will fail.")
        # max_retries=2: SDK inner-layer reads Retry-After headers on 429/5xx
        # before our tenacity outer-layer (in _retry.py) takes over.
        self._client: OpenAI | None = OpenAI(api_key=api_key, max_retries=2) if api_key else None

    def _require_client(self) -> OpenAI:
        """Return the configured client or raise clearly if no API key was set."""
        if self._client is None:
            raise RuntimeError(
                "OpenAISearchService is not configured: OPENAI_API_KEY is missing. "
                "Set it in your .env file and restart the server."
            )
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def structured_llm_call(self, request: BaseModel, model_class: Type[T]) -> T:
        """
        Execute the three-step research + extraction + verification pipeline
        for the target and return a validated instance of *model_class*.

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
        if country:
            context_parts.append(f"Country: {country}")
        if zip_code:
            context_parts.append(f"Zip Code: {zip_code}")
        if url:
            context_parts.append(f"URL: {url}")

        target_context = " | ".join(context_parts)

        # ── Step 1: Multi-Turn Deep Research (The "Gatherer") ─────────────────
        raw_research = self._gather(target_context, company_name, url)
        logger.debug(
            f"[OpenAISearchService] Total multi-turn research length: {len(raw_research)} chars"
        )

        # ── Step 2: Extraction (The "Filter & Formatter") ─────────────────────
        extraction_prompt = (
            f"Based on this research:\n{raw_research}\n\n"
            "TASK: Extract the info into JSON.\n"
            "STRICT RULES:\n"
            "1. Extract ALL valid emails, phone numbers, fax numbers, and physical addresses you can find into arrays.\n"
            f"   - For each extracted contact, assign a 'tag' from the following allowed values ONLY: {_ALLOWED_TAGS}.\n"
            "   - Pick the tag that best describes the department or purpose of the contact.\n"
            "   - If it is an administrative assistant, tag it as 'Admin'.\n"
            "   - Use 'context' to optionally provide the webpage section or text where it was found (e.g., 'Found on Careers page').\n"
            "2. For addresses, populate the structured fields (address1, address2, city, state, zip, country, countryCode).\n"
            f"3. The response MUST be a VALID JSON object matching exactly this schema:\n{json.dumps(model_class.model_json_schema(), indent=2)}\n"
            f"4. IMPORTANT: DO NOT return the schema definition itself. Return the ACTUAL extracted data values.\n"
            f"5. Make sure to fill in the 'company_name' key with the target company: {company_name}.\n"
            "6. EXCLUDE PLACEHOLDERS AND PLATFORM ACCESS EMAILS:\n"
            "   - Do NOT extract dummy/example emails (e.g., 'email@...', 'example@...', 'name@...', 'abc@...').\n"
            "   - Do NOT extract emails that appear as instructions for logging into or accessing a third-party "
            "software platform, LMS, or tool (e.g., 'use training@vendor.com to access the training portal', "
            "'log in at support@softwareplatform.com'). These are platform credentials, not HR contacts.\n"
            "   - DO keep emails from outsourced HR, payroll, or benefits providers if they are explicitly "
            "presented as a contact address for reaching that function on behalf of the target company "
            "(e.g., a PEO, staffing agency, or parent company managing payroll for this entity).\n"
            "   Only extract real, verified contact emails used to reach a person or department."
        )

        @retry_openai()
        def _extract_call():
            return self._require_client().responses.create(
                model=_OPENAI_MODEL,
                input=extraction_prompt,
                text={"format": {"type": "json_object"}},
            )

        final_response = _extract_call()

        # ── Parse + clean + validate ───────────────────────────────────────────
        data = json.loads(final_response.output_text)
        cleaned_data = self._clean_dict(data)
        result = model_class.model_validate(cleaned_data)

        # ── Step 3: Verification (The "Verifier") ─────────────────────────────
        result = self._verify(raw_research, result, company_name, model_class)

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

    def _gather(
        self,
        target_context: str,
        company_name: str,
        url: str | None = None,
    ) -> str:
        """
        Multi-turn deep research using the OpenAI Responses API with chained
        context (``previous_response_id``).

        Three turns are executed sequentially; each turn builds on the previous
        one's context so the model avoids revisiting sources already covered:

          Turn 1 – Broad HR / General / Admin / Business directories (a–i)
          Turn 2 – PDFs / BBB / schema-markup / directory sites      (j–o)
          Turn 3 – LinkedIn / social / press releases / databases     (p–t)

        All three turns use ``max_num_results=25`` and
        ``search_context_size="high"`` for maximum data surface area.
        When *url* is provided the model is instructed to crawl specific
        subpages of the official site as its mandatory first action (this is
        done via the LLM's built-in web_search tool — no Python HTTP calls).
        """
        client = self._require_client()

        # ── Optional: URL-specific subpage crawl instruction (Opp 5) ──────────
        url_instruction = ""
        if url:
            url_instruction = (
                f"\nPRIORITY: The official website is confirmed to be: {url}\n"
                "MANDATORY FIRST STEP: Before running any broad searches, visit the "
                "official website and specifically check ALL of the following subpages "
                "(the LLM's web_search tool handles the actual visiting — no scraping):\n"
                f"  - {url}/contact        |  {url}/contact-us\n"
                f"  - {url}/about          |  {url}/about-us\n"
                f"  - {url}/careers        |  {url}/jobs\n"
                f"  - {url}/hr             |  {url}/human-resources\n"
                f"  - {url}/staff          |  {url}/team   |  {url}/our-people\n"
                f"  - {url}/locations      |  {url}/directory\n"
                "Extract ALL contact information visible on each page before proceeding "
                "to the searches below.\n"
            )

        # ── Turn 1: Broad HR / General / Directories ───────────────────────────
        turn1_prompt = (
            f"Deep research task: Find contact information for the following target:\n{target_context}\n"
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
            "6. Note the context (page section or surrounding label) for every contact found.\n\n"

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

        logger.debug(f"[OpenAISearchService] Turn 1: HR/General/Directories for {company_name!r}")

        @retry_openai()
        def _t1():
            return client.responses.create(
                model=_OPENAI_MODEL,
                tools=[_WEB_SEARCH_TOOL],
                input=turn1_prompt,
            )

        r1 = _t1()

        # ── Turn 2: PDFs / BBB / Schema Markup / Business Directories ─────────
        turn2_prompt = (
            f"Continue researching '{company_name}'. Focus on sources not yet covered: "
            "PDFs, business directories, schema-markup phone tags, and BBB listings. "
            "Find phone numbers, fax numbers, emails, or addresses not yet identified.\n\n"
            "SEARCH STRATEGY — run ALL of the following:\n"
            f"  j) '{company_name}' filetype:pdf HR contact\n"
            f"  k) '{company_name}' filetype:pdf payroll department contact\n"
            f"  l) '{company_name}' inurl:contact OR inurl:staff OR inurl:team\n"
            f"  m) '{company_name}' \"tel:\" OR \"fax:\" contact\n"
            f"  n) '{company_name} fax number' site:bbb.org\n"
            f"  o) '{company_name}' site:yellowpages.com OR site:manta.com phone fax\n"
            "Do NOT skip any query. Report ALL new contact details found that were not "
            "already identified in the previous research turn."
        )

        logger.debug(f"[OpenAISearchService] Turn 2: PDFs/BBB/Schema for {company_name!r}")

        @retry_openai()
        def _t2():
            return client.responses.create(
                model=_OPENAI_MODEL,
                tools=[_WEB_SEARCH_TOOL],
                previous_response_id=r1.id,
                input=turn2_prompt,
            )

        r2 = _t2()

        # ── Turn 3: LinkedIn / Social / Press Releases ─────────────────────────
        turn3_prompt = (
            f"Final research pass for '{company_name}'. Focus on professional networks, "
            "press releases, and contact databases. Find any remaining contacts not yet identified.\n\n"
            "SEARCH STRATEGY:\n"
            f"  p) '{company_name}' HR email site:linkedin.com\n"
            f"  q) '{company_name}' payroll contact site:linkedin.com\n"
            f"  r) '{company_name}' 'for more information contact' press release email\n"
            f"  s) '{company_name}' site:glassdoor.com OR site:indeed.com HR contact email\n"
            f"  t) '{company_name}' site:zoominfo.com OR site:apollo.io HR phone email\n"
            "Do NOT skip any query. Report ALL remaining contact details that were not "
            "already identified in prior research turns.\n\n"
            "CRITICAL: Do NOT return '[email protected]'. Only real, verified emails."
        )

        logger.debug(f"[OpenAISearchService] Turn 3: LinkedIn/Social/Press for {company_name!r}")

        @retry_openai()
        def _t3():
            return client.responses.create(
                model=_OPENAI_MODEL,
                tools=[_WEB_SEARCH_TOOL],
                previous_response_id=r2.id,
                input=turn3_prompt,
            )

        r3 = _t3()

        logger.debug(
            f"[OpenAISearchService] Multi-turn complete for {company_name!r}: "
            f"T1={len(r1.output_text)}c, T2={len(r2.output_text)}c, T3={len(r3.output_text)}c"
        )

        return (
            "=== RESEARCH TURN 1 (HR / General / Directories) ===\n"
            + r1.output_text
            + "\n\n=== RESEARCH TURN 2 (PDFs / BBB / Schema Markup) ===\n"
            + r2.output_text
            + "\n\n=== RESEARCH TURN 3 (LinkedIn / Social / Press Releases) ===\n"
            + r3.output_text
        )

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
        Phone and fax values are compared after digit-only normalisation (strips
        dashes, spaces, parentheses, dots) so that formatting differences between
        the extracted value and the research text do not cause valid contacts to
        be incorrectly removed.
        Contacts that cannot be found (even after normalisation) are removed,
        preventing pattern-completion and domain-guessing hallucinations.
        """
        extracted_json = extracted.model_dump_json(indent=2)

        verification_prompt = (
            f"You are a precise fact-checker for contact information about '{company_name}'.\n\n"
            "You will be given:\n"
            "  1. RAW RESEARCH TEXT — text retrieved from live web searches.\n"
            "  2. EXTRACTED CONTACTS — a JSON object of contacts pulled from that research.\n\n"
            "YOUR TASK: Review each contact and decide whether to KEEP or REMOVE it.\n\n"
            "PRE-FILTER (apply FIRST, before all other rules):\n"
            "  ALWAYS REMOVE any email whose value is '[email protected]' or contains "
            "'cloudflare' in its domain. These are Cloudflare email-obfuscation placeholders, "
            "never real contact addresses.\n\n"
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
            "- Preserve 'tag', 'context', 'company_name', and 'official_site' unchanged.\n"
            "- IMPORTANT: Return ONLY the filtered data values in the EXACT SAME JSON structure "
            "as EXTRACTED CONTACTS below. Do NOT return a schema or type definition.\n\n"
            f"RAW RESEARCH TEXT:\n{research_text}\n\n"
            f"EXTRACTED CONTACTS:\n{extracted_json}\n\n"
            "Return the filtered result as a JSON object with the same keys and structure as EXTRACTED CONTACTS above."
        )

        logger.debug(
            f"[OpenAISearchService] Step 3: verifying extracted contacts for {company_name!r}"
        )

        @retry_openai()
        def _verify_call():
            return self._require_client().responses.create(
                model=_OPENAI_MODEL,
                input=verification_prompt,
                text={"format": {"type": "json_object"}},
            )

        verify_response = _verify_call()

        data = json.loads(verify_response.output_text)
        cleaned_data = self._clean_dict(data)
        try:
            verified = model_class.model_validate(cleaned_data)
            logger.debug(
                f"[OpenAISearchService] Step 3: verification complete for {company_name!r}"
            )
            return verified
        except Exception as exc:
            # If the LLM returned a schema definition or garbage, fall back to the
            # pre-verification result rather than crashing the whole pipeline.
            logger.warning(
                f"[OpenAISearchService] Step 3: verification parse failed ({exc}); "
                f"returning pre-verification result for {company_name!r}"
            )
            return extracted

    @staticmethod
    def _clean_val(value: object) -> object:
        """Strip residual markdown citation artefacts from string values."""
        if isinstance(value, str):
            return re.sub(r"\s*\(\[.*?\]\(.*?\)\)", "", value).strip()
        return value

    def _clean_dict(self, data: Any) -> Any:
        """Recursively clean all string values inside *data* (handles dicts and lists)."""
        if isinstance(data, dict):
            return {k: self._clean_dict(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._clean_dict(item) for item in data]
        return self._clean_val(data)
