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
    Service wrapper around the three-step OpenAI contact-research workflow.

    Step 1 – "Gatherer"  : performs a live web_search tool call to collect raw
             contact information for the queried company.
    Step 2 – "Extractor" : distils the raw research into a validated Pydantic
             model instance.
    Step 3 – "Verifier"  : cross-checks each extracted contact value against the
             raw research text and removes anything that cannot be found verbatim,
             preventing pattern-completion and domain-guessing hallucinations.
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
            "6. Note the context (page section or surrounding label) for every contact found.\n\n"

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
            "   - For each extracted contact, assign a 'tag' from the following allowed values ONLY: 'Human Resource', 'Payroll', 'Admin', 'Careers', 'Personnel', 'Finance', 'Secretary', 'Labor relations', or 'Others'.\n"
            "   - If it is an administrative assistant, tag it as 'Admin'.\n"
            "   - Use 'context' to optionally provide the webpage section or text where it was found (e.g., 'Found on Careers page').\n"
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

        # ── Step 3: Verification (The "Verifier") ─────────────────────────
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
        Contacts whose exact value cannot be found verbatim are removed,
        preventing pattern-completion and domain-guessing hallucinations.
        """
        extracted_json = extracted.model_dump_json(indent=2)

        verification_prompt = (
            f"You are a precise fact-checker for contact information about '{company_name}'.\n\n"
            "You will be given:\n"
            "  1. RAW RESEARCH TEXT — text retrieved from live web searches.\n"
            "  2. EXTRACTED CONTACTS — a JSON object of contacts pulled from that research.\n\n"
            "YOUR TASK: Review each contact and decide whether to KEEP or REMOVE it.\n\n"
            "KEEP a contact if ANY of these conditions are true:\n"
            "  a) Its exact value appears verbatim anywhere in the research text.\n"
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
            "- Preserve 'tag', 'context', 'company_name', and 'official_site' unchanged.\n\n"
            f"RAW RESEARCH TEXT:\n{research_text}\n\n"
            f"EXTRACTED CONTACTS:\n{extracted_json}\n\n"
            f"Return the filtered result as a VALID JSON object matching this schema:\n"
            f"{json.dumps(model_class.model_json_schema(), indent=2)}"
        )

        logger.debug(
            f"[OpenAISearchService] Step 3: verifying extracted contacts for {company_name!r}"
        )
        verify_response = self._client.responses.create(
            model="gpt-4o-mini",
            input=verification_prompt,
            text={"format": {"type": "json_object"}},
        )

        data = json.loads(verify_response.output_text)
        cleaned_data = self._clean_dict(data)
        verified = model_class.model_validate(cleaned_data)
        logger.debug(
            f"[OpenAISearchService] Step 3: verification complete for {company_name!r}"
        )
        return verified

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
