"""
Combined Search Service
-----------------------
Executes both OpenAI and Gemini search pipelines concurrently and aggregates their results.
"""

import asyncio
import re
from urllib.parse import urlparse, urlunparse, urlencode, parse_qsl
from loguru import logger
from typing import Optional

from app.models import (
    CombinedSearchRequest,
    CombinedSearchResult,
    CombinedCompanyInfo,
    CombinedSearchSummary,
    SourceStats,
    OpenAISearchRequest,
    OpenAISearchResult,
    GeminiSearchRequest,
    GeminiSearchResult,
    TaggedContact,
    StructuredAddress,
    ProviderTokenUsage,
    TokenUsage,
)
from app.services.openai_search import OpenAISearchService
from app.services.gemini_search import GeminiSearchService


# ---------------------------------------------------------------------------
# URL sanitisation helpers
# ---------------------------------------------------------------------------

# Query-parameter keys whose presence indicates a tracking or attribution
# parameter that should be stripped before the URL is returned to the caller.
# All comparisons are done case-insensitively.
_TRACKING_PARAMS: frozenset[str] = frozenset({
    # UTM family (Google Analytics)
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "utm_id", "utm_source_platform", "utm_creative_format", "utm_marketing_tactic",
    # Common referral / attribution params
    "ref", "source", "referrer", "via", "origin",
    # OpenAI / Bing tracking
    "msclkid", "gclid", "fbclid", "ttclid",
})

# Hostnames (or hostname suffixes) that are never the company's own official
# website.  Any URL whose parsed hostname ends with one of these strings is
# treated as a third-party or provider-internal URL and discarded outright.
# Keys are matched as exact hostname suffixes (e.g. ".bbb.org" catches
# "www.bbb.org", "bbb.org", and any subdomain).
_REJECTED_HOSTS: tuple[str, ...] = (
    # Vertex AI / Gemini grounding redirect infrastructure
    "vertexaisearch.cloud.google.com",
    # BBB — directory listing, not the company's own site
    "bbb.org",
    # Common third-party business directories / social platforms
    "yelp.com",
    "linkedin.com",
    "glassdoor.com",
    "indeed.com",
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "x.com",
    "youtube.com",
    "wikipedia.org",
    "yellowpages.com",
    "manta.com",
    "zoominfo.com",
    "dnb.com",
    "hoovers.com",
    "corporationwiki.com",
    "bizapedia.com",
    "opencorporates.com",
    "sec.gov",
)


def _sanitize_official_site(raw_url: str) -> str:
    """
    Return a clean, canonical official-site URL, or ``""`` if the input is
    unusable.

    Two classes of problems are addressed:

    1. **Tracking query parameters** — Parameters such as ``utm_source=openai``
       are stripped from the URL's query string.  The rest of the URL
       (scheme, host, path, fragment) is preserved unchanged.  If removing
       tracking params leaves the query string empty, the ``?`` separator is
       also removed.

    2. **Third-party / non-official hostnames** — URLs pointing to known
       business-directory, social-media, or provider-internal infrastructure
       hosts (e.g. ``bbb.org``, ``vertexaisearch.cloud.google.com``,
       ``linkedin.com``) are discarded entirely (returns ``""``).
       These URLs originate from the LLM's grounding sources and represent
       *where the data was found*, not the company's own web presence.

    Args:
        raw_url: The URL string produced by the LLM pipeline, possibly
                 containing tracking params or pointing to a non-official host.

    Returns:
        A sanitised URL string, or ``""`` if the URL should be suppressed.
    """
    if not raw_url or not raw_url.strip():
        return ""

    url = raw_url.strip()

    # Must have a valid http(s) scheme — bare hostnames, relative paths, and
    # provider-specific protocol strings (e.g. "grounding://...") are rejected.
    if not url.startswith(("http://", "https://")):
        logger.debug(
            f"[_sanitize_official_site] Rejected (no http(s) scheme): {url!r}"
        )
        return ""

    try:
        parsed = urlparse(url)
    except Exception:  # pragma: no cover — malformed URL edge case
        logger.debug(
            f"[_sanitize_official_site] Rejected (URL parse error): {url!r}"
        )
        return ""

    hostname: str = parsed.netloc.lower()
    # Strip port suffix if present (e.g. "example.com:8080" → "example.com")
    hostname = hostname.split(":")[0]

    # ── 1. Reject known third-party / infrastructure hosts ─────────────────
    for rejected in _REJECTED_HOSTS:
        # Match on exact hostname or any subdomain of a rejected host.
        if hostname == rejected or hostname.endswith("." + rejected):
            logger.debug(
                f"[_sanitize_official_site] Rejected (non-official host {rejected!r}): "
                f"{url!r}"
            )
            return ""

    # ── 2. Strip tracking query parameters ─────────────────────────────────
    if parsed.query:
        clean_params = [
            (k, v)
            for k, v in parse_qsl(parsed.query, keep_blank_values=True)
            if k.lower() not in _TRACKING_PARAMS
        ]
        clean_query = urlencode(clean_params) if clean_params else ""
        parsed = parsed._replace(query=clean_query)
        cleaned_url = urlunparse(parsed)
        if cleaned_url != url:
            logger.debug(
                f"[_sanitize_official_site] Stripped tracking params: "
                f"{url!r} → {cleaned_url!r}"
            )
        return cleaned_url

    return url


class CombinedSearchService:
    """
    Service that delegates to both OpenAI and Gemini search services concurrently,
    then combines the results into a single comprehensive response.
    """

    def __init__(self) -> None:
        self.openai_service = OpenAISearchService()
        self.gemini_service = GeminiSearchService()

    async def search(self, request: CombinedSearchRequest) -> CombinedSearchResult:
        company_name = request.company_name
        max_limit = getattr(request, "max_limit", None)
        logger.info(f"[CombinedSearchService] Starting combined research for: {company_name!r}")

        # Pass max_limit=None so sub-services return their full uncapped result sets.
        # The cap is applied once here after deduplication so that the best
        # max_limit items are selected from the merged pool, not from each source
        # individually before merging.
        req_dict = {**request.model_dump(), "max_limit": None}
        openai_req = OpenAISearchRequest(**req_dict)
        gemini_req = GeminiSearchRequest(**req_dict)

        # Execute both concurrently in threads to prevent blocking
        openai_task = asyncio.to_thread(
            self.openai_service.structured_llm_call, openai_req, OpenAISearchResult
        )
        gemini_task = asyncio.to_thread(
            self.gemini_service.structured_llm_call, gemini_req, GeminiSearchResult
        )

        openai_result: Optional[OpenAISearchResult] = None
        gemini_result: Optional[GeminiSearchResult] = None

        # Await tasks and handle exceptions individually
        results = await asyncio.gather(openai_task, gemini_task, return_exceptions=True)

        if isinstance(results[0], Exception):
            logger.error(f"[CombinedSearchService] OpenAI search failed: {results[0]}")
        else:
            openai_result = results[0]

        if isinstance(results[1], Exception):
            logger.error(f"[CombinedSearchService] Gemini search failed: {results[1]}")
        else:
            gemini_result = results[1]

        if not openai_result and not gemini_result:
            raise RuntimeError("Both OpenAI and Gemini search pipelines failed.")

        # Aggregate the results
        phones: dict[str, TaggedContact] = {}
        faxes: dict[str, TaggedContact] = {}
        emails: dict[str, TaggedContact] = {}
        addresses: dict[str, StructuredAddress] = {}
        official_site = ""

        # --- Per-source stats (raw counts before deduplication) ---
        openai_stats = SourceStats()
        gemini_stats = SourceStats()

        # Process OpenAI Result
        if openai_result:
            if openai_result.official_site:
                official_site = _sanitize_official_site(openai_result.official_site)

            openai_stats.total_phones = len([p for p in openai_result.company_info.phones if p])
            openai_stats.total_faxes = len([f for f in openai_result.company_info.faxes if f])
            openai_stats.total_emails = len([e for e in openai_result.company_info.emails if e])
            openai_stats.total_addresses = len([a for a in openai_result.company_info.addresses if a])

            for p in openai_result.company_info.phones:
                if p:
                    key = re.sub(r"\D", "", p.value)  # normalize to digits for dedup
                    if key not in phones: phones[key] = p
            for f in openai_result.company_info.faxes:
                if f:
                    key = re.sub(r"\D", "", f.value)
                    if key not in faxes: faxes[key] = f
            for e in openai_result.company_info.emails:
                if e:
                    e_val = e.value.lower()
                    if e_val not in emails: emails[e_val] = e
            for a in openai_result.company_info.addresses:
                if a:
                    key = f"{a.address1}|{a.city}|{a.zip}".lower()
                    if key not in addresses: addresses[key] = a

        # Process Gemini Result
        if gemini_result:
            if not official_site and gemini_result.official_site:
                official_site = _sanitize_official_site(gemini_result.official_site)

            gemini_stats.total_phones = len([p for p in gemini_result.company_info.phones if p])
            gemini_stats.total_faxes = len([f for f in gemini_result.company_info.faxes if f])
            gemini_stats.total_emails = len([e for e in gemini_result.company_info.emails if e])
            gemini_stats.total_addresses = len([a for a in gemini_result.company_info.addresses if a])

            for p in gemini_result.company_info.phones:
                if p:
                    key = re.sub(r"\D", "", p.value)
                    if key not in phones: phones[key] = p
            for f in gemini_result.company_info.faxes:
                if f:
                    key = re.sub(r"\D", "", f.value)
                    if key not in faxes: faxes[key] = f
            for e in gemini_result.company_info.emails:
                if e:
                    e_val = e.value.lower()
                    if e_val not in emails: emails[e_val] = e
            for a in gemini_result.company_info.addresses:
                if a:
                    key = f"{a.address1}|{a.city}|{a.zip}".lower()
                    if key not in addresses: addresses[key] = a

        # Sort values
        phones_list = sorted(list(phones.values()), key=lambda x: x.value)
        faxes_list = sorted(list(faxes.values()), key=lambda x: x.value)
        emails_list = sorted(list(emails.values()), key=lambda x: x.value)
        addresses_list = sorted(list(addresses.values()), key=lambda x: f"{x.address1} {x.city} {x.zip}")

        # Apply max_limit cap
        apply_limit = max_limit is not None and max_limit > 0
        final_phones = phones_list[:max_limit] if apply_limit else phones_list
        final_faxes = faxes_list[:max_limit] if apply_limit else faxes_list
        final_emails = emails_list[:max_limit] if apply_limit else emails_list
        final_addresses = addresses_list[:max_limit] if apply_limit else addresses_list

        company_info = CombinedCompanyInfo(
            phones=final_phones,
            faxes=final_faxes,
            emails=final_emails,
            addresses=final_addresses,
        )

        # --- Build summary (combined counts reflect final capped+deduped lists) ---
        summary = CombinedSearchSummary(
            openai=openai_stats,
            gemini=gemini_stats,
            combined=SourceStats(
                total_phones=len(final_phones),
                total_faxes=len(final_faxes),
                total_emails=len(final_emails),
                total_addresses=len(final_addresses),
            ),
        )

        # --- Aggregate token usage from both providers ---
        openai_provider_usage: Optional[ProviderTokenUsage] = (
            openai_result.token_usage.openai
            if openai_result and openai_result.token_usage and openai_result.token_usage.openai
            else None
        )
        gemini_provider_usage: Optional[ProviderTokenUsage] = (
            gemini_result.token_usage.gemini
            if gemini_result and gemini_result.token_usage and gemini_result.token_usage.gemini
            else None
        )
        grand_total = (
            (openai_provider_usage or ProviderTokenUsage())
            + (gemini_provider_usage or ProviderTokenUsage())
        )
        combined_token_usage = TokenUsage(
            openai=openai_provider_usage,
            gemini=gemini_provider_usage,
            grand_total=grand_total,
        )

        logger.info(
            f"[CombinedSearchService] Completed combined research for: {company_name!r} | "
            f"OpenAI(phones={openai_stats.total_phones}, faxes={openai_stats.total_faxes}, "
            f"emails={openai_stats.total_emails}, addresses={openai_stats.total_addresses}) | "
            f"Gemini(phones={gemini_stats.total_phones}, faxes={gemini_stats.total_faxes}, "
            f"emails={gemini_stats.total_emails}, addresses={gemini_stats.total_addresses}) | "
            f"Combined(phones={summary.combined.total_phones}, faxes={summary.combined.total_faxes}, "
            f"emails={summary.combined.total_emails}, addresses={summary.combined.total_addresses}) | "
            f"tokens(openai={openai_provider_usage.total_tokens if openai_provider_usage else 0}, "
            f"gemini={gemini_provider_usage.total_tokens if gemini_provider_usage else 0}, "
            f"grand_total={grand_total.total_tokens})"
        )

        # --- Resolve official company name (prefer OpenAI → Gemini → input fallback) ---
        resolved_official_name: str = (
            (openai_result.official_company_name if openai_result and openai_result.official_company_name else None)
            or (gemini_result.official_company_name if gemini_result and gemini_result.official_company_name else None)
            or company_name
        )
        if resolved_official_name != company_name:
            logger.debug(
                f"[CombinedSearchService] Resolved official name: "
                f"{company_name!r} → {resolved_official_name!r}"
            )

        return CombinedSearchResult(
            company_name=company_name,
            official_company_name=resolved_official_name,
            official_site=official_site,
            company_info=company_info,
            summary=summary,
            openai_result=openai_result,
            gemini_result=gemini_result,
            token_usage=combined_token_usage,
        )
