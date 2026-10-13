"""
Company Matcher Service
-----------------------
Determines whether two company name strings refer to the same real-world
entity using an LLM.

Supported matching cases  (standard mode)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Acronym expansion     — "IBM"       ↔ "International Business Machines Corporation"
* Dot normalization     — "I.B.M."   ↔ "IBM"
* Legal-suffix noise    — "Microsoft" ↔ "Microsoft Corporation"
* Typos                 — "nicrosoft" ↔ "Microsoft"
* OCR character swaps   — "Mic0rosoft"↔ "Microsoft"
* Semantic / world knowledge — "openi" ↔ "OpenAI Group PBC"

Deep-search mode  (deep_search=True)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For trade names, DBA names, and local businesses that an LLM's training data
may not recognise (e.g. "Thomas Electrical" → "THOMAS P. OWENS, LLC"), a
two-step web-research pass is added before the verdict:

  Step 1 — Research each company independently via live web search:
           • OpenAI: ``responses.create`` with the ``web_search`` tool.
           • Gemini: ``generate_content`` with ``google_search`` grounding.
           • Groq:   no native search → uses OpenAI web search for research,
                     then routes the final verdict back through Groq.
  Step 2 — Verdict LLM call receives the enriched research as context.

Provider choice
~~~~~~~~~~~~~~~
* groq   — AsyncGroq  (llama-3.1-8b-instant): fastest, cheapest, good for clear cases.
* openai — AsyncOpenAI (settings.OPENAI_MODEL): higher accuracy + built-in web search.
* gemini — google-genai (settings.GEMINI_MODEL): Google Search grounding.

All providers return the same structured JSON:
    { "is_same_company": bool, "confidence": float, "reasoning": str }

Retry strategy
~~~~~~~~~~~~~~
Each provider method is decorated with the project's standard dual-layer
tenacity retry (3 outer attempts × 2–3 SDK inner retries = ≤9 API calls).
See app/services/_retry.py for full documentation.
"""

from __future__ import annotations

import asyncio
import json
import time
from functools import partial
from typing import Any

from fastapi import HTTPException, status
from groq import (
    AsyncGroq,
    RateLimitError as GroqRateLimitError,
    APIStatusError as GroqAPIStatusError,
    APIConnectionError as GroqAPIConnectionError,
    APITimeoutError as GroqAPITimeoutError,
)
from openai import (
    AsyncOpenAI,
    OpenAI,
    RateLimitError as OAIRateLimitError,
    APIStatusError as OAIAPIStatusError,
    APIConnectionError as OAIAPIConnectionError,
    APITimeoutError as OAIAPITimeoutError,
)
from loguru import logger

from app.core.config import settings
from app.models import CompanyMatchRequest, CompanyMatchResult
from app.services._retry import retry_groq, retry_openai, retry_gemini


# ---------------------------------------------------------------------------
# Prompt template (shared across all providers)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a company identity resolution expert. "
    "Your only job is to decide whether two company name strings refer to the "
    "same real-world legal entity or operating brand. "
    "You have broad world knowledge of company names, abbreviations, and history."
)

_USER_PROMPT_TEMPLATE = """\
Compare the following two company name strings and determine whether they refer
to the same real-world company.

Company A: {company_a}
Company B: {company_b}

Rules you MUST apply:
1. Acronyms and full names are the same entity  — "IBM" = "International Business Machines Corporation".
2. Strip legal suffixes before comparing        — "Corp", "Corporation", "Inc", "LLC",
   "Ltd", "Group", "PBC", "Holdings", "Co.", "S.A.", "GmbH", "plc", etc. are irrelevant.
3. Dot-separated initials equal the acronym     — "I.B.M." = "IBM".
4. Tolerate minor typos (1–2 character edits)   — "nicrosoft" → "Microsoft".
5. Tolerate common OCR errors (digit-for-letter swap, etc.) — "Mic0rosoft" → "Microsoft".
6. Use world knowledge for partial / informal names — "openi" is clearly "OpenAI".
7. Two genuinely different companies MUST return false — "Microsoft" ≠ "Apple".
8. Do NOT match companies that are merely in the same industry.

Return ONLY a valid JSON object with exactly these three keys — no extra text:
{{
  "is_same_company": <true|false>,
  "confidence": <float 0.0–1.0>,
  "reasoning": "<one sentence>"
}}
"""


# Deep-search enriched prompt template.
# Injected when deep_search=True and research has been gathered.
_USER_PROMPT_DEEP_TEMPLATE = """\
Compare the following two company name strings and determine whether they refer
to the same real-world company.

Company A: {company_a}
Company B: {company_b}

You have been given three blocks of live web research below.  Use this research
as your PRIMARY source of truth — it overrides your training-data assumptions.

--- JOINT RESEARCH (searched both names together) ---
{research_joint}

--- RESEARCH: Company A ("{company_a}") ---
{research_a}

--- RESEARCH: Company B ("{company_b}") ---
{research_b}

DECISION RULES — apply ALL of them in order:

RULE 1 — Geographic filtering (IMPORTANT):
  If a location context is visible in the research or the names, IGNORE any
  results about companies in OTHER states or cities.
  Only consider businesses in the same geographic area.

RULE 2 — Two-hop DBA resolution:
  Company B may have an OFFICIAL DBA or trade name that is a longer/fuller form
  of Company A's informal name.  If research shows this chain:
    Company A ≈ (partial match) ≈ Company B's DBA → Company B's legal entity
  then return true.
  Example:
    Company A = "Thomas Electrical"
    Research shows: "Thomas Owens Electrical Service" is a DBA for "THOMAS P. OWENS, LLC"
    "Thomas Electrical" shares "Thomas" + "Electrical" with the DBA → SAME ENTITY.

RULE 3 — Informal shorthand:
  Local businesses are routinely called by shortened informal names.
  If the only electrical/plumbing/construction contractor named "{company_a}"
  in the relevant area is the one behind "{company_b}", treat them as the same
  entity.  Do NOT require an official record to explicitly spell out the informal name.

RULE 4 — Acronyms and legal suffixes:
  Acronyms equal their expanded form — "IBM" = "International Business Machines".
  Ignore legal suffixes — Corp, Inc, LLC, Ltd, Group, PBC, Holdings, etc.

RULE 5 — Typos / OCR:
  Tolerate 1–2 character edits or digit-for-letter substitutions.

RULE 6 — Different companies:
  Two genuinely DIFFERENT companies in DIFFERENT areas must still return false,
  even if they share some words.

Return ONLY a valid JSON object with exactly these three keys — no extra text:
{{
  "is_same_company": <true|false>,
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one sentence citing the specific research finding that drove the verdict>"
}}
"""


def _build_user_prompt(
    company_a: str,
    company_b: str,
    research_a: str = "",
    research_b: str = "",
    research_joint: str = "",
) -> str:
    """Build the verdict prompt, injecting research context when available."""
    if research_a or research_b or research_joint:
        return _USER_PROMPT_DEEP_TEMPLATE.format(
            company_a=company_a.strip(),
            company_b=company_b.strip(),
            research_a=research_a.strip() or "No results found.",
            research_b=research_b.strip() or "No results found.",
            research_joint=research_joint.strip() or "No results found.",
        )
    return _USER_PROMPT_TEMPLATE.format(
        company_a=company_a.strip(),
        company_b=company_b.strip(),
    )


def _parse_llm_json(raw: str, provider: str) -> dict[str, Any]:
    """
    Parse and validate the LLM's JSON response.

    Strips accidental markdown fences (```json ... ```) that some models
    emit despite explicit instructions not to add extra text.
    """
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        # Drop the opening fence line and any closing fence
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error(f"[{provider}] Failed to parse JSON response: {exc!r}\nRaw: {raw!r}")
        raise ValueError(f"LLM returned non-JSON output: {raw[:200]!r}") from exc

    if not isinstance(data.get("is_same_company"), bool):
        raise ValueError(
            f"'is_same_company' missing or not a boolean in LLM response: {data}"
        )
    confidence = data.get("confidence", 0.5)
    if not isinstance(confidence, (int, float)):
        confidence = 0.5
    data["confidence"] = float(max(0.0, min(1.0, confidence)))
    data.setdefault("reasoning", "")
    return data


# ---------------------------------------------------------------------------
# CompanyMatcherService
# ---------------------------------------------------------------------------

class CompanyMatcherService:
    """
    Facade that routes a company-match request to the selected LLM provider
    and returns a normalised ``CompanyMatchResult``.

    Instantiated per-request (stateless aside from the clients, which are
    created lazily on first call and are safe to reuse across coroutines).
    """

    def __init__(self) -> None:
        # Clients are created lazily so that a missing API key only raises
        # at call time, not at import / startup.
        self._groq_client: AsyncGroq | None = None
        self._openai_client: AsyncOpenAI | None = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def match(self, request: CompanyMatchRequest) -> CompanyMatchResult:
        """
        Route the request to the appropriate provider and return the result.

        When ``request.deep_search`` is True, a live web-research pass is run
        for each company name concurrently BEFORE the verdict call.  The
        research text is injected into the verdict prompt so the LLM can
        resolve trade names, DBA names, and local businesses that are absent
        from its training data.

        Raises:
            HTTPException 503 — if the required API key is not configured.
            HTTPException 500 — if the LLM returns a malformed response after
                                all retries are exhausted.
        """
        provider = request.provider
        deep = request.deep_search
        location = (request.location_hint or "").strip()

        logger.info(
            f"[company-match] '{request.company_a}' vs '{request.company_b}' "
            f"via {provider} | deep_search={deep}"
            + (f" | location='{location}'" if location else "")
        )

        start = time.perf_counter()
        research_a = research_b = research_joint = ""

        # ------------------------------------------------------------------
        # Step 1 (optional): three concurrent web research calls
        # ------------------------------------------------------------------
        if deep:
            logger.info("[company-match] Running deep-search research step (3 parallel calls)...")
            try:
                research_a, research_b, research_joint = await asyncio.gather(
                    self._deep_research_single(request.company_a, location, provider),
                    self._deep_research_single(request.company_b, location, provider),
                    self._deep_research_joint(
                        request.company_a, request.company_b, location, provider
                    ),
                )
                logger.debug(
                    f"[company-match] Research complete: "
                    f"A={len(research_a)} chars, B={len(research_b)} chars, "
                    f"joint={len(research_joint)} chars"
                )
            except HTTPException:
                raise
            except Exception as exc:
                # Non-fatal: log and fall back to standard (no-research) mode
                # so the endpoint still returns a result rather than a 500.
                logger.warning(
                    f"[company-match] Deep research failed, falling back to "
                    f"standard mode: {exc}"
                )

        # ------------------------------------------------------------------
        # Step 2: verdict LLM call (with or without research context)
        # ------------------------------------------------------------------
        try:
            if provider == "groq":
                data = await self._call_groq(
                    request.company_a, request.company_b,
                    research_a, research_b, research_joint,
                )
            elif provider == "openai":
                data = await self._call_openai(
                    request.company_a, request.company_b,
                    research_a, research_b, research_joint,
                )
            else:  # "gemini"
                data = await self._call_gemini(
                    request.company_a, request.company_b,
                    research_a, research_b, research_joint,
                )
        except HTTPException:
            raise  # propagate 503 key-missing errors unchanged
        except ValueError as exc:
            # Malformed JSON from LLM after all retries
            logger.error(f"[company-match/{provider}] Malformed LLM output: {exc}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"LLM returned an unparseable response: {exc}",
            ) from exc
        except Exception as exc:
            logger.error(f"[company-match/{provider}] Unexpected error: {exc}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"LLM provider error: {exc}",
            ) from exc

        elapsed = round(time.perf_counter() - start, 3)

        result = CompanyMatchResult(
            company_a=request.company_a,
            company_b=request.company_b,
            is_same_company=data["is_same_company"],
            confidence=data["confidence"],
            reasoning=data["reasoning"],
            provider=provider,
            deep_search_used=deep,
            processing_time_seconds=elapsed,
        )
        logger.info(
            f"[company-match] result={result.is_same_company} "
            f"confidence={result.confidence:.2f} time={elapsed}s"
        )
        return result

    # ------------------------------------------------------------------
    # Deep-search research helpers
    # ------------------------------------------------------------------

    async def _deep_research_single(self, company_name: str, location_hint: str, provider: str) -> str:
        """
        Research a single company name to discover its legal entity name, DBA,
        trade name, and registered details.

        Passes a rich task prompt (like find_poe's _investigate) rather than a
        bare keyword query, so Gemini/OpenAI autonomously searches the right
        source types (state license registries, contractor databases, BBB, etc.).
        """
        loc_context = f" Location context: {location_hint}." if location_hint else ""
        task_prompt = (
            f"Business entity research task:\n"
            f"Company name to research: {company_name!r}{loc_context}\n\n"
            f"YOUR GOAL: Find the official legal/registered name of this business and any "
            f"known trade names or DBA ('doing business as') names it operates under.\n\n"
            f"RESEARCH INSTRUCTIONS:\n"
            f"1. Search state contractor license databases, secretary of state business "
            f"   registries, and license verification portals (e.g. myfloridalicense.com, "
            f"   sos.state.*.us).\n"
            f"2. Search the Better Business Bureau (bbb.org), Manta, YellowPages, and "
            f"   local business directories.\n"
            f"3. Search for the company's official website and any 'About Us' or "
            f"   'Contact' pages that mention the legal entity name.\n"
            f"4. Note if the search name '{company_name}' appears as a trade name or DBA "
            f"   for a differently-named legal entity.\n"
            f"5. If location context is given, use it to narrow results to the right area.\n"
            f"6. Search at least 3 distinct sources before concluding nothing was found.\n\n"
            f"Return a concise summary (under 250 words) of: the legal entity name, "
            f"any known trade names / DBA names, the registered state, and the source URLs."
        )
        logger.debug(f"[company-match/deep] Single research: '{company_name}'")

        if provider == "gemini":
            return await self._research_gemini(task_prompt, company_name)
        return await self._research_openai(task_prompt, company_name)

    async def _deep_research_joint(self, company_a: str, company_b: str, location_hint: str, provider: str) -> str:
        """
        Joint research — searches both names together to surface DBA/trade-name links.

        This is the most targeted call for resolving local trade names:
        'Thomas Electrical' (trade) vs 'THOMAS P. OWENS, LLC' (legal entity).
        Asking directly 'are these the same?' is the fastest path to an answer.
        """
        loc_context = f" Location context: {location_hint}." if location_hint else ""
        task_prompt = (
            f"Business identity verification task:\n"
            f"Name A: {company_a!r}\n"
            f"Name B: {company_b!r}\n"
            f"{loc_context}\n\n"
            f"YOUR GOAL: Determine whether '{company_a}' and '{company_b}' refer to the "
            f"same real-world business — including the possibility that one is an informal "
            f"shorthand for a trade name / DBA of the other.\n\n"
            f"RESEARCH INSTRUCTIONS:\n"
            f"1. Search state contractor license databases and business registries for "
            f"   '{company_b}' — list ALL registered DBA names / trade names on file.\n"
            f"2. Check if '{company_a}' shares key words with any DBA of '{company_b}'.\n"
            f"   Example: if '{company_b}' has a DBA containing '{company_a.split()[0]}' "
            f"   and an industry word, that is a strong match signal.\n"
            f"3. Search for '{company_a}' in the same location — if the ONLY business "
            f"   matching '{company_a}' in that area is '{company_b}', they are the same.\n"
            f"4. Search BBB, Manta, YellowPages, and local directories for '{company_a}' "
            f"   to see if it resolves to '{company_b}'.\n"
            f"5. Extract the owner/principal name from '{company_b}' if present "
            f"   (e.g. 'Thomas P. Owens' from 'THOMAS P. OWENS, LLC') and search for "
            f"   that person alongside '{company_a}'.\n"
            f"6. Restrict results to the location context if provided.\n\n"
            f"Return a concise summary (under 300 words) stating:\n"
            f"- ALL known trade names / DBA names of '{company_b}' from registries\n"
            f"- Whether '{company_a}' matches or partially matches any of those names\n"
            f"- Whether the same location has a business matching '{company_a}'\n"
            f"- The source URLs for your findings"
        )
        label = f"{company_a} + {company_b}"
        logger.debug(f"[company-match/deep] Joint research: '{label}'")

        if provider == "gemini":
            return await self._research_gemini(task_prompt, label)
        return await self._research_openai(task_prompt, label)

    def _get_openai_sync_client(self) -> OpenAI:
        """Sync OpenAI client for the Responses API (web_search tool)."""
        if not settings.OPENAI_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="OPENAI_API_KEY is not configured — required for deep_search.",
            )
        return OpenAI(api_key=settings.OPENAI_API_KEY, max_retries=2)

    @retry_openai()
    async def _research_openai(self, task_prompt: str, label: str) -> str:
        """
        Single-turn OpenAI Responses API call with the web_search tool.

        Receives a rich task prompt (not a bare keyword query) so the model
        autonomously runs the right searches across the right source types.
        Returns the model's free-text research summary.
        """
        def _sync() -> str:
            client = self._get_openai_sync_client()

            @retry_openai()
            def _call():
                return client.responses.create(
                    model=settings.OPENAI_MODEL,
                    tools=[{"type": "web_search", "search_context_size": "medium"}],
                    input=task_prompt,
                )

            response = _call()
            return response.output_text or ""

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync)

    @retry_gemini()
    async def _research_gemini(self, task_prompt: str, label: str) -> str:
        """
        Single grounded Gemini call with google_search tool.

        Receives a rich task prompt (mirroring GeminiFindPoeService._investigate)
        so Gemini autonomously decides what searches to run and which sources to
        check, rather than treating the input as a literal keyword query.
        Returns the model's free-text research summary.
        """
        if not settings.GEMINI_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="GEMINI_API_KEY is not configured — required for deep_search.",
            )

        def _sync() -> str:
            from google import genai
            from google.genai import types as genai_types

            client = genai.Client(api_key=settings.GEMINI_API_KEY)
            grounding_tool = genai_types.Tool(
                google_search=genai_types.GoogleSearch()
            )
            response = client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=task_prompt,
                config=genai_types.GenerateContentConfig(
                    tools=[grounding_tool],
                    max_output_tokens=1024,
                ),
            )
            return response.text or ""

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync)


    # ------------------------------------------------------------------
    # Provider: Groq
    # ------------------------------------------------------------------

    def _get_groq_client(self) -> AsyncGroq:
        if not settings.GROQ_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="GROQ_API_KEY is not configured on this server.",
            )
        if self._groq_client is None:
            # max_retries=2 — inner SDK retry layer (reads Retry-After headers)
            self._groq_client = AsyncGroq(
                api_key=settings.GROQ_API_KEY, max_retries=2
            )
        return self._groq_client

    @retry_groq()
    async def _call_groq(
        self,
        company_a: str,
        company_b: str,
        research_a: str = "",
        research_b: str = "",
        research_joint: str = "",
    ) -> dict[str, Any]:
        client = self._get_groq_client()
        prompt = _build_user_prompt(company_a, company_b, research_a, research_b, research_joint)
        try:
            response = await client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
        except (GroqRateLimitError, GroqAPIStatusError, GroqAPIConnectionError, GroqAPITimeoutError):
            raise  # let retry_groq() handle it
        raw = response.choices[0].message.content or ""
        return _parse_llm_json(raw, "groq")

    # ------------------------------------------------------------------
    # Provider: OpenAI
    # ------------------------------------------------------------------

    def _get_openai_client(self) -> AsyncOpenAI:
        if not settings.OPENAI_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="OPENAI_API_KEY is not configured on this server.",
            )
        if self._openai_client is None:
            self._openai_client = AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY, max_retries=2
            )
        return self._openai_client

    @retry_openai()
    async def _call_openai(
        self,
        company_a: str,
        company_b: str,
        research_a: str = "",
        research_b: str = "",
        research_joint: str = "",
    ) -> dict[str, Any]:
        client = self._get_openai_client()
        prompt = _build_user_prompt(company_a, company_b, research_a, research_b, research_joint)
        try:
            response = await client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                # max_tokens was deprecated for o-series / gpt-4.1+ models;
                # max_completion_tokens is the correct param for all current models.
                # 1024 leaves room for the deep-search prompt + verbose reasoning.
                max_completion_tokens=1024,
                response_format={"type": "json_object"},
            )
        except (OAIRateLimitError, OAIAPIStatusError, OAIAPIConnectionError, OAIAPITimeoutError):
            raise  # let retry_openai() handle it
        raw = response.choices[0].message.content or ""
        return _parse_llm_json(raw, "openai")

    # ------------------------------------------------------------------
    # Provider: Gemini
    # ------------------------------------------------------------------

    @retry_gemini()
    async def _call_gemini(
        self,
        company_a: str,
        company_b: str,
        research_a: str = "",
        research_b: str = "",
        research_joint: str = "",
    ) -> dict[str, Any]:
        """
        Gemini's google-genai SDK is synchronous, so we offload the blocking
        call to the default ThreadPoolExecutor rather than blocking the event loop.

        response_schema is required alongside response_mime_type — without it,
        newer Gemini models still prepend conversational text ("Here is the JSON..")
        before the JSON block, breaking JSON parsing.  With response_schema the SDK
        returns structured JSON directly via response.parsed.
        """
        if not settings.GEMINI_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="GEMINI_API_KEY is not configured on this server.",
            )

        def _sync_call() -> dict[str, Any]:
            from google import genai
            from google.genai import types as genai_types
            from pydantic import BaseModel as _PydanticBase

            # Local schema — tells Gemini the exact JSON shape to produce.
            # Defined inside _sync_call so it stays private and doesn't pollute
            # the module namespace.
            class _MatchVerdict(_PydanticBase):
                is_same_company: bool
                confidence: float
                reasoning: str

            client = genai.Client(api_key=settings.GEMINI_API_KEY)
            prompt = _build_user_prompt(company_a, company_b, research_a, research_b, research_joint)

            response = client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    # system_instruction keeps the role separation clean —
                    # avoids concatenating system + user text into one blob.
                    system_instruction=_SYSTEM_PROMPT,
                    temperature=0,
                    # 1024 gives ample headroom for deep-search prompt + reasoning.
                    max_output_tokens=1024,
                    response_mime_type="application/json",
                    # response_schema forces the model to emit only structured JSON;
                    # the SDK populates response.parsed with a validated instance.
                    response_schema=_MatchVerdict,
                ),
            )

            # Prefer response.parsed (already validated by the SDK); fall back to
            # response.text only if parsed is unavailable (older SDK versions).
            if response.parsed is not None:
                verdict: _MatchVerdict = response.parsed
                return {
                    "is_same_company": verdict.is_same_company,
                    "confidence": float(verdict.confidence),
                    "reasoning": verdict.reasoning,
                }
            return {"_raw": response.text or ""}

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _sync_call)

        # If the SDK returned a pre-validated dict, pass it straight through;
        # otherwise fall back to JSON parsing the raw text.
        if "_raw" in result:
            return _parse_llm_json(result["_raw"], "gemini")
        return result
