"""
Shared Registered Agent helpers for the OpenAI and Gemini search pipelines.

Keeps the prompt wording and the anti-hallucination grounding check in one
place so both providers stay in sync (mirrors how EIN is handled).
"""

import re
from typing import Optional

from loguru import logger

from app.models import RegisteredAgent

# States use different names for the same role.
REGISTERED_AGENT_TERMS = (
    "Registered Agent, Resident Agent, Statutory Agent, Agent for Service of Process, "
    "Commercial Clerk, Registered Office Provider, National Registered Agent"
)


def research_query(label: str, company_name: str) -> str:
    """Search-strategy line (prefixed with *label*, e.g. ``"o3"``) used by the gather turns."""
    return (
        f"  {label}) '{company_name}' (\"registered agent\" OR \"resident agent\" OR \"statutory agent\" "
        f"OR \"agent for service of process\") secretary of state business entity filing "
        f"(also try site:opencorporates.com OR site:bizapedia.com)\n"
    )


RESEARCH_GOAL = (
    "Identify the company's Registered Agent (the person or company designated to receive legal "
    "notices, as recorded with the Secretary of State / state business registry). "
    f"States call this role by different names: {REGISTERED_AGENT_TERMS}. "
    "Record the agent's NAME, the term the source used, the state, and the agent's address and PHONE NUMBER if listed."
)

EXTRACTION_RULE = (
    "Fill 'registered_agent' with the company's Registered Agent as an object "
    "{\"name\", \"designation\", \"state\", \"address\", \"phone\"}. The registered agent is the person or company "
    "(e.g. 'CT Corporation System') designated to receive legal notices, found in state business "
    "registry / Secretary of State filings, SEC filings, or business-entity directories. "
    f"The role may be labelled any of: {REGISTERED_AGENT_TERMS} — treat them all as the registered agent. "
    "'name' is the agent's name exactly as written in the research; 'designation' is the label the source "
    "used; 'state' is the state the agent is registered in; 'address' is a single-line address or null; 'phone' is the agent's own phone number as written in the research, or null (never use the company's phone number unless the source lists it for the agent). "
    "Do NOT use the company's own officers, owners, or contacts unless the research explicitly says they are the "
    "registered agent. If not found or uncertain, set 'registered_agent' to null. NEVER invent or guess an agent.\n"
)

VERIFY_NOTE = (
    "- Preserve 'registered_agent' unchanged (or set it to null if the agent's name does not appear "
    "in RAW RESEARCH TEXT; set its 'phone' to null if the phone's digits do not appear in RAW RESEARCH TEXT).\n"
)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def is_grounded(name: str, research_text: str) -> bool:
    """
    True if *name* is supported by *research_text*.

    Accepts a normalised substring match, or — for multi-word names — every
    word present in the text, so reordered forms ("Smith, John" vs
    "John Smith") are not rejected.  Single-word names need a substring match.
    """
    n = _norm(name)
    if not n:
        return False
    haystack = _norm(research_text)
    if n in haystack:
        return True
    tokens = n.split()
    if len(tokens) < 2:
        return False
    words = set(haystack.split())
    return all(t in words for t in tokens)


def phone_is_grounded(phone: str, research_text: str) -> bool:
    """Digit-only match (ignores formatting). Needs at least 7 digits to count as a phone."""
    digits = re.sub(r"\D", "", phone)
    if len(digits) < 7:
        return False
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]  # tolerate a leading US country code
    return digits in re.sub(r"\D", "", research_text)


def ground_registered_agent(
    result: object, research_text: str, log_prefix: str, company_name: str
) -> None:
    """Null out ``result.registered_agent`` if its name is not found in *research_text*."""
    agent: Optional[RegisteredAgent] = getattr(result, "registered_agent", None)
    if not agent:
        return
    if not is_grounded(agent.name, research_text):
        logger.warning(
            f"[{log_prefix}] Step 3: removing unverified registered agent {agent.name!r} "
            f"not found in raw research text for {company_name!r}"
        )
        result.registered_agent = None  # type: ignore[attr-defined]
        return
    if agent.phone and not phone_is_grounded(agent.phone, research_text):
        logger.warning(
            f"[{log_prefix}] Step 3: removing unverified registered agent phone {agent.phone!r} "
            f"not found in raw research text for {company_name!r}"
        )
        agent.phone = None
