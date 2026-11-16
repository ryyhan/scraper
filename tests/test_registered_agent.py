"""Tests for Registered Agent extraction, grounding, and combined resolution."""

from app.models import (
    CombinedSearchResult,
    GeminiSearchResult,
    OpenAISearchResult,
    RegisteredAgent,
)
from app.services._registered_agent import ground_registered_agent, is_grounded, phone_is_grounded
from app.services.combined_search import _resolve_registered_agent


# --- Model ---

def test_registered_agent_cleans_whitespace_and_empty_optionals():
    a = RegisteredAgent(name="  CT   Corporation  System ", designation="", state=" DE ", address=None)
    assert a.name == "CT Corporation System"
    assert a.designation is None
    assert a.state == "DE"
    assert RegisteredAgent(name="CSC", phone="  ").phone is None
    assert RegisteredAgent(name="CSC", phone=" (800) 927-9801 ").phone == "(800) 927-9801"


def test_result_models_coerce_bad_registered_agent_to_none():
    for cls in (OpenAISearchResult, GeminiSearchResult, CombinedSearchResult):
        assert cls(company_name="Acme", registered_agent={"name": "N/A"}).registered_agent is None
        assert cls(company_name="Acme", registered_agent={"state": "DE"}).registered_agent is None
        assert cls(company_name="Acme", registered_agent=42).registered_agent is None
        assert cls(company_name="Acme").registered_agent is None


def test_result_models_accept_valid_and_bare_string_agent():
    r = OpenAISearchResult(
        company_name="Acme",
        registered_agent={"name": "Corporation Service Company", "designation": "Registered Agent", "state": "DE"},
    )
    assert r.registered_agent.name == "Corporation Service Company"
    assert OpenAISearchResult(company_name="Acme", registered_agent="Jane Doe").registered_agent.name == "Jane Doe"


# --- Grounding ---

def test_is_grounded():
    text = "Registered Agent: C T Corporation System, 1209 Orange St. Resident agent John Q. Smith."
    assert is_grounded("C T Corporation System", text)
    assert is_grounded("c t corporation system.", text)
    assert is_grounded("Smith, John", text)            # reordered tokens
    assert not is_grounded("Corporation Service Company", text)
    assert not is_grounded("Bob", "Robert Smith")      # single token needs substring
    assert not is_grounded("", text)


def test_ground_registered_agent_clears_hallucination_only():
    r = OpenAISearchResult(company_name="Acme", registered_agent={"name": "Invented Agents LLC"})
    ground_registered_agent(r, "Registered agent: Real Agent Inc", "test", "Acme")
    assert r.registered_agent is None

    r = OpenAISearchResult(company_name="Acme", registered_agent={"name": "Real Agent Inc"})
    ground_registered_agent(r, "Registered agent: Real Agent Inc", "test", "Acme")
    assert r.registered_agent is not None


def test_phone_is_grounded():
    text = "Agent: CSC, tel 800.927.9801, 251 Little Falls Dr"
    assert phone_is_grounded("(800) 927-9801", text)
    assert phone_is_grounded("+1 800 927 9801", text)      # leading country code tolerated
    assert not phone_is_grounded("(800) 111-2222", text)
    assert not phone_is_grounded("123", "123")              # too short to be a phone


def test_ground_drops_ungrounded_phone_but_keeps_agent():
    r = OpenAISearchResult(company_name="Acme", registered_agent={"name": "Real Agent Inc", "phone": "800-111-2222"})
    ground_registered_agent(r, "Registered agent: Real Agent Inc, 800-927-9801", "test", "Acme")
    assert r.registered_agent is not None and r.registered_agent.phone is None

    r = OpenAISearchResult(company_name="Acme", registered_agent={"name": "Real Agent Inc", "phone": "(800) 927-9801"})
    ground_registered_agent(r, "Registered agent: Real Agent Inc, 800-927-9801", "test", "Acme")
    assert r.registered_agent.phone == "(800) 927-9801"


# --- Combined resolution ---

def test_resolve_single_provider():
    a = RegisteredAgent(name="CSC")
    assert _resolve_registered_agent(a, None, "Acme") is a
    assert _resolve_registered_agent(None, a, "Acme") is a
    assert _resolve_registered_agent(None, None, "Acme") is None


def test_resolve_agreement_merges_missing_fields():
    oai = RegisteredAgent(name="C T Corporation System", state="DE")
    gem = RegisteredAgent(name="CT CORPORATION SYSTEM.", designation="Statutory Agent", state="TX", address="1 Main St", phone="800-927-9801")
    out = _resolve_registered_agent(oai, gem, "Acme")
    assert out.name == "C T Corporation System"
    assert out.state == "DE"                       # OpenAI value wins when present
    assert out.designation == "Statutory Agent"    # filled from Gemini
    assert out.address == "1 Main St"
    assert out.phone == "800-927-9801"


def test_resolve_mismatch_defaults_to_openai():
    oai = RegisteredAgent(name="Agent One")
    gem = RegisteredAgent(name="Agent Two")
    assert _resolve_registered_agent(oai, gem, "Acme") is oai
