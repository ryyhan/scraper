"""
Unit tests for data models, validation, and sanitization in app.models.models.
"""

import pytest
from app.models import (
    ContactTag,
    TaggedContact,
    ContactInfo,
    HintAddress,
    ProviderTokenUsage,
    TokenUsage,
    TaskRecord,
)
from app.models.models import _resolve_contact_tag


def test_contact_tag_fuzzy_resolution():
    """Verify robust resolution of LLM departmental tags."""
    # Exact match
    assert _resolve_contact_tag("Human Resource") == ContactTag.HUMAN_RESOURCE
    assert _resolve_contact_tag("Payroll") == ContactTag.PAYROLL
    assert _resolve_contact_tag("Headquarters") == ContactTag.HEADQUARTERS

    # Case-insensitive
    assert _resolve_contact_tag("human resource") == ContactTag.HUMAN_RESOURCE
    assert _resolve_contact_tag("payroll") == ContactTag.PAYROLL

    # Normalised (strips non-alpha)
    assert _resolve_contact_tag("Human_Resource") == ContactTag.HUMAN_RESOURCE
    assert _resolve_contact_tag("HumanResources") == ContactTag.HUMAN_RESOURCE
    assert _resolve_contact_tag("human-resource") == ContactTag.HUMAN_RESOURCE

    # Short abbreviation prefix match
    assert _resolve_contact_tag("HR") == ContactTag.HUMAN_RESOURCE
    assert _resolve_contact_tag("IT") == ContactTag.IT

    # Unknown label coercion
    assert _resolve_contact_tag("unknown_department_xyz") == ContactTag.OTHERS
    assert _resolve_contact_tag(None) == ContactTag.OTHERS
    assert _resolve_contact_tag(12345) == ContactTag.OTHERS


def test_tagged_contact_email_validation():
    """Verify TaggedContact accepts valid emails and rejects placeholders/dummy domains."""
    # Valid email
    c1 = TaggedContact(value="hr@company.org", tag="Human Resource")
    assert c1.value == "hr@company.org"
    assert c1.tag == ContactTag.HUMAN_RESOURCE

    # Cloudflare obfuscation placeholder rejected
    with pytest.raises(ValueError):
        TaggedContact(value="[email protected]", tag="Human Resource")

    # Blocked dummy domains rejected
    with pytest.raises(ValueError):
        TaggedContact(value="john@example.com", tag="General")
    with pytest.raises(ValueError):
        TaggedContact(value="test@cloudflare.com", tag="General")
    with pytest.raises(ValueError):
        TaggedContact(value="ceo@company.com", tag="Management")

    # Blocked placeholder prefixes rejected
    with pytest.raises(ValueError):
        TaggedContact(value="email@somefirm.com", tag="General")
    with pytest.raises(ValueError):
        TaggedContact(value="example@somefirm.com", tag="General")

    # Malformed email strings (containing @ but invalid format) rejected
    with pytest.raises(ValueError):
        TaggedContact(value="user@...com", tag="General")
    with pytest.raises(ValueError):
        TaggedContact(value="@domain.com", tag="General")
    with pytest.raises(ValueError):
        TaggedContact(value="missing-domain@", tag="General")

    # Phone numbers (no @) are passed through without email checks
    c_phone = TaggedContact(value="(555) 234-5678", tag="Headquarters")
    assert c_phone.value == "(555) 234-5678"


def test_contact_info_phone_and_email_validation():
    """Verify ContactInfo filters invalid phones and emails."""
    info = ContactInfo(
        Phone=["(555) 234-5678", "123", "(555) 234-5678"],  # "123" too short, duplicate removed
        Email=["info@legit.com", "[email protected]", "info@legit.com", "bad-format"],
    )
    assert len(info.Phone) == 1
    assert info.Phone[0] == "(555) 234-5678"

    assert len(info.Email) == 1
    assert info.Email[0] == "info@legit.com"


def test_hint_address_to_prompt_string():
    """Verify HintAddress renders correctly into prompt format."""
    addr = HintAddress(
        address1="100 Main St",
        city="Austin",
        state="TX",
        country="United States",
    )
    assert addr.to_prompt_string() == "100 Main St, Austin, TX, United States"

    empty_addr = HintAddress()
    assert empty_addr.to_prompt_string() == ""


def test_token_usage_arithmetic():
    """Verify ProviderTokenUsage accumulation."""
    u1 = ProviderTokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
    u2 = ProviderTokenUsage(input_tokens=200, output_tokens=80, total_tokens=280)
    combined = u1 + u2
    assert combined.input_tokens == 300
    assert combined.output_tokens == 130
    assert combined.total_tokens == 430

    token_usage = TokenUsage(openai=u1, gemini=u2, grand_total=combined)
    assert token_usage.grand_total.total_tokens == 430


def test_task_record_model():
    """Verify TaskRecord default values and field assignment."""
    task = TaskRecord(task_id="test-uuid-123")
    assert task.status == "PENDING"
    assert task.task_type == "general"
    assert task.message is None
    assert task.result_data is None
    assert task.created_at is not None
    assert task.updated_at is not None


def test_company_info_address_sanitization():
    """Verify that malformed address items are dropped without failing the entire model."""
    from app.models import OpenAICompanyInfo, GeminiCompanyInfo, StructuredAddress

    raw_addresses = [
        {"address1": "123 Main St", "city": "Dallas", "state": "TX", "tag": "Headquarters"},
        "invalid-address-format",
        StructuredAddress(address1="456 Side St", city="Plano", state="TX", tag=ContactTag.CORPORATE),
    ]

    info = OpenAICompanyInfo(addresses=raw_addresses)
    assert len(info.addresses) == 2
    assert info.addresses[0].address1 == "123 Main St"
    assert info.addresses[1].address1 == "456 Side St"

    gemini_info = GeminiCompanyInfo(addresses=raw_addresses)
    assert len(gemini_info.addresses) == 2


def test_validate_and_normalize_ein():
    """Verify EIN/FEIN validation, normalisation, and dummy/SSN rejection."""
    from app.models import validate_and_normalize_ein

    # Valid standard formatted EIN (prefix 12 is valid IRS campus, non-dummy sequence)
    assert validate_and_normalize_ein("12-3456780") == "12-3456780"
    assert validate_and_normalize_ein("54-1234567") == "54-1234567"
    assert validate_and_normalize_ein("  20-9876543  ") == "20-9876543"

    # Raw 9 digits normalised to XX-XXXXXXX
    assert validate_and_normalize_ein("123456780") == "12-3456780"
    assert validate_and_normalize_ein("541234567") == "54-1234567"

    # SSN format (3-2-4) strictly rejected
    assert validate_and_normalize_ein("123-45-6789") is None
    assert validate_and_normalize_ein("987-65-4321") is None

    # Dummy/placeholder patterns rejected
    assert validate_and_normalize_ein("00-0000000") is None
    assert validate_and_normalize_ein("12-3456789") is None
    assert validate_and_normalize_ein("11-1111111") is None
    assert validate_and_normalize_ein("99-9999999") is None

    # Invalid / unassigned IRS prefixes rejected
    assert validate_and_normalize_ein("00-1234567") is None
    assert validate_and_normalize_ein("07-1234567") is None
    assert validate_and_normalize_ein("96-1234567") is None

    # Non-digits or wrong length rejected
    assert validate_and_normalize_ein("12-3456") is None
    assert validate_and_normalize_ein("12-345678901") is None
    assert validate_and_normalize_ein("12-ABCDEFG") is None

    # Null / empty / placeholder strings coerced to None
    assert validate_and_normalize_ein(None) is None
    assert validate_and_normalize_ein("") is None
    assert validate_and_normalize_ein("   ") is None
    assert validate_and_normalize_ein("N/A") is None
    assert validate_and_normalize_ein("None") is None
    assert validate_and_normalize_ein("not_found") is None


def test_ein_on_search_result_models():
    """Verify that all POE search result models accept and validate the ein field."""
    from app.models import (
        ScrapeResult,
        OpenAISearchResult,
        GeminiSearchResult,
        CombinedSearchResult,
        ContactInfo,
    )

    # ScrapeResult
    sr1 = ScrapeResult(poe_name="Acme", official_site="https://acme.com", ein="12-3456780")
    assert sr1.ein == "12-3456780"

    # OpenAISearchResult with raw digits
    oai1 = OpenAISearchResult(company_name="Acme", ein="541234567")
    assert oai1.ein == "54-1234567"

    # GeminiSearchResult with invalid EIN coerces to None
    gem1 = GeminiSearchResult(company_name="Acme", ein="invalid-ein-placeholder")
    assert gem1.ein is None

    # CombinedSearchResult with valid EIN
    comb1 = CombinedSearchResult(company_name="Acme", ein="20-9876543")
    assert comb1.ein == "20-9876543"

    # ContactInfo with EIN
    ci = ContactInfo(EIN="12-3456780")
    assert ci.EIN == "12-3456780"

