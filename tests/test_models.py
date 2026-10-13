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
    assert task.message is None
    assert task.result_data is None
    assert task.created_at is not None
    assert task.updated_at is not None
