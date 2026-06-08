from typing import Optional, Dict, Any, List, Literal
from datetime import datetime, timezone
from sqlmodel import SQLModel, Field, JSON
from pydantic import BaseModel, field_validator, ConfigDict
import re
from enum import Enum

# --- API Request Model ---
class SearchRequest(BaseModel):
    poe_name: str
    timeout: Optional[int] = 120  # Default timeout in seconds

# --- Shared Contact Models ---
class ContactTag(str, Enum):
    # Core HR / Employment
    HUMAN_RESOURCE = "Human Resource"
    PAYROLL = "Payroll"
    PERSONNEL = "Personnel"
    BENEFITS = "Benefits"
    RECRUITING = "Recruiting"
    CANDIDATE_SUPPORT = "Candidate Support"
    TRAINING = "Training"
    BACKGROUND_CHECK = "Background Check"
    EMPLOYMENT_VERIFICATION = "Employment Verification"
    LABOR_RELATIONS = "Labor relations"
    # Business / Operations
    ADMIN = "Admin"
    MANAGEMENT = "Management"
    OPERATIONS = "Operations"
    FINANCE = "Finance"
    ACCOUNTING = "Accounting"
    LEGAL = "Legal"
    COMPLIANCE = "Compliance"
    IT = "IT"
    SAFETY = "Safety"
    # Customer-facing
    SALES = "Sales"
    CUSTOMER_SERVICE = "Customer Service"
    SUPPORT = "Support"
    MARKETING = "Marketing"
    PR = "Public Relations"
    # Administrative
    CAREERS = "Careers"
    SECRETARY = "Secretary"
    GENERAL = "General"
    OTHERS = "Others"


def _norm_tag(s: str) -> str:
    """Normalise a tag string: keep only letters, lowercase. Module-level for reuse."""
    return re.sub(r"[^a-z]", "", s.lower())


def _resolve_contact_tag(raw: Any) -> ContactTag:
    """
    Fuzzy resolver for ContactTag that survives LLM label drift.

    Resolution order:
      1. Already a ContactTag instance – return as-is.
      2. Exact match against enum *values*  (e.g. "Human Resource").
      3. Case-insensitive match against enum values (e.g. "human resource").
      4. Normalised match – strip all non-alpha chars, lowercase, compare
         against normalised enum values AND enum keys
         (e.g. "HumanResources", "HUMANRESOURCES", "human_resource").
      5. Short-abbreviation prefix match (input ≤ 4 chars only, e.g. "HR", "IT").
         The length guard prevents broad false matches like "itmanager" → IT.
      6. Fall back to ContactTag.OTHERS and log a warning.
    """
    if isinstance(raw, ContactTag):
        return raw

    if not isinstance(raw, str):
        return ContactTag.OTHERS

    # Step 2 – exact
    try:
        return ContactTag(raw)
    except ValueError:
        pass

    stripped = raw.strip()

    # Step 3 – case-insensitive value match
    lower = stripped.lower()
    for member in ContactTag:
        if member.value.lower() == lower:
            return member

    # Step 4 – normalised match (keep letters only, lowercase)
    norm_input = _norm_tag(stripped)
    for member in ContactTag:
        if _norm_tag(member.value) == norm_input or _norm_tag(member.name) == norm_input:
            return member

    # Step 5 – short abbreviation prefix match only (≤ 4 chars, e.g. "HR", "IT").
    # The length guard prevents false positives like "itmanager" → IT.
    if len(norm_input) <= 4:
        for member in ContactTag:
            if _norm_tag(member.value).startswith(norm_input):
                return member

    # Step 6 – unknown; coerce gracefully
    from loguru import logger
    logger.warning(
        f"[ContactTag] Unknown tag {raw!r} from LLM — coercing to 'Others'"
    )
    return ContactTag.OTHERS

# Cloudflare obfuscated placeholder and other known fake patterns to reject outright.
_BLOCKED_EMAILS: set[str] = {
    "[email protected]",
}
_BLOCKED_EMAIL_DOMAINS: tuple[str, ...] = ("cloudflare.com",)
_PLACEHOLDER_PREFIXES: tuple[str, ...] = (
    "email@", "example@", "name@", "abc@", "user@", "test@", "noreply@",
)
# Compiled once at module level for efficiency.
# Domain part requires proper labels (alphanumeric + hyphens) separated by single dots.
# This rejects patterns like "name@...com" or "user@foo..bar.com".
_EMAIL_REGEX = re.compile(
    r"^[a-zA-Z0-9_.+\-]+"           # local part
    r"@"
    r"[a-zA-Z0-9]"                   # domain: must start with alphanumeric
    r"([a-zA-Z0-9\-]*[a-zA-Z0-9])?" # domain: optional middle chars (no leading/trailing hyphen)
    r"(\.[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?)+$"  # one or more .label segments
)


class TaggedContact(BaseModel):
    value: str
    tag: ContactTag
    context: Optional[str] = Field(default="", description="Context about where this contact was found or who it belongs to")

    @field_validator("value", mode="before")
    @classmethod
    def reject_placeholder_emails(cls, v: Any) -> Any:
        """Validate email format and block Cloudflare obfuscation placeholders and dummy emails.

        NOTE: TaggedContact is shared by phones, faxes, and emails.
        Email-format checks are only applied when the value contains '@'.
        Phone/fax values (no '@') are passed through untouched.
        """
        if not isinstance(v, str):
            return v
        stripped = v.strip()
        lower = stripped.lower()

        # --- Not an email (phone/fax/etc.) — skip all email checks ---
        if "@" not in lower:
            return stripped

        # --- Step 1: Must be a valid email address format ---
        # Rejects hallucinations like "contact form on website" or "name@...com"
        if not _EMAIL_REGEX.match(stripped):
            raise ValueError(f"Not a valid email address format: {v!r}")

        # --- Step 2: Block known Cloudflare obfuscation placeholders ---
        if lower in _BLOCKED_EMAILS:
            raise ValueError(f"Blocked placeholder email: {v!r}")

        # --- Step 3: Block known bad domains ---
        for domain in _BLOCKED_EMAIL_DOMAINS:
            if lower.endswith(f"@{domain}") or f"@{domain}" in lower:
                raise ValueError(f"Blocked Cloudflare domain email: {v!r}")

        # --- Step 4: Block placeholder-style prefixes ---
        for prefix in _PLACEHOLDER_PREFIXES:
            if lower.startswith(prefix):
                raise ValueError(f"Blocked placeholder-prefix email: {v!r}")

        return stripped

    @field_validator("tag", mode="before")
    @classmethod
    def coerce_tag(cls, v: Any) -> ContactTag:
        """Fuzzy-resolve LLM tag labels; falls back to OTHERS on no match."""
        return _resolve_contact_tag(v)

class StructuredAddress(BaseModel):
    address1: str = Field(default="", description="First line of the address (e.g., street, building)")
    address2: str = Field(default="", description="Second line of the address (e.g., suite, apartment)")
    city: str = Field(default="")
    state: str = Field(default="")
    zip: str = Field(default="")
    country: str = Field(default="")
    countryCode: str = Field(default="")
    tag: ContactTag
    context: Optional[str] = Field(default=None, description="Context about where this contact was found or who it belongs to")

    @field_validator("tag", mode="before")
    @classmethod
    def coerce_tag(cls, v: Any) -> ContactTag:
        """Fuzzy-resolve LLM tag labels; falls back to OTHERS on no match."""
        return _resolve_contact_tag(v)

# --- Legacy / Result Models ---
class ContactInfo(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    
    Phone: List[str] = Field(default_factory=list)
    Fax: List[str] = Field(default_factory=list)
    Email: List[str] = Field(default_factory=list)
    Address: List[str] = Field(default_factory=list)
    DeptContacts: Optional[Dict[str, Any]] = None

    @field_validator('Phone', 'Fax', mode='before')
    @classmethod
    def validate_phone(cls, v: Any) -> List[str]:
        if not v:
            return []
        if isinstance(v, str):
            v = [v]
        elif not isinstance(v, list):
            return []
            
        valid_numbers = []
        for item in v:
            if not item or not isinstance(item, str):
                continue
            item = item.strip()
            digits_only = re.sub(r"[^0-9]", "", item)
            if 6 <= len(digits_only) <= 18:
                valid_numbers.append(item)
        return list(dict.fromkeys(valid_numbers))  # Remove duplicates

    @field_validator('Email', mode='before')
    @classmethod
    def validate_email(cls, v: Any) -> List[str]:
        if not v:
            return []
        if isinstance(v, str):
            v = [v]
        elif not isinstance(v, list):
            return []

        # Known Cloudflare obfuscation placeholder — always reject.
        _CF_PLACEHOLDER = "[email protected]"

        valid_emails = []
        for item in v:
            if not item or not isinstance(item, str):
                continue
            item = item.strip()
            if item.lower() == _CF_PLACEHOLDER:
                continue
            # Use the same authoritative regex as TaggedContact — rejects
            # formats like "contact form on website", "name@...com", etc.
            if _EMAIL_REGEX.match(item):
                valid_emails.append(item.lower())

        return list(dict.fromkeys(valid_emails))  # Remove duplicates

class ScrapeResult(BaseModel):
    poe_name: str
    official_site: str
    poe_info: Optional[ContactInfo] = None


# --- OpenAI Search Models ---

class OpenAISearchRequest(BaseModel):
    """Request body for the POST /openai-search/ endpoint."""
    company_name: str  # e.g. "DEMOULAS MARKET BASKET"
    country: Optional[str] = None
    zip_code: Optional[str] = None
    url: Optional[str] = None
    max_limit: Optional[int] = None


class OpenAICompanyInfo(BaseModel):
    """Structured company contact extracted by the OpenAI two-step pipeline."""
    phones: List[TaggedContact] = Field(default_factory=list)
    faxes: List[TaggedContact] = Field(default_factory=list)
    emails: List[TaggedContact] = Field(default_factory=list)
    addresses: List[StructuredAddress] = Field(default_factory=list)



class OpenAISearchResult(BaseModel):
    """Full response envelope returned by the /openai-search/ endpoint."""
    company_name: str
    official_site: str = ""
    company_info: OpenAICompanyInfo = Field(default_factory=OpenAICompanyInfo)


# --- Gemini Search Models ---

class GeminiSearchRequest(BaseModel):
    """Request body for the POST /gemini-search/ endpoint."""
    company_name: str  # e.g. "DEMOULAS MARKET BASKET"
    country: Optional[str] = None
    zip_code: Optional[str] = None
    url: Optional[str] = None
    max_limit: Optional[int] = None


class GeminiCompanyInfo(BaseModel):
    """Structured company contact extracted by the Gemini two-step pipeline."""
    phones: List[TaggedContact] = Field(default_factory=list)
    faxes: List[TaggedContact] = Field(default_factory=list)
    emails: List[TaggedContact] = Field(default_factory=list)
    addresses: List[StructuredAddress] = Field(default_factory=list)


class GeminiSearchResult(BaseModel):
    """Full response envelope returned by the /gemini-search/ endpoint."""
    company_name: str
    official_site: str = ""
    company_info: GeminiCompanyInfo = Field(default_factory=GeminiCompanyInfo)


# --- VOE (Verification of Employment) Models ---

class VoeRequest(BaseModel):
    """Request body for the POST /verify-voe/ endpoint."""
    full_name: str          # e.g. "Jane Doe"
    job_title: str          # e.g. "Senior Software Engineer"
    company: str            # e.g. "Acme Corp"
    zip_code: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None


class VoeVerificationResult(BaseModel):
    """Structured result returned by the /verify-voe/ endpoint."""
    full_name: str
    company: str
    job_title: str
    confidence_score: float     # 0.0 – 10.0  (calibrated rubric applied in prompt)
    verdict: Literal["VERIFIED", "LIKELY", "UNVERIFIED", "CONTRADICTED"]
    evidence_summary: str       # 2–3 sentence human-readable explanation
    sources_found: List[str]    # URLs or source names used as evidence


# --- Combined Search Models ---

class CombinedSearchRequest(BaseModel):
    """Request body for the POST /combined-search/ endpoint."""
    company_name: str
    country: Optional[str] = "United States"
    zip_code: Optional[str] = None
    url: Optional[str] = None
    max_limit: Optional[int] = None


class CombinedCompanyInfo(BaseModel):
    """Structured company contact extracted by combining OpenAI and Gemini."""
    phones: List[TaggedContact] = Field(default_factory=list)
    faxes: List[TaggedContact] = Field(default_factory=list)
    emails: List[TaggedContact] = Field(default_factory=list)
    addresses: List[StructuredAddress] = Field(default_factory=list)


class SourceStats(BaseModel):
    """Count of each contact type returned by a single source (OpenAI or Gemini)."""
    total_phones: int = 0
    total_faxes: int = 0
    total_emails: int = 0
    total_addresses: int = 0


class CombinedSearchSummary(BaseModel):
    """
    High-level count summary for the /combined-search/ response.
    
    - ``openai``  – raw counts as returned by the OpenAI pipeline (before dedup).
    - ``gemini``  – raw counts as returned by the Gemini pipeline (before dedup).
    - ``combined`` – deduplicated counts that appear in ``company_info``
                     (these respect ``max_limit`` when applied).
    """
    openai: SourceStats = Field(default_factory=SourceStats)
    gemini: SourceStats = Field(default_factory=SourceStats)
    combined: SourceStats = Field(default_factory=SourceStats)


class CombinedSearchResult(BaseModel):
    """Full response envelope returned by the /combined-search/ endpoint."""
    company_name: str
    official_site: str = ""
    company_info: CombinedCompanyInfo = Field(default_factory=CombinedCompanyInfo)
    summary: CombinedSearchSummary = Field(default_factory=CombinedSearchSummary)
    openai_result: Optional[OpenAISearchResult] = None
    gemini_result: Optional[GeminiSearchResult] = None


# --- External Webhook Payload Model ---
class WebhookPayload(BaseModel):
    status: str
    message: str
    result: ScrapeResult

# --- PDF Extraction Models ---

class PdfExtractionResult(BaseModel):
    """Response envelope returned by the synchronous POST /extract-pdf/ endpoint."""
    filename: str                                          # Original uploaded filename
    provider: Literal["gemini", "openai"]                 # LLM provider used
    page_count: int                                        # Number of pages detected
    extracted_text: str                                    # Full concatenated text
    processing_time_seconds: float                         # Wall-clock time for the LLM call



# --- Background Check Parser Models ---

class BgCheckFields(BaseModel):
    """
    Structured fields extracted from a background check / screening report.

    All fields default to ``""`` when not found in the document so that
    downstream consumers always receive a consistent flat JSON object.
    Dates are normalised to ISO 8601 (YYYY-MM-DD) by the extraction prompt.
    """
    file_number: str = Field(
        default="",
        description="Case / order / reference identifier (e.g. 'BGC-2024-00123').",
    )
    employee_name: str = Field(
        default="",
        description="Full name of the subject / applicant being screened.",
    )
    date_of_birth: str = Field(
        default="",
        description="Subject's date of birth in YYYY-MM-DD format, or ''.",
    )
    requested_by: str = Field(
        default="",
        description="Name or organisation that ordered the background check.",
    )
    employer_name: str = Field(
        default="",
        description="Employer / client company named on the report.",
    )
    position: str = Field(
        default="",
        description="Job title or position the applicant is being considered for.",
    )
    report_date: str = Field(
        default="",
        description="Report generation / order date in YYYY-MM-DD format, or ''.",
    )
    status: str = Field(
        default="",
        description=(
            "Overall report status as it appears on the document "
            "(e.g. 'Clear', 'Consider', 'Adverse Action')."
        ),
    )


class BgCheckParseResult(BaseModel):
    """Full response envelope returned by POST /parse-background-check/."""
    filename: str = Field(description="Original uploaded filename.")
    provider: Literal["gemini", "openai"] = Field(description="LLM provider used.")
    processing_time_seconds: float = Field(
        description="Total wall-clock time for both pipeline stages."
    )
    data: BgCheckFields = Field(
        default_factory=BgCheckFields,
        description="Extracted structured fields from the background check PDF.",
    )


# --- Database Model ---

class TaskRecord(SQLModel, table=True):
    task_id: str = Field(primary_key=True)
    status: str = Field(default="PENDING")  # PENDING, IN_PROGRESS, SUCCESS, FAILURE
    message: Optional[str] = Field(default=None)
    result_data: Optional[Dict[str, Any]] = Field(default=None, sa_type=JSON)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
