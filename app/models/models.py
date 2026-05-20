from typing import Optional, Dict, Any, List
from datetime import datetime
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
    HUMAN_RESOURCE = "Human Resource"
    PAYROLL = "Payroll"
    ADMIN = "Admin"
    CAREERS = "Careers"
    PERSONNEL = "Personnel"
    FINANCE = "Finance"
    SECRETARY = "Secretary"
    LABOR_RELATIONS = "Labor relations"
    OTHERS = "Others"

class TaggedContact(BaseModel):
    value: str
    tag: ContactTag
    context: Optional[str] = Field(default="", description="Context about where this contact was found or who it belongs to")

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
            
        valid_emails = []
        for item in v:
            if not item or not isinstance(item, str):
                continue
            item = item.strip()
            match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", item)
            if match:
                valid_emails.append(match.group(0).lower())
                
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
    company_info: OpenAICompanyInfo = OpenAICompanyInfo()


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
    verdict: str                # "VERIFIED" | "LIKELY" | "UNVERIFIED" | "CONTRADICTED"
    evidence_summary: str       # 2–3 sentence human-readable explanation
    sources_found: List[str]    # URLs or source names used as evidence


# --- Combined Search Models ---

class CombinedSearchRequest(BaseModel):
    """Request body for the POST /combined-search/ endpoint."""
    company_name: str
    country: Optional[str] = None
    zip_code: Optional[str] = None
    url: Optional[str] = None
    max_limit: Optional[int] = None


class CombinedCompanyInfo(BaseModel):
    """Structured company contact extracted by combining OpenAI and Gemini."""
    phones: List[TaggedContact] = Field(default_factory=list)
    faxes: List[TaggedContact] = Field(default_factory=list)
    emails: List[TaggedContact] = Field(default_factory=list)
    addresses: List[StructuredAddress] = Field(default_factory=list)


class CombinedSearchResult(BaseModel):
    """Full response envelope returned by the /combined-search/ endpoint."""
    company_name: str
    official_site: str = ""
    company_info: CombinedCompanyInfo = Field(default_factory=CombinedCompanyInfo)
    openai_result: Optional[OpenAISearchResult] = None
    gemini_result: Optional[GeminiSearchResult] = None


# --- External Webhook Payload Model ---
class WebhookPayload(BaseModel):
    status: str
    message: str
    result: ScrapeResult

# --- Database Model ---
class TaskRecord(SQLModel, table=True):
    task_id: str = Field(primary_key=True)
    status: str = Field(default="PENDING")  # PENDING, IN_PROGRESS, SUCCESS, FAILURE
    message: Optional[str] = Field(default=None)
    result_data: Optional[Dict[str, Any]] = Field(default=None, sa_type=JSON)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
