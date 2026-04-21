from typing import Optional, Dict, Any, List
from datetime import datetime
from sqlmodel import SQLModel, Field, JSON
from pydantic import BaseModel, field_validator, ConfigDict
import re

# --- API Request Model ---
class SearchRequest(BaseModel):
    poe_name: str
    timeout: Optional[int] = 120  # Default timeout in seconds

# --- LLM / Result Models ---
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


class OpenAICompanyInfo(BaseModel):
    """Structured company contact extracted by the OpenAI two-step pipeline."""
    phones: List[str] = Field(default_factory=list)
    faxes: List[str] = Field(default_factory=list)
    emails: List[str] = Field(default_factory=list)
    addresses: List[str] = Field(default_factory=list)



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


class GeminiCompanyInfo(BaseModel):
    """Structured company contact extracted by the Gemini two-step pipeline."""
    phones: List[str] = Field(default_factory=list)
    faxes: List[str] = Field(default_factory=list)
    emails: List[str] = Field(default_factory=list)
    addresses: List[str] = Field(default_factory=list)


class GeminiSearchResult(BaseModel):
    """Full response envelope returned by the /gemini-search/ endpoint."""
    company_name: str
    official_site: str = ""
    company_info: GeminiCompanyInfo = Field(default_factory=GeminiCompanyInfo)


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
