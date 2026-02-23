from typing import Optional, Dict, Any
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
    
    Phone: str
    Fax: str = ""
    Email: str
    Address: str
    City: str = ""
    State: str = ""
    ZipCode: str = ""
    DeptContacts: Optional[Dict[str, Any]] = None

    @field_validator('Phone', 'Fax', mode='before')
    @classmethod
    def validate_phone(cls, v: Any) -> str:
        if not v or not isinstance(v, str):
            return ""
        v = v.strip()
        
        # Strip all formatting characters to see if we have actual digits left
        digits_only = re.sub(r"[^0-9]", "", v)
        
        # Most North American / International numbers range between 6 and 18 digits
        if len(digits_only) < 6 or len(digits_only) > 18:
            return ""
            
        # Return the original formatted string if it passed the length check
        return v

    @field_validator('Email', mode='before')
    @classmethod
    def validate_email(cls, v: Any) -> str:
        if not v or not isinstance(v, str):
            return ""
        v = v.strip()
        
        # Extract the FIRST valid email address from the string using regex
        match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", v)
        if match:
            return match.group(0).lower()
            
        return ""

class ScrapeResult(BaseModel):
    poe_name: str
    official_site: str
    poe_info: Optional[ContactInfo] = None
    
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
