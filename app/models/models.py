from typing import Optional, Dict, Any
from datetime import datetime
from sqlmodel import SQLModel, Field, JSON
from pydantic import BaseModel

# --- API Request Model ---
class SearchRequest(BaseModel):
    poe_name: str
    timeout: Optional[int] = 120  # Default timeout in seconds

# --- LLM / Result Models ---
class ContactInfo(BaseModel):
    Phone: str
    Email: str
    Address: str
    DeptContacts: Optional[Dict[str, Any]] = None

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
    result_data: Optional[Dict[str, Any]] = Field(default=None, sa_type=JSON)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
