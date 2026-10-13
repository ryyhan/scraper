import json
from typing import Type, TypeVar
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI

client = OpenAI()

T = TypeVar("T", bound=BaseModel)

# ----------------------------
# 1. Updated Schema with Singular Constraints
# ----------------------------
class CompanyInfo(BaseModel):
    # The 'description' helps the model understand it should only pick ONE.
    phone: str = Field(
        default="", 
        description="A single primary phone number. Prefer HR. No lists or multiple numbers."
    )
    fax: str = Field(
        default="", 
        description="A single fax number. Empty string if not found."
    )
    email: str = Field(
        default="", 
        description="A single primary email address. Prefer HR. Do not include multiple."
    )
    address: str = Field(
        default="", 
        description="The single main office or headquarters address."
    )

class Result(BaseModel):
    company_name: str = ""
    official_site: str = ""
    company_info: CompanyInfo = Field(default_factory=CompanyInfo)

# ----------------------------
# 2. Refined Two-Step Call
# ----------------------------
def structured_llm_call(company_query: str, model_class: Type[T]) -> T:
    # STEP 1: Deep Research (The "Gatherer")
    research_prompt = (
        f"Deep research task: Find contact information for {company_query}.\n\n"
        "GOALS:\n"
        "1. Prioritize Human Resources (HR) contact info.\n"
        "2. If HR is unavailable, find General/Corporate info.\n"
        "3. Look for a Fax number.\n"
        "4. Find as much as possible so the next step can choose the best single option."
    )

    research_response = client.responses.create(
        model="gpt-5-mini",
        tools=[{"type": "web_search"}],
        input=research_prompt
    )
    raw_research = research_response.output_text

    # STEP 2: Extraction (The "Filter & Formatter")
    extraction_prompt = (
        f"Based on this research:\n{raw_research}\n\n"
        "TASK: Extract the info into JSON.\n"
        "STRICT RULES:\n"
        "1. Pick ONLY ONE (the best/most relevant) value for each field.\n"
        "2. If multiple emails/phones exist, prioritize HR, then pick only the primary one.\n"
        "3. Do NOT use commas, slashes, or lists to include multiple contacts.\n"
        f"4. Schema: {model_class.model_json_schema()}"
    )

    final_response = client.responses.create(
        model="gpt-5-mini",
        input=extraction_prompt,
        text={"format": {"type": "json_object"}} 
    )

    # Convert and validate
    data = json.loads(final_response.output_text)
    
    # We strip any residual " (Link)" citations just in case
    def clean_val(v):
        import re
        if isinstance(v, str):
            return re.sub(r'\s*\(\[.*?\]\(.*?\)\)', "", v).strip()
        return v

    # Clean the dictionary before Pydantic validation
    cleaned_data = {k: (clean_val(v) if not isinstance(v, dict) else {nk: clean_val(nv) for nk, nv in v.items()}) 
                    for k, v in data.items()}

    return model_class.model_validate(cleaned_data)

# ----------------------------
# 3. Usage
# ----------------------------
if __name__ == "__main__":
    query = "DEMOULAS MARKET BASKET"
    try:
        result = structured_llm_call(query, Result)
        print(result.model_dump_json(indent=2))
    except Exception as e:
        print(f"Error: {e}")