import json
from typing import List, Optional
from groq import AsyncGroq, RateLimitError
from loguru import logger
from app.models import ContactInfo
from app.core.config import settings

class LLMService:
    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
        if not self.api_key:
            logger.warning("GROQ_API_KEY not found in settings.")
        self.client = AsyncGroq(api_key=self.api_key)

    async def verify_official_site(self, search_results: List[str], company_name: str) -> str:
        if not search_results:
            return ""

        prompt = f"""
        I am looking for the official homepage of "{company_name}".
        Here are the search results:
        {json.dumps(search_results, indent=2)}

        Return ONLY the URL that is most likely the official homepage.
        If none look correct, return "NOT_FOUND".
        Do not output any explanation.
        """

        try:
            chat_completion = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that identifies official company websites."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0,
            )
            result = chat_completion.choices[0].message.content.strip()
            if "NOT_FOUND" in result:
                return ""
            return result
        except RateLimitError:
            logger.error("Groq Rate Limit exceeded during site verification.")
            return ""
        except Exception as e:
            logger.error(f"LLM site verification error: {e}")
            return ""

    async def extract_contact_info(self, page_text: str) -> Optional[ContactInfo]:
        if not page_text:
            return None
        
        prompt = f"""
        Extract contact information for the company from the following text.
        
        Text Content (Truncated):
        {page_text}
        
        Return a valid JSON object with the following keys:
        - "Phone": The phone number (string)
        - "Email": The email address (string)
        - "Address": The full physical address (string)
        - "DeptContacts": A dictionary of specific department contacts if available (e.g. {{"Sales": "123-456"}})

        If a field is not found, use an empty string or null.
        Ensure the output is strictly valid JSON.
        """

        try:
            chat_completion = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a data extraction assistant. Output valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            content = chat_completion.choices[0].message.content
            data = json.loads(content)
            
            return ContactInfo(
                Phone=str(data.get("Phone", "") or ""),
                Email=str(data.get("Email", "") or ""),
                Address=str(data.get("Address", "") or ""),
                DeptContacts=data.get("DeptContacts", {})
            )
            
        except RateLimitError:
            logger.error("Groq Rate Limit exceeded during contact extraction.")
            return None
        except json.JSONDecodeError:
            logger.error("Failed to decode JSON from LLM response.")
            return None
        except Exception as e:
            logger.error(f"LLM extraction error: {e}")
            return None
