import json
from typing import List, Optional
from groq import AsyncGroq, RateLimitError, APIStatusError
from loguru import logger
from app.models import ContactInfo
from app.core.config import settings
from app.services._retry import retry_groq

class LLMService:
    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
        if not self.api_key:
            logger.warning("GROQ_API_KEY not found in settings.")
        # max_retries=2: SDK inner-layer reads Retry-After headers on 429/5xx
        # before our tenacity outer-layer (in _retry.py) takes over.
        self.client = AsyncGroq(api_key=self.api_key, max_retries=2)

    @retry_groq()
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
                model=settings.GROQ_MODEL,
                temperature=0,
            )
            result = chat_completion.choices[0].message.content.strip()
            if "NOT_FOUND" in result:
                return ""
            return result
        except (RateLimitError, APIStatusError) as e:
            logger.error(f"Groq API error (will be retried by tenacity if attempts remaining): {e}")
            raise # Re-raise so tenacity can catch it
        except Exception as e:
            logger.error(f"LLM site verification error: {e}")
            return ""

    @retry_groq()
    async def extract_contact_info(self, page_text: str) -> Optional[ContactInfo]:
        if not page_text:
            return None
        
        prompt = f"""
        Extract contact information for the company from the following text.
        
        Text Content (Truncated):
        {page_text}
        
        Return a valid JSON object with the following keys:
        - "Phone": Array of phone numbers (list of strings)
        - "Fax": Array of fax numbers (list of strings)
        - "Email": Array of email addresses (list of strings)
        - "Address": Array of full physical street addresses (list of strings, e.g., ["123 Main St, City, State 12345"])
        - "DeptContacts": A dictionary of specific department contacts if available (e.g. {{"Sales": "123-456"}})

        If a field is not found, use an empty array or null (or empty object for DeptContacts).
        Ensure the output is strictly valid JSON.
        """

        try:
            chat_completion = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a data extraction assistant. Output valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                model=settings.GROQ_MODEL,
                temperature=0,
                max_tokens=8000,
                response_format={"type": "json_object"}
            )
            
            data = json.loads(chat_completion.choices[0].message.content)
            
            return ContactInfo(
                Phone=data.get("Phone", []),
                Fax=data.get("Fax", []),
                Email=data.get("Email", []),
                Address=data.get("Address", []),
                DeptContacts=data.get("DeptContacts", {})
            )
            
        except json.JSONDecodeError:
            logger.error("Failed to decode JSON from LLM response.")
            return None
        except (RateLimitError, APIStatusError) as e:
            logger.error(f"Groq API error (will be retried by tenacity if attempts remaining): {e}")
            raise # Re-raise so tenacity can catch it
        except Exception as e:
            logger.error(f"LLM extraction error: {e}")
            return None

    @retry_groq()
    async def extract_fallback_email(self, snippets_text: str, current_info: ContactInfo) -> ContactInfo:
        if not snippets_text:
            return current_info
            
        prompt = f"""
        I am trying to find the contact email address for a company. I searched the web and here are the text snippets from the search results:
        
        {snippets_text}
        
        If you see official-looking email addresses in these snippets, please return them.
        Return a valid JSON object with EXACTLY ONE key named "Email". The value must be an array of strings (e.g., ["email1@example.com", "email2@example.com"]).
        Ignore "[email protected]" or obfuscated Cloudflare text.
        If you cannot find any, return an empty array.
        """

        try:
            chat_completion = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You extract email addresses from text. Output valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                model=settings.GROQ_MODEL,
                temperature=0,
                max_tokens=8000,
                response_format={"type": "json_object"}
            )
            
            data = json.loads(chat_completion.choices[0].message.content)
            
            found_emails = data.get("Email", [])
            if isinstance(found_emails, str):
                found_emails = [found_emails]
            elif not isinstance(found_emails, list):
                found_emails = []
                
            if found_emails:
                # Reassign (not .extend) so Pydantic's validate_assignment fires the
                # validate_email validator on the new values, rejecting any hallucinated
                # or malformed addresses the LLM may have returned.
                current_info.Email = current_info.Email + [
                    e for e in found_emails if isinstance(e, str) and e.strip()
                ]
                logger.info(f"Fallback search found missing email(s): {found_emails}")
                
            return current_info
            
        except (RateLimitError, APIStatusError) as e:
            logger.error(f"Groq API error (will be retried by tenacity if attempts remaining): {e}")
            raise
        except Exception as e:
            logger.error(f"LLM fallback extraction error: {e}")
            return current_info
