import json
from typing import List, Optional
from groq import AsyncGroq, RateLimitError, APIStatusError
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.models import ContactInfo
from app.core.config import settings

class LLMService:
    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
        if not self.api_key:
            logger.warning("GROQ_API_KEY not found in settings.")
        self.client = AsyncGroq(api_key=self.api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RateLimitError, APIStatusError)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying Groq verify_official_site. Attempt {retry_state.attempt_number}"
        )
    )
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
                model="llama-3.1-8b-instant",
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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RateLimitError, APIStatusError)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying Groq extract_contact_info. Attempt {retry_state.attempt_number}"
        )
    )
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
                model="llama-3.1-8b-instant",
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
            
        except json.JSONDecodeError:
            logger.error("Failed to decode JSON from LLM response.")
            return None
        except (RateLimitError, APIStatusError) as e:
            logger.error(f"Groq API error (will be retried by tenacity if attempts remaining): {e}")
            raise # Re-raise so tenacity can catch it
        except Exception as e:
            logger.error(f"LLM extraction error: {e}")
            return None

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=6),
        retry=retry_if_exception_type((RateLimitError, APIStatusError)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying Groq extract_fallback_email. Attempt {retry_state.attempt_number}"
        )
    )
    async def extract_fallback_email(self, snippets_text: str, current_info: ContactInfo) -> ContactInfo:
        if not snippets_text:
            return current_info
            
        prompt = f"""
        I am trying to find the contact email address for a company. I searched the web and here are the text snippets from the search results:
        
        {snippets_text}
        
        If you see an official-looking email address in these snippets, please return it.
        Return a valid JSON object with a single key "Email". If you cannot find one, return an empty string for the value.
        """

        try:
            chat_completion = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You extract email addresses from text. Output valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.1-8b-instant",
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            content = chat_completion.choices[0].message.content
            data = json.loads(content)
            
            found_email = str(data.get("Email", "") or "").strip()
            if found_email:
                logger.info(f"Fallback search found missing email: {found_email}")
                current_info.Email = found_email
                
            return current_info
            
        except (RateLimitError, APIStatusError) as e:
            logger.error(f"Groq API error (will be retried by tenacity if attempts remaining): {e}")
            raise
        except Exception as e:
            logger.error(f"LLM fallback extraction error: {e}")
            return current_info
