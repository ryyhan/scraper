import asyncio
from typing import List, Optional
from urllib.parse import urljoin
from playwright.async_api import async_playwright, Error as PlaywrightError
from loguru import logger
from playwright_stealth import Stealth
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.core.config import settings

class ScraperService:
    """
    Manages Playwright browser sessions and provides scraping methods.
    """
    
    def __init__(self):
        self.playwright = None
        self.browser = None

    async def __aenter__(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=False)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def perform_duckduckgo_search(self, query: str) -> List[str]:
        """
        Performs a DuckDuckGo search using httpx (HTML version) to avoid bot detection.
        Returns a list of direct URLs.
        """
        import httpx
        from bs4 import BeautifulSoup
        import urllib.parse
        
        url = "https://html.duckduckgo.com/html/"
        params = {"q": query}
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://html.duckduckgo.com/"
        }
        
        results = []
        try:
            logger.info(f"Performing DuckDuckGo HTML Search for: {query}")
            async with httpx.AsyncClient() as client:
                response = await client.post(url, data=params, headers=headers, follow_redirects=True)
                
            if response.status_code != 200:
                logger.error(f"DDG Search failed with status {response.status_code}")
                return []
                
            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.select(".result__a")
            
            logger.info(f"Found {len(links)} raw results.")
            
            for link in links:
                raw_href = link.get('href')
                if raw_href:
                    # DDG HTML links are often /l/?uddg=...
                    if "uddg=" in raw_href:
                        parsed = urllib.parse.urlparse(raw_href)
                        qs = urllib.parse.parse_qs(parsed.query)
                        if 'uddg' in qs:
                            clean_url = qs['uddg'][0]
                            if "duckduckgo.com" not in clean_url:
                                results.append(clean_url)
                    else:
                        # Direct link or ad
                        if "duckduckgo.com" not in raw_href:
                            results.append(raw_href)
                        
                if len(results) >= 5:
                    break
                    
        except Exception as e:
            logger.error(f"Error during DuckDuckGo search for '{query}': {e}")
            
        logger.info(f"DuckDuckGo Search returned {len(results)} valid URLs.")
        return results

    async def perform_duckduckgo_snippet_search(self, query: str) -> str:
        """
        Performs a DuckDuckGo search and extracts the visible text snippets.
        Used as a fallback to bypass scraping and directly ask the LLM to find emails in search results.
        """
        import httpx
        from bs4 import BeautifulSoup
        
        url = "https://html.duckduckgo.com/html/"
        params = {"q": query}
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://html.duckduckgo.com/"
        }
        
        snippets_text = ""
        try:
            logger.info(f"Performing DuckDuckGo Snippet Search for: {query}")
            async with httpx.AsyncClient() as client:
                response = await client.post(url, data=params, headers=headers, follow_redirects=True, timeout=15)
                
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                snippets = soup.select(".result__snippet")
                for s in snippets:
                    snippets_text += s.get_text(strip=True) + "\n---\n"
                    
        except Exception as e:
            logger.error(f"Error during DuckDuckGo snippet search for '{query}': {e}")
            
        return snippets_text

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((PlaywrightError, asyncio.TimeoutError)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying harvest_contact_links. Attempt {retry_state.attempt_number} for {retry_state.args[1]}"
        )
    )
    async def harvest_contact_links(self, homepage_url: str) -> List[str]:
        """
        Visits the homepage and extracts links that might contain contact info.
        """
        context = await self.browser.new_context()
        page = await context.new_page()
        await Stealth().apply_stealth_async(page)
        links = set()
        links.add(homepage_url)
        
        try:
            # wait_until="networkidle" handles Single Page Apps better than domcontentloaded
            await page.goto(homepage_url, wait_until="networkidle", timeout=20000)
        except Exception:
            logger.debug(f"Timeout waiting for networkidle on {homepage_url}, proceeding anyway.")
            
        try:
            # Give it 3 extra seconds just in case of animations/heavy React hydration
            await asyncio.sleep(3)
            
            keywords = ["contact", "about", "location", "team", "connect", "회사소개", "연락처"]
            anchors = page.locator("a[href]")
            count = await anchors.count()
            
            for i in range(count):
                href = await anchors.nth(i).get_attribute("href")
                text = await anchors.nth(i).inner_text()
                
                if href:
                    # Robust URL completion (handles /about, mailto:, etc.)
                    full_url = urljoin(homepage_url, href)
                    text_lower = text.lower()
                    href_lower = href.lower()
                    
                    if any(k in text_lower or k in href_lower for k in keywords):
                        links.add(full_url)
                        
        except Exception as e:
            logger.error(f"Error harvesting links from {homepage_url}: {e}")
        finally:
            await context.close()
            
        return list(links)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((PlaywrightError, asyncio.TimeoutError)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying extract_page_text. Attempt {retry_state.attempt_number} for {retry_state.args[1]}"
        )
    )
    async def extract_page_text(self, url: str) -> str:
        """
        Visits a URL and extracts its visible text, truncated to 15k chars.
        Handles SPAs by waiting for networkidle.
        """
        context = await self.browser.new_context()
        page = await context.new_page()
        await Stealth().apply_stealth_async(page)
        text = ""
        try:
            try:
                await page.goto(url, wait_until="networkidle", timeout=20000)
            except Exception:
                logger.debug(f"Timeout waiting for networkidle on {url}, proceeding anyway.")
            
            await asyncio.sleep(3) # Extra buffer for React/Vue hydration
            
            body = page.locator("body")
            text = await body.inner_text()
            text = " ".join(text.split())
        except Exception as e:
            logger.error(f"Error extracting text from {url}: {e}")
        finally:
            await context.close()
        
        return text[:15000]
