import asyncio
import re
from typing import List, Optional
from urllib.parse import urljoin
from playwright.async_api import (
    async_playwright,
    Browser,
    BrowserContext,
    Playwright,
    Error as PlaywrightError,
)
from loguru import logger
from playwright_stealth import Stealth
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.core.config import settings
from app.models.models import _EMAIL_REGEX, validate_and_normalize_ein


class BrowserPool:
    """
    Manages a persistent, shared Playwright Chromium browser process.

    Instead of launching and tearing down a full Chromium process for every
    individual scrape task (which incurs 1.5–3.0 s of cold-start latency and
    heavy CPU churn), the pool keeps a persistent browser alive and creates
    lightweight, isolated incognito contexts (``browser.new_context()``) per task.
    """

    _instance: Optional["BrowserPool"] = None

    def __init__(self) -> None:
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._lock: asyncio.Lock = asyncio.Lock()

    @classmethod
    def get_instance(cls) -> "BrowserPool":
        """Return the process-wide BrowserPool singleton."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def is_running(self) -> bool:
        """True if the persistent browser is launched and connected."""
        return self._browser is not None and self._browser.is_connected()

    async def start(self) -> Browser:
        """
        Initialize Playwright and launch the persistent Chromium browser if not already running.
        """
        async with self._lock:
            if self._browser and self._browser.is_connected():
                return self._browser

            if self._playwright is None:
                self._playwright = await async_playwright().start()

            logger.info("[BrowserPool] Launching persistent Chromium browser...")
            self._browser = await self._playwright.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                ],
            )
            logger.info("[BrowserPool] Persistent Chromium browser launched successfully.")
            return self._browser

    async def get_browser(self) -> Browser:
        """
        Return the running browser instance, automatically restarting it if disconnected.
        """
        if self._browser and self._browser.is_connected():
            return self._browser
        return await self.start()

    async def new_context(self, **kwargs) -> BrowserContext:
        """
        Create a new isolated incognito browser context from the persistent browser.
        """
        browser = await self.get_browser()
        return await browser.new_context(**kwargs)

    async def close(self) -> None:
        """
        Gracefully close the persistent browser and stop Playwright on application shutdown.
        """
        async with self._lock:
            if self._browser:
                try:
                    await self._browser.close()
                    logger.info("[BrowserPool] Persistent Chromium browser closed.")
                except Exception as e:
                    logger.debug(f"[BrowserPool] Error closing browser: {e}")
                finally:
                    self._browser = None

            if self._playwright:
                try:
                    await self._playwright.stop()
                    logger.info("[BrowserPool] Playwright stopped.")
                except Exception as e:
                    logger.debug(f"[BrowserPool] Error stopping playwright: {e}")
                finally:
                    self._playwright = None


class ScraperService:
    """
    Manages Playwright browser sessions and provides scraping methods.
    Uses the persistent BrowserPool by default to avoid per-task cold-start latency.
    """

    def __init__(self, browser: Optional[Browser] = None):
        self.browser: Optional[Browser] = browser
        self._owned_browser: bool = False

    async def __aenter__(self):
        if self.browser is None:
            self.browser = await BrowserPool.get_instance().get_browser()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._owned_browser and self.browser:
            await self.browser.close()

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
                        
                if len(results) >= 4:
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

    async def perform_serper_search(self, query: str) -> List[str]:
        """
        Performs a Google search using the Serper.dev API.
        Returns a list of direct URLs.
        """
        import httpx
        import json
        
        if not settings.SERPER_API_KEY:
            logger.error("SERPER_API_KEY is missing, falling back to DuckDuckGo")
            return await self.perform_duckduckgo_search(query)
            
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query, "gl": "us", "hl": "en", "num": 4})
        headers = {
            'X-API-KEY': settings.SERPER_API_KEY,
            'Content-Type': 'application/json'
        }
        
        results = []
        try:
            logger.info(f"Performing Serper Search for: {query}")
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, data=payload, timeout=15)
                
            if response.status_code == 200:
                data = response.json()
                for item in data.get("organic", []):
                    if "link" in item:
                        results.append(item["link"])
            else:
                logger.error(f"Serper API failed with status {response.status_code}: {response.text}")
                
        except Exception as e:
            logger.error(f"Error during Serper search for '{query}': {e}")
            
        return results

    async def perform_serper_snippet_search(self, query: str) -> str:
        """
        Performs a Google search using Serper and extracts the snippets.
        Used as a fallback to bypass scraping and directly ask the LLM to find emails.
        """
        import httpx
        import json
        
        if not settings.SERPER_API_KEY:
            return await self.perform_duckduckgo_snippet_search(query)
            
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query, "gl": "us", "hl": "en", "num": 10})
        headers = {
            'X-API-KEY': settings.SERPER_API_KEY,
            'Content-Type': 'application/json'
        }
        
        snippets_text = ""
        try:
            logger.info(f"Performing Serper Snippet Search for: {query}")
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, data=payload, timeout=15)
                
            if response.status_code == 200:
                data = response.json()
                for item in data.get("organic", []):
                    if "snippet" in item:
                        snippets_text += item["snippet"] + "\n---\n"
            else:
                logger.error(f"Serper API failed with status {response.status_code}: {response.text}")
                
        except Exception as e:
            logger.error(f"Error during Serper snippet search for '{query}': {e}")
            
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
        
        # Block heavy resources to speed up page loading
        await page.route(
            "**/*",
            lambda route: route.abort()
            if route.request.resource_type in ["image", "media", "font"]
            else route.continue_()
        )
        
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
            
            # Using browser-side evaluation is much faster and bypasses "Node is not an HTMLElement" exceptions
            links_data = []
            for attempt in range(3):
                try:
                    links_data = await page.evaluate('''() => {
                        const anchors = Array.from(document.querySelectorAll("a[href]"));
                        return anchors.map(a => ({
                            href: a.getAttribute("href") || "",
                            text: a.innerText || a.textContent || ""
                        }));
                    }''')
                    break  # Evaluation succeeded
                except Exception as e:
                    if "Execution context was destroyed" in str(e) and attempt < 2:
                        logger.warning(f"Context destroyed natively (redirecting). Retrying {attempt+1}/3 after 3s...")
                        await asyncio.sleep(3)
                    else:
                        raise e # Re-raise if we are out of attempts or it is a different error
            
            for item in links_data:
                href = item.get("href", "")
                text = item.get("text", "")
                
                if href:
                    # Robust URL completion (handles /about, mailto:, etc.)
                    full_url = urljoin(homepage_url, href)
                    text_lower = text.lower()
                    href_lower = href.lower()
                    
                    if any(k in text_lower or k in href_lower for k in keywords):
                        # Ensure we don't try to visit mailto: or tel: links as web pages
                        if full_url.startswith("http"):
                            links.add(full_url)
                        
        except Exception as e:
            logger.error(f"Error harvesting links from {homepage_url}: {e}")
        finally:
            await context.close()
            
        # Prioritize Links before returning
        def score_link(url: str) -> int:
            url_lower = url.lower()
            if "contact" in url_lower: return 1
            if "about" in url_lower: return 2
            if "team" in url_lower or "staff" in url_lower: return 3
            if "location" in url_lower or "office" in url_lower: return 4
            if url_lower == homepage_url.lower(): return 5
            return 6
            
        sorted_links = sorted(list(links), key=score_link)
        return sorted_links

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
        
        # Block heavy resources to speed up page loading
        await page.route(
            "**/*",
            lambda route: route.abort()
            if route.request.resource_type in ["image", "media", "font"]
            else route.continue_()
        )
        
        text = ""
        try:
            try:
                await page.goto(url, wait_until="networkidle", timeout=20000)
            except Exception:
                logger.debug(f"Timeout waiting for networkidle on {url}, proceeding anyway.")
            
            await asyncio.sleep(3) # Extra buffer for React/Vue hydration
            
            try:
                # 1. Force Auto-Scroll to trigger lazy-loaded footers where emails often live
                await page.evaluate("""
                    window.scrollTo(0, document.body.scrollHeight);
                    setTimeout(() => window.scrollTo(0, 0), 500);
                """)
                await asyncio.sleep(1.5) # Wait for lazy elements to render
                
                body = page.locator("body")
                # Wait up to 5 seconds for the body to be present
                await body.wait_for(state="attached", timeout=5000)
                text = await body.inner_text()
            except Exception:
                logger.warning(f"Could not locate <body> on {url}, falling back to full page extraction.")
                # Fallback for sites using <frameset> or heavily broken HTML
                text = await page.evaluate("document.documentElement.innerText") or ""
                
            # 2. Raw HTML Regex Parsing for Hidden Emails
            import re
            html_content = await page.content() or ""
            
            # Aggressively hunt for mailto: links in the raw HTML source
            mailto_matches = re.findall(r'mailto:([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', html_content, re.IGNORECASE)
            
            # Also look for any standard email pattern in the raw source (can be noisy, so we filter it)
            raw_emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', html_content)
            
            found_emails = set(mailto_matches + raw_emails)
            # Filter out common false-positives and obfuscated/malformed addresses.
            _CF_PLACEHOLDER = "[email protected]"
            invalid_domains = ['.png', '.jpg', '.jpeg', '.gif', '.css', '.js', 'sentry', 'example', 'domain.com', '.webp', 'wixpress']
            valid_emails = [
                e for e in found_emails
                if e.lower() != _CF_PLACEHOLDER                        # Cloudflare placeholder
                and not any(bad in e.lower() for bad in invalid_domains)  # image/tracker domains
                and _EMAIL_REGEX.match(e)                               # must be a valid email format
            ]

            if valid_emails:
                logger.info(f"Regex found {len(valid_emails)} hidden emails in raw HTML on {url}")
                # Append them forcefully to the bottom of the visible text so the LLM is guaranteed to see them
                text += "\n\n--- HIDDEN EMAILS FOUND IN RAW HTML SOURCE ---\n" + "\n".join(valid_emails)

            # Aggressively hunt for EIN/FEIN in raw HTML source (footers, legal disclosures)
            ein_raw_matches = re.findall(
                r'(?:EIN|FEIN|Tax\s*ID|Employer\s*ID|IRS\s*#?)[\s:#]*([0-9]{2}-[0-9]{7})\b',
                html_content,
                re.IGNORECASE,
            )
            valid_eins = [
                validate_and_normalize_ein(e)
                for e in set(ein_raw_matches)
                if validate_and_normalize_ein(e)
            ]
            if valid_eins:
                logger.info(f"Regex found {len(valid_eins)} possible EIN/FEIN in raw HTML on {url}")
                text += "\n\n--- POSSIBLE EIN/FEIN FOUND IN RAW HTML SOURCE ---\n" + "\n".join(valid_eins)
                
            text = " ".join(text.split())
        except Exception as e:
            logger.error(f"Error extracting text from {url}: {e}")
        finally:
            await context.close()
        
        return text[:15000]
