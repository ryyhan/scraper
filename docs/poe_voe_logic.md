# High-Level Logics: Place of Employment (POE) and Verification of Employment (VOE)

This document provides a comprehensive, independent logical description of the **Place of Employment (POE)** search and scrape pipelines and the **Verification of Employment (VOE)** pipeline implemented in this application. Each system is detailed autonomously below under its own respective heading.

---

## 1. Place of Employment (POE) Pipeline Logic

### A. Executive Summary & Core Concept
The Place of Employment (POE) pipeline is designed to search for a target company (Place of Employment) online, identify its verified official website, and extract highly reliable, structured, and tagged contact details (such as emails, phone numbers, faxes, and physical addresses) categorized by corporate department (HR, Payroll, Careers, Admin, Finance, etc.).

This pipeline can run in three different operational modes depending on the endpoint triggered: a Playwright-based browser scraping agent (`/google-search/`), dedicated API search agents utilizing grounded LLM search tools (`/gemini-search/` and `/openai-search/`), or a combined hybrid search engine (`/combined-search/`).

### B. Architecture & Infrastructure
The POE system is built upon an asynchronous background task queue. When a client initiates a POE request:
- The FastAPI router receives the payload and generates a unique `task_id`.
- It writes a new task record with an `IN_PROGRESS` status to the SQLite database (`tasks.db`).
- The API immediately responds to the client with the tracking ID and status, avoiding blocked HTTP connections.
- The router dispatches the execution logic to the background as a FastAPI `BackgroundTask`.
- To prevent CPU and RAM exhaustion, the task first waits to acquire a slot from a global semaphore (`BROWSER_SEMAPHORE`), which controls active Playwright browser instances.
- Once a slot is acquired, the service offloads the actual synchronous scraper or LLM calls to a thread pool via `asyncio.to_thread` to prevent blocking the single-threaded event loop.
- Upon completion, the task status in SQLite is updated to `SUCCESS` or `FAILURE`, the extracted contact details are stored in the database record, and a webhook payload is sent to the client's configured endpoint.

### C. The Playwright Scraping Pipeline Logic (`/google-search/`)
This workflow operates from the bottom up by executing web searches, identifying the homepage, crawling relevant pages, and extracting text for analysis:
- **Search Execution**: The scraper triggers a query (`"{company_name} official site contact"`) via DuckDuckGo or Serper to retrieve the top 5 ranking URLs.
- **Homepage Verification**: The LLM Service (powered by Groq Llama3) reviews these candidate URLs and verifies which one represents the genuine, official homepage of the company. If verification fails, the task terminates.
- **Link Harvesting**: The scraper initializes a Playwright browser instance with Stealth evasion plugins to avoid bot detection. It loads the official homepage and extracts all hyperlinks. Heuristic filters scan the links to prioritize up to 4 deep pages containing keywords such as `contact`, `about`, `careers`, `support`, `staff`, `team`, `location`, `office`, `hr`, or `payroll`.
- **Text Extraction**: The browser sequentially loads each harvested page, waits for dynamic Javascript to render, isolates the visible page body text, strips unnecessary code, and aggregates the text up to a 15,000-character limit.
- **Structured LLM Extraction**: The accumulated text is analyzed by the LLM in JSON mode to construct structured contact arrays.
- **Fallback Email Search**: If no email address is extracted in the primary stage, a fallback web search is initiated for `"{company_name}" (email OR "contact us at" OR "reach us at" OR "@")`. The LLM extracts any emails from these search snippets to enrich the final output.
- **Regex Timeout Safeguard**: If the scraper times out (default 120 seconds), a regex-based parser scans any text extracted prior to the timeout to salvage valid email addresses, avoiding a hard failure.

### D. The LLM Grounded Search Pipeline Logic (`/gemini-search/` and `/openai-search/`)
This workflow utilizes the LLM's native Google Search grounding tool (Gemini 2.5) or Web Search tool (OpenAI GPT-4o-mini) to research, structure, and verify contacts:
- **Step 1: The Grounded Investigator**: The service prompts the grounded LLM to execute a sequence of targeted searches, seeking department-specific lines, careers contact details, payroll, administrative assistants, and fax numbers. It checks official sites, LinkedIn, ZoomInfo, Apollo, RocketReach, the Better Business Bureau (BBB), press releases, and job boards. If it encounters a Cloudflare obfuscation placeholder (`[email protected]`), it ignores it and targets alternative sources (press releases, public PDFs, SEC filings) where emails are listed in plain text. It compiles its findings into a rich, cited Markdown text document.
- **Step 2: The Extractor**: The research text is passed to a second LLM call forced to output structured JSON matching the Pydantic schema. It populates emails, phones, faxes, and addresses, tagging each with a departmental category (`Human Resource`, `Payroll`, `Admin`, `Careers`, `Personnel`, `Finance`, `Secretary`, `Labor relations`, or `Others`).
- **Step 3: The Hallucination Verifier**: To prevent "domain-guessing" (where the LLM invents plausible-looking emails like `hr@company.com`), a third "Fact-Checker" call is made. It compares the extracted contacts against the raw cited research. It keeps the contact if its value appears verbatim in the research text, or if the research explicitly attributes it to the company, or if it is a generic catch-all (like `info@` or `contact@`) matching the verified domain. Any departmental or specific contact not found verbatim is discarded.

### E. The Combined Search Pipeline Logic (`/combined-search/`)
This serves as a high-concurrency coordinator that initiates both OpenAI and Gemini pipelines in parallel using `asyncio.to_thread`. Once both complete, it combines the outputs, normalizes the casing and formatting of contact values, dedupes them, sorts the arrays, and truncates the lists to a client-specified `max_limit` parameter before returning.

---

## 2. Verification of Employment (VOE) Pipeline Logic

### A. Executive Summary & Core Concept
The Verification of Employment (VOE) pipeline is a focused decision engine designed to verify whether a specific named individual currently holds (or recently held) a designated job title at a target company (Place of Employment). 

Unlike the POE pipeline which harvests contact details, the VOE pipeline performs directed background research on an individual's professional history, evaluates the evidence, and outputs a calibrated confidence score and verdict.

### B. Current Status
- **Work in Progress**: The VOE pipeline is under active development. The underlying logic, prompts, and score rubrics are currently undergoing calibration, tuning, and rigorous testing to achieve optimal verification accuracy.

### C. Architecture & Infrastructure
The VOE system operates asynchronously:
- When a client issues a POST request to `/verify-voe/`, the router generates a unique `task_id` and records the task as `IN_PROGRESS` in the SQLite database (`tasks.db`).
- The router immediately returns a success status and the `task_id` to the client.
- The verification workload is dispatched as a background task.
- Because the Gemini SDK is synchronous, the worker offloads the API calls to a thread pool via `asyncio.to_thread` to preserve FastAPI's event-loop concurrency.
- Once the dual-stage evaluation completes, the task record in SQLite is updated to `SUCCESS` or `FAILURE` with the validation payload, and the results are pushed to the client via a webhook.

### D. Step-by-Step Verification Logic
The VOE process is divided into two distinct logical stages:

#### Step 1: The Grounded Investigator (Research Stage)
- **Context Construction**: The service compiles the individual's full name, job title, and company name, along with any optional location parameters (city, zip code, country) to avoid common-name collisions.
- **Evidence Gathering**: Gemini is prompted to perform live Google searches targeting professional networking profiles, official corporate team directories, press releases, company news articles, and SEC filings.
- **Disambiguation**: The model uses location indicators to ensure it is research-matching the correct individual.
- **Deep Search**: The model is instructed to consult at least 3–5 distinct sources, documenting all affirmative evidence as well as any **contradictory evidence** (such as profiles showing the person has retired, changed companies, holds a different title, or works at a competitor).
- **Output**: The investigator compiles a detailed, cited Markdown research summary.

#### Step 2: The Calibrated Analyst (Scoring & Verdict Stage)
The cited research summary is passed to a second Gemini call configured to output structured JSON matching the Pydantic verification schema. The LLM evaluates the evidence against a strict, calibrated scoring rubric:

- **Score 9.0 to 10.0 (Verdict: `VERIFIED`)**: Direct confirmation. A credible source (LinkedIn profile, official company directory, bio page, or press release) explicitly names the person in the exact stated role at the target company.
- **Score 6.0 to 8.0 (Verdict: `LIKELY`)**: Indirect or partial evidence. Clear records link the individual to the company, but their exact current title or active status remains slightly ambiguous.
- **Score 3.0 to 5.0 (Verdict: `UNVERIFIED`)**: Inconclusive. Both the company and the individual exist publicly, but no connection between them can be verified across the searched web sources.
- **Score 0.0 to 2.0 (Verdict: `CONTRADICTED`)**: Direct contradiction or no public footprint. The evidence proves the individual has departed the company, holds a different title, works elsewhere, OR the individual leaves no public/professional trace online.

The final response object returned by the service includes the original request details, a numeric confidence score, the calibrated verdict string, a concise 2–3 sentence explanation of the findings, and the array of verified URLs or publication names cited as proof.
