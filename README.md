# Async Scraper & Intelligence API

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.129.0-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Playwright](https://img.shields.io/badge/Playwright-1.58.0-45ba4b.svg?logo=playwright&logoColor=white)](https://playwright.dev)
[![Docker](https://img.shields.io/badge/Docker-Multi--Stage-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](#)

An enterprise-grade, asynchronous AI intelligence and web scraping service built with **FastAPI**, **Playwright (stealth mode)**, **SQLite (WAL mode)**, and a multi-provider LLM orchestration layer powered by **OpenAI**, **Google Gemini**, and **Groq**.

The service provides automated discovery, deep crawling, employment verification, place-of-employment identification, entity resolution, and computer vision PDF processing with robust concurrency gating, sliding-window rate limiting, and dual-layer retry resiliency.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Core Feature Matrix](#core-feature-matrix)
3. [Architecture & Reliability Features](#architecture--reliability-features)
   - [Asynchronous Task Lifecycle & Polling](#asynchronous-task-lifecycle--polling)
   - [Browser Concurrency Semaphore](#browser-concurrency-semaphore)
   - [Sliding-Window Rate Limiting](#sliding-window-rate-limiting)
   - [Dual-Layer Retry Strategy](#dual-layer-retry-strategy)
   - [SQLite WAL Mode & Connection Pool](#sqlite-wal-mode--connection-pool)
   - [Anti-Hallucination & Data Hygiene](#anti-hallucination--data-hygiene)
   - [Unified Token Tracking](#unified-token-tracking)
4. [Environment Configuration](#environment-configuration)
5. [Installation & Local Setup](#installation--local-setup)
6. [Docker Deployment](#docker-deployment)
7. [API Reference](#api-reference)
   - [1. Headless Browser Scraper (`/google-search/`)](#1-headless-browser-scraper-google-search)
   - [2. OpenAI Grounded Search (`/openai-search/`)](#2-openai-grounded-search-openai-search)
   - [3. Gemini Grounded Search (`/gemini-search/`)](#3-gemini-grounded-search-gemini-search)
   - [4. Combined Grounded Search (`/combined-search/`)](#4-combined-grounded-search-combined-search)
   - [5. Verification of Employment (`/verify-voe/`)](#5-verification-of-employment-verify-voe)
   - [6. Find Place of Employment (`/find-poe/`)](#6-find-place-of-employment-find-poe)
   - [7. Company Name Matcher (`/match-companies/`)](#7-company-name-matcher-match-companies)
   - [8. Vision PDF Text Extraction (`/extract-pdf/`)](#8-vision-pdf-text-extraction-extract-pdf)
   - [9. Background Check Report Parser (`/parse-background-check/`)](#9-background-check-report-parser-parse-background-check)
   - [10. Webhook Mock (`/webhook-mock`)](#10-webhook-mock-webhook-mock)
8. [Data Models & Taxonomy](#data-models--taxonomy)
   - [Departmental Contact Taxonomy (`ContactTag`)](#departmental-contact-taxonomy-contacttag)
   - [Disambiguation Address Input (`HintAddress`)](#disambiguation-address-input-hintaddress)
   - [Database Schema (`TaskRecord`)](#database-schema-taskrecord)
9. [Observability & Error Handling](#observability--error-handling)
10. [Repository Structure](#repository-structure)

---

## System Architecture

```mermaid
flowchart TD
    Client[Client / Consumer] -->|HTTP Request| API[FastAPI Application]

    subgraph "Ingress & Protection"
        API --> RL{Sliding-Window Rate Limiter<br/>15 req / 60s per IP}
        RL -->|Exceeded| E429[HTTP 429 Too Many Requests]
        RL -->|Allowed| Router[FastAPI Router]
    end

    subgraph "Execution Modes"
        Router -->|Synchronous Endpoints| SyncHandlers[Sync Parsers & Utilities]
        SyncHandlers --> PDF[PdfExtractor / BgCheckParser]
        SyncHandlers --> Matcher[CompanyMatcherService]

        Router -->|Asynchronous Task Creation| TaskInit[Create TaskRecord<br/>Status: IN_PROGRESS]
        TaskInit --> DB[(SQLite tasks.db<br/>WAL Mode + 30s busy_timeout)]
        TaskInit --> Response[Return 200 OK with task_id]
        TaskInit --> BG[FastAPI BackgroundTask]
    end

    subgraph "Background Worker Pipelines"
        BG --> ScraperPipe[Scraper Pipeline]
        BG --> OAIPipe[OpenAI Search Pipeline]
        BG --> GemPipe[Gemini Search Pipeline]
        BG --> CombPipe[Combined Search Pipeline]
        BG --> VOEPipe[VOE Verification Pipeline]
        BG --> POEPipe[Find POE Pipeline]

        ScraperPipe --> SemGate{Browser Semaphore<br/>Max 4 instances}
        SemGate --> Playwright[Playwright Stealth Engine]
        SemGate --> GroqSite[Groq LLaMA-3.1 Verification]

        OAIPipe --> OAIWeb[OpenAI Responses API + web_search]
        GemPipe --> GemWeb[Gemini + google_search Grounding]
        CombPipe --> BothConcur[Asyncio Gather: OpenAI + Gemini]
        VOEPipe --> VOEEngine[Gemini / OpenAI Grounded Investigator]
        POEPipe --> POEEngine[Gemini / OpenAI Employer Discovery]
    end

    subgraph "Persistence & Egress"
        ScraperPipe & OAIPipe & GemPipe & CombPipe & VOEPipe & POEPipe --> ResVal[Anti-Hallucination & Normalization]
        ResVal --> UpdateDB[(Update SQLite Record<br/>Status: SUCCESS / FAILURE)]
        UpdateDB --> Webhook{Webhook Configured?}
        Webhook -->|Yes| PushHook[POST WebhookPayload to Client]
        Webhook -->|No| Done[Ready for Polling GET /{endpoint}/{task_id}]
    end
```

---

## Core Feature Matrix

| Endpoint | Provider | Execution | Primary Capabilities | Concurrency Gate |
|---|---|---|---|---|
| `POST /google-search/` | DuckDuckGo / Serper + Playwright + Groq (`llama-3.1-8b-instant`) | Async (Background) | Deep web crawling, JS execution, stealth anti-bot evasion, regex fallback | `BROWSER_SEMAPHORE` (default: 4) |
| `POST /openai-search/` | OpenAI (`gpt-4o-mini`) | Async (Background) | 3-turn chained web search Responses API, contact extraction & fact-checking | Event loop / thread pool |
| `POST /gemini-search/` | Google Gemini (`gemini-2.5-flash-lite`) | Async (Background) | 3 parallel sub-gatherers with live Google Search grounding (`MODE_ALWAYS`) | Event loop / thread pool |
| `POST /combined-search/` | OpenAI + Gemini | Async (Background) | Concurrent multi-LLM search, URL sanitisation, deduplication, summary stats | Event loop / thread pool |
| `POST /verify-voe/` | Gemini, OpenAI, or Both | Async (Background) | Employment verification, deep web research, calibrated scoring rubric (0–10) | Event loop / thread pool |
| `POST /find-poe/` | Gemini, OpenAI, or Both | Async (Background) | Professional footprint discovery, ranked candidate employers list | Event loop / thread pool |
| `POST /match-companies/` | Groq, OpenAI, or Gemini | Synchronous | Entity resolution, acronym resolution, legal suffix stripping, deep search mode | Direct call |
| `POST /extract-pdf/` | Gemini Files API or OpenAI Vision (`pypdfium2`) | Synchronous | Whole-document OCR, form checkbox & radio state transcribing (`[SELECTED]`) | Direct call |
| `POST /parse-background-check/` | Gemini or OpenAI | Synchronous | Two-stage parsing for HR/screening reports (HireRight, Sterling, Checkr, etc.) | Direct call |
| `POST /webhook-mock` | Local | Synchronous | Echo & test endpoint for webhook callback inspection | Direct call |

---

## Architecture & Reliability Features

### Asynchronous Task Lifecycle & Polling

Task-generating endpoints follow the **Asynchronous Request-Reply Pattern**:
1. The client issues a `POST` request.
2. The endpoint creates a `TaskRecord` in SQLite (`status="IN_PROGRESS"`), schedules the pipeline as a FastAPI `BackgroundTask`, and returns HTTP 200 with `{"task_id": "...", "status": "IN_PROGRESS"}` immediately.
3. The client polls `GET /{pipeline}/{task_id}` until `status` transitions to `"SUCCESS"` or `"FAILURE"`.
4. If a global `WEBHOOK_URL` (or task-specific webhook) is defined, the worker pushes a `WebhookPayload` on completion.
5. Clients can inspect failed executions via `GET /{pipeline}/failed/?limit=10`.

```
PENDING ──► IN_PROGRESS ──┬──► SUCCESS  (result_data populated)
                          └──► FAILURE  (error message recorded)
```

### Browser Concurrency Semaphore

Playwright instances are resource-intensive. To prevent CPU saturation and out-of-memory crashes:
- A global in-process `asyncio.Semaphore(settings.MAX_CONCURRENT_BROWSERS)` gates every browser launch in `process_scraping_task`.
- Tasks queue safely in the asyncio event loop until an active browser completes its work and exits its context manager.

### Sliding-Window Rate Limiting

To safeguard external LLM budgets and internal system resources:
- Implemented as a reusable FastAPI dependency (`task_rate_limit`) in `app/api/rate_limit.py`.
- **Sliding-window window**: 15 requests per 60 seconds per IP address.
- **Client IP Resolution**: Evaluates the leftmost IP in `X-Forwarded-For` (for reverse proxies like Nginx, Cloudflare, or AWS ALB) with fallback to `request.client.host`.
- When limit is breached, returns **HTTP 429 Too Many Requests** with standard `Retry-After: <seconds>` headers.
- Health checks, polling `GET` endpoints, and synchronous utility endpoints are deliberately unthrottled or individually configured.

### Dual-Layer Retry Strategy

Every external LLM interaction is protected by an outer Tenacity wrapper and an inner SDK configuration (`app/services/_retry.py`):

```
Layer 1 — SDK-Native Retries (Inner):
    • OpenAI & Groq: max_retries=2 (evaluates Retry-After headers and server backoff)
    • Gemini: HttpRetryOptions(attempts=2)
Layer 2 — Tenacity Exponential Jitter (Outer):
    • 3 attempts, exponential backoff with full random jitter (wait 3s to 60s)
    • Catches DeadlineExceeded, connection drops, and SDK-exhausted 429/5xx errors
Combined Ceiling:
    3 (tenacity) × 3 (1 initial + 2 SDK retries) = 9 max attempts per logical request
```

### SQLite WAL Mode & Connection Pool

To support concurrent background writes without database lock contention (`SQLITE_BUSY`):
- PRAGMAs configured on connection:
  - `PRAGMA journal_mode=WAL;` — Write-Ahead Logging allows readers to proceed without blocking writers, and vice-versa.
  - `PRAGMA busy_timeout=30000;` — SQLite automatically retries locked operations for up to 30 seconds before raising an error.
  - `PRAGMA synchronous=NORMAL;` — Optimal balance between write performance and durability.
- Connection Pool: `pool_size=5`, `max_overflow=10`, `pool_recycle=1800` (30 minutes), and `check_same_thread=False` to hand connections cleanly across asyncio worker threads.

### Anti-Hallucination & Data Hygiene

LLM-extracted data is passed through rigorous sanitization stages:
1. **Cloudflare Obfuscation Scrubbing**: Rejects `[email protected]` and dummy domains (`example.com`, `test.com`, `acme.com`, etc.).
2. **Authoritative Email Format Regex**: Validates mailbox local-parts and multi-segment RFC domains, rejecting truncated strings or inline descriptions.
3. **Phone Normalization & Length Validation**: Strips formatting symbols to ensure standard numeric digit lengths (6 to 18 digits).
4. **Fuzzy Tag Coercion**: `_resolve_contact_tag()` maps variations (e.g., `"Human Resources"`, `"human_resource"`, `"HR"`) to canonical `ContactTag` enum values, defaulting safely to `ContactTag.OTHERS`.
5. **URL Sanitization & Directory Discard**: The combined search service strips tracking parameters (`utm_*`, `gclid`, `ref`, etc.) and discards directory listings (e.g., `bbb.org`, `yelp.com`, `vertexaisearch.cloud.google.com`) so that only verified direct company domains are returned.

### Unified Token Tracking

All search and verification pipelines track token usage across providers into a standard schema (`TokenUsage`):
- `openai.input_tokens`, `openai.output_tokens`, `openai.total_tokens`
- `gemini.input_tokens`, `gemini.output_tokens`, `gemini.total_tokens`
- `grand_total` representing the overall cumulative consumption.

---

## Environment Configuration

Configuration is managed via Pydantic Settings in `app/core/config.py` and loaded from a `.env` file.

| Variable | Type | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | `string` | `None` | API key for Groq Cloud (used in `/google-search/` and default `/match-companies/`) |
| `OPENAI_API_KEY` | `string` | `None` | API key for OpenAI (used in `/openai-search/`, `/verify-voe/`, `/find-poe/`, etc.) |
| `GEMINI_API_KEY` | `string` | `None` | API key from [Google AI Studio](https://aistudio.google.com/) |
| `SERPER_API_KEY` | `string` | `""` | Optional Serper Google Search API key |
| `SEARCH_PROVIDER` | `string` | `"duckduckgo"` | Search provider for Playwright scraper: `"duckduckgo"` or `"serper"` |
| `DATABASE_URL` | `string` | `"sqlite:///./tasks.db"` | Database connection string. In Docker: `"sqlite:////data/tasks.db"` |
| `WEBHOOK_URL` | `string` | `None` | Optional default webhook destination URL for completed tasks |
| `MAX_CONCURRENT_BROWSERS` | `integer` | `4` | Maximum concurrent Chromium browser instances |
| `PDF_MAX_FILE_SIZE_MB` | `integer` | `20` | Maximum upload file size for `/extract-pdf/` and `/parse-background-check/` |
| `OPENAI_MODEL` | `string` | `"gpt-4o-mini"` | Model used for OpenAI search, verification, and POE discovery |
| `OPENAI_OCR_MODEL` | `string` | `"gpt-4o-mini"` | Model used for OpenAI vision/OCR document extraction |
| `GEMINI_MODEL` | `string` | `"gemini-2.5-flash-lite"` | Model used for Gemini search, verification, and POE discovery |
| `GEMINI_OCR_MODEL` | `string` | `"gemini-2.5-flash-lite"` | Model used for Gemini Files API PDF extraction (use `gemini-2.5-pro` for handwriting) |
| `GROQ_MODEL` | `string` | `"llama-3.1-8b-instant"` | Model used for Groq website verification and company matching |

---

## Installation & Local Setup

### Prerequisites
- **Python**: 3.11 or higher
- **Node/OS Packages**: Chromium dependencies for Playwright

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd scraper

# 2. Create and activate a virtual environment
python3.11 -m venv scapEnv
source scapEnv/bin/activate  # On Windows: scapEnv\Scripts\activate

# 3. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Install Playwright Chromium browser and OS dependencies
playwright install chromium
playwright install-deps chromium  # Installs system libraries on Linux

# 5. Configure environment variables
cp .env.example .env
# Edit .env with your GROQ_API_KEY, OPENAI_API_KEY, and GEMINI_API_KEY

# 6. Run the application
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation will be available at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## Docker Deployment

The application features a production-ready, multi-stage `Dockerfile` and `docker-compose.yml` configuration.

### Multi-Stage Build
1. **Stage 1 (`builder`)**: Installs native build dependencies (`gcc`, `libffi-dev`) and compiles wheels into an isolated `/install` prefix.
2. **Stage 2 (`runtime`)**: Minimal `python:3.11-slim` runtime containing only Chromium system dependencies, Playwright browser binaries, and the application source.

### Running with Docker Compose

```bash
# 1. Prepare environment
cp .env.example .env
# Fill in API keys in .env

# 2. Build and launch the container
docker compose up --build -d

# 3. Check container logs
docker compose logs -f scraper

# 4. Verify health check status
docker ps --filter "name=scraper_api"
```

### Persistent Volumes
- `db-data`: Mounts to `/data` in the container to persist `tasks.db` across container rebuilds.
- `logs`: Mounts to `/app/scraper_logs` for audit and post-mortem analysis.

> [!IMPORTANT]
> **Single Worker Rule (`--workers 1`)**:
> The service uses an in-process `asyncio.Semaphore` (browser gate) and an in-memory sliding-window rate limiter. Both are per-process singletons and will not synchronize across multiple OS workers. Scale capacity via `MAX_CONCURRENT_BROWSERS` and vertical CPU/RAM allocation rather than adding Uvicorn workers.

---

## API Reference

### 1. Headless Browser Scraper (`/google-search/`)

Crawls the web using Playwright with stealth evasion plugins to discover a company's official homepage, harvests up to 4 contact-related subpages, and extracts structured contact info using Groq.

#### Submit Task
- **Endpoint**: `POST /google-search/`
- **Rate Limited**: Yes (15 req/60s)
- **Request Body**:
```json
{
  "poe_name": "Acme Industrial Supplies",
  "timeout": 120
}
```
- **Response** (`200 OK`):
```json
{
  "task_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "status": "IN_PROGRESS"
}
```

#### Poll Task Status
- **Endpoint**: `GET /google-search/{task_id}`
- **Response** (`200 OK`):
```json
{
  "task_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "status": "SUCCESS",
  "message": "Successfully extracted contact info",
  "result": {
    "poe_name": "Acme Industrial Supplies",
    "official_site": "https://www.acmeindustrial.com",
    "poe_info": {
      "Phone": ["(555) 234-5678"],
      "Fax": ["(555) 234-5679"],
      "Email": ["contact@acmeindustrial.com", "careers@acmeindustrial.com"],
      "Address": ["100 Innovation Way, Suite 200, Austin, TX 78701"],
      "DeptContacts": null
    }
  },
  "created_at": "2026-10-13T10:00:00Z",
  "updated_at": "2026-10-13T10:00:35Z"
}
```

#### List Failed Tasks
- **Endpoint**: `GET /google-search/failed/?limit=10`

---

### 2. OpenAI Grounded Search (`/openai-search/`)

Uses OpenAI's Responses API with multi-turn `web_search` tool execution across 3 chained turns, extracting tagged contacts and verifying them against raw citations.

#### Submit Task
- **Endpoint**: `POST /openai-search/`
- **Rate Limited**: Yes
- **Request Body**:
```json
{
  "company_name": "DEMOULAS MARKET BASKET",
  "country": "United States",
  "zip_code": "01876",
  "url": "https://www.shopmarketbasket.com",
  "max_limit": 5,
  "phone": "(978) 640-8300",
  "address": {
    "address1": "87 Prescott Rd",
    "city": "Tewksbury",
    "state": "MA",
    "zip": "01876",
    "country": "United States"
  }
}
```
- **Response** (`200 OK`):
```json
{
  "task_id": "4cf510b1-4f11-4776-8809-5c4d3fa52a3b",
  "status": "IN_PROGRESS"
}
```

#### Poll Task Status
- **Endpoint**: `GET /openai-search/{task_id}`
- **Response** (`200 OK`):
```json
{
  "task_id": "4cf510b1-4f11-4776-8809-5c4d3fa52a3b",
  "status": "SUCCESS",
  "message": "Successfully extracted contact info via OpenAI",
  "result": {
    "company_name": "DEMOULAS MARKET BASKET",
    "official_company_name": "Demoulas Super Markets, Inc.",
    "official_site": "https://www.shopmarketbasket.com",
    "company_info": {
      "phones": [
        {
          "value": "(978) 640-8300",
          "tag": "Headquarters",
          "context": "Main corporate office telephone"
        }
      ],
      "faxes": [
        {
          "value": "(978) 640-8390",
          "tag": "Human Resource",
          "context": "Corporate HR and benefits fax"
        }
      ],
      "emails": [
        {
          "value": "hr@shopmarketbasket.com",
          "tag": "Human Resource",
          "context": "Personnel department inquiry"
        }
      ],
      "addresses": [
        {
          "address1": "87 Prescott Road",
          "address2": "",
          "city": "Tewksbury",
          "state": "MA",
          "zip": "01876",
          "country": "United States",
          "countryCode": "US",
          "tag": "Headquarters",
          "context": "Corporate Headquarters"
        }
      ]
    },
    "token_usage": {
      "openai": {
        "input_tokens": 12450,
        "output_tokens": 820,
        "total_tokens": 13270
      },
      "gemini": null,
      "grand_total": {
        "input_tokens": 12450,
        "output_tokens": 820,
        "total_tokens": 13270
      }
    }
  },
  "created_at": "2026-10-13T10:05:00Z",
  "updated_at": "2026-10-13T10:05:22Z"
}
```

---

### 3. Gemini Grounded Search (`/gemini-search/`)

Executes 3 concurrent specialized sub-gatherers using Google Gemini with live Google Search grounding (`MODE_ALWAYS`), followed by native JSON extraction and hallucination verification.

#### Submit Task
- **Endpoint**: `POST /gemini-search/`
- **Rate Limited**: Yes
- **Request Body**:
```json
{
  "company_name": "Troopr",
  "country": "United States",
  "url": "https://www.troopr.ai"
}
```
- **Response** (`200 OK`):
```json
{
  "task_id": "1c7a8b6e-2134-4b55-a9a7-9e7f86f8c7b1",
  "status": "IN_PROGRESS"
}
```

#### Poll Task Status
- **Endpoint**: `GET /gemini-search/{task_id}`
- **Response** (`200 OK`):
Returns the `GeminiSearchResult` schema with structured contacts, official company name, and Gemini token usage metadata.

---

### 4. Combined Grounded Search (`/combined-search/`)

Runs both OpenAI and Gemini pipelines in parallel, sanitises URLs, filters third-party directories, merges and deduplicates contacts, and provides aggregate statistics.

#### Submit Task
- **Endpoint**: `POST /combined-search/`
- **Rate Limited**: Yes
- **Request Body**:
```json
{
  "company_name": "TRUCKSMART",
  "country": "United States",
  "zip_code": "95677",
  "max_limit": 5,
  "type": "rolodex"
}
```
- **Response** (`200 OK`):
```json
{
  "task_id": "5e1347a1-76cc-4d1f-97d7-be3796eb1e4f",
  "status": "IN_PROGRESS"
}
```

#### Poll Task Status
- **Endpoint**: `GET /combined-search/{task_id}`
- **Response** (`200 OK`):
```json
{
  "task_id": "5e1347a1-76cc-4d1f-97d7-be3796eb1e4f",
  "status": "SUCCESS",
  "message": "Successfully extracted contact info via combined search",
  "result": {
    "company_name": "TRUCKSMART",
    "official_company_name": "Trucksmart Inc.",
    "official_site": "https://www.trucksmart.com",
    "company_info": {
      "phones": [
        {"value": "(916) 773-7333", "tag": "Headquarters", "context": "Main telephone"}
      ],
      "faxes": [
        {"value": "(916) 781-7725", "tag": "Admin", "context": "Office fax"}
      ],
      "emails": [
        {"value": "info@trucksmart.com", "tag": "General", "context": "General inquiries"}
      ],
      "addresses": [
        {
          "address1": "6455 Pacific St",
          "city": "Rocklin",
          "state": "CA",
          "zip": "95677",
          "country": "United States",
          "countryCode": "US",
          "tag": "Headquarters",
          "context": "Corporate office"
        }
      ]
    },
    "summary": {
      "openai": {"total_phones": 1, "total_faxes": 1, "total_emails": 2, "total_addresses": 1},
      "gemini": {"total_phones": 1, "total_faxes": 1, "total_emails": 1, "total_addresses": 1},
      "combined": {"total_phones": 1, "total_faxes": 1, "total_emails": 2, "total_addresses": 1}
    },
    "token_usage": {
      "openai": {"input_tokens": 8100, "output_tokens": 520, "total_tokens": 8620},
      "gemini": {"input_tokens": 7400, "output_tokens": 490, "total_tokens": 7890},
      "grand_total": {"input_tokens": 15500, "output_tokens": 1010, "total_tokens": 16510}
    }
  },
  "created_at": "2026-10-13T10:10:00Z",
  "updated_at": "2026-10-13T10:10:28Z"
}
```

---

### 5. Verification of Employment (`/verify-voe/`)

Evaluates whether a named individual holds a designated position at a company using grounded web research and a calibrated scoring rubric.

#### Submit Task
- **Endpoint**: `POST /verify-voe/`
- **Rate Limited**: Yes
- **Request Body**:
```json
{
  "full_name": "Jane Doe",
  "job_title": "Chief Financial Officer",
  "company": "Acme Corp",
  "city": "New York",
  "state": "NY",
  "country": "United States",
  "provider": "both"
}
```
- **Response** (`200 OK`):
```json
{
  "task_id": "7a3e9c2b-8812-4f30-b1d5-998877665544",
  "status": "IN_PROGRESS",
  "provider": "both"
}
```

#### Scoring & Verdict Rubric

| Confidence Score | Verdict | Meaning |
|---|---|---|
| **9.0 – 10.0** | `VERIFIED` | Direct evidence (LinkedIn, corporate bio, press release) confirms person and role |
| **6.0 – 8.0** | `LIKELY` | Strong indirect evidence ties individual to company, but exact current title is ambiguous |
| **3.0 – 5.0** | `UNVERIFIED` | Both entity and person exist, but no public connection found |
| **0.0 – 2.0** | `CONTRADICTED` | Evidence proves individual departed, works elsewhere, or has no public footprint |

#### Poll Task Status
- **Endpoint**: `GET /verify-voe/{task_id}`
- **Response** (`200 OK` when `provider="both"`):
```json
{
  "task_id": "7a3e9c2b-8812-4f30-b1d5-998877665544",
  "status": "SUCCESS",
  "message": "VOE verification complete (both providers)",
  "result": {
    "full_name": "Jane Doe",
    "company": "Acme Corp",
    "job_title": "Chief Financial Officer",
    "best_result": {
      "full_name": "Jane Doe",
      "company": "Acme Corp",
      "job_title": "Chief Financial Officer",
      "confidence_score": 9.5,
      "verdict": "VERIFIED",
      "evidence_summary": "Official corporate leadership page and recent Q3 earnings press release confirm Jane Doe as CFO.",
      "sources_found": [
        "https://acme.com/leadership/jane-doe",
        "https://www.prnewswire.com/news-releases/acme-q3-results-2026.html"
      ]
    },
    "gemini": {
      "provider": "gemini",
      "result": { ... },
      "error": null
    },
    "openai": {
      "provider": "openai",
      "result": { ... },
      "error": null
    },
    "token_usage": {
      "openai": { "input_tokens": 4200, "output_tokens": 310, "total_tokens": 4510 },
      "gemini": { "input_tokens": 3900, "output_tokens": 280, "total_tokens": 4180 },
      "grand_total": { "input_tokens": 8100, "output_tokens": 590, "total_tokens": 8690 }
    }
  }
}
```

---

### 6. Find Place of Employment (`/find-poe/`)

Discovers where a named individual is currently employed by investigating public web sources and returning a ranked list of employer candidates.

#### Submit Task
- **Endpoint**: `POST /find-poe/`
- **Rate Limited**: Yes
- **Request Body**:
```json
{
  "full_name": "Alex Mercer",
  "job_title": "Director of Engineering",
  "city": "Seattle",
  "state": "WA",
  "country": "United States",
  "provider": "gemini"
}
```
- **Response** (`200 OK`):
```json
{
  "task_id": "f2a1b3c4-d5e6-4789-90ab-cdef12345678",
  "status": "IN_PROGRESS",
  "provider": "gemini"
}
```

#### Poll Task Status
- **Endpoint**: `GET /find-poe/{task_id}`
- **Response** (`200 OK`):
```json
{
  "task_id": "f2a1b3c4-d5e6-4789-90ab-cdef12345678",
  "status": "SUCCESS",
  "message": "FindPOE pipeline complete (gemini)",
  "result": {
    "full_name": "Alex Mercer",
    "job_title": "Director of Engineering",
    "best_match": {
      "company_name": "CloudScale Systems",
      "confidence_score": 9.2,
      "evidence_summary": "LinkedIn profile and company engineering blog confirm Alex Mercer currently serves as Director of Engineering at CloudScale Systems since 2024.",
      "sources_found": [
        "https://www.linkedin.com/in/alexmercer-eng",
        "https://blog.cloudscale.io/authors/alex-mercer"
      ]
    },
    "candidates": [
      {
        "company_name": "CloudScale Systems",
        "confidence_score": 9.2,
        "evidence_summary": "Current Director of Engineering.",
        "sources_found": ["https://www.linkedin.com/in/alexmercer-eng"]
      },
      {
        "company_name": "Legacy Tech Corp",
        "confidence_score": 3.5,
        "evidence_summary": "Former Staff Engineer from 2020-2023.",
        "sources_found": ["https://techcrunch.com/2021/legacy-tech-team"]
      }
    ],
    "token_usage": {
      "gemini": {"input_tokens": 3800, "output_tokens": 410, "total_tokens": 4210},
      "grand_total": {"input_tokens": 3800, "output_tokens": 410, "total_tokens": 4210}
    }
  }
}
```

---

### 7. Company Name Matcher (`/match-companies/`)

Determines whether two company name strings refer to the same real-world entity, handling acronyms, legal suffixes, typos, and OCR glitches.

#### Synchronous Request
- **Endpoint**: `POST /match-companies/`
- **Execution**: Synchronous
- **Rate Limited**: Yes
- **Request Body**:
```json
{
  "company_a": "IBM",
  "company_b": "International Business Machines Corp.",
  "provider": "groq",
  "deep_search": false
}
```

#### Deep Search Mode
Set `"deep_search": true` for difficult local businesses, DBA names, or trade names (e.g., comparing `"Thomas Electrical"` against `"THOMAS P. OWENS, LLC"`). The engine executes live web research before delivering the verdict.

- **Response** (`200 OK`):
```json
{
  "company_a": "IBM",
  "company_b": "International Business Machines Corp.",
  "is_same_company": true,
  "confidence": 0.99,
  "reasoning": "IBM is the universally recognized acronym for International Business Machines Corporation.",
  "provider": "groq",
  "deep_search_used": false,
  "processing_time_seconds": 0.42
}
```

---

### 8. Vision PDF Text Extraction (`/extract-pdf/`)

Extracts text from scanned or digital PDFs using vision-capable LLMs, transcribing form checkboxes and radio buttons as `[SELECTED]` or `[ ]`.

- **Endpoint**: `POST /extract-pdf/`
- **Execution**: Synchronous
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file`: PDF binary (max 20 MB)
  - `provider`: `"gemini"` (default) or `"openai"`
- **Response** (`200 OK`):
```json
{
  "filename": "employment_application.pdf",
  "provider": "gemini",
  "page_count": 3,
  "extracted_text": "--- Page 1 ---\nEMPLOYMENT APPLICATION\nFull Name: Jane Doe\nEligible to work in US: [SELECTED] Yes  [ ] No\n...",
  "processing_time_seconds": 3.85
}
```

---

### 9. Background Check Report Parser (`/parse-background-check/`)

A specialized two-stage pipeline for pre-employment screening reports (HireRight, Sterling, Checkr, Accurate, First Advantage). Stage 1 runs OCR extraction; Stage 2 normalizes 7 target fields into structured JSON.

- **Endpoint**: `POST /parse-background-check/`
- **Execution**: Synchronous
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file`: PDF binary (max 20 MB)
  - `provider`: `"gemini"` (default) or `"openai"`
- **Response** (`200 OK`):
```json
{
  "filename": "hireright_report_sample.pdf",
  "provider": "gemini",
  "processing_time_seconds": 4.12,
  "data": {
    "file_number": "BGC-2026-99124",
    "employee_name": "Jane Elizabeth Doe",
    "date_of_birth": "1988-04-12",
    "requested_by": "Global Logistics Corp HR",
    "employer_name": "Apex Delivery Systems",
    "report_date": "2026-09-18",
    "status": "Clear"
  }
}
```

---

### 10. Webhook Mock (`/webhook-mock`)

Test endpoint that accepts incoming `WebhookPayload` objects and logs them. Useful for local testing of webhook delivery.

- **Endpoint**: `POST /webhook-mock`
- **Request Body**: `WebhookPayload`
- **Response** (`200 OK`):
```json
{
  "received": true
}
```

---

## Data Models & Taxonomy

### Departmental Contact Taxonomy (`ContactTag`)

The system categorizes every email, phone, fax, and address into 31 standardized tags:

```
Core HR & Employment:
  • "Human Resource"          • "Payroll"                 • "Personnel"
  • "Benefits"                • "Recruiting"              • "Candidate Support"
  • "Training"                • "Background Check"        • "Employment Verification"
  • "Labor relations"

Business & Operations:
  • "Admin"                   • "Management"              • "Operations"
  • "Corporate"               • "Headquarters"            • "Finance"
  • "Accounting"              • "Legal"                   • "Compliance"
  • "IT"                      • "Safety"

Customer-Facing:
  • "Sales"                   • "Customer Service"        • "Support"
  • "Marketing"               • "Public Relations"

Administrative & Catch-All:
  • "Careers"                 • "Secretary"               • "General"
  • "Others"
```

### Disambiguation Address Input (`HintAddress`)

Used in `/openai-search/`, `/gemini-search/`, and `/combined-search/` to anchor multi-location companies:

```json
{
  "address1": "100 Main St",
  "address2": "Suite 400",
  "city": "Boston",
  "state": "MA",
  "zip": "02110",
  "country": "United States"
}
```

### Database Schema (`TaskRecord`)

Defined using **SQLModel** (`tasks.db`):

| Column | Type | Constraints | Description |
|---|---|---|---|
| `task_id` | `VARCHAR` | Primary Key | UUID v4 tracking identifier |
| `status` | `VARCHAR` | Default: `"PENDING"` | `"PENDING"`, `"IN_PROGRESS"`, `"SUCCESS"`, `"FAILURE"` |
| `message` | `VARCHAR` | Nullable | Human-readable outcome or failure description |
| `result_data` | `JSON` | Nullable | Complete JSON serialization of the result model |
| `created_at` | `DATETIME` | UTC timestamp | Record creation timestamp |
| `updated_at` | `DATETIME` | UTC timestamp | Last state change timestamp |

---

## Observability & Error Handling

### Logging Configuration
Logs are handled by **Loguru** with both stdout streaming and rotating file output:
- **File**: `scraper.log`
- **Rotation**: `100 MB`
- **Retention**: `30 days`
- **Level**: `INFO` (switch to `DEBUG` via environment if needed)

### Standard HTTP Status Codes

| Code | Meaning | Context |
|---|---|---|
| `200 OK` | Success | Immediate result returned or task created successfully |
| `400 Bad Request` | Invalid Input | Malformed JSON, non-PDF file upload, or empty required field |
| `404 Not Found` | Not Found | Task ID not found in SQLite database |
| `413 Payload Too Large` | File Too Large | PDF exceeds `PDF_MAX_FILE_SIZE_MB` limit |
| `415 Unsupported Media Type` | Bad Media Type | File magic bytes do not match `%PDF-` |
| `429 Too Many Requests` | Rate Limited | Sliding-window limit exceeded (includes `Retry-After` header) |
| `500 Internal Server Error` | Server Error | Unhandled exception during pipeline execution |

---

## Repository Structure

```
scraper/
├── app/
│   ├── api/
│   │   ├── deps.py               # SQLite WAL engine & get_session dependency
│   │   ├── rate_limit.py          # In-process sliding-window rate limiter
│   │   └── routes.py              # Endpoint routing & background task handlers
│   ├── core/
│   │   └── config.py              # Pydantic BaseSettings environment loader
│   ├── models/
│   │   └── models.py              # Pydantic v2 schemas & SQLModel TaskRecord
│   └── services/
│       ├── _retry.py              # Dual-layer Tenacity retry factories
│       ├── _token_utils.py        # Token usage normalization helpers
│       ├── bg_check_parser.py     # Background check report parsing pipeline
│       ├── combined_search.py     # Parallel OpenAI + Gemini aggregator
│       ├── company_matcher.py     # Company identity matching service
│       ├── gemini_find_poe.py     # Gemini employer discovery pipeline
│       ├── gemini_search.py       # Gemini 3-sub-gatherer search pipeline
│       ├── llm.py                 # Groq LLaMA-3.1 inference service
│       ├── openai_find_poe.py     # OpenAI employer discovery pipeline
│       ├── openai_search.py       # OpenAI 3-turn search pipeline
│       ├── openai_voe.py          # OpenAI employment verification pipeline
│       ├── pdf_extractor.py       # Vision PDF extraction (Gemini / OpenAI)
│       ├── scraper.py             # Playwright stealth browser engine
│       ├── voe_verification.py    # Gemini employment verification pipeline
│       └── webhook.py             # Outbound HTTP webhook delivery
├── docs/                          # Specialized API guides & architectural notes
├── .dockerignore                  # Docker build exclusions
├── .env.example                   # Annotated template for environment variables
├── ARCHITECTURE.md                # System sequence and workflow diagrams
├── Dockerfile                     # Multi-stage production container definition
├── docker-compose.yml             # Docker Compose orchestration with health checks
├── main.py                        # Uvicorn entrypoint & lifespan database init
├── requirements.txt               # Pinned Python package dependencies
└── README.md                      # Primary project documentation
```
