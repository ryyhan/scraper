# Scraper Service Architecture

This document describes the architectural flow of the asynchronous, FastAPI-based scraping application.

## Core Technologies
- **API Framework**: FastAPI
- **Database**: SQLite (managed via SQLModel)
- **Scraping Engine**: HTTPX (Initial Search) & Playwright Async + Stealth (Deep interaction)
- **AI Extraction**: Groq API (`llama-3.3-70b-versatile`)
- **Logging**: Loguru

---

## 1. High-Level Flow Diagram

The following flowchart illustrates the entire lifecycle of a scraping request, from incoming API call to the final webhook delivery.

```mermaid
flowchart TD
    A[Client Request POST /google-search/] --> B[FastAPI Router]
    
    subgraph Initial Request
        B --> C{Create TaskRecord}
        C -->|Save to DB| D[(SQLite tasks.db)]
        C -->|Return task_id| E[Response 200 OK]
    end
    
    C -->|Trigger| F[Background Task]
    
    subgraph Async Background Process (Controlled via Semaphore)
        F --> G[Scraper Service: DuckDuckGo HTML via HTTPX]
        G --> H{Results Found?}
        
        H -->|No| I[Update Status: FAILURE]
        H -->|Yes| J[LLM Service: Verify Official Site]
        
        J --> K{Site Confirmed?}
        K -->|No| I
        K -->|Yes| L[Scraper Service: Playwright Stealth Harvest]
        
        L --> M[Extract deepest links 'contact', 'about', etc.]
        M --> N[Scraper Service: Extract Page Text Max 15k]
        
        N --> O[LLM Service: Extract Structued Data to JSON]
        O --> P[Validation success?]
        
        P -->|No| I
        P -->|Yes| Q[Update Status: SUCCESS + Data]
    end
    
    Q --> R
    I --> R[Update DB Record]
    
    R --> S[Webhook Service: Push payload to External Client]
```

---

## 2. Sequence Diagram

This sequence diagram details the interaction between the core internal services.

```mermaid
sequenceDiagram
    participant User
    participant App as FastAPI
    participant DB as SQLite
    participant Worker as Background Task
    participant Scraper as ScraperService
    participant LLM as LLMService
    participant Webhook as WebhookService

    User->>App: POST /google-search/ {"poe_name": "Example Corp"}
    App->>DB: INSERT Task (id, status=IN_PROGRESS)
    App->>Worker: Dispatch `process_scraping_task`
    App-->>User: HTTP 200 {"task_id": "...", "status": "IN_PROGRESS"}
    
    rect rgb(245, 245, 245)
        Note right of Worker: Background Processing Flow
        
        Worker->>Scraper: perform_duckduckgo_search("Example Corp")
        Note right of Scraper: Uses httpx to bypass JS bot-detection
        Scraper-->>Worker: List of top 5 result URLs
        
        Worker->>LLM: verify_official_site([urls])
        Note right of LLM: Groq Llama3 evaluates URLs
        LLM-->>Worker: Verified Homepage URL
        
        Worker->>Scraper: harvest_contact_links(homepage)
        Note right of Scraper: Playwright + Stealth async renders page
        Scraper-->>Worker: [homepage, /about, /contact]
        
        loop For each link (max 3)
            Worker->>Scraper: extract_page_text(url)
            Scraper-->>Worker: Visible body text (truncated)
        end
        
        Worker->>LLM: extract_contact_info(combined_text)
        Note right of LLM: Prompts for JSON output mode
        LLM-->>Worker: Structured ContactInfo Dictionary
    end
    
    Worker->>DB: UPDATE Task (status, result_data)
    Worker->>Webhook: submit_result(webhook_url, final_payload)
```

## 3. Package Structure

The project has been refactored into a scalable package structure:

- `app/` - Core application root
  - `api/` - Routing (`routes.py`) and Dependency Injection (`deps.py`).
  - `core/` - Application-wide configuration and settings logic (`config.py`).
  - `models/` - Pydantic request/response schemas and SQLModel database entities (`models.py`).
  - `services/` - Isolated business logic:
    - `scraper.py` (Manages Playwright browser lifecycles and HTTPX stealth searches).
    - `llm.py` (Interacts with the Groq inference engine).
    - `webhook.py` (Handles HTTP webhook pushing logic).
- `main.py` - The lightweight uvicorn entrypoint.

## 4. Concurrency Management
To prevent browser processes from overwhelming the server's RAM, the background task queue enforces a global `asyncio.Semaphore`. Even if a user throws 1,000 tasks at the POST endpoint, only `MAX_CONCURRENT_BROWSERS` (configurable via `.env`) instances will process concurrently. The rest will wait safely in the asyncio event loop queue.
