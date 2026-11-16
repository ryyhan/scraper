# Phase 2: Search Engine Expansion Plan

## 🎯 Executive Summary
In Phase 2, we will evolve the Scraper API from a single-engine utility into a **Multi-Provider Search Platform**. 

Currently, the system uses a combined Serper + Groq + Playwright approach (the "Google Search" API). In this 3-month phase, we will introduce two brand new, independent modules:
1. **OpenAI Search API**: Powered by OpenAI foundational models and their native web search tools.
2. **Gemini Search API**: Powered by Google Gemini models and native Google Search grounding.

This modular architecture will empower the frontend client to dynamically route tasks to the best provider based on cost, latency, or target website complexity without altering their ingestion flow.

---

## 🗓️ Month 1: Foundation & Modularization
*Goal: Prepare the architecture for multiple independent AI providers without breaking the existing V1 logic.*

### Week 1-2: Architectural Refactoring
- **Define Interface Contracts**: Create abstract base classes (e.g., `BaseSearchProvider`) that define the required input/output behavior for *all* search engines.
- **Standardize Data Models**: Refactor `SearchRequest` and `ScrapeResult` into generic payloads that can accommodate slight variations between providers while maintaining a strict `ContactInfo` schema.
- **Isolate Legacy Code**: Sandbox the existing Serper/Groq/Playwright codebase into its own self-contained service namespace (e.g., `LegacyGoogleSearchService`) to prevent regressions.

### Week 3-4: Database & Infrastructure Readiness
- **Database Schema Updates**: Update the SQLite `TaskRecord` to include `provider_type` (e.g., `openai`, `gemini`, `legacy`) and `execution_cost` mapping.
- **Environment Management**: Introduce robust config validation for the upcoming `.env` variables (`OPENAI_API_KEY`, `GEMINI_API_KEY`).
- **Billing & Token Tracking**: Add middleware to track token usage across different providers to allow for future cost analysis.

---

## 🗓️ Month 2: OpenAI Integration
*Goal: Build, test, and release the OpenAI-native search module.*

### Week 1-2: OpenAI Search Service
- **SDK Integration**: Install and configure the official OpenAI Python SDK.
- **The "Two-Step" AI Pipeline**:
  - **Step 1 (Gather)**: Prompt `gpt-4o-mini` (or similar) using the native `web_search` tool to dynamically browse the web for company background and contact data.
  - **Step 2 (Extract)**: Pass the raw gathered text into a strict structured output schema (JSON mode combined with Pydantic validation) to guarantee exact data extraction.
- **Concurrency Management**: Because the OpenAI Python SDK is synchronous, wrap all network calls in `asyncio.to_thread` to ensure the FastAPI event loop remains unblocked.

### Week 3-4: Endpoint Setup & Hardening
- **Route Implementation**: Expose `POST /openai-search/` and `GET /openai-search/{task_id}` endpoints.
- **Error Handling**: Build specific resilience around OpenAI's rate limiting (`HTTP 429`) and context-window errors using Tenacity retry blocks.
- **E2E Testing**: Push test workloads through the system and verify external Webhook compatibility. 

---

## 🗓️ Month 3: Gemini Integration & Frontend Handoff
*Goal: Deliver the Gemini module and finalize client documentation.*

### Week 1-2: Gemini Search Service
- **SDK Integration**: Integrate the Google GenAI SDK.
- **Search Grounding**: Implement LLM calls utilizing Google's native Search Grounding (`tools=[{"google_search_retrieval": ...}]`) to accurately look up missing contact domains without needing a headless browser.
- **Extraction Formatting**: Map the Gemini function-calling/schema output natively to the standard `ContactInfo` response format.

### Week 3-4: API Finalization & Frontend Delivery
- **Route Implementation**: Expose `POST /gemini-search/` and `GET /gemini-search/{task_id}`.
- **Swagger / OpenAPI Polish**: Extensively document the three separate API branches in the Swagger UI (`/docs`). Include payload examples so the Frontend team understands exact differences.
- **Handoff & Client Routing Setup**:
  - Advise the frontend team on how to implement conditional logic (e.g., UI toggle: "Use OpenAI (More Accurate) vs Use Gemini (Faster)").
- **Production Deployment**: Roll phase 2 into production monitors and observe platform stability.

---

## 📊 End of Phase 2 Deliverables
1. **Three isolated REST API pipelines** (`/google-search/`, `/openai-search/`, `/gemini-search/`).
2. **Unified Polling and Webhook architectures** for robust async UX.
3. Fully unblocked, horizontally scalable background worker logic.
4. Extensive documentation for the Frontend engineering team.
