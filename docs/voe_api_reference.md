# VOE — Verification of Employment API Reference

A comprehensive reference for the `/verify-voe/` endpoints: what to send, what you get back, and how the verification logic works under the hood.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Endpoints](#endpoints)
3. [Request Body](#request-body)
4. [Provider Modes](#provider-modes)
5. [Pipeline Logic](#pipeline-logic)
6. [Response Shapes](#response-shapes)
7. [Confidence Score & Verdict Rubric](#confidence-score--verdict-rubric)
8. [Polling Flow](#polling-flow)
9. [Error Handling](#error-handling)
10. [Full Examples](#full-examples)

---

## Quick Start

```bash
# 1. Submit a verification job
curl -X POST http://localhost:8000/verify-voe/ \
  -H "Content-Type: application/json" \
  -d '{"full_name": "Jane Doe", "job_title": "CFO", "company": "Acme Corp"}'

# → {"task_id": "a1b2c3...", "status": "IN_PROGRESS", "provider": "gemini"}

# 2. Poll until complete
curl http://localhost:8000/verify-voe/a1b2c3...
```

---

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/verify-voe/` | Submit a new verification job |
| `GET` | `/verify-voe/{task_id}` | Poll the status / result of a job |
| `GET` | `/verify-voe/failed/` | List recently failed VOE jobs |

---

## Request Body

`POST /verify-voe/`

```json
{
  "full_name":  "Jane Doe",
  "job_title":  "Chief Financial Officer",
  "company":    "Acme Corp",
  "zip_code":   "10001",
  "city":       "New York",
  "state":      "NY",
  "country":    "United States",
  "provider":   "gemini"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `full_name` | `string` | ✅ | Full legal name of the person to verify |
| `job_title` | `string` | ✅ | Job title claimed by / for the person |
| `company` | `string` | ✅ | Employer name |
| `zip_code` | `string` | ❌ | Postal code — improves disambiguation for common names |
| `city` | `string` | ❌ | City — improves disambiguation for common names |
| `state` | `string` | ❌ | State or province (e.g. `"TX"`, `"Ontario"`) — improves disambiguation for common names |
| `country` | `string` | ❌ | Country — improves disambiguation for common names |
| `provider` | `"gemini"` \| `"openai"` \| `"both"` | ❌ | LLM provider to use. **Default: `"gemini"`** |

> **Tip:** Always supply at least one of `city`, `state`, `zip_code`, or `country` for common names (e.g. "John Smith") to prevent the model from researching the wrong person.

---

## Provider Modes

### `gemini` *(default)*

Uses Google Gemini with its native **`google_search` grounding tool**. Gemini issues live Google Search queries mid-generation and cites its sources directly in the response.

- **Model:** configurable via `GEMINI_MODEL` in `.env` (default: `gemini-2.5-flash-lite`)
- **Best for:** broad web coverage, strong LinkedIn/company-site research, fast turnaround

### `openai`

Uses OpenAI's **Responses API with the `web_search` tool** (`search_context_size: "high"`). OpenAI performs live web searches and consolidates findings before generating the structured result.

- **Model:** configurable via `OPENAI_MODEL` in `.env`
- **Best for:** cases where Gemini grounding is blocked or returns sparse results

### `both`

Runs both providers **concurrently** (via `asyncio.gather`). Both pipelines start at the same time and their results are compared. The provider with the **higher `confidence_score`** is surfaced as `best_result`.

- **Requires:** both `GEMINI_API_KEY` and `OPENAI_API_KEY` set
- **Best for:** high-stakes verifications where you want cross-validated evidence
- **Partial success:** if one provider fails, the task still succeeds using the working provider's result; the failed provider's error is recorded in its slot

---

## Pipeline Logic

Every provider (Gemini or OpenAI) runs a **two-step pipeline**:

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1 – "Investigator"  (live web research)                   │
│                                                                 │
│  The model is given the subject context and instructed to:      │
│  • Search LinkedIn, the company website, press releases,        │
│    news articles, SEC filings, and professional directories     │
│  • Document the person's name + title appearing together        │
│    in any credible source                                       │
│  • Note contradictory evidence (changed role, left company)     │
│  • Use location context to disambiguate common names            │
│  • Check at least 3–5 distinct sources                          │
│                                                                 │
│  Output: free-text evidence summary with cited source URLs      │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                    (empty evidence? → early UNVERIFIED return,
                     Step 2 is skipped to avoid a wasted call)
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2 – "Analyst"  (structured scoring)                       │
│                                                                 │
│  The raw evidence from Step 1 is passed back to the model       │
│  alongside a strict scoring rubric. The model must produce      │
│  a validated JSON object (VoeVerificationResult) containing:    │
│  • confidence_score  (0.0 – 10.0, calibrated to the rubric)    │
│  • verdict           (VERIFIED / LIKELY / UNVERIFIED /          │
│                       CONTRADICTED)                             │
│  • evidence_summary  (2–3 sentences)                            │
│  • sources_found     (list of URLs / publication names)         │
│                                                                 │
│  Gemini: uses response_schema= for native Pydantic parsing      │
│  OpenAI: uses json_object mode + manual model_validate()        │
└─────────────────────────────────────────────────────────────────┘
```

### `provider="both"` flow

```
POST /verify-voe/  (provider="both")
         │
         ├── asyncio.to_thread(gemini_svc.verify, request)  ─┐
         │                                                    ├── run concurrently
         └── asyncio.to_thread(openai_svc.verify, request)  ─┘
                                  │
                      asyncio.gather(return_exceptions=True)
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
              gemini result               openai result
              (or error)                  (or error)
                    │                           │
                    └──────────┬────────────────┘
                               │
                   both failed? → RuntimeError → task FAILURE
                               │
                   pick winner: max(candidates, key=confidence_score)
                               │
                        CombinedVoeResult
                       ┌───────────────────┐
                       │ best_result        │  ← winner
                       │ gemini { result }  │  ← full gemini output
                       │ openai { result }  │  ← full openai output
                       └───────────────────┘
```

---

## Response Shapes

### Immediate response (POST)

```json
{
  "task_id":  "a1b2c3d4-...",
  "status":   "IN_PROGRESS",
  "provider": "gemini"
}
```

---

### Polled response — single provider (`gemini` or `openai`)

`GET /verify-voe/{task_id}` once complete:

```json
{
  "task_id":    "a1b2c3d4-...",
  "status":     "SUCCESS",
  "message":    "Verification complete (gemini) – score=8.5 verdict=VERIFIED",
  "result": {
    "full_name":         "Jane Doe",
    "company":           "Acme Corp",
    "job_title":         "Chief Financial Officer",
    "confidence_score":  8.5,
    "verdict":           "VERIFIED",
    "evidence_summary":  "Jane Doe is listed as CFO on Acme Corp's official leadership page and was quoted in a 2024 Business Wire press release in that role. Her LinkedIn profile corroborates this with a tenure start of March 2022.",
    "sources_found": [
      "https://www.acmecorp.com/about/leadership",
      "https://www.businesswire.com/news/...",
      "https://www.linkedin.com/in/jane-doe-acme"
    ]
  },
  "created_at": "2026-06-25T10:30:00Z",
  "updated_at": "2026-06-25T10:31:42Z"
}
```

---

### Polled response — `provider="both"`

```json
{
  "task_id":    "a1b2c3d4-...",
  "status":     "SUCCESS",
  "message":    "Combined verification complete – best=VERIFIED (score=9.0, provider=gemini)",
  "result": {
    "full_name":  "Jane Doe",
    "company":    "Acme Corp",
    "job_title":  "Chief Financial Officer",
    "best_result": {
      "full_name":         "Jane Doe",
      "company":           "Acme Corp",
      "job_title":         "Chief Financial Officer",
      "confidence_score":  9.0,
      "verdict":           "VERIFIED",
      "evidence_summary":  "...",
      "sources_found":     ["..."]
    },
    "gemini": {
      "provider": "gemini",
      "result": {
        "confidence_score": 9.0,
        "verdict":          "VERIFIED",
        "evidence_summary": "...",
        "sources_found":    ["..."]
      },
      "error": null
    },
    "openai": {
      "provider": "openai",
      "result": {
        "confidence_score": 7.5,
        "verdict":          "LIKELY",
        "evidence_summary": "...",
        "sources_found":    ["..."]
      },
      "error": null
    }
  },
  "created_at": "2026-06-25T10:30:00Z",
  "updated_at": "2026-06-25T10:32:10Z"
}
```

If **one provider fails**, its slot looks like:

```json
"openai": {
  "provider": "openai",
  "result":   null,
  "error":    "OpenAIVoeService is not configured: OPENAI_API_KEY is missing."
}
```

---

### Failed response (any provider)

```json
{
  "task_id": "a1b2c3d4-...",
  "status":  "FAILURE",
  "message": "VOE verification error: 503 Service Unavailable",
  "result": {
    "full_name": "Jane Doe",
    "company":   "Acme Corp",
    "job_title": "Chief Financial Officer"
  }
}
```

---

### Failed tasks list — `GET /verify-voe/failed/`

```json
{
  "failed_tasks": [
    {
      "task_id":   "a1b2c3d4-...",
      "full_name": "Jane Doe",
      "company":   "Acme Corp",
      "message":   "VOE verification error: ...",
      "failed_at": "2026-06-25T10:31:00Z"
    }
  ]
}
```

---

## Confidence Score & Verdict Rubric

Both providers are calibrated against the **same rubric** so scores are directly comparable when using `provider="both"`.

| Score range | Verdict | Meaning |
|-------------|---------|---------|
| **9 – 10** | `VERIFIED` | **Direct confirmation found.** LinkedIn profile, official company bio, press release, or news article explicitly names the person in the stated role at this company. |
| **6 – 8** | `LIKELY` | **Indirect or partial evidence.** Person and company are connected but the exact title or current employment status is ambiguous. |
| **3 – 5** | `UNVERIFIED` | **No clear link found.** Company and person both exist publicly but no credible source ties them together. |
| **0 – 2** | `CONTRADICTED` | **Evidence contradicts the claim.** Person left the company, holds a different title, or their public profile lists a different employer. Also assigned when the person leaves no public footprint at all. |

> **Note on `0.0` scores:** A `confidence_score` of exactly `0.0` combined with `verdict=UNVERIFIED` and an `evidence_summary` mentioning "blocked" or "no results" means Step 1 returned empty — either a safety filter was triggered or the search returned nothing. This is **not** the same as `CONTRADICTED`; it means the system could not gather any evidence, not that the claim was disproved.

---

## Polling Flow

The endpoint is fully asynchronous. The expected client flow is:

```
POST /verify-voe/
  └→ 200 { task_id, status: "IN_PROGRESS" }

  loop:
    GET /verify-voe/{task_id}
      └→ 200 { status: "IN_PROGRESS" }   ← still running
      └→ 200 { status: "SUCCESS", result: {...} }   ← done
      └→ 200 { status: "FAILURE", message: "..." }  ← error
```

Typical durations:

| Provider | Expected duration |
|----------|------------------|
| `gemini` | 15 – 40 seconds |
| `openai` | 20 – 50 seconds |
| `both` | 20 – 50 seconds (both run concurrently, bounded by the slower one) |

---

## Error Handling

| Scenario | HTTP code | When it occurs |
|----------|-----------|----------------|
| Missing required field (`full_name`, `job_title`, `company`) | `422` | Pydantic validation at the route level — before any task is created |
| Invalid `provider` value | `422` | Pydantic validation |
| `GEMINI_API_KEY` not set (provider is `gemini` or `both`) | `400` | Checked synchronously before task creation |
| `OPENAI_API_KEY` not set (provider is `openai` or `both`) | `400` | Checked synchronously before task creation |
| Rate limit exceeded | `429` | >15 requests/60s from the same IP; includes `Retry-After` header |
| Task not found | `404` | `GET /verify-voe/{task_id}` with an unknown ID |
| LLM API error / timeout | Task `FAILURE` | Caught inside background task; poll to observe |
| Both providers fail (only for `provider="both"`) | Task `FAILURE` | Both pipelines raised exceptions; neither produced a result |

---

## Full Examples

### Example 1 — Single provider (Gemini), all fields

```bash
curl -X POST http://localhost:8000/verify-voe/ \
  -H "Content-Type: application/json" \
  -d '{
    "full_name":  "John Smith",
    "job_title":  "VP of Engineering",
    "company":    "TechCorp Inc",
    "city":       "Austin",
    "state":      "TX",
    "country":    "United States",
    "provider":   "gemini"
  }'
```

### Example 2 — OpenAI provider

```bash
curl -X POST http://localhost:8000/verify-voe/ \
  -H "Content-Type: application/json" \
  -d '{
    "full_name":  "Sarah Lee",
    "job_title":  "HR Director",
    "company":    "Global Logistics LLC",
    "provider":   "openai"
  }'
```

### Example 3 — Combined (cross-validated)

```bash
curl -X POST http://localhost:8000/verify-voe/ \
  -H "Content-Type: application/json" \
  -d '{
    "full_name":  "Robert Kim",
    "job_title":  "Chief Executive Officer",
    "company":    "Sunrise Capital Group",
    "country":    "United States",
    "provider":   "both"
  }'
```

### Example 4 — Poll for result

```bash
curl http://localhost:8000/verify-voe/a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

### Example 5 — List recent failures

```bash
curl "http://localhost:8000/verify-voe/failed/?limit=5"
```
