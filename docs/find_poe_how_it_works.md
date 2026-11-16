# How Find POE Works

The `/find-poe/` endpoint finds where a person works by giving their name (and any helpful context) to **one or two AI systems**, which then search the live web and return a ranked list of likely employers.

---

## The Big Picture

```
You submit a person's name (+ optional job title, location)
         │
         ├──► OpenAI searches the web ──────────────────┐
         │         (if selected)                        │
         └──► Gemini searches the web ─────────────────►│
                   (if selected)                        │
                                              Each provider returns
                                              a ranked list of employers
                                                        │
                                                        ▼
                                          Best match surfaced at the top
```

When `provider=both`, both searches run **at the same time**, so the total wait is only as long as the slower of the two — not both added together.

---

## Step-by-Step

### 1. You Submit a Request

You send the person's name along with any context you have — job title, city, state, zip code, or country. The system immediately returns a **task ID** and starts working in the background.

```
POST /find-poe/
→ { "task_id": "abc-123", "status": "IN_PROGRESS", "provider": "gemini" }
```

> **Tip:** The more context you provide, the better. For a common name like "John Smith", supplying a city and state dramatically narrows the search to the right person.

### 2. The AI Pipeline Runs (Per Provider)

Each AI goes through its own **two-step process**:

**Step A — Research**
The AI searches the live web — LinkedIn, company websites, press releases, news articles, SEC filings, and professional directories — specifically looking for where this person works *right now*. It is instructed to:
- Distinguish current roles from past ones
- Document **all** possible employers found (not just the top hit)
- Flag contradictions when different sources disagree
- Note the exact URL or source for every piece of evidence

**Step B — Extraction**
The raw research notes are passed to a second AI call which reads through everything and produces a **structured, scored list** of employer candidates. Each candidate gets:

| Field | What it contains |
|---|---|
| `company_name` | The employer's official name as it appears in sources |
| `confidence_score` | A score from 0.0 to 10.0 (see rubric below) |
| `evidence_summary` | 2–3 sentences explaining what was found and why this score was given |
| `sources_found` | The URLs or publications that mentioned this person at this employer |

### 3. Candidates Are Ranked

The candidates list is sorted by `confidence_score` from highest to lowest. The top result is surfaced as `best_match` for quick access — but the full ranked list is always returned so you can inspect every lead the AI found.

### 4. You Poll for the Result

You check the status by calling `GET /find-poe/{task_id}` until `status` is `SUCCESS`.

```
GET /find-poe/abc-123
→ { "status": "SUCCESS", "result": { ... } }
```

---

## Confidence Score Rubric

Both providers are calibrated against the **same rubric**, so scores mean the same thing regardless of which AI produced them.

| Score | Label | Meaning |
|---|---|---|
| **9 – 10** | Confirmed | LinkedIn, an official company bio, or a news article **explicitly names** this person as a current employee at this employer |
| **6 – 8** | Likely | Multiple credible sources connect this person to the employer, but the exact current role or tenure is not entirely clear |
| **3 – 5** | Possible | A single non-authoritative mention, a past employment connection, or a name collision that can't be ruled out |
| **0 – 2** | Unlikely | Only coincidental or very tenuous mentions — this employer is almost certainly not correct |

---

## Provider Modes

### `gemini` *(default)*
Uses Google Gemini with live Google Search grounding. Gemini issues real-time search queries mid-generation and cites its sources directly in the output.

### `openai`
Uses the OpenAI Responses API with a `web_search` tool set to high-context mode, which pulls deeper page content that a standard search might miss.

### `both`
Runs both providers **concurrently**. Each returns its own independent ranked list. The winner is the provider whose **top candidate scored highest**. In the event of a tie, Gemini is preferred. The full output from both providers is included in the response so you can compare them side by side.

---

## What Happens If One Provider Fails?

The system is resilient. If one AI provider encounters an error during the `both` mode, the other's results are still returned as the final output. The task only fails if **both** providers fail simultaneously.

If Step A (research) returns no results at all — for example due to a safety filter or a transient API issue — Step B is skipped entirely and an empty candidates list is returned rather than wasting an additional AI call.

---

## Why Provide More Context?

The `full_name` field is the only required input, but the quality of results improves significantly with additional fields:

| Field | Why it helps |
|---|---|
| `job_title` | Narrows the search to a specific role — essential for common names |
| `city` / `state` | Rules out people with the same name in different locations |
| `zip_code` | Adds precision when city alone isn't enough |
| `country` | Scopes the search geographically (defaults to `"United States"`) |

---

## Quick Reference

| What | Detail |
|---|---|
| Endpoint | `POST /find-poe/` |
| Type | Asynchronous (background task, poll for result) |
| Providers | Gemini, OpenAI, or both — running concurrently when `both` is selected |
| Only required field | `full_name` |
| Output | Ranked list of employer candidates, each with a confidence score and evidence |
| Winner selection (`both`) | Provider whose top candidate has the highest confidence score |
| Partial failure | One provider failing still returns a result |
| Full failure | Both providers failing marks the task as `FAILURE` |
| Empty evidence guard | If research returns nothing, extraction is skipped and an empty list is returned |
