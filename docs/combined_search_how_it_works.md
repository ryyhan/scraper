# How Combined Search Works

The `/combined-search/` endpoint finds a company's contact information by asking **two different AI systems** — OpenAI and Google Gemini — the same question at the same time, then intelligently merging what each one found.

---

## The Big Picture

```
You submit a company name
         │
         ├──► OpenAI searches the web ──────────────────┐
         │                                              │
         └──► Gemini searches the web ─────────────────►│
                                                        │
                                              Results are merged
                                            (duplicates removed)
                                                        │
                                                        ▼
                                          One clean, combined result
```

Both searches happen **at the same time**, so the total wait is only as long as the slower of the two — not both added together.

---

## Step-by-Step

### 1. You Submit a Request

You send the company name (and optionally a country, zip code, or known website URL) to the endpoint. The system immediately returns a **task ID** and starts working in the background.

```
POST /combined-search/
→ { "task_id": "abc-123", "status": "IN_PROGRESS" }
```

### 2. Both AI Pipelines Run Simultaneously

Each AI (OpenAI and Gemini) independently goes through its own **two-step process**:

**Step A — Research**
The AI searches the live web — company websites, directories, LinkedIn, press releases — to gather raw contact information for the company.

**Step B — Extraction**
The raw findings are handed to a second AI call that reads through everything and pulls out structured contacts: phone numbers, fax numbers, email addresses, and physical addresses. Each contact is tagged with a department label (e.g. `"Human Resource"`, `"Payroll"`, `"Admin"`) and a note about where it was found.

### 3. Results Are Merged and Deduplicated

Once both pipelines finish, their results are combined into a single list. Duplicates are removed intelligently:

| Contact type | How duplicates are detected |
|---|---|
| Phone & Fax | Digit-only comparison — `(800) 555-1234` and `8005551234` are treated as the same number |
| Email | Case-insensitive — `HR@Company.com` and `hr@company.com` are the same |
| Address | Matched on street + city + zip — minor formatting differences are ignored |

When both providers found the same contact, **the first one seen is kept** (OpenAI takes priority, then Gemini). No information is lost — the raw results from each provider are also returned in the response if you want to inspect them individually.

### 4. Official Company Name Is Resolved

The AI often finds the legally registered name of a company, which may differ from what you searched. For example, searching `"Market Basket"` might resolve to `"Demoulas Market Basket Inc."`. OpenAI's resolved name is preferred; Gemini's is used as a fallback; and if neither found one, your original search term is returned.

### 5. A Limit Is Applied (If Requested)

If you specified a `max_limit` in your request (e.g. `max_limit: 5`), it is applied **after** deduplication — meaning the limit is enforced on the final merged pool, not on each provider's results separately. This ensures you always get the best available contacts up to your limit.

### 6. You Poll for the Result

You check the status by calling `GET /combined-search/{task_id}` until `status` is `SUCCESS`.

```
GET /combined-search/abc-123
→ { "status": "SUCCESS", "result": { ... } }
```

The result contains:
- **`official_company_name`** — the resolved legal name
- **`official_site`** — the company's website
- **`company_info`** — the deduplicated, tagged contact list (phones, faxes, emails, addresses)
- **`summary`** — a count breakdown showing how many contacts each provider found vs. the final combined count
- **`openai_result` / `gemini_result`** — the full raw output from each provider, for reference

---

## What Happens If One Provider Fails?

The system is resilient. If one AI provider encounters an error, the other's results are still returned as the combined output. The task only fails if **both** providers fail simultaneously.

---

## Why Use Both Providers?

Different AI systems search different parts of the web with different strategies. Running both gives:

- **Better coverage** — one provider may surface contacts the other missed
- **Higher confidence** — contacts found by both providers are more likely to be accurate
- **Built-in fallback** — if one provider has an outage, results still come through

---

## Quick Reference

| What | Detail |
|---|---|
| Endpoint | `POST /combined-search/` |
| Type | Asynchronous (background task, poll for result) |
| Providers | OpenAI + Google Gemini, running concurrently |
| Deduplication | Automatic — phones by digits, emails case-insensitive, addresses by street+city+zip |
| Partial failure | One provider failing still returns a result |
| Full failure | Both providers failing marks the task as `FAILURE` |
