# Combined Search API Payload Guide

This guide outlines the request and response payloads for the automated Combined Search (`combined-search`) endpoint, which concurrently gathers and merges contact information from both OpenAI and Gemini pipelines.

---

## 1. Request Payload (`POST /combined-search/`)

Send this payload to start a background combined search task.

### Fields
* **`company_name`** (string, required): The target company name to research.
* **`country`** (string, optional): Country context to narrow down contacts.
* **`zip_code`** (string, optional): Postal code context.
* **`url`** (string, optional): Known website URL to direct extraction.
* **`max_limit`** (integer, optional): Maximum number of contacts to return per category (e.g., max 3 emails, max 3 phones).

### Example JSON Request
```json
{
  "company_name": "TRUCKSMART",
  "country": "United States",
  "max_limit": 5
}
```

### Immediate Response
Returns a tracking ID to poll for the completed combined search.
```json
{
  "task_id": "5e1347a1-76cc-4d1f-97d7-be3796eb1e4f",
  "status": "IN_PROGRESS"
}
```

---

## 2. Completed Result Payload (`GET /combined-search/{task_id}`)

Once the task status changes to `"SUCCESS"`, the aggregated contact details are returned inside the `result` object.

### Result Fields
* **`company_name`** (string): The requested company name.
* **`official_site`** (string): Verified official website URL.
* **`company_info`** (object): Contains arrays of distinct contacts (`phones`, `faxes`, `emails`, `addresses`). Each item is a tagged object containing:
  * **`value`** (string): The extracted contact info (e.g., email or phone number).
  * **`tag`** (string): Assigned departmental category (e.g., `"Human Resource"`, `"Payroll"`, `"Admin"`, `"Careers"`, `"Finance"`, `"Others"`).
  * **`context`** (string): Surrounding label or webpage section where it was found.
* **`openai_result`** / **`gemini_result`** (objects, optional): Raw individual results from each provider pipeline for complete visibility.

### Example JSON Completed Response
```json
{
  "task_id": "5e1347a1-76cc-4d1f-97d7-be3796eb1e4f",
  "status": "SUCCESS",
  "message": "Successfully extracted contact info via combined search",
  "result": {
    "company_name": "TRUCKSMART",
    "official_site": "http://www.trucksmart.com",
    "company_info": {
      "phones": [
        {
          "value": "(916) 773-7333",
          "tag": "Others",
          "context": "Main switchboard listed on footer"
        }
      ],
      "faxes": [
        {
          "value": "(916) 781-7725",
          "tag": "Others",
          "context": "Contact Us page"
        }
      ],
      "emails": [
        {
          "value": "john.brassfield@trucksmart.com",
          "tag": "Others",
          "context": "Staff directory"
        },
        {
          "value": "info@trucksmart.com",
          "tag": "Others",
          "context": "General inquiries"
        }
      ],
      "addresses": [
        {
          "value": "6455 Pacific St, Rocklin, CA 95677, United States",
          "tag": "Others",
          "context": "Corporate Headquarters"
        }
      ]
    },
    "openai_result": null,
    "gemini_result": null
  },
  "created_at": "2026-05-13T14:30:00.123Z",
  "updated_at": "2026-05-13T14:30:12.456Z"
}
```
