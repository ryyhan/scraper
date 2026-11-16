# Scraper Service Technical Documentation Index

This directory contains in-depth architectural guides, API payload references, benchmark comparisons, and project planning documents.

For general setup, deployment, and overall API reference, see the main [README.md](../README.md).

---

## Documentation Directory

### 1. API References & Payload Guides
- **[voe_api_reference.md](./voe_api_reference.md)**  
  Comprehensive payload and integration guide for the `/verify-voe/` endpoints. Details request parameters, single vs. dual provider execution modes, response envelopes, confidence score rubrics, and error codes.
- **[combined_search_api_reference.md](./combined_search_api_reference.md)**  
  Integration guide for the `/combined-search/` endpoint. Covers asynchronous task submission, polling, deduplication rules, and contact tagging.

### 2. Deep Dive Architecture & Pipeline Logics
- **[poe_voe_logic.md](./poe_voe_logic.md)**  
  Detailed conceptual breakdown of the Place of Employment (POE) crawler/search pipelines and the Verification of Employment (VOE) decision engine.
- **[find_poe_how_it_works.md](./find_poe_how_it_works.md)**  
  Step-by-step explanation of how the `/find-poe/` pipeline investigates public professional footprints, ranks employer candidates, and resolves the best match.
- **[combined_search_how_it_works.md](./combined_search_how_it_works.md)**  
  Explanation of the concurrent multi-LLM gatherer architecture, tracking parameter stripping, and deduplication logic.
- **[openai_poe_comparison.md](./openai_poe_comparison.md)**  
  Comparative analysis and performance characteristics of the OpenAI search pipeline vs. standard scraping methods.

### 3. Expansion & Project Plans
- **[phase_2_expansion_plan.md](./phase_2_expansion_plan.md)**  
  Roadmap for Phase 2 feature expansion, including multi-vendor background check extraction, document intelligence, and enterprise scaling.
- **[phase_2_project_plan.csv](./phase_2_project_plan.csv)** / **[phase_2_project_plan_detailed.csv](./phase_2_project_plan_detailed.csv)**  
  Task tracking and sprint deliverables for Phase 2 milestones.
