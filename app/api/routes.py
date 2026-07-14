import asyncio
import time
import uuid
from typing import Literal
from loguru import logger
from datetime import datetime, timezone
from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends, UploadFile, File, Form
from sqlmodel import Session, select, desc, String

from app.models import SearchRequest, TaskRecord, ScrapeResult, WebhookPayload
from app.models import OpenAISearchRequest, OpenAISearchResult, OpenAICompanyInfo
from app.models import GeminiSearchRequest, GeminiSearchResult
from app.models import VoeRequest, VoeVerificationResult, VoeProviderResult, CombinedVoeResult
from app.models import CombinedSearchRequest, CombinedSearchResult
from app.models import PdfExtractionResult
from app.models import BgCheckFields, BgCheckParseResult
from app.api.deps import get_session
from app.api.rate_limit import task_rate_limit
from app.services import ScraperService, LLMService, WebhookService
from app.services import OpenAISearchService, GeminiSearchService
from app.services import VoeVerificationService
from app.services import OpenAIVoeService
from app.services import CombinedSearchService
from app.services import PdfExtractorService
from app.services import BgCheckParserService
from app.core.config import settings

router = APIRouter()
BROWSER_SEMAPHORE = asyncio.Semaphore(settings.MAX_CONCURRENT_BROWSERS)

async def process_scraping_task(task_id: str, request: SearchRequest, webhook_url: str):
    logger.info(f"Task {task_id}: Waiting for browser slot...")
    async with BROWSER_SEMAPHORE:
        logger.info(f"Task {task_id}: Acquired browser slot. Starting scrape.")
        
        scraper = ScraperService()
        llm_client = LLMService()
        
        final_result = ScrapeResult(
            poe_name=request.poe_name,
            official_site="Information not available",
            poe_info=None
        )
        status = "FAILURE"
        message = "Unknown error"

        try:
            combined_text = ""
            # Wrap core logic to enforce global timeout
            async def _run_scrape():
                nonlocal message, status, combined_text
                async with scraper:
                    # 1. Search (Dynamic Provider)
                    smart_query = f"{request.poe_name} official site contact"
                    if settings.SEARCH_PROVIDER.lower() == "serper":
                        search_results = await scraper.perform_serper_search(smart_query)
                    else:
                        search_results = await scraper.perform_duckduckgo_search(smart_query)
                        
                    if not search_results:
                        message = "No search results found"
                        raise ValueError(message)
                    
                    # 2. Verify Official Site
                    official_site = await llm_client.verify_official_site(search_results, request.poe_name)
                    if not official_site:
                        message = "Official site not found by LLM"
                        raise ValueError(message)
                    
                    final_result.official_site = official_site
                    
                    # 3. Harvest Links
                    links_to_visit = await scraper.harvest_contact_links(official_site)
                    logger.info(f"Task {task_id}: Found links to visit: {links_to_visit}")

                    # 4. Visit pages
                    for link in links_to_visit[:4]:
                        text = await scraper.extract_page_text(link)
                        combined_text += f"\n--- Source: {link} ---\n{text}\n"

                    if len(combined_text) > 60000:
                        combined_text = combined_text[:60000]

                    # 5. Extract Contact Info
                    contact_info = await llm_client.extract_contact_info(combined_text)
                    if contact_info:
                        final_result.poe_info = contact_info
                    
                    # 6. Fallback Search for missing email
                    if final_result.poe_info and not final_result.poe_info.Email:
                        logger.info(f"Task {task_id}: Primary extraction missed Email. Triggering targeted fallback search.")
                        fallback_query = f'"{request.poe_name}" (email OR "contact us at" OR "reach us at" OR "@")'
                        
                        if settings.SEARCH_PROVIDER.lower() == "serper":
                            snippets_text = await scraper.perform_serper_snippet_search(fallback_query)
                        else:
                            snippets_text = await scraper.perform_duckduckgo_snippet_search(fallback_query)
                            
                        if snippets_text:
                            final_result.poe_info = await llm_client.extract_fallback_email(snippets_text, final_result.poe_info)
                    
                    if final_result.poe_info:
                        status = "SUCCESS"
                        message = "Successfully extracted contact info"
                    else:
                        message = "Failed to extract contact info via LLM"

            timeout_seconds = request.timeout if request.timeout is not None else 120
            
            try:
                await asyncio.wait_for(_run_scrape(), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                logger.warning(f"Task {task_id}: Reached payload timeout of {timeout_seconds}s.")
                message = f"Task reached timeout limit ({timeout_seconds}s). Returning partial data."
                status = "SUCCESS"
                
                if final_result.poe_info is None and combined_text:
                    import re
                    from app.models import ContactInfo
                    from app.models.models import _EMAIL_REGEX  # reuse authoritative validator
                    _CF_PLACEHOLDER = "[email protected]"
                    raw_emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', combined_text)
                    invalid_domains = ['.png', '.jpg', '.jpeg', '.gif', '.css', '.js', 'sentry', 'example', 'domain.com', '.webp', 'wixpress']
                    valid_emails = [
                        e for e in set(raw_emails)
                        if e.lower() != _CF_PLACEHOLDER
                        and not any(bad in e.lower() for bad in invalid_domains)
                        and _EMAIL_REGEX.match(e)  # enforce full format validation
                    ]
                    if valid_emails:
                        logger.info(f"Task {task_id}: Extracted valid email from partial text post-timeout.")
                        final_result.poe_info = ContactInfo(
                            Phone=[],
                            Fax=[],
                            Email=valid_emails,
                            Address=[]
                        )
                
        except Exception as e:
            logger.error(f"Task {task_id}: Error during processing: {e}")
            message = f"[Scraper] {e}"
            status = "FAILURE"
        
        # --- Save Result to DB ---
        # Note: We create a new session here because this runs in background
        from app.api.deps import engine
        with Session(engine) as session:
            task = session.get(TaskRecord, task_id)
            if task:
                task.status = status
                task.message = message
                task.result_data = final_result.model_dump()
                task.updated_at = datetime.now(timezone.utc)
                session.add(task)
                session.commit()
            else:
                logger.error(
                    f"[task-persist] Task {task_id} not found in DB — result discarded."
                )
                
        # --- Send External Webhook ---
        if webhook_url:
            payload = WebhookPayload(
                status=status,
                message=message,
                result=final_result
            )
            await WebhookService.submit_result(webhook_url, payload.model_dump())

        logger.info(f"Task {task_id}: Completed with status {status}")

@router.post("/google-search/")
async def create_search_task(
    request: SearchRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
    _rl: None = Depends(task_rate_limit),
):
    task_id = str(uuid.uuid4())
    task_record = TaskRecord(task_id=task_id, status="IN_PROGRESS")
    session.add(task_record)
    session.commit()

    # Use config-defined webhook or mock
    webhook_url = settings.WEBHOOK_URL or "http://localhost:8000/webhook-mock"

    background_tasks.add_task(process_scraping_task, task_id, request, webhook_url)

    return {"task_id": task_id, "status": "IN_PROGRESS"}

@router.get("/google-search/failed/")
async def get_failed_tasks(limit: int = 10, session: Session = Depends(get_session)):
    statement = (
        select(TaskRecord)
        .where(TaskRecord.status == "FAILURE")
        .where(
            (TaskRecord.message.contains("Scraper")) | 
            (TaskRecord.message.contains("Timeout")) |
            (TaskRecord.message == "Unknown error")
        )
        .order_by(desc(TaskRecord.updated_at))
        .limit(limit)
    )
    tasks = session.exec(statement).all()
    
    results = []
    for t in tasks:
        poe_name = "Unknown"
        if t.result_data and isinstance(t.result_data, dict):
            poe_name = t.result_data.get("poe_name", "Unknown")
            
        results.append({
            "task_id": t.task_id,
            "poe_name": poe_name,
            "message": t.message,
            "failed_at": t.updated_at
        })
        
    return {"failed_tasks": results}

@router.get("/google-search/{task_id}")
async def get_task_status(task_id: str, session: Session = Depends(get_session)):
    task = session.get(TaskRecord, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
        
    return {
        "task_id": task.task_id,
        "status": task.status,
        "message": task.message,
        "result": task.result_data,
        "created_at": task.created_at,
        "updated_at": task.updated_at
    }

@router.post("/webhook-mock")
async def webhook_mock(payload: WebhookPayload):
    logger.info(f"RECEIVED WEBHOOK: {payload}")
    return {"received": True}


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI Search Endpoints
# ─────────────────────────────────────────────────────────────────────────────

async def _process_openai_search_task(task_id: str, request: OpenAISearchRequest) -> None:
    """
    Background coroutine that executes the synchronous OpenAI two-step pipeline
    inside a thread pool so the event loop is never blocked, then persists the
    result to the database.
    """
    logger.info(f"[openai-search] Task {task_id}: starting for company={request.company_name!r}")

    status = "FAILURE"
    message = "Unknown error"
    result_data: dict = {"company_name": request.company_name}

    try:
        service = OpenAISearchService()

        # Run the synchronous SDK calls in a thread so we don't block the loop
        result: OpenAISearchResult = await asyncio.to_thread(
            service.structured_llm_call,
            request,
            OpenAISearchResult,
        )

        status = "SUCCESS"
        message = "Successfully extracted contact info via OpenAI"
        result_data = result.model_dump()
        logger.info(f"[openai-search] Task {task_id}: completed successfully")

    except Exception as exc:
        message = f"OpenAI error: {exc}"
        logger.error(f"[openai-search] Task {task_id}: failed – {exc}")

    # Persist outcome to the shared TaskRecord table
    from app.api.deps import engine
    with Session(engine) as session:
        task = session.get(TaskRecord, task_id)
        if task:
            task.status = status
            task.message = message
            task.result_data = result_data
            task.updated_at = datetime.now(timezone.utc)
            session.add(task)
            session.commit()
        else:
            logger.error(
                f"[task-persist] Task {task_id} not found in DB — result discarded."
            )


@router.post("/openai-search/", summary="Search company contact info via OpenAI")
async def create_openai_search_task(
    request: OpenAISearchRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
    _rl: None = Depends(task_rate_limit),
):
    """
    Enqueue an asynchronous OpenAI-powered contact-info lookup for *company_name*.

    Returns a *task_id* that can be polled via **GET /openai-search/{task_id}**.
    """
    task_id = str(uuid.uuid4())
    task_record = TaskRecord(task_id=task_id, status="IN_PROGRESS")
    session.add(task_record)
    session.commit()

    background_tasks.add_task(
        _process_openai_search_task,
        task_id,
        request,
    )

    return {"task_id": task_id, "status": "IN_PROGRESS"}


@router.get("/openai-search/failed/", summary="List failed OpenAI search tasks")
async def get_failed_openai_tasks(
    limit: int = 10,
    session: Session = Depends(get_session),
):
    """
    Return the most recently failed OpenAI search tasks (most recent first).
    """
    statement = (
        select(TaskRecord)
        .where(TaskRecord.status == "FAILURE")
        .where(TaskRecord.message.contains("OpenAI"))  # type: ignore[union-attr]
        .order_by(desc(TaskRecord.updated_at))
        .limit(limit)
    )
    tasks = session.exec(statement).all()

    results = []
    for t in tasks:
        company_name = "Unknown"
        if t.result_data and isinstance(t.result_data, dict):
            company_name = t.result_data.get("company_name", "Unknown")

        results.append({
            "task_id": t.task_id,
            "company_name": company_name,
            "message": t.message,
            "failed_at": t.updated_at,
        })

    return {"failed_tasks": results}


@router.get("/openai-search/{task_id}", summary="Get OpenAI search task status")
async def get_openai_task_status(
    task_id: str,
    session: Session = Depends(get_session),
):
    """
    Poll the status and result of an OpenAI search task by *task_id*.
    """
    task = session.get(TaskRecord, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "task_id": task.task_id,
        "status": task.status,
        "message": task.message,
        "result": task.result_data,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Gemini Search Endpoints
# ─────────────────────────────────────────────────────────────────────────────

async def _process_gemini_search_task(task_id: str, request: GeminiSearchRequest) -> None:
    """
    Background coroutine that executes the synchronous Gemini two-step pipeline
    inside a thread pool so the event loop is never blocked, then persists the
    result to the database.

    The Gemini SDK is synchronous, so ``asyncio.to_thread`` is used to offload
    the blocking network I/O — identical to the pattern used for the OpenAI service.
    """
    logger.info(
        f"[gemini-search] Task {task_id}: starting for company={request.company_name!r}"
    )

    status = "FAILURE"
    message = "Unknown error"
    result_data: dict = {"company_name": request.company_name}

    try:
        service = GeminiSearchService()

        # Run the synchronous SDK calls in a thread so we don't block the event loop
        result: GeminiSearchResult = await asyncio.to_thread(
            service.structured_llm_call,
            request,
            GeminiSearchResult,
        )

        status = "SUCCESS"
        message = "Successfully extracted contact info via Gemini"
        result_data = result.model_dump()
        logger.info(f"[gemini-search] Task {task_id}: completed successfully")

    except Exception as exc:
        message = f"Gemini error: {exc}"
        logger.error(f"[gemini-search] Task {task_id}: failed – {exc}")

    # Persist outcome to the shared TaskRecord table
    from app.api.deps import engine
    with Session(engine) as session:
        task = session.get(TaskRecord, task_id)
        if task:
            task.status = status
            task.message = message
            task.result_data = result_data
            task.updated_at = datetime.now(timezone.utc)
            session.add(task)
            session.commit()
        else:
            logger.error(
                f"[task-persist] Task {task_id} not found in DB — result discarded."
            )


@router.post("/gemini-search/", summary="Search company contact info via Gemini")
async def create_gemini_search_task(
    request: GeminiSearchRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
    _rl: None = Depends(task_rate_limit),
):
    """
    Enqueue an asynchronous Gemini-powered contact-info lookup for *company_name*.

    Gemini performs live Google Search grounding (Step 1) and then distils the
    raw research into a validated structured JSON result (Step 2).

    Returns a *task_id* that can be polled via **GET /gemini-search/{task_id}**.
    """
    task_id = str(uuid.uuid4())
    task_record = TaskRecord(task_id=task_id, status="IN_PROGRESS")
    session.add(task_record)
    session.commit()

    background_tasks.add_task(
        _process_gemini_search_task,
        task_id,
        request,
    )

    return {"task_id": task_id, "status": "IN_PROGRESS"}


@router.get("/gemini-search/failed/", summary="List failed Gemini search tasks")
async def get_failed_gemini_tasks(
    limit: int = 10,
    session: Session = Depends(get_session),
):
    """
    Return the most recently failed Gemini search tasks (most recent first).
    """
    statement = (
        select(TaskRecord)
        .where(TaskRecord.status == "FAILURE")
        .where(
            TaskRecord.message.contains("Gemini")  # type: ignore[union-attr]
        )
        .order_by(desc(TaskRecord.updated_at))
        .limit(limit)
    )
    tasks = session.exec(statement).all()

    results = []
    for t in tasks:
        company_name = "Unknown"
        if t.result_data and isinstance(t.result_data, dict):
            company_name = t.result_data.get("company_name", "Unknown")

        results.append({
            "task_id": t.task_id,
            "company_name": company_name,
            "message": t.message,
            "failed_at": t.updated_at,
        })

    return {"failed_tasks": results}


@router.get("/gemini-search/{task_id}", summary="Get Gemini search task status")
async def get_gemini_task_status(
    task_id: str,
    session: Session = Depends(get_session),
):
    """
    Poll the status and result of a Gemini search task by *task_id*.
    """
    task = session.get(TaskRecord, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "task_id": task.task_id,
        "status": task.status,
        "message": task.message,
        "result": task.result_data,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
    }


# ─────────────────────────────────────────────────────────────────────────────
# VOE (Verification of Employment) Endpoints
# ─────────────────────────────────────────────────────────────────────────────

async def _process_voe_task(task_id: str, request: VoeRequest) -> None:
    """
    Background coroutine that routes to the correct VOE provider pipeline(s)
    based on ``request.provider`` then persists the result to the DB.

    - ``gemini`` — single Gemini two-step pipeline (original behaviour).
    - ``openai`` — single OpenAI two-step pipeline (web_search grounding).
    - ``both``   — both pipelines run concurrently via asyncio.gather;
                   the higher-confidence result is surfaced as ``best_result``.
    """
    provider = request.provider
    logger.info(
        f"[verify-voe] Task {task_id}: starting (provider={provider!r}) for "
        f"{request.full_name!r} @ {request.company!r}"
    )

    status = "FAILURE"
    message = "Unknown error"
    result_data: dict = {
        "full_name": request.full_name,
        "company": request.company,
        "job_title": request.job_title,
    }

    try:
        if provider == "gemini":
            # ── Single-provider: Gemini ────────────────────────────────────
            service = VoeVerificationService()
            result: VoeVerificationResult = await asyncio.to_thread(
                service.verify, request
            )
            status = "SUCCESS"
            message = (
                f"Verification complete (gemini) – score={result.confidence_score} "
                f"verdict={result.verdict}"
            )
            result_data = result.model_dump()

        elif provider == "openai":
            # ── Single-provider: OpenAI ────────────────────────────────────
            service_oai = OpenAIVoeService()
            result_oai: VoeVerificationResult = await asyncio.to_thread(
                service_oai.verify, request
            )
            status = "SUCCESS"
            message = (
                f"Verification complete (openai) – score={result_oai.confidence_score} "
                f"verdict={result_oai.verdict}"
            )
            result_data = result_oai.model_dump()

        else:  # provider == "both"
            # ── Concurrent dual-provider ───────────────────────────────────
            gemini_svc = VoeVerificationService()
            openai_svc = OpenAIVoeService()

            gemini_task = asyncio.to_thread(gemini_svc.verify, request)
            openai_task = asyncio.to_thread(openai_svc.verify, request)

            raw_results = await asyncio.gather(
                gemini_task, openai_task, return_exceptions=True
            )

            gemini_res: VoeVerificationResult | None = None
            gemini_err: str | None = None
            openai_res: VoeVerificationResult | None = None
            openai_err: str | None = None

            if isinstance(raw_results[0], Exception):
                gemini_err = str(raw_results[0])
                logger.error(f"[verify-voe] Task {task_id}: Gemini failed – {gemini_err}")
            else:
                gemini_res = raw_results[0]

            if isinstance(raw_results[1], Exception):
                openai_err = str(raw_results[1])
                logger.error(f"[verify-voe] Task {task_id}: OpenAI failed – {openai_err}")
            else:
                openai_res = raw_results[1]

            if not gemini_res and not openai_res:
                raise RuntimeError(
                    "Both Gemini and OpenAI VOE pipelines failed. "
                    f"Gemini: {gemini_err}. OpenAI: {openai_err}."
                )

            # Pick the result with the higher confidence score as best_result
            candidates = [r for r in (gemini_res, openai_res) if r is not None]
            best = max(candidates, key=lambda r: r.confidence_score)

            # Aggregate token usage from both providers
            from app.models import ProviderTokenUsage, TokenUsage
            openai_prov_usage: ProviderTokenUsage | None = (
                openai_res.token_usage.openai
                if openai_res and openai_res.token_usage and openai_res.token_usage.openai
                else None
            )
            gemini_prov_usage: ProviderTokenUsage | None = (
                gemini_res.token_usage.gemini
                if gemini_res and gemini_res.token_usage and gemini_res.token_usage.gemini
                else None
            )
            voe_grand_total = (
                (openai_prov_usage or ProviderTokenUsage())
                + (gemini_prov_usage or ProviderTokenUsage())
            )

            combined = CombinedVoeResult(
                full_name=request.full_name,
                company=request.company,
                job_title=request.job_title,
                best_result=best,
                gemini=VoeProviderResult(
                    provider="gemini",
                    result=gemini_res,
                    error=gemini_err,
                ),
                openai=VoeProviderResult(
                    provider="openai",
                    result=openai_res,
                    error=openai_err,
                ),
                token_usage=TokenUsage(
                    openai=openai_prov_usage,
                    gemini=gemini_prov_usage,
                    grand_total=voe_grand_total,
                ),
            )

            status = "SUCCESS"
            message = (
                f"Combined verification complete – "
                f"best={best.verdict} (score={best.confidence_score}, "
                f"provider={'gemini' if best is gemini_res else 'openai'})"
            )
            result_data = combined.model_dump()

        logger.info(f"[verify-voe] Task {task_id}: completed successfully ({provider})")

    except Exception as exc:
        message = f"VOE verification error: {exc}"
        logger.error(f"[verify-voe] Task {task_id}: failed – {exc}")

    # Persist outcome to the shared TaskRecord table
    from app.api.deps import engine
    with Session(engine) as session:
        task = session.get(TaskRecord, task_id)
        if task:
            task.status = status
            task.message = message
            task.result_data = result_data
            task.updated_at = datetime.now(timezone.utc)
            session.add(task)
            session.commit()
        else:
            logger.error(
                f"[task-persist] Task {task_id} not found in DB — result discarded."
            )


@router.post("/verify-voe/", summary="Verify employment of a person via Gemini, OpenAI, or both")
async def create_voe_task(
    request: VoeRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
    _rl: None = Depends(task_rate_limit),
):
    """
    Enqueue an asynchronous employment-verification job for the given person.

    The LLM performs live web research to gather evidence across LinkedIn,
    company directories, press releases, and news articles, then scores its
    confidence (0–10) against a calibrated rubric.

    Returns a *task_id* that can be polled via **GET /verify-voe/{task_id}**.

    **Required fields:** `full_name`, `job_title`, `company`  
    **Optional fields:** `zip_code`, `city`, `country` (improve disambiguation)

    **Provider options** (`provider` field):
    - `gemini` *(default)* — Gemini with live Google Search grounding.
    - `openai` — OpenAI Responses API with web_search tool.
    - `both` — Runs both concurrently. Returns each provider's result plus a
      `best_result` chosen by the highest `confidence_score`.

    When `provider=both`, both `GEMINI_API_KEY` and `OPENAI_API_KEY` must be set.
    """
    # ── Fail fast if required API key(s) are missing ────────────────────────────
    # Mirrors the pattern used by /extract-pdf/ and /parse-background-check/.
    # Returning HTTP 400 here is far more useful than returning 200 IN_PROGRESS
    # only for the background task to fail immediately with a cryptic RuntimeError.
    if request.provider in ("gemini", "both") and not settings.GEMINI_API_KEY:
        raise HTTPException(
            status_code=400,
            detail=(
                "GEMINI_API_KEY is not configured. "
                "Set it in your .env file and restart the server."
            ),
        )
    if request.provider in ("openai", "both") and not settings.OPENAI_API_KEY:
        raise HTTPException(
            status_code=400,
            detail=(
                "OPENAI_API_KEY is not configured. "
                "Set it in your .env file and restart the server."
            ),
        )

    task_id = str(uuid.uuid4())
    task_record = TaskRecord(task_id=task_id, status="IN_PROGRESS")
    session.add(task_record)
    session.commit()

    background_tasks.add_task(_process_voe_task, task_id, request)

    return {"task_id": task_id, "status": "IN_PROGRESS", "provider": request.provider}


@router.get("/verify-voe/failed/", summary="List failed VOE verification tasks")
async def get_failed_voe_tasks(
    limit: int = 10,
    session: Session = Depends(get_session),
):
    """
    Return the most recently failed VOE verification tasks (most recent first).
    """
    statement = (
        select(TaskRecord)
        .where(TaskRecord.status == "FAILURE")
        .where(
            # "VOE verification error: ..." — normal failure path through except block.
            # "Unknown error"              — task crashed before reaching except block
            #                               (e.g. import error, OOM, task cancellation).
            # Both must be covered so no failed VOE task is invisible in this endpoint.
            TaskRecord.message.contains("verification")  # type: ignore[union-attr]
            | (TaskRecord.message == "Unknown error")     # type: ignore[union-attr]
        )
        .order_by(desc(TaskRecord.updated_at))
        .limit(limit)
    )
    tasks = session.exec(statement).all()

    results = []
    for t in tasks:
        full_name = "Unknown"
        company = "Unknown"
        if t.result_data and isinstance(t.result_data, dict):
            full_name = t.result_data.get("full_name", "Unknown")
            company = t.result_data.get("company", "Unknown")

        results.append({
            "task_id": t.task_id,
            "full_name": full_name,
            "company": company,
            "message": t.message,
            "failed_at": t.updated_at,
        })

    return {"failed_tasks": results}


@router.get("/verify-voe/{task_id}", summary="Get VOE verification task status")
async def get_voe_task_status(
    task_id: str,
    session: Session = Depends(get_session),
):
    """
    Poll the status and result of a VOE verification task by *task_id*.

    Once `status` is `SUCCESS`, the `result` object contains:
    - `confidence_score` (0.0–10.0)
    - `verdict` (VERIFIED / LIKELY / UNVERIFIED / CONTRADICTED)
    - `evidence_summary` (human-readable explanation)
    - `sources_found` (list of URLs/publications used as evidence)
    """
    task = session.get(TaskRecord, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "task_id": task.task_id,
        "status": task.status,
        "message": task.message,
        "result": task.result_data,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Combined Search Endpoints
# ─────────────────────────────────────────────────────────────────────────────

async def _process_combined_search_task(task_id: str, request: CombinedSearchRequest) -> None:
    """
    Background coroutine that executes the combined OpenAI and Gemini search pipeline
    then persists the result to the shared TaskRecord table.
    """
    logger.info(
        f"[combined-search] Task {task_id}: starting for company={request.company_name!r}"
    )

    status = "FAILURE"
    message = "Unknown error"
    result_data: dict = {"company_name": request.company_name}

    try:
        service = CombinedSearchService()
        result: CombinedSearchResult = await service.search(request)

        status = "SUCCESS"
        message = "Successfully extracted contact info via combined search"
        result_data = result.model_dump()
        logger.info(f"[combined-search] Task {task_id}: completed successfully")

    except Exception as exc:
        message = f"Combined search error: {exc}"
        logger.error(f"[combined-search] Task {task_id}: failed – {exc}")

    # Persist outcome to the shared TaskRecord table
    from app.api.deps import engine
    with Session(engine) as session:
        task = session.get(TaskRecord, task_id)
        if task:
            task.status = status
            task.message = message
            task.result_data = result_data
            task.updated_at = datetime.now(timezone.utc)
            session.add(task)
            session.commit()
        else:
            logger.error(
                f"[task-persist] Task {task_id} not found in DB — result discarded."
            )


@router.post("/combined-search/", summary="Search company contact info via combined OpenAI and Gemini")
async def create_combined_search_task(
    request: CombinedSearchRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
    _rl: None = Depends(task_rate_limit),
):
    """
    Enqueue an asynchronous combined search job for the given company.

    This calls both the OpenAI and Gemini pipelines concurrently and aggregates their results.

    Returns a *task_id* that can be polled via **GET /combined-search/{task_id}**.
    """
    task_id = str(uuid.uuid4())
    task_record = TaskRecord(task_id=task_id, status="IN_PROGRESS")
    session.add(task_record)
    session.commit()

    background_tasks.add_task(
        _process_combined_search_task,
        task_id,
        request,
    )

    return {"task_id": task_id, "status": "IN_PROGRESS"}


@router.get("/combined-search/failed/", summary="List failed Combined search tasks")
async def get_failed_combined_tasks(
    limit: int = 10,
    session: Session = Depends(get_session),
):
    """
    Return the most recently failed combined search tasks (most recent first).
    """
    statement = (
        select(TaskRecord)
        .where(TaskRecord.status == "FAILURE")
        .where(
            TaskRecord.message.contains("combined")  # type: ignore[union-attr]
        )
        .order_by(desc(TaskRecord.updated_at))
        .limit(limit)
    )
    tasks = session.exec(statement).all()

    results = []
    for t in tasks:
        company_name = "Unknown"
        if t.result_data and isinstance(t.result_data, dict):
            company_name = t.result_data.get("company_name", "Unknown")

        results.append({
            "task_id": t.task_id,
            "company_name": company_name,
            "message": t.message,
            "failed_at": t.updated_at,
        })

    return {"failed_tasks": results}


@router.get("/combined-search/{task_id}", summary="Get Combined search task status")
async def get_combined_task_status(
    task_id: str,
    session: Session = Depends(get_session),
):
    """
    Poll the status and result of a combined search task by *task_id*.
    """
    task = session.get(TaskRecord, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "task_id": task.task_id,
        "status": task.status,
        "message": task.message,
        "result": task.result_data,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PDF Text Extraction Endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/extract-pdf/",
    response_model=PdfExtractionResult,
    summary="Extract text from a PDF using a vision LLM",
    tags=["PDF Extraction"],
)
async def extract_pdf(
    file: UploadFile = File(
        ...,
        description="The PDF file to extract text from (max 20 MB).",
    ),
    provider: Literal["gemini", "openai"] = Form(
        default="gemini",
        description="Vision LLM provider: 'gemini' (default) or 'openai'.",
    ),
) -> PdfExtractionResult:
    """
    Extract full text from every page of an uploaded PDF file using a vision LLM.

    **Gemini provider** (default)
    - Uploads the raw PDF to the Gemini Files API and processes the entire
      document natively — no page splitting required.
    - Best for mixed text/image PDFs and scanned documents.
    - Requires ``GEMINI_API_KEY`` in your environment.

    **OpenAI provider**
    - Renders each page to a JPEG image with ``pypdfium2`` and sends all
      images to ``gpt-4o-mini`` vision in a single call.
    - Requires ``OPENAI_API_KEY`` in your environment and
      ``pypdfium2`` installed (``pip install pypdfium2``).

    **Constraints**
    - File must be a valid PDF (``.pdf`` extension + ``%PDF`` magic bytes).
    - Maximum file size: ``PDF_MAX_FILE_SIZE_MB`` MB (default 20 MB;
      override via environment variable).

    **Response** is returned synchronously — no task polling required.
    """
    # ── 1. Read bytes eagerly so we can validate before any LLM call ───────
    pdf_bytes = await file.read()
    filename = file.filename or "upload.pdf"

    # ── 2. Validate: extension, file size, PDF magic bytes ─────────────────
    try:
        PdfExtractorService.validate(pdf_bytes, filename)
    except ValueError as exc:
        status_code = 413 if "too large" in str(exc).lower() else 400
        raise HTTPException(status_code=status_code, detail=str(exc))

    # ── 3. Fail fast if the requested provider key is missing ──────────────
    if provider == "gemini" and not settings.GEMINI_API_KEY:
        raise HTTPException(
            status_code=400,
            detail=(
                "GEMINI_API_KEY is not configured. "
                "Set it in your .env file and restart the server."
            ),
        )
    if provider == "openai" and not settings.OPENAI_API_KEY:
        raise HTTPException(
            status_code=400,
            detail=(
                "OPENAI_API_KEY is not configured. "
                "Set it in your .env file and restart the server."
            ),
        )

    # ── 4. Run extraction in a thread (SDKs are synchronous) ──────────────
    logger.info(
        f"[extract-pdf] Starting: file={filename!r}, "
        f"size={len(pdf_bytes) / 1024:.1f} KB, provider={provider!r}"
    )

    service = PdfExtractorService()
    t0 = time.perf_counter()

    try:
        extracted_text, page_count = await asyncio.to_thread(
            service.extract,
            pdf_bytes,
            filename,
            provider,
        )
    except RuntimeError as exc:
        # Raised by the service when a key is missing or pypdfium2 is absent
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"[extract-pdf] Unhandled error for file={filename!r}: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Text extraction failed: {exc}",
        )

    elapsed = round(time.perf_counter() - t0, 3)
    logger.info(
        f"[extract-pdf] Done: file={filename!r}, pages={page_count}, "
        f"chars={len(extracted_text)}, elapsed={elapsed}s"
    )

    return PdfExtractionResult(
        filename=filename,
        provider=provider,
        page_count=page_count,
        extracted_text=extracted_text,
        processing_time_seconds=elapsed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Background Check PDF Parser Endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/parse-background-check/",
    response_model=BgCheckParseResult,
    summary="Parse a background check PDF and extract structured fields",
    tags=["PDF Extraction"],
)
async def parse_background_check(
    file: UploadFile = File(
        ...,
        description="Background check / screening report PDF (max 20 MB).",
    ),
    provider: Literal["gemini", "openai"] = Form(
        default="gemini",
        description="Vision LLM provider: 'gemini' (default) or 'openai'.",
    ),
) -> BgCheckParseResult:
    """
    Extract structured fields from a background check / screening report PDF.

    Supports reports from **HireRight**, **Sterling**, **Checkr**,
    **Accurate**, **First Advantage**, and similar vendors.

    **Two-stage pipeline:**
    1. **Raw extraction** — the full PDF is processed by the selected vision
       LLM (Gemini Files API or OpenAI gpt-4o-mini) to obtain complete text.
    2. **Field extraction** — a focused, structured LLM call identifies the
       6 target fields from the raw text using vendor-agnostic label synonyms.

    **Fields returned** (all fields default to `""` if not found):
    - `file_number` — Case / order / reference ID
    - `employee_name` — Full name of the subject / applicant
    - `date_of_birth` — Subject's DOB (normalised to YYYY-MM-DD)
    - `requested_by` — Requester name or organisation
    - `employer_name` — Employer / client company
    - `report_date` — Report date (normalised to YYYY-MM-DD)
    - `status` — Overall status (e.g. Clear, Consider, Full Time Active, No Longer Employed)

    **Providers:**
    - `gemini` (default) — uses Gemini Files API + native JSON schema output.
      Requires `GEMINI_API_KEY`.
    - `openai` — renders pages via `pypdfium2` + `gpt-4o-mini` vision.
      Requires `OPENAI_API_KEY`.

    **Constraints:**
    - File must be a valid PDF (`.pdf` extension + `%PDF` magic bytes).
    - Maximum file size: `PDF_MAX_FILE_SIZE_MB` MB (default 20 MB).
    """
    # ── 1. Read bytes eagerly ────────────────────────────────────────────────────
    pdf_bytes = await file.read()
    filename = file.filename or "upload.pdf"

    # ── 2. Validate: extension, size, magic bytes ────────────────────────────
    try:
        PdfExtractorService.validate(pdf_bytes, filename)
    except ValueError as exc:
        status_code = 413 if "too large" in str(exc).lower() else 400
        raise HTTPException(status_code=status_code, detail=str(exc))

    # ── 3. Fail fast if the requested provider key is missing ──────────────
    if provider == "gemini" and not settings.GEMINI_API_KEY:
        raise HTTPException(
            status_code=400,
            detail=(
                "GEMINI_API_KEY is not configured. "
                "Set it in your .env file and restart the server."
            ),
        )
    if provider == "openai" and not settings.OPENAI_API_KEY:
        raise HTTPException(
            status_code=400,
            detail=(
                "OPENAI_API_KEY is not configured. "
                "Set it in your .env file and restart the server."
            ),
        )

    # ── 4. Run two-stage pipeline in a thread (both SDKs are synchronous) ───
    logger.info(
        f"[parse-background-check] Starting: file={filename!r}, "
        f"size={len(pdf_bytes) / 1024:.1f} KB, provider={provider!r}"
    )

    service = BgCheckParserService()
    t0 = time.perf_counter()

    try:
        fields_dict: dict = await asyncio.to_thread(
            service.parse,
            pdf_bytes,
            filename,
            provider,
        )
    except RuntimeError as exc:
        # Missing API key or pypdfium2 not installed
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(
            f"[parse-background-check] Unhandled error for file={filename!r}: {exc}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Background check parsing failed: {exc}",
        )

    elapsed = round(time.perf_counter() - t0, 3)
    logger.info(
        f"[parse-background-check] Done: file={filename!r}, elapsed={elapsed}s, "
        f"employee={fields_dict.get('employee_name', '')!r}, "
        f"status={fields_dict.get('status', '')!r}"
    )

    return BgCheckParseResult(
        filename=filename,
        provider=provider,
        processing_time_seconds=elapsed,
        data=BgCheckFields(**fields_dict),
    )
