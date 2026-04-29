import asyncio
import uuid
from loguru import logger
from datetime import datetime
from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from sqlmodel import Session, select, desc, String

from app.models import SearchRequest, TaskRecord, ScrapeResult, WebhookPayload
from app.models import OpenAISearchRequest, OpenAISearchResult, OpenAICompanyInfo
from app.models import GeminiSearchRequest, GeminiSearchResult
from app.models import VoeRequest, VoeVerificationResult
from app.models import CombinedSearchRequest, CombinedSearchResult
from app.api.deps import get_session
from app.services import ScraperService, LLMService, WebhookService
from app.services import OpenAISearchService, GeminiSearchService
from app.services import VoeVerificationService
from app.services import CombinedSearchService
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

                    if len(combined_text) > 15000:
                        combined_text = combined_text[:15000]

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
                    emails = set(re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', combined_text))
                    invalid_domains = ['.png', '.jpg', '.jpeg', '.gif', '.css', '.js', 'sentry', 'example', 'domain.com', '.webp', 'wixpress']
                    valid_emails = [e for e in emails if not any(bad in e.lower() for bad in invalid_domains)]
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
            message = str(e)
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
                task.updated_at = datetime.utcnow()
                session.add(task)
                session.commit()
                
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
async def create_search_task(request: SearchRequest, background_tasks: BackgroundTasks, session: Session = Depends(get_session)):
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
    statement = select(TaskRecord).where(TaskRecord.status == "FAILURE").order_by(desc(TaskRecord.updated_at)).limit(limit)
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
        message = str(exc)
        logger.error(f"[openai-search] Task {task_id}: failed – {exc}")

    # Persist outcome to the shared TaskRecord table
    from app.api.deps import engine
    with Session(engine) as session:
        task = session.get(TaskRecord, task_id)
        if task:
            task.status = status
            task.message = message
            task.result_data = result_data
            task.updated_at = datetime.utcnow()
            session.add(task)
            session.commit()


@router.post("/openai-search/", summary="Search company contact info via OpenAI")
async def create_openai_search_task(
    request: OpenAISearchRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
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
        .where(TaskRecord.message.contains("OpenAI") | TaskRecord.result_data.cast(String).contains("company_name"))  # type: ignore[union-attr]
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
        message = str(exc)
        logger.error(f"[gemini-search] Task {task_id}: failed – {exc}")

    # Persist outcome to the shared TaskRecord table
    from app.api.deps import engine
    with Session(engine) as session:
        task = session.get(TaskRecord, task_id)
        if task:
            task.status = status
            task.message = message
            task.result_data = result_data
            task.updated_at = datetime.utcnow()
            session.add(task)
            session.commit()


@router.post("/gemini-search/", summary="Search company contact info via Gemini")
async def create_gemini_search_task(
    request: GeminiSearchRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
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
            | TaskRecord.result_data.cast(String).contains("company_name")  # type: ignore[union-attr]
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
    Background coroutine that runs the two-step VOE verification pipeline
    inside a thread pool (Gemini SDK is synchronous) then persists the
    result to the shared TaskRecord table.
    """
    logger.info(
        f"[verify-voe] Task {task_id}: starting for "
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
        service = VoeVerificationService()

        # Offload synchronous Gemini SDK calls to a thread pool
        result: VoeVerificationResult = await asyncio.to_thread(
            service.verify,
            request,
        )

        status = "SUCCESS"
        message = (
            f"Verification complete – score={result.confidence_score} "
            f"verdict={result.verdict}"
        )
        result_data = result.model_dump()
        logger.info(f"[verify-voe] Task {task_id}: completed successfully")

    except Exception as exc:
        message = str(exc)
        logger.error(f"[verify-voe] Task {task_id}: failed – {exc}")

    # Persist outcome to the shared TaskRecord table
    from app.api.deps import engine
    with Session(engine) as session:
        task = session.get(TaskRecord, task_id)
        if task:
            task.status = status
            task.message = message
            task.result_data = result_data
            task.updated_at = datetime.utcnow()
            session.add(task)
            session.commit()


@router.post("/verify-voe/", summary="Verify employment of a person via Gemini web research")
async def create_voe_task(
    request: VoeRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
):
    """
    Enqueue an asynchronous employment-verification job for the given person.

    Gemini performs live Google Search grounding to gather evidence across
    LinkedIn, company directories, press releases, and news articles, then
    scores its confidence (0–10) against a calibrated rubric.

    Returns a *task_id* that can be polled via **GET /verify-voe/{task_id}**.

    **Required fields:** `full_name`, `job_title`, `company`  
    **Optional fields:** `zip_code`, `city`, `country` (improve disambiguation)
    """
    task_id = str(uuid.uuid4())
    task_record = TaskRecord(task_id=task_id, status="IN_PROGRESS")
    session.add(task_record)
    session.commit()

    background_tasks.add_task(_process_voe_task, task_id, request)

    return {"task_id": task_id, "status": "IN_PROGRESS"}


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
            TaskRecord.result_data.cast(String).contains("full_name")  # type: ignore[union-attr]
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
        message = str(exc)
        logger.error(f"[combined-search] Task {task_id}: failed – {exc}")

    # Persist outcome to the shared TaskRecord table
    from app.api.deps import engine
    with Session(engine) as session:
        task = session.get(TaskRecord, task_id)
        if task:
            task.status = status
            task.message = message
            task.result_data = result_data
            task.updated_at = datetime.utcnow()
            session.add(task)
            session.commit()


@router.post("/combined-search/", summary="Search company contact info via combined OpenAI and Gemini")
async def create_combined_search_task(
    request: CombinedSearchRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
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
            | TaskRecord.result_data.cast(String).contains("company_name")  # type: ignore[union-attr]
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

