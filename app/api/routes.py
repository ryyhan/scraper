import asyncio
import uuid
from loguru import logger
from datetime import datetime
from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from sqlmodel import Session, select, desc

from app.models import SearchRequest, TaskRecord, ScrapeResult, WebhookPayload
from app.models import OpenAISearchRequest, OpenAISearchResult, OpenAICompanyInfo
from app.api.deps import get_session
from app.services import ScraperService, LLMService, WebhookService
from app.services import OpenAISearchService
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
                            Phone="",
                            Fax="",
                            Email=valid_emails[0],
                            Address="",
                            City="",
                            State="",
                            ZipCode=""
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

async def _process_openai_search_task(task_id: str, company_name: str) -> None:
    """
    Background coroutine that executes the synchronous OpenAI two-step pipeline
    inside a thread pool so the event loop is never blocked, then persists the
    result to the database.
    """
    logger.info(f"[openai-search] Task {task_id}: starting for company={company_name!r}")

    status = "FAILURE"
    message = "Unknown error"
    result_data: dict = {"company_name": company_name}

    try:
        service = OpenAISearchService()

        # Run the synchronous SDK calls in a thread so we don't block the loop
        result: OpenAISearchResult = await asyncio.to_thread(
            service.structured_llm_call,
            company_name,
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
        request.company_name,
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
        .where(TaskRecord.message.contains("OpenAI") | TaskRecord.result_data.cast(str).contains("company_name"))  # type: ignore[union-attr]
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
