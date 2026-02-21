import asyncio
import uuid
from loguru import logger
from datetime import datetime
from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from sqlmodel import Session

from app.models import SearchRequest, TaskRecord, ScrapeResult, WebhookPayload
from app.api.deps import get_session
from app.services import ScraperService, LLMService, WebhookService
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
            async with scraper:
                # 1. Search (DuckDuckGo now)
                search_results = await scraper.perform_duckduckgo_search(f"{request.poe_name}")
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
                combined_text = ""
                for link in links_to_visit[:3]:
                    text = await scraper.extract_page_text(link)
                    combined_text += f"\n--- Source: {link} ---\n{text}\n"

                if len(combined_text) > 15000:
                    combined_text = combined_text[:15000]

                # 5. Extract Contact Info
                contact_info = await llm_client.extract_contact_info(combined_text)
                
                # 6. Fallback Search for missing email
                if contact_info and not contact_info.Email:
                    logger.info(f"Task {task_id}: Primary extraction missed Email. Triggering targeted fallback search.")
                    fallback_query = f'"{request.poe_name}" contact email address'
                    snippets_text = await scraper.perform_duckduckgo_snippet_search(fallback_query)
                    if snippets_text:
                        contact_info = await llm_client.extract_fallback_email(snippets_text, contact_info)
                
                if contact_info:
                    final_result.poe_info = contact_info
                    status = "SUCCESS"
                    message = "Successfully extracted contact info"
                else:
                    message = "Failed to extract contact info via LLM"

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

@router.get("/google-search/{task_id}")
async def get_task_status(task_id: str, session: Session = Depends(get_session)):
    task = session.get(TaskRecord, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
        
    return {
        "task_id": task.task_id,
        "status": task.status,
        "result": task.result_data,
        "created_at": task.created_at,
        "updated_at": task.updated_at
    }

@router.post("/webhook-mock")
async def webhook_mock(payload: WebhookPayload):
    logger.info(f"RECEIVED WEBHOOK: {payload}")
    return {"received": True}
