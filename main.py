import sys
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from sqlmodel import SQLModel
from loguru import logger

# Configure Loguru to write to a local file, rotating when it reaches 10MB
# and keeping the last 7 days of logs.
logger.add("scraper.log", rotation="100 MB", retention="30 days", level="INFO")

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from app.core.config import settings
from app.api.deps import engine
from app.api.routes import router as api_router
from app.services.scraper import BrowserPool

from sqlalchemy import text

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)
    # Lightweight auto-migration for existing databases
    with engine.connect() as conn:
        try:
            result = conn.execute(text("PRAGMA table_info(taskrecord)"))
            columns = [row[1] for row in result.fetchall()]
            if columns and "task_type" not in columns:
                logger.info("Auto-migrating taskrecord table: adding 'task_type' column...")
                conn.execute(text("ALTER TABLE taskrecord ADD COLUMN task_type VARCHAR DEFAULT 'general'"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_taskrecord_task_type ON taskrecord (task_type)"))
                conn.commit()
                logger.info("Auto-migration complete: 'task_type' column added.")
        except Exception as e:
            logger.warning(f"Auto-migration check encountered an issue: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    logger.info("Application startup: DB connected, tables checked.")
    try:
        await BrowserPool.get_instance().start()
    except Exception as e:
        logger.warning(
            f"Failed to pre-warm persistent Playwright browser pool on startup (will initialize on demand): {e}"
        )
    yield
    await BrowserPool.get_instance().close()
    logger.info("Application shutdown: browser pool closed.")

app = FastAPI(title="Async Scraper API", lifespan=lifespan)
app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
