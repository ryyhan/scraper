import sys
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from sqlmodel import SQLModel
from loguru import logger

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from app.core.config import settings
from app.api.deps import engine
from app.api.routes import router as api_router

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    logger.info("Application startup: DB connected, tables checked.")
    yield
    logger.info("Application shutdown.")

app = FastAPI(title="Async Scraper API", lifespan=lifespan)
app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
