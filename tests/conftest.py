"""
Shared pytest fixtures and test configuration.
"""

import os
import pytest
import pytest_asyncio
import httpx
from sqlmodel import SQLModel, Session, create_engine
from sqlalchemy.pool import StaticPool

# Set dummy API keys for testing if not already present in environment
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
os.environ.setdefault("DATABASE_URL", "sqlite://")

from main import app
from app.api.deps import get_session
from app.api.rate_limit import reset_rate_limit_store


# Create an isolated in-memory SQLite engine for tests
test_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)


import app.api.deps as api_deps
api_deps.engine = test_engine


@pytest.fixture(autouse=True)
def setup_test_db():
    """Create all tables before each test and drop after."""
    SQLModel.metadata.create_all(test_engine)
    yield
    SQLModel.metadata.drop_all(test_engine)


@pytest.fixture
def db_session():
    """Provide a direct DB session connected to test_engine."""
    with Session(test_engine) as session:
        yield session


@pytest_asyncio.fixture(autouse=True)
async def reset_rate_limiter():
    """Ensure in-memory rate limiter store is clean before each test."""
    await reset_rate_limit_store()
    yield
    await reset_rate_limit_store()


def override_get_session():
    with Session(test_engine) as session:
        yield session


app.dependency_overrides[get_session] = override_get_session


@pytest_asyncio.fixture
async def client():
    """Async HTTP test client bound to the FastAPI application."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
