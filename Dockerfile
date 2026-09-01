# =============================================================================
# Stage 1: Builder
#   Installs Python dependencies into /install so the compiler and build tools
#   are NOT present in the final image.
# =============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Build-time system deps (compiler for native wheels such as greenlet/SQLAlchemy)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Dependency manifest first — improves Docker layer cache hits
COPY requirements.txt .

# Install everything into an isolated prefix; Stage 2 will COPY it in
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# =============================================================================
# Stage 2: Runtime
#   Lean final image — app code, installed packages, Playwright + Chromium.
# =============================================================================
FROM python:3.11-slim AS runtime

# -----------------------------------------------------------------------------
# Playwright / Chromium runtime system libraries
# Derived from: playwright install-deps chromium
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        libnss3 \
        libnspr4 \
        libatk1.0-0 \
        libatk-bridge2.0-0 \
        libcups2 \
        libdrm2 \
        libdbus-1-3 \
        libxcb1 \
        libxkbcommon0 \
        libx11-6 \
        libxcomposite1 \
        libxdamage1 \
        libxext6 \
        libxfixes3 \
        libxrandr2 \
        libgbm1 \
        libpango-1.0-0 \
        libcairo2 \
        libasound2 \
        libatspi2.0-0 \
        libxshmfence1 \
        fonts-liberation \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Pull installed Python packages from the builder stage
COPY --from=builder /install /usr/local

# Playwright looks in PLAYWRIGHT_BROWSERS_PATH for Chromium binaries at runtime
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN playwright install chromium

# -----------------------------------------------------------------------------
# Application
# -----------------------------------------------------------------------------
WORKDIR /app

COPY . .

EXPOSE 8000

# Default env vars — all can be overridden in docker-compose.yml or .env
ENV DATABASE_URL="sqlite:////data/tasks.db" \
    MAX_CONCURRENT_BROWSERS=4 \
    PDF_MAX_FILE_SIZE_MB=20 \
    SEARCH_PROVIDER="duckduckgo" \
    OPENAI_MODEL="gpt-4o-mini" \
    OPENAI_OCR_MODEL="gpt-4o-mini" \
    GEMINI_MODEL="gemini-2.5-flash-lite" \
    GEMINI_OCR_MODEL="gemini-2.5-flash-lite" \
    GROQ_MODEL="llama-3.1-8b-instant"

# -----------------------------------------------------------------------------
# Entrypoint
#
# --workers 1 is intentional.
#   The app uses an in-process asyncio.Semaphore (browser gate) and an
#   in-memory sliding-window rate-limiter.  Both are per-process and will
#   silently break under multiple OS workers.  Scale concurrency via
#   MAX_CONCURRENT_BROWSERS instead.
# -----------------------------------------------------------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
