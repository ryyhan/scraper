"""
Database engine and session dependency.

SQLite is configured with:
  - WAL journal mode   — allows concurrent reads during a write, removing the
                         default "exclusive lock" that causes SQLITE_BUSY errors
                         when multiple background tasks try to commit simultaneously.
  - busy_timeout 30 s  — instead of immediately raising SQLITE_BUSY, SQLite will
                         retry internally for up to 30 seconds before giving up.
                         This is the recommended production setting for SQLite under
                         moderate concurrent write load.
  - check_same_thread  — disabled so the same connection can be handed to any
                         asyncio task / thread that holds the Session.
"""

from sqlmodel import Session, create_engine
from sqlalchemy import event, text
from loguru import logger
from app.core.config import settings


def _set_sqlite_pragmas(dbapi_connection, _connection_record):
    """
    Fired once per new raw DBAPI connection.

    WAL mode and busy_timeout must be set at the DBAPI level (not via the
    SQLAlchemy/SQLModel layer) because they are connection-scoped PRAGMAs that
    cannot be deferred inside a transaction.
    """
    cursor = dbapi_connection.cursor()
    # Write-Ahead Logging: concurrent readers never block a writer and vice-versa.
    cursor.execute("PRAGMA journal_mode=WAL;")
    # If the DB is locked, wait up to 30 000 ms before raising OperationalError.
    cursor.execute("PRAGMA busy_timeout=30000;")
    # Recommended alongside WAL for data integrity on power loss.
    cursor.execute("PRAGMA synchronous=NORMAL;")
    cursor.close()


# StaticPool is intentionally NOT used here: we want a real connection pool so
# that background tasks running in separate threads each get their own connection
# (required by SQLite's threading model) rather than sharing a single in-memory one.
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},
    # SQLite doesn't benefit from a large pool; 5 + 10 overflow covers all
    # concurrent background tasks without opening too many file handles.
    pool_size=5,
    max_overflow=10,
    # Recycle connections every 30 min to avoid stale file-handle issues on
    # long-running deployments.
    pool_recycle=1800,
)

# Register the PRAGMA hook for every new connection the pool opens.
event.listen(engine, "connect", _set_sqlite_pragmas)

logger.info("SQLite engine initialised (WAL mode, busy_timeout=30 s, pool_size=5).")


def get_session():
    """FastAPI dependency that yields a SQLModel Session and closes it on exit."""
    with Session(engine) as session:
        yield session
