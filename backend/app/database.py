"""
Database configuration and session management.
"""

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.config import get_settings
from app.models import Base

settings = get_settings()


def get_database_url() -> str:
    """Get the properly formatted async database URL."""
    url = settings.database_url
    
    # Handle SQLite URLs
    if "sqlite" in url.lower():
        # Extract the database path
        if ":///" in url:
            # Format: sqlite:///path or sqlite+aiosqlite:///path
            path = url.split(":///")[-1]
        elif "://" in url:
            path = url.split("://")[-1]
        else:
            path = "translator.db"
        
        # Return properly formatted aiosqlite URL
        return f"sqlite+aiosqlite:///{path}"
    
    # For other databases, return as-is
    return url


# Create async engine
database_url = get_database_url()

# SQLite requires special configuration for async operations
if "sqlite" in database_url.lower():
    # SQLite-specific settings to avoid "Already borrowed" errors
    # For async SQLite, we use NullPool to avoid connection sharing issues
    from sqlalchemy.pool import NullPool
    engine = create_async_engine(
        database_url,
        echo=settings.debug,
        poolclass=NullPool,  # No connection pooling for SQLite
        connect_args={
            "check_same_thread": False,  # Allow multi-threaded access
            "timeout": 30,  # Wait up to 30 seconds for locks
        },
    )
else:
    # For other databases, use default pool
    engine = create_async_engine(
        database_url,
        echo=settings.debug,
        pool_pre_ping=True,
    )

# Session factory
async_session = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def init_db():
    """Initialize database tables."""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception as e:
        # If database initialization fails, log but don't crash
        # (database is optional for this application)
        import structlog
        logger = structlog.get_logger()
        logger.warning("Database initialization failed (this is OK if not using database)", error=str(e))


async def get_session() -> AsyncSession:
    """Get database session."""
    async with async_session() as session:
        yield session
