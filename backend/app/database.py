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
engine = create_async_engine(
    database_url,
    echo=settings.debug,
)

# Session factory
async_session = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def init_db():
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    """Get database session."""
    async with async_session() as session:
        yield session
