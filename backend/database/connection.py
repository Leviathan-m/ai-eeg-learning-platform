"""
Database Connection Management

Provides async database connection pooling and session management
for PostgreSQL using asyncpg and SQLAlchemy.

Author: AI-EEG Learning Platform Team
"""

import asyncpg
import json
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, Dict, Any
from urllib.parse import urlparse
import numpy as np

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, JSON, Float
from sqlalchemy.sql import func

from utils.config import settings
from utils.logging_config import get_request_logger


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


# Global database engine and session factory
_engine = None
_async_session_maker = None


async def init_db() -> None:
    """
    Initialize database connection and create tables.
    """
    global _engine, _async_session_maker

    logger = get_request_logger("database")

    try:
        # Create async engine
        _engine = create_async_engine(
            settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://"),
            echo=settings.DEBUG,
            pool_size=settings.DATABASE_POOL_SIZE,
            max_overflow=settings.DATABASE_MAX_OVERFLOW,
        )

        # Create session factory
        _async_session_maker = async_sessionmaker(
            _engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

        # Create tables
        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all, checkfirst=True)

        logger.info("Database initialized successfully")

    except Exception as e:
        logger.error("Database initialization failed", error=str(e))
        raise


async def close_db() -> None:
    """
    Close database connections.
    """
    global _engine

    if _engine:
        await _engine.dispose()
        logger = get_request_logger("database")
        logger.info("Database connections closed")


@asynccontextmanager
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Get database session with automatic cleanup.

    Yields:
        AsyncSession: Database session
    """
    if not _async_session_maker:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    session = _async_session_maker()
    try:
        yield session
    except Exception as e:
        await session.rollback()
        logger = get_request_logger("database")
        logger.error("Database session error", error=str(e))
        raise
    finally:
        await session.close()


# Raw SQL connection for high-performance operations
_raw_pool = None


async def get_raw_connection() -> asyncpg.Connection:
    """
    Get raw asyncpg connection for high-performance operations.

    Returns:
        asyncpg.Connection: Raw database connection
    """
    global _raw_pool

    if not _raw_pool:
        # Parse database URL
        parsed = urlparse(settings.DATABASE_URL)

        _raw_pool = await asyncpg.create_pool(
            host=parsed.hostname,
            port=parsed.port,
            user=parsed.username,
            password=parsed.password,
            database=parsed.path.lstrip('/'),
            min_size=5,
            max_size=20,
        )

    return await _raw_pool.acquire()


async def release_raw_connection(conn: asyncpg.Connection) -> None:
    """
    Release raw database connection.

    Args:
        conn: Connection to release
    """
    global _raw_pool

    if _raw_pool:
        await _raw_pool.release(conn)


@asynccontextmanager
async def get_raw_db() -> AsyncGenerator[asyncpg.Connection, None]:
    """
    Get raw database connection with automatic cleanup.

    Yields:
        asyncpg.Connection: Raw database connection
    """
    conn = await get_raw_connection()
    try:
        yield conn
    finally:
        await release_raw_connection(conn)


async def health_check() -> Dict[str, Any]:
    """
    Perform database health check.

    Returns:
        Health status dictionary
    """
    try:
        async with get_raw_db() as conn:
            result = await conn.fetchval("SELECT 1")
            return {
                "status": "healthy",
                "database": "connected",
                "response": result
            }
    except Exception as e:
        logger = get_request_logger("database")
        logger.error("Database health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e)
        }


# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder that handles numpy data types.
    """

    def default(self, obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def serialize_for_db(data: Any) -> str:
    """
    Serialize data for database storage.

    Args:
        data: Data to serialize

    Returns:
        JSON string representation
    """
    return json.dumps(data, cls=NumpyEncoder)


def deserialize_from_db(data: str) -> Any:
    """
    Deserialize data from database.

    Args:
        data: JSON string to deserialize

    Returns:
        Deserialized data
    """
    return json.loads(data)
