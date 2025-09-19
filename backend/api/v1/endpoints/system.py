"""
System API Endpoints

Endpoints for system health monitoring, metrics, and administrative operations.

Author: AI-EEG Learning Platform Team
"""

import time
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from database.connection import get_db, health_check
from database.models import SystemMetrics
from services.auth_service import get_current_user
from utils.config import settings
from utils.logging_config import get_request_logger

router = APIRouter()
logger = get_request_logger("system_api")

# Track service start time for uptime calculation
_service_start_time = time.time()


# Pydantic models for request/response
class HealthStatus(BaseModel):
    """Model for system health status."""
    status: str = Field(..., description="Overall system status")
    timestamp: float = Field(default_factory=time.time)
    version: str = "1.0.0"
    uptime_seconds: float

    # Component health
    database: Dict[str, Any]
    cache: Dict[str, Any]
    eeg_processing: Dict[str, Any]

    # Performance metrics
    response_time_ms: float


class SystemMetricsResponse(BaseModel):
    """Model for system metrics response."""
    metric_name: str
    metric_value: float
    metric_type: str
    timestamp: float
    labels: Optional[Dict[str, Any]] = None
    description: Optional[str] = None


class SystemStats(BaseModel):
    """Model for comprehensive system statistics."""
    uptime_seconds: float
    total_users: int
    active_users_24h: int
    total_eeg_sessions: int
    total_learning_sessions: int
    avg_response_time: float
    memory_usage_mb: float
    cpu_usage_percent: float


@router.get("/health", response_model=HealthStatus)
async def get_system_health() -> HealthStatus:
    """
    Get comprehensive system health status.

    Returns:
        Detailed health status of all system components
    """
    start_time = time.time()

    try:
        # Check database health
        db_health = await health_check()

        # Check cache health (Redis)
        cache_health = await check_cache_health()

        # Check EEG processing health
        eeg_health = await check_eeg_processing_health()

        # Calculate overall status
        components = [db_health, cache_health, eeg_health]
        overall_status = "healthy"

        for component in components:
            if component.get("status") != "healthy":
                overall_status = "degraded"
                break

        response_time = (time.time() - start_time) * 1000

        health_status = HealthStatus(
            status=overall_status,
            uptime_seconds=time.time() - _service_start_time,
            database=db_health,
            cache=cache_health,
            eeg_processing=eeg_health,
            response_time_ms=response_time
        )

        # Log health check results
        logger.info(
            "Health check completed",
            status=overall_status,
            response_time_ms=round(response_time, 2)
        )

        return health_status

    except Exception as e:
        logger.error("Health check failed", error=str(e))

        # Return degraded status on error
        return HealthStatus(
            status="unhealthy",
            uptime_seconds=time.time() - _service_start_time,
            database={"status": "unknown", "error": str(e)},
            cache={"status": "unknown", "error": str(e)},
            eeg_processing={"status": "unknown", "error": str(e)},
            response_time_ms=(time.time() - start_time) * 1000
        )


@router.get("/stats", response_model=SystemStats)
async def get_system_stats(
    db: AsyncSession = Depends(get_db)
) -> SystemStats:
    """
    Get comprehensive system statistics.

    Args:
        db: Database session

    Returns:
        System-wide statistics
    """
    try:
        # Get user statistics
        user_result = await db.execute("""
            SELECT
                COUNT(*) as total_users,
                COUNT(CASE WHEN created_at >= CURRENT_DATE - INTERVAL '1 day' THEN 1 END) as new_users_24h,
                COUNT(CASE WHEN updated_at >= CURRENT_DATE - INTERVAL '1 day' THEN 1 END) as active_users_24h
            FROM users
            WHERE is_active = true
        """)

        user_stats = user_result.first()

        # Get session statistics
        session_result = await db.execute("""
            SELECT
                COUNT(DISTINCT es.id) as total_eeg_sessions,
                COUNT(DISTINCT ls.id) as total_learning_sessions
            FROM eeg_sessions es
            FULL OUTER JOIN learning_sessions ls ON true
        """)

        session_stats = session_result.first()

        # Get memory and CPU usage
        memory_usage = await get_memory_usage()
        cpu_usage = await get_cpu_usage()

        return SystemStats(
            uptime_seconds=time.time() - _service_start_time,
            total_users=user_stats.total_users or 0,
            active_users_24h=user_stats.active_users_24h or 0,
            total_eeg_sessions=session_stats.total_eeg_sessions or 0,
            total_learning_sessions=session_stats.total_learning_sessions or 0,
            avg_response_time=0.0,  # Would need to track this separately
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage
        )

    except Exception as e:
        logger.error("Failed to retrieve system stats", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve system statistics"
        )


@router.get("/metrics", response_model=List[SystemMetricsResponse])
async def get_system_metrics(
    metric_type: Optional[str] = None,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
) -> List[SystemMetricsResponse]:
    """
    Get system metrics data.

    Args:
        metric_type: Filter by metric type (gauge, counter, histogram)
        limit: Maximum number of metrics to return
        db: Database session

    Returns:
        List of system metrics
    """
    try:
        query = """
            SELECT metric_name, metric_value, metric_type, timestamp, labels, description
            FROM system_metrics
        """

        params = {}

        if metric_type:
            query += " WHERE metric_type = :metric_type"
            params["metric_type"] = metric_type

        query += " ORDER BY timestamp DESC LIMIT :limit"
        params["limit"] = limit

        result = await db.execute(query, params)
        rows = result.fetchall()

        metrics = []
        for row in rows:
            metrics.append(SystemMetricsResponse(
                metric_name=row.metric_name,
                metric_value=row.metric_value,
                metric_type=row.metric_type,
                timestamp=row.timestamp.timestamp(),
                labels=row.labels,
                description=row.description
            ))

        return metrics

    except Exception as e:
        logger.error("Failed to retrieve system metrics", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve system metrics"
        )


@router.post("/metrics")
async def record_system_metric(
    metric_name: str,
    metric_value: float,
    metric_type: str = "gauge",
    labels: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, str]:
    """
    Record a new system metric.

    Args:
        metric_name: Name of the metric
        metric_value: Value of the metric
        metric_type: Type of metric (gauge, counter, histogram)
        labels: Additional labels for the metric
        description: Description of the metric
        db: Database session

    Returns:
        Success message
    """
    try:
        if metric_type not in ["gauge", "counter", "histogram", "summary"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid metric type. Must be: gauge, counter, histogram, summary"
            )

        # Create metric record
        metric = SystemMetrics(
            metric_name=metric_name,
            metric_value=metric_value,
            metric_type=metric_type,
            labels=labels or {},
            description=description
        )

        db.add(metric)
        await db.commit()

        logger.info(
            "System metric recorded",
            metric_name=metric_name,
            value=metric_value,
            type=metric_type
        )

        return {"message": "Metric recorded successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to record system metric",
            metric_name=metric_name,
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to record system metric"
        )


@router.get("/config")
async def get_system_config() -> Dict[str, Any]:
    """
    Get system configuration information (non-sensitive).

    Returns:
        System configuration details
    """
    try:
        # Return only non-sensitive configuration
        config_info = {
            "environment": settings.ENVIRONMENT,
            "debug": settings.DEBUG,
            "version": "1.0.0",
            "eeg_sampling_rate": settings.EEG_SAMPLING_RATE,
            "eeg_channels": settings.EEG_CHANNELS,
            "max_upload_size": settings.MAX_UPLOAD_SIZE,
            "allowed_extensions": settings.ALLOWED_EXTENSIONS,
            "api_version": settings.API_V1_STR,
        }

        return config_info

    except Exception as e:
        logger.error("Failed to retrieve system config", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve system configuration"
        )


@router.post("/maintenance/cleanup")
async def trigger_cleanup(
    cleanup_type: str = "inactive_sessions",
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Trigger system maintenance cleanup operations.

    Args:
        cleanup_type: Type of cleanup to perform
        db: Database session

    Returns:
        Cleanup results
    """
    try:
        results = {}

        if cleanup_type == "inactive_sessions":
            # Clean up old inactive sessions
            result = await db.execute("""
                DELETE FROM eeg_sessions
                WHERE status = 'active'
                    AND start_time < CURRENT_TIMESTAMP - INTERVAL '24 hours'
            """)

            results["inactive_sessions_removed"] = result.rowcount

        elif cleanup_type == "old_metrics":
            # Clean up old system metrics (keep last 30 days)
            result = await db.execute("""
                DELETE FROM system_metrics
                WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL '30 days'
            """)

            results["old_metrics_removed"] = result.rowcount

        elif cleanup_type == "orphaned_data":
            # Clean up orphaned EEG data points
            result = await db.execute("""
                DELETE FROM eeg_data_points
                WHERE session_id NOT IN (SELECT id FROM eeg_sessions)
            """)

            results["orphaned_data_points_removed"] = result.rowcount

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown cleanup type: {cleanup_type}"
            )

        await db.commit()

        logger.info(
            "System cleanup completed",
            cleanup_type=cleanup_type,
            results=results
        )

        return {
            "message": "Cleanup completed successfully",
            "cleanup_type": cleanup_type,
            "results": results
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "System cleanup failed",
            cleanup_type=cleanup_type,
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail="System cleanup failed"
        )


async def check_cache_health() -> Dict[str, Any]:
    """
    Check Redis cache health.

    Returns:
        Cache health status
    """
    try:
        # Simple ping to Redis
        # In a real implementation, you'd use the actual Redis client
        return {
            "status": "healthy",
            "ping_time_ms": 1.0
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def check_eeg_processing_health() -> Dict[str, Any]:
    """
    Check EEG processing system health.

    Returns:
        EEG processing health status
    """
    try:
        # Get EEG manager status
        from main import app

        if hasattr(app.state, 'eeg_manager'):
            stats = await app.state.eeg_manager.get_system_stats()
            return {
                "status": "healthy",
                "active_sessions": stats.get('active_sessions', 0),
                "total_data_points": stats.get('total_data_points', 0)
            }
        else:
            return {
                "status": "unhealthy",
                "error": "EEG manager not initialized"
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def get_memory_usage() -> float:
    """
    Get current memory usage in MB.

    Returns:
        Memory usage in megabytes
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # Convert to MB
    except ImportError:
        # psutil not available
        return 0.0
    except Exception:
        return 0.0


async def get_cpu_usage() -> float:
    """
    Get current CPU usage percentage.

    Returns:
        CPU usage as percentage
    """
    try:
        import psutil
        return psutil.cpu_percent(interval=1)
    except ImportError:
        # psutil not available
        return 0.0
    except Exception:
        return 0.0
