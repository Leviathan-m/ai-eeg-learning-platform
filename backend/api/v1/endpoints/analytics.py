"""
Analytics API Endpoints

Endpoints for learning analytics, performance insights, and data visualization.

Author: AI-EEG Learning Platform Team
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from database.connection import get_db
from database.models import EEGSession, LearningSession, Recommendation, User
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from services.auth_service import get_current_user
from sqlalchemy.ext.asyncio import AsyncSession
from utils.logging_config import get_request_logger

router = APIRouter()
logger = get_request_logger("analytics_api")


# Pydantic models for request/response
class LearningAnalytics(BaseModel):
    """Model for learning analytics data."""

    user_id: int
    period_days: int
    total_sessions: int
    completed_sessions: int
    avg_session_duration: float
    completion_rate: float
    avg_performance_score: Optional[float]
    favorite_subjects: List[str]
    learning_streak: int
    improvement_trend: str  # "improving", "stable", "declining"


class EEGAnalytics(BaseModel):
    """Model for EEG analytics data."""

    user_id: int
    period_days: int
    total_sessions: int
    avg_attention_score: float
    avg_stress_level: float
    avg_cognitive_load: float
    signal_quality_trend: str
    peak_performance_times: List[str]
    attention_distribution: Dict[str, float]


class ContentAnalytics(BaseModel):
    """Model for content performance analytics."""

    content_id: str
    title: str
    subject: str
    difficulty: int
    view_count: int
    completion_rate: float
    avg_rating: Optional[float]
    avg_learning_time: float
    recommendation_success_rate: float


class DashboardData(BaseModel):
    """Model for user dashboard data."""

    user_profile: Dict[str, Any]
    recent_activity: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    recommendations: List[Dict[str, Any]]
    learning_goals: List[Dict[str, Any]]
    achievements: List[Dict[str, Any]]


@router.get("/learning", response_model=LearningAnalytics)
async def get_learning_analytics(
    period_days: int = Query(default=30, ge=1, le=365),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> LearningAnalytics:
    """
    Get comprehensive learning analytics for the user.

    Args:
        period_days: Analysis period in days
        current_user: Authenticated user
        db: Database session

    Returns:
        Learning analytics data
    """
    try:
        since_date = datetime.utcnow() - timedelta(days=period_days)

        # Get learning session statistics
        session_result = await db.execute(
            """
            SELECT
                COUNT(*) as total_sessions,
                COUNT(CASE WHEN completed = true THEN 1 END) as completed_sessions,
                AVG(EXTRACT(EPOCH FROM (end_time - start_time))) as avg_duration,
                AVG(score) as avg_score
            FROM learning_sessions
            WHERE user_id = :user_id AND start_time >= :since
        """,
            {"user_id": current_user.id, "since": since_date},
        )

        session_stats = session_result.first()

        # Get favorite subjects
        subject_result = await db.execute(
            """
            SELECT lc.subject, COUNT(*) as count
            FROM learning_sessions ls
            JOIN learning_content lc ON ls.content_id = lc.content_id
            WHERE ls.user_id = :user_id AND ls.start_time >= :since
            GROUP BY lc.subject
            ORDER BY count DESC
            LIMIT 5
        """,
            {"user_id": current_user.id, "since": since_date},
        )

        favorite_subjects = [row.subject for row in subject_result.fetchall()]

        # Calculate completion rate
        total_sessions = session_stats.total_sessions or 0
        completed_sessions = session_stats.completed_sessions or 0
        completion_rate = (
            (completed_sessions / total_sessions * 100) if total_sessions > 0 else 0
        )

        # Calculate learning streak (simplified)
        streak_result = await db.execute(
            """
            SELECT COUNT(*) as streak
            FROM (
                SELECT DATE(start_time) as session_date
                FROM learning_sessions
                WHERE user_id = :user_id AND completed = true
                ORDER BY session_date DESC
                LIMIT 30
            ) daily_sessions
            WHERE session_date >= CURRENT_DATE - INTERVAL '30 days'
        """,
            {"user_id": current_user.id},
        )

        streak = streak_result.first().streak or 0

        # Determine improvement trend (simplified)
        trend_result = await db.execute(
            """
            SELECT
                AVG(CASE WHEN start_time >= :recent THEN score END) as recent_avg,
                AVG(CASE WHEN start_time < :recent THEN score END) as older_avg
            FROM learning_sessions
            WHERE user_id = :user_id AND score IS NOT NULL
        """,
            {
                "user_id": current_user.id,
                "recent": datetime.utcnow() - timedelta(days=period_days // 2),
            },
        )

        trend_stats = trend_result.first()
        if trend_stats.recent_avg and trend_stats.older_avg:
            if trend_stats.recent_avg > trend_stats.older_avg * 1.05:
                improvement_trend = "improving"
            elif trend_stats.recent_avg < trend_stats.older_avg * 0.95:
                improvement_trend = "declining"
            else:
                improvement_trend = "stable"
        else:
            improvement_trend = "stable"

        return LearningAnalytics(
            user_id=current_user.id,
            period_days=period_days,
            total_sessions=total_sessions,
            completed_sessions=completed_sessions,
            avg_session_duration=session_stats.avg_duration or 0,
            completion_rate=round(completion_rate, 2),
            avg_performance_score=session_stats.avg_score,
            favorite_subjects=favorite_subjects,
            learning_streak=streak,
            improvement_trend=improvement_trend,
        )

    except Exception as e:
        logger.error(
            "Failed to retrieve learning analytics",
            user_id=current_user.id,
            error=str(e),
        )
        raise HTTPException(
            status_code=500, detail="Failed to retrieve learning analytics"
        )


@router.get("/eeg", response_model=EEGAnalytics)
async def get_eeg_analytics(
    period_days: int = Query(default=7, ge=1, le=90),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> EEGAnalytics:
    """
    Get EEG-based cognitive analytics for the user.

    Args:
        period_days: Analysis period in days
        current_user: Authenticated user
        db: Database session

    Returns:
        EEG analytics data
    """
    try:
        since_date = datetime.utcnow() - timedelta(days=period_days)

        # Get EEG statistics
        eeg_result = await db.execute(
            """
            SELECT
                COUNT(DISTINCT es.id) as total_sessions,
                AVG((edp.processed_features->>'attention_score')::float) as avg_attention,
                AVG((edp.processed_features->>'stress_level')::float) as avg_stress,
                AVG((edp.processed_features->>'cognitive_load')::float) as avg_cognitive_load,
                AVG(edp.signal_quality) as avg_quality
            FROM eeg_sessions es
            LEFT JOIN eeg_data_points edp ON es.id = edp.session_id
            WHERE es.user_id = :user_id AND es.start_time >= :since
        """,
            {"user_id": current_user.id, "since": since_date},
        )

        eeg_stats = eeg_result.first()

        # Determine signal quality trend
        quality_trend = "stable"
        if eeg_stats.avg_quality:
            if eeg_stats.avg_quality > 0.8:
                quality_trend = "excellent"
            elif eeg_stats.avg_quality > 0.6:
                quality_trend = "good"
            elif eeg_stats.avg_quality > 0.4:
                quality_trend = "fair"
            else:
                quality_trend = "poor"

        # Get peak performance times (simplified)
        peak_times_result = await db.execute(
            """
            SELECT
                EXTRACT(HOUR FROM edp.timestamp) as hour,
                AVG((edp.processed_features->>'attention_score')::float) as avg_attention
            FROM eeg_data_points edp
            JOIN eeg_sessions es ON edp.session_id = es.id
            WHERE es.user_id = :user_id AND edp.timestamp >= :since
            GROUP BY EXTRACT(HOUR FROM edp.timestamp)
            ORDER BY avg_attention DESC
            LIMIT 3
        """,
            {"user_id": current_user.id, "since": since_date},
        )

        peak_times = [f"{int(row.hour):02d}:00" for row in peak_times_result.fetchall()]

        # Get attention distribution
        attention_dist = await db.execute(
            """
            SELECT
                CASE
                    WHEN (processed_features->>'attention_score')::float < 0.3 THEN 'low'
                    WHEN (processed_features->>'attention_score')::float < 0.7 THEN 'medium'
                    ELSE 'high'
                END as attention_level,
                COUNT(*) as count
            FROM eeg_data_points edp
            JOIN eeg_sessions es ON edp.session_id = es.id
            WHERE es.user_id = :user_id AND edp.timestamp >= :since
            GROUP BY attention_level
        """,
            {"user_id": current_user.id, "since": since_date},
        )

        attention_distribution = {
            row.attention_level: row.count for row in attention_dist.fetchall()
        }

        return EEGAnalytics(
            user_id=current_user.id,
            period_days=period_days,
            total_sessions=eeg_stats.total_sessions or 0,
            avg_attention_score=round(eeg_stats.avg_attention or 0, 3),
            avg_stress_level=round(eeg_stats.avg_stress or 0, 3),
            avg_cognitive_load=round(eeg_stats.avg_cognitive_load or 0, 3),
            signal_quality_trend=quality_trend,
            peak_performance_times=peak_times,
            attention_distribution=attention_distribution,
        )

    except Exception as e:
        logger.error(
            "Failed to retrieve EEG analytics", user_id=current_user.id, error=str(e)
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve EEG analytics")


@router.get("/content", response_model=List[ContentAnalytics])
async def get_content_analytics(
    subject_filter: Optional[str] = None,
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> List[ContentAnalytics]:
    """
    Get content performance analytics.

    Args:
        subject_filter: Filter by subject
        limit: Maximum number of content items to return
        db: Database session

    Returns:
        Content analytics data
    """
    try:
        query = """
            SELECT
                lc.content_id,
                lc.title,
                lc.subject,
                lc.difficulty,
                lc.view_count,
                AVG(ls.score) as avg_rating,
                AVG(EXTRACT(EPOCH FROM (ls.end_time - ls.start_time))) as avg_learning_time,
                COUNT(ls.id) as total_views,
                COUNT(CASE WHEN ls.completed = true THEN 1 END)::float /
                    NULLIF(COUNT(ls.id), 0) as completion_rate,
                COUNT(CASE WHEN r.accepted = true THEN 1 END)::float /
                    NULLIF(COUNT(r.id), 0) as recommendation_success_rate
            FROM learning_content lc
            LEFT JOIN learning_sessions ls ON lc.content_id = ls.content_id
            LEFT JOIN recommendations r ON lc.content_id = r.content_id
            WHERE 1=1
        """

        params = {}

        if subject_filter:
            query += " AND lc.subject = :subject"
            params["subject"] = subject_filter

        query += """
            GROUP BY lc.content_id, lc.title, lc.subject, lc.difficulty, lc.view_count
            ORDER BY total_views DESC
            LIMIT :limit
        """
        params["limit"] = limit

        result = await db.execute(query, params)
        rows = result.fetchall()

        analytics = []
        for row in rows:
            analytics.append(
                ContentAnalytics(
                    content_id=row.content_id,
                    title=row.title,
                    subject=row.subject,
                    difficulty=row.difficulty,
                    view_count=row.total_views or 0,
                    completion_rate=round((row.completion_rate or 0) * 100, 2),
                    avg_rating=row.avg_rating,
                    avg_learning_time=row.avg_learning_time or 0,
                    recommendation_success_rate=round(
                        (row.recommendation_success_rate or 0) * 100, 2
                    ),
                )
            )

        return analytics

    except Exception as e:
        logger.error("Failed to retrieve content analytics", error=str(e))
        raise HTTPException(
            status_code=500, detail="Failed to retrieve content analytics"
        )


@router.get("/dashboard", response_model=DashboardData)
async def get_dashboard_data(
    current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> DashboardData:
    """
    Get comprehensive dashboard data for the user.

    Args:
        current_user: Authenticated user
        db: Database session

    Returns:
        Dashboard data
    """
    try:
        # Get recent learning activity
        activity_result = await db.execute(
            """
            SELECT
                ls.id,
                lc.title,
                ls.start_time,
                ls.completed,
                ls.score,
                lc.subject,
                lc.difficulty
            FROM learning_sessions ls
            JOIN learning_content lc ON ls.content_id = lc.content_id
            WHERE ls.user_id = :user_id
            ORDER BY ls.start_time DESC
            LIMIT 10
        """,
            {"user_id": current_user.id},
        )

        recent_activity = []
        for row in activity_result.fetchall():
            recent_activity.append(
                {
                    "id": row.id,
                    "title": row.title,
                    "start_time": row.start_time.isoformat(),
                    "completed": row.completed,
                    "score": row.score,
                    "subject": row.subject,
                    "difficulty": row.difficulty,
                }
            )

        # Get current performance metrics
        metrics_result = await db.execute(
            """
            SELECT
                AVG(ls.score) as avg_score,
                COUNT(CASE WHEN ls.completed = true THEN 1 END)::float /
                    NULLIF(COUNT(ls.id), 0) as completion_rate,
                AVG((edp.processed_features->>'attention_score')::float) as avg_attention
            FROM learning_sessions ls
            LEFT JOIN eeg_sessions es ON ls.eeg_session_id = es.id
            LEFT JOIN eeg_data_points edp ON es.id = edp.session_id
            WHERE ls.user_id = :user_id AND ls.start_time >= CURRENT_DATE - INTERVAL '30 days'
        """,
            {"user_id": current_user.id},
        )

        metrics = metrics_result.first()

        # Get pending recommendations
        rec_result = await db.execute(
            """
            SELECT r.id, lc.title, r.confidence_score, r.reasoning
            FROM recommendations r
            JOIN learning_content lc ON r.content_id = lc.content_id
            WHERE r.user_id = :user_id AND r.accepted IS NULL
            ORDER BY r.recommended_at DESC
            LIMIT 5
        """,
            {"user_id": current_user.id},
        )

        recommendations = []
        for row in rec_result.fetchall():
            recommendations.append(
                {
                    "id": row.id,
                    "title": row.title,
                    "confidence_score": row.confidence_score,
                    "reasoning": row.reasoning,
                }
            )

        # Mock learning goals and achievements (would be stored in database)
        learning_goals = [
            {
                "goal": "Complete 5 math lessons",
                "progress": 60,
                "target_date": "2025-10-01",
            },
            {
                "goal": "Improve attention score by 20%",
                "progress": 75,
                "target_date": "2025-09-30",
            },
        ]

        achievements = [
            {"name": "First Lesson Completed", "unlocked_at": "2025-09-15"},
            {"name": "Week Streak", "unlocked_at": "2025-09-20"},
        ]

        return DashboardData(
            user_profile={
                "id": current_user.id,
                "username": current_user.username,
                "full_name": current_user.full_name,
                "created_at": current_user.created_at.isoformat(),
            },
            recent_activity=recent_activity,
            performance_metrics={
                "avg_score": round(metrics.avg_score or 0, 2),
                "completion_rate": round((metrics.completion_rate or 0) * 100, 2),
                "avg_attention": round(metrics.avg_attention or 0, 3),
            },
            recommendations=recommendations,
            learning_goals=learning_goals,
            achievements=achievements,
        )

    except Exception as e:
        logger.error(
            "Failed to retrieve dashboard data", user_id=current_user.id, error=str(e)
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve dashboard data")
