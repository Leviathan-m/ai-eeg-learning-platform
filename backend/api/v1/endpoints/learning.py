"""
Learning API Endpoints

Endpoints for learning content management, session tracking, and progress monitoring.

Author: AI-EEG Learning Platform Team
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from database.connection import get_db
from database.models import EEGSession, LearningContent, LearningSession, User
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from services.auth_service import get_current_user
from sqlalchemy.ext.asyncio import AsyncSession
from utils.logging_config import get_request_logger

router = APIRouter()
logger = get_request_logger("learning_api")


# Pydantic models for request/response
class LearningContentResponse(BaseModel):
    """Model for learning content response."""

    content_id: str
    title: str
    subject: str
    difficulty: int
    description: Optional[str]
    content_type: str
    duration_minutes: int
    prerequisites: List[str]
    learning_objectives: List[str]
    tags: List[str]
    created_at: datetime


class LearningSessionRequest(BaseModel):
    """Model for learning session creation."""

    content_id: str
    eeg_session_id: Optional[str] = None
    notes: Optional[str] = Field(None, max_length=500)


class LearningSessionResponse(BaseModel):
    """Model for learning session response."""

    id: int
    user_id: int
    content_id: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: Optional[int]
    completed: bool
    progress_percentage: float
    score: Optional[float]
    notes: Optional[str]
    eeg_session_id: Optional[str]


class LearningProgress(BaseModel):
    """Model for learning progress tracking."""

    content_id: str
    title: str
    progress_percentage: float
    time_spent_seconds: int
    estimated_completion_time: str
    last_accessed: Optional[datetime]
    avg_score: Optional[float]
    completed_sections: List[str]


@router.get("/content", response_model=List[LearningContentResponse])
async def get_learning_content(
    subject: Optional[str] = None,
    difficulty: Optional[int] = Query(None, ge=1, le=10),
    content_type: Optional[str] = None,
    tags: Optional[List[str]] = Query(None),
    skip: int = 0,
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> List[LearningContentResponse]:
    """
    Get available learning content with optional filtering.

    Args:
        subject: Filter by subject
        difficulty: Filter by difficulty level
        content_type: Filter by content type
        tags: Filter by tags
        skip: Number of items to skip
        limit: Maximum number of items to return
        db: Database session

    Returns:
        List of learning content
    """
    try:
        query = "SELECT * FROM learning_content WHERE 1=1"
        params = {}

        # Add filters
        if subject:
            query += " AND subject = :subject"
            params["subject"] = subject

        if difficulty:
            query += " AND difficulty = :difficulty"
            params["difficulty"] = difficulty

        if content_type:
            query += " AND content_type = :content_type"
            params["content_type"] = content_type

        if tags:
            # Filter by tags (array contains)
            query += " AND tags && :tags"
            params["tags"] = tags

        # Add ordering and pagination
        query += " ORDER BY created_at DESC LIMIT :limit OFFSET :skip"
        params.update({"limit": limit, "skip": skip})

        result = await db.execute(query, params)
        content_items = result.fetchall()

        content_list = []
        for item in content_items:
            content_list.append(
                LearningContentResponse(
                    content_id=item.content_id,
                    title=item.title,
                    subject=item.subject,
                    difficulty=item.difficulty,
                    description=item.description,
                    content_type=item.content_type,
                    duration_minutes=item.duration_minutes,
                    prerequisites=item.prerequisites or [],
                    learning_objectives=item.learning_objectives or [],
                    tags=item.tags or [],
                    created_at=item.created_at,
                )
            )

        return content_list

    except Exception as e:
        logger.error("Failed to retrieve learning content", error=str(e))
        raise HTTPException(
            status_code=500, detail="Failed to retrieve learning content"
        )


@router.get("/content/{content_id}", response_model=LearningContentResponse)
async def get_content_details(
    content_id: str, db: AsyncSession = Depends(get_db)
) -> LearningContentResponse:
    """
    Get detailed information about specific learning content.

    Args:
        content_id: Content identifier
        db: Database session

    Returns:
        Detailed content information
    """
    try:
        result = await db.execute(
            """
            SELECT * FROM learning_content WHERE content_id = :content_id
        """,
            {"content_id": content_id},
        )

        content = result.first()

        if not content:
            raise HTTPException(status_code=404, detail="Learning content not found")

        return LearningContentResponse(
            content_id=content.content_id,
            title=content.title,
            subject=content.subject,
            difficulty=content.difficulty,
            description=content.description,
            content_type=content.content_type,
            duration_minutes=content.duration_minutes,
            prerequisites=content.prerequisites or [],
            learning_objectives=content.learning_objectives or [],
            tags=content.tags or [],
            created_at=content.created_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to retrieve content details", content_id=content_id, error=str(e)
        )
        raise HTTPException(
            status_code=500, detail="Failed to retrieve content details"
        )


@router.post("/sessions", response_model=LearningSessionResponse)
async def start_learning_session(
    session_request: LearningSessionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> LearningSessionResponse:
    """
    Start a new learning session.

    Args:
        session_request: Learning session parameters
        current_user: Authenticated user
        db: Database session

    Returns:
        Created learning session
    """
    try:
        # Verify content exists
        content_result = await db.execute(
            """
            SELECT id FROM learning_content WHERE content_id = :content_id
        """,
            {"content_id": session_request.content_id},
        )

        if not content_result.first():
            raise HTTPException(status_code=404, detail="Learning content not found")

        # Verify EEG session if provided
        if session_request.eeg_session_id:
            eeg_result = await db.execute(
                """
                SELECT id FROM eeg_sessions
                WHERE session_id = :session_id AND user_id = :user_id
            """,
                {
                    "session_id": session_request.eeg_session_id,
                    "user_id": current_user.id,
                },
            )

            if not eeg_result.first():
                raise HTTPException(status_code=404, detail="EEG session not found")

        # Create learning session
        session = LearningSession(
            user_id=current_user.id,
            content_id=session_request.content_id,
            eeg_session_id=session_request.eeg_session_id,
            start_time=datetime.utcnow(),
            notes=session_request.notes,
        )

        db.add(session)
        await db.commit()
        await db.refresh(session)

        logger.info(
            "Learning session started",
            user_id=current_user.id,
            content_id=session_request.content_id,
            session_id=session.id,
        )

        return LearningSessionResponse(
            id=session.id,
            user_id=session.user_id,
            content_id=session.content_id,
            start_time=session.start_time,
            end_time=session.end_time,
            duration_seconds=session.duration_seconds,
            completed=session.completed,
            progress_percentage=session.progress_percentage,
            score=session.score,
            notes=session.notes,
            eeg_session_id=session.eeg_session_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to start learning session",
            user_id=current_user.id,
            content_id=session_request.content_id,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail="Failed to start learning session")


@router.put("/sessions/{session_id}", response_model=LearningSessionResponse)
async def update_learning_session(
    session_id: int,
    progress_percentage: Optional[float] = Query(None, ge=0, le=100),
    score: Optional[float] = Query(None, ge=0, le=100),
    completed: Optional[bool] = None,
    notes: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> LearningSessionResponse:
    """
    Update an existing learning session.

    Args:
        session_id: Learning session ID
        progress_percentage: Progress percentage (0-100)
        score: Performance score (0-100)
        completed: Whether session is completed
        notes: Additional notes
        current_user: Authenticated user
        db: Database session

    Returns:
        Updated learning session
    """
    try:
        # Get session
        result = await db.execute(
            """
            SELECT * FROM learning_sessions
            WHERE id = :session_id AND user_id = :user_id
        """,
            {"session_id": session_id, "user_id": current_user.id},
        )

        session = result.first()

        if not session:
            raise HTTPException(status_code=404, detail="Learning session not found")

        # Update fields
        update_data = {}
        if progress_percentage is not None:
            update_data["progress_percentage"] = progress_percentage
        if score is not None:
            update_data["score"] = score
        if completed is not None:
            update_data["completed"] = completed
            if completed and not session.end_time:
                update_data["end_time"] = datetime.utcnow()
        if notes is not None:
            update_data["notes"] = notes

        if update_data:
            update_data["updated_at"] = datetime.utcnow()

            await db.execute(
                """
                UPDATE learning_sessions
                SET progress_percentage = COALESCE(:progress_percentage, progress_percentage),
                    score = COALESCE(:score, score),
                    completed = COALESCE(:completed, completed),
                    end_time = COALESCE(:end_time, end_time),
                    notes = COALESCE(:notes, notes)
                WHERE id = :session_id AND user_id = :user_id
            """,
                {"session_id": session_id, "user_id": current_user.id, **update_data},
            )

            # Refresh session data
            result = await db.execute(
                """
                SELECT * FROM learning_sessions WHERE id = :session_id
            """,
                {"session_id": session_id},
            )
            session = result.first()

        await db.commit()

        return LearningSessionResponse(
            id=session.id,
            user_id=session.user_id,
            content_id=session.content_id,
            start_time=session.start_time,
            end_time=session.end_time,
            duration_seconds=session.duration_seconds,
            completed=session.completed,
            progress_percentage=session.progress_percentage,
            score=session.score,
            notes=session.notes,
            eeg_session_id=session.eeg_session_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to update learning session",
            session_id=session_id,
            user_id=current_user.id,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail="Failed to update learning session")


@router.get("/sessions", response_model=List[LearningSessionResponse])
async def get_learning_sessions(
    content_id: Optional[str] = None,
    completed: Optional[bool] = None,
    skip: int = 0,
    limit: int = Query(default=20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> List[LearningSessionResponse]:
    """
    Get user's learning sessions with optional filtering.

    Args:
        content_id: Filter by content ID
        completed: Filter by completion status
        skip: Number of sessions to skip
        limit: Maximum number of sessions to return
        current_user: Authenticated user
        db: Database session

    Returns:
        List of learning sessions
    """
    try:
        query = "SELECT * FROM learning_sessions WHERE user_id = :user_id"
        params = {"user_id": current_user.id}

        # Add filters
        if content_id:
            query += " AND content_id = :content_id"
            params["content_id"] = content_id

        if completed is not None:
            query += " AND completed = :completed"
            params["completed"] = completed

        # Add ordering and pagination
        query += " ORDER BY start_time DESC LIMIT :limit OFFSET :skip"
        params.update({"limit": limit, "skip": skip})

        result = await db.execute(query, params)
        sessions = result.fetchall()

        session_list = []
        for session in sessions:
            session_list.append(
                LearningSessionResponse(
                    id=session.id,
                    user_id=session.user_id,
                    content_id=session.content_id,
                    start_time=session.start_time,
                    end_time=session.end_time,
                    duration_seconds=session.duration_seconds,
                    completed=session.completed,
                    progress_percentage=session.progress_percentage,
                    score=session.score,
                    notes=session.notes,
                    eeg_session_id=session.eeg_session_id,
                )
            )

        return session_list

    except Exception as e:
        logger.error(
            "Failed to retrieve learning sessions",
            user_id=current_user.id,
            error=str(e),
        )
        raise HTTPException(
            status_code=500, detail="Failed to retrieve learning sessions"
        )


@router.get("/progress", response_model=List[LearningProgress])
async def get_learning_progress(
    current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> List[LearningProgress]:
    """
    Get learning progress for all user's content.

    Args:
        current_user: Authenticated user
        db: Database session

    Returns:
        List of learning progress for each content
    """
    try:
        result = await db.execute(
            """
            SELECT
                lc.content_id,
                lc.title,
                lc.duration_minutes,
                MAX(ls.progress_percentage) as max_progress,
                SUM(EXTRACT(EPOCH FROM (
                    CASE
                        WHEN ls.end_time IS NOT NULL THEN ls.end_time
                        ELSE CURRENT_TIMESTAMP
                    END - ls.start_time
                ))) as total_time_spent,
                MAX(ls.start_time) as last_accessed,
                AVG(ls.score) as avg_score,
                COUNT(CASE WHEN ls.completed = true THEN 1 END) as completed_sessions
            FROM learning_content lc
            LEFT JOIN learning_sessions ls ON lc.content_id = ls.content_id
                AND ls.user_id = :user_id
            GROUP BY lc.content_id, lc.title, lc.duration_minutes
            HAVING MAX(ls.progress_percentage) > 0 OR COUNT(ls.id) > 0
            ORDER BY last_accessed DESC
        """,
            {"user_id": current_user.id},
        )

        progress_items = result.fetchall()

        progress_list = []
        for item in progress_items:
            # Estimate completion time
            if item.max_progress and item.max_progress > 0:
                remaining_percentage = 100 - item.max_progress
                estimated_seconds = (
                    item.total_time_spent / item.max_progress
                ) * remaining_percentage
                estimated_completion = f"{int(estimated_seconds // 3600)}h {int((estimated_seconds % 3600) // 60)}m"
            else:
                estimated_completion = "Unknown"

            progress_list.append(
                LearningProgress(
                    content_id=item.content_id,
                    title=item.title,
                    progress_percentage=item.max_progress or 0,
                    time_spent_seconds=int(item.total_time_spent or 0),
                    estimated_completion_time=estimated_completion,
                    last_accessed=item.last_accessed,
                    avg_score=item.avg_score,
                    completed_sections=[],  # Would need additional tracking
                )
            )

        return progress_list

    except Exception as e:
        logger.error(
            "Failed to retrieve learning progress",
            user_id=current_user.id,
            error=str(e),
        )
        raise HTTPException(
            status_code=500, detail="Failed to retrieve learning progress"
        )


@router.get("/subjects")
async def get_available_subjects(db: AsyncSession = Depends(get_db)) -> List[str]:
    """
    Get list of available subjects.

    Args:
        db: Database session

    Returns:
        List of available subjects
    """
    try:
        result = await db.execute(
            """
            SELECT DISTINCT subject FROM learning_content
            ORDER BY subject
        """
        )

        subjects = [row.subject for row in result.fetchall()]
        return subjects

    except Exception as e:
        logger.error("Failed to retrieve subjects", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve subjects")


@router.get("/stats")
async def get_learning_stats(
    period_days: int = Query(default=30, ge=1, le=365),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Get comprehensive learning statistics.

    Args:
        period_days: Analysis period in days
        current_user: Authenticated user
        db: Database session

    Returns:
        Learning statistics
    """
    try:
        since_date = datetime.utcnow() - timedelta(days=period_days)

        # Get session statistics
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

        stats = session_result.first()

        # Get subject breakdown
        subject_result = await db.execute(
            """
            SELECT lc.subject, COUNT(*) as count
            FROM learning_sessions ls
            JOIN learning_content lc ON ls.content_id = lc.content_id
            WHERE ls.user_id = :user_id AND ls.start_time >= :since
            GROUP BY lc.subject
            ORDER BY count DESC
        """,
            {"user_id": current_user.id, "since": since_date},
        )

        subjects = [
            {"subject": row.subject, "count": row.count}
            for row in subject_result.fetchall()
        ]

        return {
            "period_days": period_days,
            "total_sessions": stats.total_sessions or 0,
            "completed_sessions": stats.completed_sessions or 0,
            "completion_rate": round(
                ((stats.completed_sessions or 0) / (stats.total_sessions or 1)) * 100, 2
            ),
            "avg_session_duration": round(stats.avg_duration or 0, 2),
            "avg_score": round(stats.avg_score or 0, 2),
            "subjects_breakdown": subjects,
        }

    except Exception as e:
        logger.error(
            "Failed to retrieve learning stats", user_id=current_user.id, error=str(e)
        )
        raise HTTPException(
            status_code=500, detail="Failed to retrieve learning statistics"
        )
