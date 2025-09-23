"""
Recommendations API Endpoints

Endpoints for AI-powered learning content recommendations based on EEG data
and user learning patterns.

Author: AI-EEG Learning Platform Team
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from database.connection import get_db
from database.models import User, LearningContent, Recommendation
from services.recommendation_service import RecommendationService
from services.auth_service import get_current_user
from utils.logging_config import get_request_logger

router = APIRouter()
logger = get_request_logger("recommendations_api")


# Pydantic models for request/response
class CognitiveLoadPrediction(BaseModel):
    """Model for cognitive load prediction from CNN-LSTM model."""
    cognitive_load_level: str = Field(..., description="Predicted cognitive load level")
    confidence_score: float = Field(..., description="Prediction confidence (0-1)")
    theta_alpha_ratio: Optional[float] = Field(None, description="Theta/Alpha power ratio")
    processing_time_ms: Optional[float] = Field(None, description="Model processing time")
    research_validation: Optional[str] = Field(None, description="Research validation notes")


class RecommendationRequest(BaseModel):
    """Model for recommendation request with research-based cognitive load integration."""
    context: Optional[str] = Field(None, description="Learning context or goal")
    max_recommendations: int = Field(default=5, ge=1, le=20)
    difficulty_preference: Optional[int] = Field(None, ge=1, le=10)
    subject_filter: Optional[List[str]] = Field(None)
    content_type_filter: Optional[List[str]] = Field(None)
    cognitive_load_prediction: Optional[CognitiveLoadPrediction] = Field(
        None,
        description="Real-time cognitive load prediction from CNN-LSTM model (research-validated)"
    )


class RecommendationResponse(BaseModel):
    """Model for recommendation response."""
    id: int
    content_id: str
    title: str
    subject: str
    difficulty: int
    content_type: str
    duration_minutes: int
    confidence_score: float
    reasoning: str
    recommended_at: datetime
    metadata: Optional[Dict[str, Any]] = None


class RecommendationFeedback(BaseModel):
    """Model for recommendation feedback."""
    recommendation_id: int
    accepted: bool
    user_rating: Optional[int] = Field(None, ge=1, le=5)
    feedback_text: Optional[str] = Field(None, max_length=500)
    performance_improvement: Optional[float] = Field(None, ge=-1.0, le=1.0)


@router.post("/", response_model=List[RecommendationResponse])
async def get_recommendations(
    request: RecommendationRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> List[RecommendationResponse]:
    """
    Get personalized learning content recommendations.

    This endpoint uses EEG data, learning history, and user preferences
    to generate personalized content recommendations.

    Args:
        request: Recommendation request parameters
        current_user: Authenticated user
        db: Database session

    Returns:
        List of personalized recommendations
    """
    try:
        # Get recommendation service from app state
        from main import app
        recommendation_service = app.state.recommendation_service

        # Get user's recent EEG features
        eeg_features = await get_user_recent_eeg_features(
            db, current_user.id, minutes=30
        )

        # Generate recommendations with research-validated cognitive load prediction
        cognitive_load_data = None
        if request.cognitive_load_prediction:
            cognitive_load_data = {
                'cognitive_load_level': request.cognitive_load_prediction.cognitive_load_level,
                'confidence_score': request.cognitive_load_prediction.confidence_score,
                'theta_alpha_ratio': request.cognitive_load_prediction.theta_alpha_ratio,
                'processing_time_ms': request.cognitive_load_prediction.processing_time_ms,
                'research_validation': request.cognitive_load_prediction.research_validation
            }

        recommendations = await recommendation_service.generate_recommendations(
            user_id=str(current_user.id),
            eeg_features=eeg_features,
            context=request.context,
            max_recommendations=request.max_recommendations,
            difficulty_preference=request.difficulty_preference,
            subject_filter=request.subject_filter,
            content_type_filter=request.content_type_filter,
            cognitive_load_prediction=cognitive_load_data
        )

        # Store recommendations in database
        stored_recommendations = []
        for rec in recommendations:
            # Create database record
            db_rec = Recommendation(
                user_id=current_user.id,
                content_id=rec['content_id'],
                confidence_score=rec['confidence_score'],
                reasoning=rec['reasoning'],
                context=request.context,
                features_used=eeg_features
            )

            db.add(db_rec)
            stored_recommendations.append(db_rec)

        await db.commit()

        # Refresh to get IDs
        for rec in stored_recommendations:
            await db.refresh(rec)

        # Format response
        response = []
        for i, rec in enumerate(recommendations):
            db_rec = stored_recommendations[i]
            response.append(RecommendationResponse(
                id=db_rec.id,
                content_id=rec['content_id'],
                title=rec['title'],
                subject=rec['subject'],
                difficulty=rec['difficulty'],
                content_type=rec['content_type'],
                duration_minutes=rec['duration_minutes'],
                confidence_score=rec['confidence_score'],
                reasoning=rec['reasoning'],
                recommended_at=db_rec.recommended_at,
                metadata=rec.get('metadata')
            ))

        logger.info(
            "Generated recommendations",
            user_id=current_user.id,
            count=len(response)
        )

        return response

    except Exception as e:
        logger.error(
            "Failed to generate recommendations",
            user_id=current_user.id,
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate recommendations: {str(e)}"
        )


@router.get("/history", response_model=List[RecommendationResponse])
async def get_recommendation_history(
    skip: int = 0,
    limit: int = 50,
    accepted_only: Optional[bool] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> List[RecommendationResponse]:
    """
    Get user's recommendation history.

    Args:
        skip: Number of recommendations to skip
        limit: Maximum number of recommendations to return
        accepted_only: Filter for accepted recommendations only
        current_user: Authenticated user
        db: Database session

    Returns:
        List of user's past recommendations
    """
    try:
        # Build query
        query = """
            SELECT r.*, lc.title, lc.subject, lc.difficulty, lc.content_type, lc.duration_minutes
            FROM recommendations r
            JOIN learning_content lc ON r.content_id = lc.content_id
            WHERE r.user_id = :user_id
        """

        params = {"user_id": current_user.id}

        if accepted_only is not None:
            query += " AND r.accepted = :accepted"
            params["accepted"] = accepted_only

        query += " ORDER BY r.recommended_at DESC LIMIT :limit OFFSET :skip"
        params.update({"limit": limit, "skip": skip})

        result = await db.execute(query, params)
        rows = result.fetchall()

        recommendations = []
        for row in rows:
            recommendations.append(RecommendationResponse(
                id=row.id,
                content_id=row.content_id,
                title=row.title,
                subject=row.subject,
                difficulty=row.difficulty,
                content_type=row.content_type,
                duration_minutes=row.duration_minutes,
                confidence_score=row.confidence_score,
                reasoning=row.reasoning,
                recommended_at=row.recommended_at,
                metadata=row.metadata
            ))

        return recommendations

    except Exception as e:
        logger.error(
            "Failed to retrieve recommendation history",
            user_id=current_user.id,
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve recommendation history"
        )


@router.post("/feedback")
async def submit_recommendation_feedback(
    feedback: RecommendationFeedback,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, str]:
    """
    Submit feedback on a recommendation.

    Args:
        feedback: Feedback data
        current_user: Authenticated user
        db: Database session

    Returns:
        Success message
    """
    try:
        # Verify recommendation ownership
        result = await db.execute("""
            SELECT id FROM recommendations
            WHERE id = :rec_id AND user_id = :user_id
        """, {"rec_id": feedback.recommendation_id, "user_id": current_user.id})

        recommendation = result.first()

        if not recommendation:
            raise HTTPException(
                status_code=404,
                detail="Recommendation not found"
            )

        # Update recommendation with feedback
        update_data = {
            "accepted": feedback.accepted,
            "user_rating": feedback.user_rating,
            "performance_improvement": feedback.performance_improvement
        }

        if feedback.accepted:
            update_data["accepted_at"] = datetime.utcnow()

        if feedback.feedback_text:
            # Store feedback text in metadata
            existing_metadata = recommendation.metadata or {}
            existing_metadata["user_feedback"] = feedback.feedback_text
            update_data["metadata"] = existing_metadata

        await db.execute("""
            UPDATE recommendations
            SET accepted = :accepted,
                accepted_at = :accepted_at,
                user_rating = :user_rating,
                performance_improvement = :performance_improvement,
                metadata = :metadata
            WHERE id = :rec_id AND user_id = :user_id
        """, {
            "rec_id": feedback.recommendation_id,
            "user_id": current_user.id,
            **update_data
        })

        await db.commit()

        logger.info(
            "Recommendation feedback submitted",
            recommendation_id=feedback.recommendation_id,
            user_id=current_user.id,
            accepted=feedback.accepted
        )

        return {"message": "Feedback submitted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to submit recommendation feedback",
            recommendation_id=feedback.recommendation_id,
            user_id=current_user.id,
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to submit feedback"
        )


@router.get("/stats")
async def get_recommendation_stats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get recommendation statistics for the user.

    Args:
        current_user: Authenticated user
        db: Database session

    Returns:
        Recommendation statistics
    """
    try:
        # Get basic stats
        result = await db.execute("""
            SELECT
                COUNT(*) as total_recommendations,
                COUNT(CASE WHEN accepted = true THEN 1 END) as accepted_count,
                AVG(confidence_score) as avg_confidence,
                AVG(user_rating) as avg_rating
            FROM recommendations
            WHERE user_id = :user_id
        """, {"user_id": current_user.id})

        stats = result.first()

        # Get recent performance
        result = await db.execute("""
            SELECT AVG(performance_improvement) as recent_performance
            FROM recommendations
            WHERE user_id = :user_id
                AND recommended_at >= :since
                AND performance_improvement IS NOT NULL
        """, {
            "user_id": current_user.id,
            "since": datetime.utcnow() - timedelta(days=30)
        })

        recent_perf = result.first()

        return {
            "total_recommendations": stats.total_recommendations or 0,
            "accepted_recommendations": stats.accepted_count or 0,
            "acceptance_rate": (
                (stats.accepted_count or 0) / (stats.total_recommendations or 1) * 100
            ),
            "average_confidence": round(stats.avg_confidence or 0, 3),
            "average_rating": round(stats.avg_rating or 0, 2),
            "recent_performance_improvement": round(recent_perf.recent_performance or 0, 3)
        }

    except Exception as e:
        logger.error(
            "Failed to retrieve recommendation stats",
            user_id=current_user.id,
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve recommendation statistics"
        )


@router.get("/trending")
async def get_trending_content(
    subject: Optional[str] = None,
    limit: int = Query(default=10, ge=1, le=50),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Get trending learning content based on community acceptance rates.

    Args:
        subject: Filter by subject
        limit: Maximum number of items to return
        current_user: Authenticated user
        db: Database session

    Returns:
        List of trending content
    """
    try:
        query = """
            SELECT
                lc.content_id,
                lc.title,
                lc.subject,
                lc.difficulty,
                lc.content_type,
                lc.duration_minutes,
                COUNT(r.id) as recommendation_count,
                AVG(r.confidence_score) as avg_confidence,
                COUNT(CASE WHEN r.accepted = true THEN 1 END)::float /
                    NULLIF(COUNT(r.id), 0) as acceptance_rate
            FROM learning_content lc
            LEFT JOIN recommendations r ON lc.content_id = r.content_id
            WHERE lc.content_id NOT IN (
                SELECT content_id FROM recommendations
                WHERE user_id = :user_id
            )
        """

        params = {"user_id": current_user.id}

        if subject:
            query += " AND lc.subject = :subject"
            params["subject"] = subject

        query += """
            GROUP BY lc.content_id, lc.title, lc.subject, lc.difficulty,
                     lc.content_type, lc.duration_minutes
            HAVING COUNT(r.id) > 0
            ORDER BY acceptance_rate DESC, recommendation_count DESC
            LIMIT :limit
        """
        params["limit"] = limit

        result = await db.execute(query, params)
        rows = result.fetchall()

        trending_content = []
        for row in rows:
            trending_content.append({
                "content_id": row.content_id,
                "title": row.title,
                "subject": row.subject,
                "difficulty": row.difficulty,
                "content_type": row.content_type,
                "duration_minutes": row.duration_minutes,
                "recommendation_count": row.recommendation_count,
                "acceptance_rate": round(row.acceptance_rate * 100, 2),
                "average_confidence": round(row.avg_confidence, 3)
            })

        return trending_content

    except Exception as e:
        logger.error(
            "Failed to retrieve trending content",
            user_id=current_user.id,
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve trending content"
        )


async def get_user_recent_eeg_features(
    db: AsyncSession,
    user_id: int,
    minutes: int = 30
) -> Dict[str, Any]:
    """
    Get user's recent EEG features for recommendation generation.

    Args:
        db: Database session
        user_id: User identifier
        minutes: Time window in minutes

    Returns:
        Recent EEG features summary
    """
    try:
        since_time = datetime.utcnow() - timedelta(minutes=minutes)

        result = await db.execute("""
            SELECT
                AVG((processed_features->>'attention_score')::float) as avg_attention,
                AVG((processed_features->>'stress_level')::float) as avg_stress,
                AVG((processed_features->>'cognitive_load')::float) as avg_cognitive_load,
                AVG((processed_features->>'signal_quality')::float) as avg_quality,
                COUNT(*) as data_points
            FROM eeg_data_points
            WHERE user_id = :user_id
                AND timestamp >= :since
                AND processed_features IS NOT NULL
        """, {"user_id": user_id, "since": since_time})

        stats = result.first()

        if not stats or not stats.data_points:
            # Return default values if no recent data
            return {
                "attention_score": 0.5,
                "stress_level": 0.5,
                "cognitive_load": 0.5,
                "signal_quality": 0.5,
                "data_points": 0
            }

        return {
            "attention_score": stats.avg_attention or 0.5,
            "stress_level": stats.avg_stress or 0.5,
            "cognitive_load": stats.avg_cognitive_load or 0.5,
            "signal_quality": stats.avg_quality or 0.5,
            "data_points": stats.data_points
        }

    except Exception as e:
        logger.error(
            "Failed to get recent EEG features",
            user_id=user_id,
            error=str(e)
        )
        # Return default values on error
        return {
            "attention_score": 0.5,
            "stress_level": 0.5,
            "cognitive_load": 0.5,
            "signal_quality": 0.5,
            "data_points": 0
        }
