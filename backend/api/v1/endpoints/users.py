"""
Users API Endpoints

Endpoints for user management, authentication, and profile operations.

Author: AI-EEG Learning Platform Team
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from database.connection import get_db
from database.models import EEGSession, LearningSession, User
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from services.auth_service import (
    authenticate_user,
    create_access_token,
    get_current_user,
    get_password_hash,
)
from sqlalchemy.ext.asyncio import AsyncSession
from utils.logging_config import get_request_logger

router = APIRouter()
logger = get_request_logger("users_api")


# Pydantic models for request/response
class UserCreate(BaseModel):
    """Model for user creation."""

    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = Field(None, max_length=100)


class UserLogin(BaseModel):
    """Model for user login."""

    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class Token(BaseModel):
    """Model for authentication token."""

    access_token: str
    token_type: str = "bearer"


class UserProfile(BaseModel):
    """Model for user profile response."""

    id: int
    username: str
    email: str
    full_name: Optional[str]
    is_active: bool
    created_at: datetime
    learning_profile: Optional[Dict[str, Any]]
    preferences: Optional[Dict[str, Any]]


class UserProfileUpdate(BaseModel):
    """Model for user profile update."""

    full_name: Optional[str] = Field(None, max_length=100)
    learning_profile: Optional[Dict[str, Any]] = None
    preferences: Optional[Dict[str, Any]] = None


class UserStats(BaseModel):
    """Model for user statistics."""

    total_eeg_sessions: int
    total_learning_sessions: int
    avg_session_duration: float
    total_content_completed: int
    current_streak: int
    favorite_subjects: List[str]


@router.post("/register", response_model=UserProfile)
async def register_user(
    user_data: UserCreate, db: AsyncSession = Depends(get_db)
) -> UserProfile:
    """
    Register a new user account.

    Args:
        user_data: User registration data
        db: Database session

    Returns:
        Created user profile
    """
    try:
        # Check if user already exists
        existing_user = await get_user_by_username_or_email(
            db, user_data.username, user_data.email
        )

        if existing_user:
            if existing_user.username == user_data.username:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already registered",
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered",
                )

        # Create new user
        hashed_password = get_password_hash(user_data.password)

        new_user = User(
            username=user_data.username,
            email=user_data.email,
            hashed_password=hashed_password,
            full_name=user_data.full_name,
        )

        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)

        logger.info(
            "User registered successfully",
            user_id=new_user.id,
            username=new_user.username,
        )

        return UserProfile(
            id=new_user.id,
            username=new_user.username,
            email=new_user.email,
            full_name=new_user.full_name,
            is_active=new_user.is_active,
            created_at=new_user.created_at,
            learning_profile=new_user.learning_profile,
            preferences=new_user.preferences,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "User registration failed", username=user_data.username, error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register user",
        )


@router.post("/login", response_model=Token)
async def login_user(
    login_data: UserLogin, db: AsyncSession = Depends(get_db)
) -> Token:
    """
    Authenticate user and return access token.

    Args:
        login_data: User login credentials
        db: Database session

    Returns:
        Authentication token
    """
    try:
        user = await authenticate_user(db, login_data.username, login_data.password)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user"
            )

        access_token = create_access_token(data={"sub": user.username})

        logger.info(
            "User logged in successfully", user_id=user.id, username=user.username
        )

        return Token(access_token=access_token, token_type="bearer")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("User login failed", username=login_data.username, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Login failed"
        )


@router.get("/me", response_model=UserProfile)
async def get_current_user_profile(
    current_user: User = Depends(get_current_user),
) -> UserProfile:
    """
    Get current user's profile information.

    Args:
        current_user: Authenticated user

    Returns:
        User profile information
    """
    return UserProfile(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        created_at=current_user.created_at,
        learning_profile=current_user.learning_profile,
        preferences=current_user.preferences,
    )


@router.put("/me", response_model=UserProfile)
async def update_user_profile(
    profile_data: UserProfileUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserProfile:
    """
    Update current user's profile information.

    Args:
        profile_data: Updated profile data
        current_user: Authenticated user
        db: Database session

    Returns:
        Updated user profile
    """
    try:
        update_data = profile_data.dict(exclude_unset=True)

        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="No fields to update"
            )

        # Update user fields
        for field, value in update_data.items():
            if hasattr(current_user, field):
                setattr(current_user, field, value)

        current_user.updated_at = datetime.utcnow()

        await db.commit()
        await db.refresh(current_user)

        logger.info(
            "User profile updated",
            user_id=current_user.id,
            updated_fields=list(update_data.keys()),
        )

        return UserProfile(
            id=current_user.id,
            username=current_user.username,
            email=current_user.email,
            full_name=current_user.full_name,
            is_active=current_user.is_active,
            created_at=current_user.created_at,
            learning_profile=current_user.learning_profile,
            preferences=current_user.preferences,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Profile update failed", user_id=current_user.id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile",
        )


@router.get("/stats", response_model=UserStats)
async def get_user_stats(
    current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> UserStats:
    """
    Get comprehensive statistics for the current user.

    Args:
        current_user: Authenticated user
        db: Database session

    Returns:
        User statistics
    """
    try:
        # Get EEG session stats
        eeg_result = await db.execute(
            """
            SELECT
                COUNT(*) as total_sessions,
                AVG(EXTRACT(EPOCH FROM (end_time - start_time))) as avg_duration
            FROM eeg_sessions
            WHERE user_id = :user_id AND status = 'completed'
        """,
            {"user_id": current_user.id},
        )

        eeg_stats = eeg_result.first()

        # Get learning session stats
        learning_result = await db.execute(
            """
            SELECT
                COUNT(*) as total_sessions,
                COUNT(CASE WHEN completed = true THEN 1 END) as completed_count,
                AVG(EXTRACT(EPOCH FROM (end_time - start_time))) as avg_duration
            FROM learning_sessions
            WHERE user_id = :user_id
        """,
            {"user_id": current_user.id},
        )

        learning_stats = learning_result.first()

        # Get favorite subjects
        subject_result = await db.execute(
            """
            SELECT lc.subject, COUNT(*) as count
            FROM learning_sessions ls
            JOIN learning_content lc ON ls.content_id = lc.content_id
            WHERE ls.user_id = :user_id AND ls.completed = true
            GROUP BY lc.subject
            ORDER BY count DESC
            LIMIT 5
        """,
            {"user_id": current_user.id},
        )

        favorite_subjects = [row.subject for row in subject_result.fetchall()]

        # Calculate current streak (simplified)
        streak_result = await db.execute(
            """
            SELECT COUNT(*) as streak
            FROM learning_sessions
            WHERE user_id = :user_id
                AND completed = true
                AND DATE(start_time) >= CURRENT_DATE - INTERVAL '7 days'
        """,
            {"user_id": current_user.id},
        )

        streak = streak_result.first().streak or 0

        return UserStats(
            total_eeg_sessions=eeg_stats.total_sessions or 0,
            total_learning_sessions=learning_stats.total_sessions or 0,
            avg_session_duration=learning_stats.avg_duration or 0,
            total_content_completed=learning_stats.completed_count or 0,
            current_streak=streak,
            favorite_subjects=favorite_subjects,
        )

    except Exception as e:
        logger.error(
            "Failed to retrieve user stats", user_id=current_user.id, error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user statistics",
        )


@router.delete("/me")
async def delete_user_account(
    current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> Dict[str, str]:
    """
    Delete the current user's account and all associated data.

    This is a destructive operation that cannot be undone.

    Args:
        current_user: Authenticated user
        db: Database session

    Returns:
        Success message
    """
    try:
        # Soft delete by deactivating the account
        # In a production system, you might want to hard delete or archive

        current_user.is_active = False
        current_user.updated_at = datetime.utcnow()

        await db.commit()

        logger.info(
            "User account deactivated",
            user_id=current_user.id,
            username=current_user.username,
        )

        return {"message": "Account deactivated successfully"}

    except Exception as e:
        logger.error("Account deletion failed", user_id=current_user.id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete account",
        )


async def get_user_by_username_or_email(
    db: AsyncSession, username: str, email: str
) -> Optional[User]:
    """
    Get user by username or email.

    Args:
        db: Database session
        username: Username to search
        email: Email to search

    Returns:
        User object or None if not found
    """
    result = await db.execute(
        """
        SELECT * FROM users
        WHERE username = :username OR email = :email
    """,
        {"username": username, "email": email},
    )

    return result.first()
