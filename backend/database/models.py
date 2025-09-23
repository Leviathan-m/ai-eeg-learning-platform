"""
Database Models

SQLAlchemy models for the AI-EEG Learning Platform database schema.
Defines all tables, relationships, and database operations.

Author: AI-EEG Learning Platform Team
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Text,
    Boolean,
    JSON,
    Float,
    ForeignKey,
    Index,
    UniqueConstraint,
    CheckConstraint,
)
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.ext.asyncio import AsyncSession

from .connection import Base, get_db, serialize_for_db, deserialize_from_db


class User(Base):
    """
    User model for storing user information and learning profiles.
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(
        String(50), unique=True, nullable=False, index=True
    )
    email: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, index=True
    )
    hashed_password: Mapped[Optional[str]] = mapped_column(String(255))
    full_name: Mapped[Optional[str]] = mapped_column(String(100))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )

    # Learning profile and preferences
    learning_profile: Mapped[Optional[Dict]] = mapped_column(JSON, default=dict)
    preferences: Mapped[Optional[Dict]] = mapped_column(JSON, default=dict)

    # Relationships
    eeg_sessions: Mapped[List["EEGSession"]] = relationship(
        "EEGSession", back_populates="user"
    )
    learning_sessions: Mapped[List["LearningSession"]] = relationship(
        "LearningSession", back_populates="user"
    )
    recommendations: Mapped[List["Recommendation"]] = relationship(
        "Recommendation", back_populates="user"
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username={self.username}, email={self.email})>"


class EEGSession(Base):
    """
    EEG session model for tracking EEG data collection sessions.
    """

    __tablename__ = "eeg_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False, index=True
    )
    session_id: Mapped[str] = mapped_column(
        String(36), unique=True, nullable=False, index=True
    )
    device_type: Mapped[str] = mapped_column(String(20), nullable=False)
    start_time: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(20), default="active")
    duration_seconds: Mapped[Optional[int]] = mapped_column(Integer)

    # Quality and metadata
    avg_signal_quality: Mapped[Optional[float]] = mapped_column(Float)
    total_data_points: Mapped[int] = mapped_column(Integer, default=0)
    session_metadata: Mapped[Optional[Dict]] = mapped_column(JSON, default=dict)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="eeg_sessions")
    data_points: Mapped[List["EEGDataPoint"]] = relationship(
        "EEGDataPoint", back_populates="session"
    )
    learning_sessions: Mapped[List["LearningSession"]] = relationship(
        "LearningSession", back_populates="eeg_session"
    )

    __table_args__ = (
        Index("idx_eeg_sessions_user_time", "user_id", "start_time"),
        CheckConstraint(
            "status IN ('active', 'completed', 'interrupted')",
            name="check_eeg_session_status",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<EEGSession(id={self.id}, user_id={self.user_id}, status={self.status})>"
        )


class EEGDataPoint(Base):
    """
    Individual EEG data point with raw data and processed features.
    """

    __tablename__ = "eeg_data_points"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("eeg_sessions.id"), nullable=False, index=True
    )
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False, index=True
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)

    # Raw EEG data
    raw_data: Mapped[Optional[Dict]] = mapped_column(JSON)

    # Processed features
    processed_features: Mapped[Optional[Dict]] = mapped_column(JSON)

    # Quality metrics
    signal_quality: Mapped[Optional[float]] = mapped_column(Float)
    processing_time_ms: Mapped[Optional[float]] = mapped_column(Float)

    # Relationships
    session: Mapped["EEGSession"] = relationship(
        "EEGSession", back_populates="data_points"
    )
    user: Mapped["User"] = relationship("User")

    __table_args__ = (
        Index("idx_eeg_data_user_time", "user_id", "timestamp"),
        Index("idx_eeg_data_session_time", "session_id", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<EEGDataPoint(id={self.id}, session_id={self.session_id}, timestamp={self.timestamp})>"


class LearningContent(Base):
    """
    Learning content model for storing educational materials.
    """

    __tablename__ = "learning_content"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    content_id: Mapped[str] = mapped_column(
        String(36), unique=True, nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    subject: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    difficulty: Mapped[int] = mapped_column(Integer, nullable=False)

    # Content details
    description: Mapped[Optional[str]] = mapped_column(Text)
    content_type: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # video, text, quiz, etc.
    duration_minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    language: Mapped[str] = mapped_column(String(10), default="en")

    # Learning metadata
    prerequisites: Mapped[Optional[List[str]]] = mapped_column(JSON, default=list)
    learning_objectives: Mapped[Optional[List[str]]] = mapped_column(JSON, default=list)
    tags: Mapped[Optional[List[str]]] = mapped_column(JSON, default=list)

    # Content URLs and resources
    content_url: Mapped[Optional[str]] = mapped_column(String(500))
    thumbnail_url: Mapped[Optional[str]] = mapped_column(String(500))
    content_metadata: Mapped[Optional[Dict]] = mapped_column(JSON, default=dict)

    # Statistics
    view_count: Mapped[int] = mapped_column(Integer, default=0)
    avg_rating: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )

    # Relationships
    learning_sessions: Mapped[List["LearningSession"]] = relationship(
        "LearningSession", back_populates="content"
    )
    recommendations: Mapped[List["Recommendation"]] = relationship(
        "Recommendation", back_populates="content"
    )

    __table_args__ = (
        CheckConstraint(
            "difficulty >= 1 AND difficulty <= 10", name="check_difficulty_range"
        ),
        CheckConstraint("duration_minutes > 0", name="check_duration_positive"),
        CheckConstraint(
            "content_type IN ('video', 'text', 'quiz', 'interactive', 'mixed')",
            name="check_content_type",
        ),
        Index("idx_content_subject_difficulty", "subject", "difficulty"),
    )

    def __repr__(self) -> str:
        return f"<LearningContent(id={self.id}, title={self.title}, difficulty={self.difficulty})>"


class LearningSession(Base):
    """
    Learning session model for tracking user learning activities.
    """

    __tablename__ = "learning_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False, index=True
    )
    content_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("learning_content.content_id"),
        nullable=False,
        index=True,
    )
    eeg_session_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("eeg_sessions.id")
    )

    # Session timing
    start_time: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime)
    duration_seconds: Mapped[Optional[int]] = mapped_column(Integer)

    # Learning progress and performance
    completed: Mapped[bool] = mapped_column(Boolean, default=False)
    progress_percentage: Mapped[float] = mapped_column(Float, default=0.0)
    score: Mapped[Optional[float]] = mapped_column(Float)

    # EEG-based metrics
    avg_attention: Mapped[Optional[float]] = mapped_column(Float)
    avg_stress: Mapped[Optional[float]] = mapped_column(Float)
    cognitive_load_avg: Mapped[Optional[float]] = mapped_column(Float)

    # Feedback and notes
    user_feedback: Mapped[Optional[Dict]] = mapped_column(JSON, default=dict)
    system_feedback: Mapped[Optional[Dict]] = mapped_column(JSON, default=dict)
    notes: Mapped[Optional[str]] = mapped_column(Text)

    # Metadata
    device_info: Mapped[Optional[Dict]] = mapped_column(JSON, default=dict)
    session_metadata: Mapped[Optional[Dict]] = mapped_column(JSON, default=dict)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="learning_sessions")
    content: Mapped["LearningContent"] = relationship(
        "LearningContent", back_populates="learning_sessions"
    )
    eeg_session: Mapped[Optional["EEGSession"]] = relationship(
        "EEGSession", back_populates="learning_sessions"
    )

    __table_args__ = (
        CheckConstraint(
            "progress_percentage >= 0 AND progress_percentage <= 100",
            name="check_progress_range",
        ),
        CheckConstraint("score >= 0 AND score <= 100", name="check_score_range"),
        Index("idx_learning_sessions_user_content", "user_id", "content_id"),
        Index("idx_learning_sessions_user_time", "user_id", "start_time"),
    )

    def __repr__(self) -> str:
        return f"<LearningSession(id={self.id}, user_id={self.user_id}, content_id={self.content_id}, completed={self.completed})>"


class Recommendation(Base):
    """
    Learning recommendation model for storing AI-generated suggestions.
    """

    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False, index=True
    )
    content_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("learning_content.content_id"),
        nullable=False,
        index=True,
    )

    # Recommendation details
    recommended_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), index=True
    )
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    reasoning: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[Optional[str]] = mapped_column(Text)

    # User interaction
    viewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    accepted: Mapped[Optional[bool]] = mapped_column(Boolean)
    accepted_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    user_rating: Mapped[Optional[int]] = mapped_column(Integer)

    # Performance after recommendation
    resulting_session_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("learning_sessions.id")
    )
    performance_improvement: Mapped[Optional[float]] = mapped_column(Float)

    # Algorithm metadata
    algorithm_version: Mapped[str] = mapped_column(String(20), default="v1.0")
    features_used: Mapped[Optional[Dict]] = mapped_column(JSON, default=dict)
    recommendation_metadata: Mapped[Optional[Dict]] = mapped_column(JSON, default=dict)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="recommendations")
    content: Mapped["LearningContent"] = relationship(
        "LearningContent", back_populates="recommendations"
    )
    resulting_session: Mapped[Optional["LearningSession"]] = relationship(
        "LearningSession"
    )

    __table_args__ = (
        CheckConstraint(
            "confidence_score >= 0 AND confidence_score <= 1",
            name="check_confidence_range",
        ),
        CheckConstraint(
            "user_rating >= 1 AND user_rating <= 5", name="check_rating_range"
        ),
        Index("idx_recommendations_user_time", "user_id", "recommended_at"),
        Index("idx_recommendations_user_content", "user_id", "content_id"),
    )

    def __repr__(self) -> str:
        return f"<Recommendation(id={self.id}, user_id={self.user_id}, content_id={self.content_id}, confidence={self.confidence_score})>"


class SystemMetrics(Base):
    """
    System metrics model for monitoring and analytics.
    """

    __tablename__ = "system_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    metric_type: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # gauge, counter, histogram
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), index=True
    )

    # Additional metadata
    labels: Mapped[Optional[Dict]] = mapped_column(JSON, default=dict)
    description: Mapped[Optional[str]] = mapped_column(String(255))

    __table_args__ = (
        CheckConstraint(
            "metric_type IN ('gauge', 'counter', 'histogram', 'summary')",
            name="check_metric_type",
        ),
        Index("idx_system_metrics_name_time", "metric_name", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<SystemMetrics(name={self.metric_name}, value={self.metric_value}, type={self.metric_type})>"


# Utility functions for database operations
async def create_user(
    username: str,
    email: str,
    hashed_password: Optional[str] = None,
    full_name: Optional[str] = None,
) -> User:
    """
    Create a new user in the database.

    Args:
        username: Unique username
        email: User email address
        hashed_password: Hashed password (optional)
        full_name: User's full name (optional)

    Returns:
        Created User object
    """
    async with get_db() as db:
        user = User(
            username=username,
            email=email,
            hashed_password=hashed_password,
            full_name=full_name,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
        return user


async def get_user_by_username(username: str) -> Optional[User]:
    """
    Get user by username.

    Args:
        username: Username to search for

    Returns:
        User object or None if not found
    """
    async with get_db() as db:
        result = await db.execute(
            "SELECT * FROM users WHERE username = :username", {"username": username}
        )
        return result.scalar_one_or_none()


async def get_user_by_email(email: str) -> Optional[User]:
    """
    Get user by email.

    Args:
        email: Email address to search for

    Returns:
        User object or None if not found
    """
    async with get_db() as db:
        result = await db.execute(
            "SELECT * FROM users WHERE email = :email", {"email": email}
        )
        return result.scalar_one_or_none()


async def create_eeg_session(
    user_id: int, session_id: str, device_type: str = "unknown"
) -> EEGSession:
    """
    Create a new EEG session.

    Args:
        user_id: User ID
        session_id: Unique session identifier
        device_type: Type of EEG device

    Returns:
        Created EEGSession object
    """
    async with get_db() as db:
        session = EEGSession(
            id=session_id,
            user_id=user_id,
            session_id=session_id,
            device_type=device_type,
            start_time=func.now(),
        )
        db.add(session)
        await db.commit()
        await db.refresh(session)
        return session
