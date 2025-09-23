"""
EEG Processing Manager

Manages multiple EEG processing sessions, coordinates data flow between
components, and handles persistence of processed EEG data.

Author: AI-EEG Learning Platform Team
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime

import numpy as np

from .processor import EEGProcessor
from database.models import EEGSession, EEGDataPoint
from database.connection import get_db
from services.cache_service import CacheService
from utils.config import settings
from utils.logging_config import get_request_logger


class EEGProcessingManager:
    """
    Manages EEG processing sessions and coordinates data flow.
    """

    def __init__(self):
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.processors: Dict[str, EEGProcessor] = {}
        self.cache_service = CacheService()
        self.logger = get_request_logger("eeg_manager")

        # Performance monitoring
        self.session_stats = {
            "total_sessions": 0,
            "active_sessions": 0,
            "total_data_points": 0,
            "avg_processing_time": 0.0,
        }

    async def create_session(self, user_id: str) -> str:
        """
        Create a new EEG processing session for a user.

        Args:
            user_id: Unique identifier for the user

        Returns:
            Session ID for the created session
        """
        session_id = str(uuid.uuid4())

        # Create EEG processor for this session
        processor = EEGProcessor(
            sampling_rate=settings.EEG_SAMPLING_RATE,
            channels=settings.EEG_CHANNELS,
            buffer_size=settings.EEG_BUFFER_SIZE,
        )

        # Store session information
        self.active_sessions[user_id] = {
            "session_id": session_id,
            "user_id": user_id,
            "processor": processor,
            "start_time": datetime.utcnow(),
            "data_points": 0,
            "last_activity": time.time(),
            "quality_score": 0.0,
        }

        self.processors[session_id] = processor

        # Update statistics
        self.session_stats["total_sessions"] += 1
        self.session_stats["active_sessions"] += 1

        # Create database session record
        await self._create_db_session(user_id, session_id)

        self.logger.info("EEG session created", user_id=user_id, session_id=session_id)

        return session_id

    async def process_eeg_data(
        self, user_id: str, session_id: str, eeg_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process incoming EEG data for a user session.

        Args:
            user_id: User identifier
            session_id: Session identifier
            eeg_data: EEG data payload

        Returns:
            Processed EEG features
        """
        # Validate session
        if user_id not in self.active_sessions:
            raise ValueError(f"No active session for user {user_id}")

        session = self.active_sessions[user_id]
        if session["session_id"] != session_id:
            raise ValueError(f"Session ID mismatch for user {user_id}")

        # Get processor
        processor = session["processor"]

        try:
            # Extract sample data
            if "channels" in eeg_data:
                sample = np.array(eeg_data["channels"], dtype=np.float64)
            elif "sample" in eeg_data:
                sample = np.array(eeg_data["sample"], dtype=np.float64)
            else:
                raise ValueError("EEG data must contain 'channels' or 'sample' field")

            # Process the sample
            features = await processor.process_sample(sample)

            # Update session statistics
            session["data_points"] += 1
            session["last_activity"] = time.time()
            session["quality_score"] = features.get("signal_quality", 0.0)

            # Cache recent features for quick access
            cache_key = f"eeg_features:{user_id}:{session_id}"
            await self.cache_service.set(
                cache_key, features, ttl=settings.REDIS_CACHE_TTL
            )

            # Asynchronously save to database (don't block real-time processing)
            asyncio.create_task(
                self._save_data_point(user_id, session_id, eeg_data, features)
            )

            # Update global statistics
            self.session_stats["total_data_points"] += 1

            return features

        except Exception as e:
            self.logger.error(
                "EEG data processing failed",
                user_id=user_id,
                session_id=session_id,
                error=str(e),
            )
            raise

    async def end_session(self, user_id: str, session_id: str) -> None:
        """
        End an EEG processing session.

        Args:
            user_id: User identifier
            session_id: Session identifier
        """
        if user_id not in self.active_sessions:
            self.logger.warning(
                "Attempted to end non-existent session",
                user_id=user_id,
                session_id=session_id,
            )
            return

        session = self.active_sessions[user_id]

        # Update session end time in database
        await self._end_db_session(user_id, session_id)

        # Clean up resources
        processor = session["processor"]
        performance_stats = processor.get_performance_stats()

        # Update global statistics
        self.session_stats["active_sessions"] -= 1
        if self.session_stats["active_sessions"] > 0:
            self.session_stats["avg_processing_time"] = (
                self.session_stats["avg_processing_time"]
                + performance_stats["avg_processing_time"]
            ) / 2

        # Remove from active sessions
        del self.active_sessions[user_id]
        del self.processors[session_id]

        self.logger.info(
            "EEG session ended",
            user_id=user_id,
            session_id=session_id,
            data_points=session["data_points"],
            duration=time.time() - session["last_activity"],
        )

    async def get_session_status(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a user's active session.

        Args:
            user_id: User identifier

        Returns:
            Session status information or None if no active session
        """
        if user_id not in self.active_sessions:
            return None

        session = self.active_sessions[user_id]
        processor = session["processor"]

        return {
            "session_id": session["session_id"],
            "user_id": user_id,
            "start_time": session["start_time"].isoformat(),
            "data_points": session["data_points"],
            "quality_score": session["quality_score"],
            "last_activity": session["last_activity"],
            "performance_stats": processor.get_performance_stats(),
        }

    async def get_recent_features(
        self, user_id: str, session_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recent EEG features for a session.

        Args:
            user_id: User identifier
            session_id: Session identifier
            limit: Maximum number of features to return

        Returns:
            List of recent EEG features
        """
        cache_key = f"eeg_features:{user_id}:{session_id}"

        # Try cache first
        cached_features = await self.cache_service.get(cache_key)
        if cached_features:
            return [cached_features]

        # Fallback to database
        async with get_db() as db:
            query = """
                SELECT processed_features, timestamp
                FROM eeg_data_points
                WHERE user_id = $1 AND session_id = $2
                ORDER BY timestamp DESC
                LIMIT $3
            """
            rows = await db.fetch(query, user_id, session_id, limit)

            features = []
            for row in rows:
                feature_data = json.loads(row["processed_features"])
                feature_data["timestamp"] = row["timestamp"].isoformat()
                features.append(feature_data)

            return features

    async def _create_db_session(self, user_id: str, session_id: str) -> None:
        """
        Create database record for EEG session.

        Args:
            user_id: User identifier
            session_id: Session identifier
        """
        async with get_db() as db:
            await db.execute(
                """
                INSERT INTO eeg_sessions (
                    id, user_id, session_id, start_time, device_type, status
                ) VALUES ($1, $2, $3, $4, $5, $6)
            """,
                session_id,
                user_id,
                session_id,
                datetime.utcnow(),
                "unknown",  # Will be updated when device info is available
                "active",
            )

    async def _end_db_session(self, user_id: str, session_id: str) -> None:
        """
        Update database record to mark session as ended.

        Args:
            user_id: User identifier
            session_id: Session identifier
        """
        async with get_db() as db:
            await db.execute(
                """
                UPDATE eeg_sessions
                SET end_time = $1, status = $2
                WHERE user_id = $3 AND session_id = $4
            """,
                datetime.utcnow(),
                "completed",
                user_id,
                session_id,
            )

    async def _save_data_point(
        self,
        user_id: str,
        session_id: str,
        raw_data: Dict[str, Any],
        processed_features: Dict[str, Any],
    ) -> None:
        """
        Save EEG data point to database.

        Args:
            user_id: User identifier
            session_id: Session identifier
            raw_data: Raw EEG data
            processed_features: Processed EEG features
        """
        try:
            async with get_db() as db:
                await db.execute(
                    """
                    INSERT INTO eeg_data_points (
                        session_id, user_id, timestamp, raw_data, processed_features
                    ) VALUES ($1, $2, $3, $4, $5)
                """,
                    session_id,
                    user_id,
                    datetime.utcnow(),
                    json.dumps(raw_data),
                    json.dumps(processed_features),
                )
        except Exception as e:
            self.logger.error(
                "Failed to save EEG data point",
                user_id=user_id,
                session_id=session_id,
                error=str(e),
            )

    async def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system-wide EEG processing statistics.

        Returns:
            System statistics dictionary
        """
        return {
            "active_sessions": len(self.active_sessions),
            "total_sessions_today": self.session_stats["total_sessions"],
            "total_data_points": self.session_stats["total_data_points"],
            "avg_processing_time": self.session_stats["avg_processing_time"],
            "memory_usage": await self._get_memory_usage(),
            "uptime": time.time(),  # Could be replaced with actual service start time
        }

    async def _get_memory_usage(self) -> float:
        """
        Get current memory usage of the EEG processing system.

        Returns:
            Memory usage in MB
        """
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # Convert to MB

    async def cleanup_inactive_sessions(self, max_inactive_time: int = 300) -> int:
        """
        Clean up sessions that have been inactive for too long.

        Args:
            max_inactive_time: Maximum allowed inactive time in seconds

        Returns:
            Number of sessions cleaned up
        """
        current_time = time.time()
        inactive_sessions = []

        for user_id, session in self.active_sessions.items():
            if current_time - session["last_activity"] > max_inactive_time:
                inactive_sessions.append((user_id, session["session_id"]))

        for user_id, session_id in inactive_sessions:
            await self.end_session(user_id, session_id)

        if inactive_sessions:
            self.logger.info(
                "Cleaned up inactive sessions", count=len(inactive_sessions)
            )

        return len(inactive_sessions)

    async def cleanup(self) -> None:
        """
        Clean up all resources and end active sessions.
        """
        self.logger.info("Starting EEG manager cleanup")

        # End all active sessions
        active_user_ids = list(self.active_sessions.keys())
        for user_id in active_user_ids:
            session = self.active_sessions[user_id]
            await self.end_session(user_id, session["session_id"])

        # Clear processors
        self.processors.clear()

        # Close cache connection
        await self.cache_service.close()

        self.logger.info("EEG manager cleanup completed")
