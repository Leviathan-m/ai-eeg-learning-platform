"""
API Routes

Main API router that combines all endpoint routers for the AI-EEG Learning Platform.

Author: AI-EEG Learning Platform Team
"""

from fastapi import APIRouter

from .v1.endpoints import analytics, eeg, learning, recommendations, system, users

# Main API router
api_router = APIRouter()

# Include all endpoint routers with versioning
api_router.include_router(users.router, prefix="/users", tags=["users"])

api_router.include_router(eeg.router, prefix="/eeg", tags=["eeg"])

api_router.include_router(
    recommendations.router, prefix="/recommendations", tags=["recommendations"]
)

api_router.include_router(learning.router, prefix="/learning", tags=["learning"])

api_router.include_router(analytics.router, prefix="/analytics", tags=["analytics"])

api_router.include_router(system.router, prefix="/system", tags=["system"])
