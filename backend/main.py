"""
AI-EEG Learning Platform Backend
===============================

Main FastAPI application for the AI-EEG Learning Platform.
Provides REST API endpoints and WebSocket support for real-time EEG data processing
and personalized learning recommendations.

Author: AI-EEG Learning Platform Team
License: MIT
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from api.routes import api_router
from database.connection import init_db, close_db
from eeg_processing.manager import EEGProcessingManager
from services.recommendation_service import RecommendationService
from utils.config import settings
from utils.logging_config import setup_logging

# Setup structured logging
setup_logging()
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    logger.info("Starting AI-EEG Learning Platform backend")

    # Initialize database connection
    await init_db()

    # Initialize services
    app.state.eeg_manager = EEGProcessingManager()
    app.state.recommendation_service = RecommendationService()

    logger.info("Application startup complete")

    yield

    # Shutdown cleanup
    logger.info("Shutting down AI-EEG Learning Platform backend")

    # Close database connections
    await close_db()

    # Cleanup services
    if hasattr(app.state, "eeg_manager"):
        await app.state.eeg_manager.cleanup()

    logger.info("Application shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="AI-EEG Learning Platform API",
    description="Real-time EEG-based personalized learning platform",
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware for security
if not settings.DEBUG:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.ALLOWED_HOSTS,
    )


@app.get("/health")
async def health_check():
    """
    Health check endpoint for load balancers and monitoring.
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
        "database": "connected",  # TODO: Add actual DB health check
    }


@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "Welcome to AI-EEG Learning Platform API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


# WebSocket endpoint for real-time EEG data
@app.websocket("/ws/eeg/{user_id}")
async def eeg_websocket_endpoint(websocket: WebSocket, user_id: str):
    """
    WebSocket endpoint for real-time EEG data streaming and processing.

    Args:
        websocket: WebSocket connection
        user_id: Unique identifier for the user
    """
    await websocket.accept()
    logger.info("EEG WebSocket connection established", user_id=user_id)

    eeg_manager = app.state.eeg_manager
    recommendation_service = app.state.recommendation_service

    try:
        # Initialize user session
        session_id = await eeg_manager.create_session(user_id)

        while True:
            # Receive EEG data from client
            raw_data = await websocket.receive_text()
            eeg_sample = parse_eeg_data(raw_data)

            # Process EEG data in real-time
            processed_features = await eeg_manager.process_eeg_data(
                user_id=user_id, session_id=session_id, eeg_data=eeg_sample
            )

            # Generate learning recommendations based on EEG features
            recommendations = await recommendation_service.generate_recommendations(
                user_id=user_id, eeg_features=processed_features, context={}
            )

            # Send processed data and recommendations back to client
            response = {
                "session_id": session_id,
                "timestamp": processed_features["timestamp"],
                "eeg_features": {
                    "attention_score": processed_features["attention_score"],
                    "stress_level": processed_features["stress_level"],
                    "cognitive_load": processed_features["cognitive_load"],
                    "band_powers": processed_features["band_powers"],
                },
                "recommendations": recommendations,
                "processing_latency_ms": processed_features["processing_time"],
            }

            await websocket.send_json(response)

    except WebSocketDisconnect:
        logger.info("EEG WebSocket connection closed", user_id=user_id)
        # Cleanup user session
        await eeg_manager.end_session(user_id, session_id)

    except Exception as e:
        logger.error("EEG WebSocket error", user_id=user_id, error=str(e))
        await websocket.send_json(
            {"error": "Internal server error", "message": "Failed to process EEG data"}
        )


# Include API routes
app.include_router(api_router, prefix=settings.API_V1_STR)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled errors.
    """
    logger.error("Unhandled exception", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
        },
    )


def parse_eeg_data(raw_data: str) -> dict:
    """
    Parse incoming EEG data from WebSocket message.

    Args:
        raw_data: Raw JSON string from client

    Returns:
        Parsed EEG data dictionary
    """
    import json

    try:
        return json.loads(raw_data)
    except json.JSONDecodeError:
        logger.error("Invalid EEG data format", raw_data=raw_data)
        raise ValueError("Invalid EEG data format")


if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info",
    )
