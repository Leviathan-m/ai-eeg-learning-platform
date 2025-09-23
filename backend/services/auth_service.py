"""
Authentication Service

Handles user authentication, JWT token management, and authorization.

Author: AI-EEG Learning Platform Team
"""

import os
from datetime import datetime, timedelta
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession

from database.connection import get_db
from database.models import User
from utils.config import settings
from utils.logging_config import get_request_logger

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token scheme
security = HTTPBearer()

logger = get_request_logger("auth_service")


def get_password_hash(password: str) -> str:
    """
    Hash a password using bcrypt.

    Args:
        password: Plain text password

    Returns:
        Hashed password
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.

    Args:
        plain_password: Plain text password
        hashed_password: Hashed password

    Returns:
        True if password matches hash
    """
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.

    Args:
        data: Data to encode in token
        expires_delta: Token expiration time

    Returns:
        JWT token string
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode.update({"exp": expire, "iat": datetime.utcnow()})

    encoded_jwt = jwt.encode(
        to_encode, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM
    )

    return encoded_jwt


def verify_token(token: str) -> Optional[dict]:
    """
    Verify and decode a JWT token.

    Args:
        token: JWT token to verify

    Returns:
        Decoded token payload or None if invalid
    """
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("Token expired")
        return None
    except jwt.JWTError as e:
        logger.warning("Token validation failed", error=str(e))
        return None


async def authenticate_user(
    db: AsyncSession, username: str, password: str
) -> Optional[User]:
    """
    Authenticate a user with username/email and password.

    Args:
        db: Database session
        username: Username or email
        password: Password

    Returns:
        User object if authentication successful, None otherwise
    """
    try:
        # Try to find user by username first, then by email
        result = await db.execute(
            """
            SELECT * FROM users WHERE username = :username OR email = :username
        """,
            {"username": username},
        )

        user = result.first()

        if not user:
            logger.warning("User not found", username=username)
            return None

        if not verify_password(password, user.hashed_password):
            logger.warning("Invalid password", username=username)
            return None

        if not user.is_active:
            logger.warning("Inactive user attempted login", username=username)
            return None

        return user

    except Exception as e:
        logger.error("Authentication error", username=username, error=str(e))
        return None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Get current authenticated user from JWT token.

    Args:
        credentials: HTTP Bearer token credentials
        db: Database session

    Returns:
        Authenticated user

    Raises:
        HTTPException: If token is invalid or user not found
    """
    try:
        payload = verify_token(credentials.credentials)

        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        username: str = payload.get("sub")
        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Get user from database
        result = await db.execute(
            """
            SELECT * FROM users WHERE username = :username
        """,
            {"username": username},
        )

        user = result.first()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user"
            )

        return user

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Token validation error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Get current active user (alias for get_current_user).

    Args:
        current_user: Authenticated user

    Returns:
        Active user
    """
    return current_user


async def get_current_superuser(current_user: User = Depends(get_current_user)) -> User:
    """
    Get current superuser.

    Args:
        current_user: Authenticated user

    Returns:
        Superuser

    Raises:
        HTTPException: If user is not a superuser
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
        )
    return current_user


def create_user_token(user: User) -> str:
    """
    Create JWT token for a user.

    Args:
        user: User object

    Returns:
        JWT token string
    """
    return create_access_token(data={"sub": user.username})


async def validate_token_and_get_user(token: str, db: AsyncSession) -> Optional[User]:
    """
    Validate token and return user if valid.

    Args:
        token: JWT token
        db: Database session

    Returns:
        User object if token is valid, None otherwise
    """
    try:
        payload = verify_token(token)
        if not payload:
            return None

        username = payload.get("sub")
        if not username:
            return None

        result = await db.execute(
            """
            SELECT * FROM users WHERE username = :username
        """,
            {"username": username},
        )

        user = result.first()
        return user if user and user.is_active else None

    except Exception as e:
        logger.error("Token validation error", error=str(e))
        return None


def hash_password(password: str) -> str:
    """
    Hash password (alias for get_password_hash).

    Args:
        password: Plain text password

    Returns:
        Hashed password
    """
    return get_password_hash(password)


def check_password_strength(password: str) -> bool:
    """
    Check if password meets strength requirements.

    Args:
        password: Password to check

    Returns:
        True if password is strong enough
    """
    if len(password) < 8:
        return False

    # Check for at least one uppercase, one lowercase, one digit
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)

    return has_upper and has_lower and has_digit


async def update_user_password(
    db: AsyncSession, user_id: int, new_password: str
) -> bool:
    """
    Update user's password.

    Args:
        db: Database session
        user_id: User ID
        new_password: New password

    Returns:
        True if password updated successfully
    """
    try:
        hashed_password = get_password_hash(new_password)

        await db.execute(
            """
            UPDATE users
            SET hashed_password = :hashed_password, updated_at = CURRENT_TIMESTAMP
            WHERE id = :user_id
        """,
            {"hashed_password": hashed_password, "user_id": user_id},
        )

        await db.commit()

        logger.info("Password updated", user_id=user_id)
        return True

    except Exception as e:
        logger.error("Password update failed", user_id=user_id, error=str(e))
        return False
