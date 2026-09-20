"""
MedExplain AI - Role-Based Authentication & Security Layer.
Module: medexplain.auth
"""

import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from medexplain.db import get_user_by_email, get_user_by_id
from medexplain.logging_config import audit, get_logger

logger = get_logger("MedExplain.Auth")

JWT_SECRET = os.getenv("JWT_SECRET", "medexplain-secure-jwt-secret-key-2026")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Roles definition
VALID_ROLES = {"patient", "lab_assistant"}

# Rate limit configuration: Max 5 failed attempts per 15 minutes (900s)
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_DURATION_SECONDS = 900

# In-memory store for login rate-limiting: key -> List[float (timestamps)]
_failed_login_attempts: Dict[str, List[float]] = {}

security_bearer = HTTPBearer()

# --- Password Hashing with Bcrypt ---

def hash_password(password: str) -> str:
    """Hash password securely using bcrypt."""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


# --- JWT Token Operations ---

def create_access_token(user_id: int, email: str, role: str, full_name: str) -> str:
    """Create a signed JWT access token containing embedded user claims and role."""
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": str(user_id),
        "email": email.lower(),
        "role": role,
        "full_name": full_name,
        "iat": now,
        "exp": expire,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decode and validate a JWT access token."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# --- Input Validation & Sanitization ---

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


def validate_signup_input(email: str, password: str, role: str, full_name: str) -> None:
    """Validate signup input fields against strict format and security rules."""
    # 1. Email format check
    if not email or not EMAIL_REGEX.match(email.strip()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid email format.",
        )

    # 2. Server-side role validation & locking
    role_clean = role.strip().lower() if role else ""
    if role_clean not in VALID_ROLES:
        audit("auth.invalid_role_attempt", attempted_role=role)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role. Role must be one of: {', '.join(sorted(VALID_ROLES))}",
        )

    # 3. Name check
    if not full_name or len(full_name.strip()) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Full name must be at least 2 characters.",
        )

    # 4. Password strength validation
    validate_password_strength(password)


def validate_password_strength(password: str) -> None:
    """Enforce password strength policy: >=8 chars, uppercase, lowercase, digit, special char."""
    if len(password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters long.",
        )
    if not re.search(r"[A-Z]", password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must contain at least one uppercase letter.",
        )
    if not re.search(r"[a-z]", password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must contain at least one lowercase letter.",
        )
    if not re.search(r"[0-9]", password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must contain at least one digit.",
        )
    if not re.search(r"[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]", password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must contain at least one special character.",
        )


def sanitize_input(text: str) -> str:
    """Sanitize text input to prevent injection payloads."""
    if not text:
        return ""
    # Strip HTML tags & null bytes
    cleaned = re.sub(r"<[^>]*>", "", text)
    cleaned = cleaned.replace("\0", "")
    return cleaned.strip()


# --- Rate Limiting on Login Attempts ---

def is_rate_limited(key: str) -> bool:
    """Check if key (IP/account) has exceeded max allowed failed attempts in window."""
    now = time.time()
    attempts = _failed_login_attempts.get(key, [])
    # Filter attempts within the 15-minute window
    valid_attempts = [ts for ts in attempts if now - ts < LOCKOUT_DURATION_SECONDS]
    _failed_login_attempts[key] = valid_attempts
    return len(valid_attempts) >= MAX_FAILED_ATTEMPTS


def record_failed_login(key: str) -> None:
    """Record a failed login attempt timestamp."""
    now = time.time()
    if key not in _failed_login_attempts:
        _failed_login_attempts[key] = []
    _failed_login_attempts[key].append(now)
    audit("auth.failed_login_attempt", key=key, count=len(_failed_login_attempts[key]))


def reset_failed_logins(key: str) -> None:
    """Clear failed login attempts for a key upon successful login."""
    if key in _failed_login_attempts:
        del _failed_login_attempts[key]


# --- Role-Based Access Control (RBAC) Middleware & Dependencies ---

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security_bearer),
) -> Dict[str, Any]:
    """Dependency to extract and verify JWT user from HTTP Bearer header."""
    token = credentials.credentials
    payload = decode_access_token(token)
    user_id = int(payload.get("sub", 0))

    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User no longer exists.",
        )
    return user


def require_role(required_role: str) -> Callable:
    """Factory dependency restricting route access to a specific user role."""
    def role_checker(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        user_role = current_user.get("role")
        if user_role != required_role:
            audit(
                "auth.forbidden_role_access",
                user_id=current_user.get("id"),
                user_role=user_role,
                required_role=required_role,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access forbidden: requires '{required_role}' role.",
            )
        return current_user

    return role_checker


# Short-hand dependencies
require_patient = require_role("patient")
require_lab_assistant = require_role("lab_assistant")
