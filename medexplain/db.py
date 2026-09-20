"""
MedExplain AI - Database Layer for User Authentication & Role Management.
Module: medexplain.db
"""

import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

DB_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_DB_PATH = str(DB_DIR / "medexplain.db")
_active_db_path: str = DEFAULT_DB_PATH


def set_active_db_path(path: str) -> None:
    """Set active database path (useful for in-memory testing)."""
    global _active_db_path
    _active_db_path = path


def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Return a SQLite connection with dict-like row access."""
    path = db_path or _active_db_path
    if path != ":memory:":
        os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize database tables for users and patient lab reports."""
    if db_path:
        set_active_db_path(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Users table with role constraint
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT CHECK(role IN ('patient', 'lab_assistant')) NOT NULL,
            full_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Patient Lab Reports table linked to patients and lab assistants
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS lab_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            trace_id TEXT UNIQUE NOT NULL,
            test_type TEXT NOT NULL,
            raw_results TEXT NOT NULL,
            ai_summary TEXT,
            uploaded_by INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(patient_id) REFERENCES users(id),
            FOREIGN KEY(uploaded_by) REFERENCES users(id)
        );
    """)

    conn.commit()
    conn.close()


def create_user(
    email: str,
    password_hash: str,
    role: str,
    full_name: str,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Insert a new user record into the database."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO users (email, password_hash, role, full_name)
            VALUES (?, ?, ?, ?)
            """,
            (email.lower().strip(), password_hash, role, full_name.strip()),
        )
        conn.commit()
        user_id = cursor.lastrowid
        return {
            "id": user_id,
            "email": email.lower().strip(),
            "role": role,
            "full_name": full_name.strip(),
        }
    finally:
        conn.close()


def get_user_by_email(email: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve user record by email."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, email, password_hash, role, full_name, created_at FROM users WHERE email = ?",
        (email.lower().strip(),),
    )
    row = cursor.fetchone()
    conn.close()

    if row:
        return dict(row)
    return None


def get_user_by_id(user_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve user record by ID."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, email, role, full_name, created_at FROM users WHERE id = ?",
        (user_id,),
    )
    row = cursor.fetchone()
    conn.close()

    if row:
        return dict(row)
    return None


# --- Encryption Helpers for Lab Reports ---

def decrypt_report_dict(row_dict: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Decrypt sensitive columns (test_type, raw_results, ai_summary) in a lab_reports row dictionary."""
    if not row_dict:
        return row_dict

    from medexplain.crypto import decrypt_field

    out = dict(row_dict)
    if "test_type" in out and out["test_type"]:
        out["test_type"] = decrypt_field(out["test_type"])

    if "raw_results" in out and out["raw_results"]:
        out["raw_results"] = decrypt_field(out["raw_results"])

    if "ai_summary" in out and out["ai_summary"]:
        out["ai_summary"] = decrypt_field(out["ai_summary"])

    return out

