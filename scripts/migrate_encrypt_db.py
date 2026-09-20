"""
MedExplain AI - Database Field Encryption Migration & Re-keying Script.
Module: scripts.migrate_encrypt_db

Reads lab_reports rows in data/medexplain.db, decrypts legacy or exposed-key data,
encrypts sensitive fields (test_type, raw_results, ai_summary) using the active
MEDEXPLAIN_DB_KEY (AES-256 Fernet), and updates the database in place.
"""

from __future__ import annotations

import os
import sys
import sqlite3
from pathlib import Path
from typing import Optional
from cryptography.fernet import Fernet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from medexplain.crypto import encrypt_field, is_encrypted, get_cipher
from medexplain.logging_config import get_logger

load_dotenv()
logger = get_logger("MedExplain.Migration")

OLD_EXPOSED_KEY = "nmw5Q9R_ZHcmO4oZEd4y7-_kXu74VdUPKXMX6uFxarU="


def _decrypt_legacy_or_old_key(value: Optional[str]) -> Optional[str]:
    """Attempt to decrypt a value using current key, fallback to old key if re-keying, or return if plaintext."""
    if not value or not isinstance(value, str):
        return value
    if not value.startswith("gAAAAA"):
        return value

    # Try current key first
    try:
        cipher = get_cipher()
        return cipher.decrypt(value.encode("utf-8")).decode("utf-8")
    except Exception:
        pass

    # Try old exposed key to re-key data safely
    try:
        old_cipher = Fernet(OLD_EXPOSED_KEY.encode("utf-8"))
        return old_cipher.decrypt(value.encode("utf-8")).decode("utf-8")
    except Exception as exc:
        raise ValueError(f"Could not decrypt field with current or legacy fallback key: {exc}") from exc


def migrate_database(db_path: Optional[str] = None) -> None:
    path = db_path or str(PROJECT_ROOT / "data" / "medexplain.db")
    if not os.path.exists(path):
        print(f"Database file not found at: {path}. Nothing to migrate.")
        return

    print(f"Connecting to database: {path}")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT id, test_type, raw_results, ai_summary FROM lab_reports")
    rows = cursor.fetchall()

    if not rows:
        print("No lab_reports rows found in database.")
        conn.close()
        return

    migrated_count = 0
    skipped_count = 0

    for row in rows:
        row_id = row["id"]
        test_type = row["test_type"]
        raw_results = row["raw_results"]
        ai_summary = row["ai_summary"]

        # Check if row is already valid with CURRENT key
        if is_encrypted(test_type) and is_encrypted(raw_results) and (not ai_summary or is_encrypted(ai_summary)):
            skipped_count += 1
            continue

        # Otherwise decrypt from old key / plaintext and re-encrypt with current key
        plain_test_type = _decrypt_legacy_or_old_key(test_type)
        plain_raw_results = _decrypt_legacy_or_old_key(raw_results)
        plain_ai_summary = _decrypt_legacy_or_old_key(ai_summary) if ai_summary else None

        new_test_type = encrypt_field(plain_test_type)
        new_raw_results = encrypt_field(plain_raw_results)
        new_ai_summary = encrypt_field(plain_ai_summary) if plain_ai_summary else None

        cursor.execute(
            """
            UPDATE lab_reports
            SET test_type = ?, raw_results = ?, ai_summary = ?
            WHERE id = ?
            """,
            (new_test_type, new_raw_results, new_ai_summary, row_id),
        )
        migrated_count += 1

    conn.commit()
    conn.close()

    print(f"Migration complete! Processed {len(rows)} rows: {migrated_count} re-encrypted with fresh key, {skipped_count} skipped (already using active key).")


if __name__ == "__main__":
    db_file = sys.argv[1] if len(sys.argv) > 1 else None
    migrate_database(db_file)
