"""
MedExplain AI - Database Field-Level Encryption Layer.
Module: medexplain.crypto

AES-256 (Fernet) field-level encryption at rest for sensitive clinical lab data
stored in SQLite (test_type, raw_results, ai_summary).
"""

from __future__ import annotations

import os
from typing import Optional
from cryptography.fernet import Fernet, InvalidToken
from medexplain.logging_config import get_logger

logger = get_logger("MedExplain.Crypto")

_fernet_instance: Optional[Fernet] = None


def get_cipher() -> Fernet:
    """Return initialized Fernet cipher instance using MEDEXPLAIN_DB_KEY env var."""
    global _fernet_instance
    if _fernet_instance is not None:
        return _fernet_instance

    raw_key = os.getenv("MEDEXPLAIN_DB_KEY")
    if not raw_key or not raw_key.strip():
        logger.critical("MEDEXPLAIN_DB_KEY environment variable is not set!")
        raise RuntimeError(
            "MEDEXPLAIN_DB_KEY environment variable is required for database encryption. "
            "Generate one with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        )

    try:
        _fernet_instance = Fernet(raw_key.strip().encode("utf-8"))
        return _fernet_instance
    except Exception as exc:
        logger.critical("Failed to initialize Fernet cipher with MEDEXPLAIN_DB_KEY: %s", exc)
        raise RuntimeError(f"Invalid MEDEXPLAIN_DB_KEY for database encryption: {exc}") from exc


def reset_cipher() -> None:
    """Reset cipher singleton (useful for testing key changes)."""
    global _fernet_instance
    _fernet_instance = None


def encrypt_field(plaintext: Optional[str]) -> Optional[str]:
    """Encrypt a plaintext string field into Fernet ciphertext."""
    if plaintext is None:
        return None
    if not isinstance(plaintext, str):
        plaintext = str(plaintext)
    if not plaintext:
        return ""

    cipher = get_cipher()
    encrypted_bytes = cipher.encrypt(plaintext.encode("utf-8"))
    return encrypted_bytes.decode("utf-8")


def decrypt_field(ciphertext: Optional[str]) -> Optional[str]:
    """
    Decrypt a Fernet ciphertext field into plaintext.
    Only allows unencrypted fallback if the text is provably legacy plaintext
    (i.e. does NOT start with 'gAAAAA'). If text starts with 'gAAAAA' but fails
    decryption, raises a ValueError.
    """
    if ciphertext is None:
        return None
    if not isinstance(ciphertext, str) or not ciphertext:
        return ciphertext

    # Legacy unencrypted plaintext check
    if not ciphertext.startswith("gAAAAA"):
        return ciphertext

    cipher = get_cipher()
    try:
        decrypted_bytes = cipher.decrypt(ciphertext.encode("utf-8"))
        return decrypted_bytes.decode("utf-8")
    except InvalidToken as exc:
        logger.critical(
            "Decryption failed for ciphertext starting with 'gAAAAA'. "
            "Key mismatch, corrupted data, or wrong MEDEXPLAIN_DB_KEY!"
        )
        raise ValueError("Decryption failed: invalid key or corrupted ciphertext token.") from exc
    except Exception as exc:
        logger.critical("Unexpected error during field decryption: %s", exc)
        raise RuntimeError(f"Field decryption error: {exc}") from exc


def is_encrypted(text: Optional[str]) -> bool:
    """Check if a string is a valid Fernet ciphertext."""
    if not text or not isinstance(text, str):
        return False
    if not text.startswith("gAAAAA"):
        return False
    try:
        cipher = get_cipher()
        cipher.decrypt(text.encode("utf-8"))
        return True
    except Exception:
        return False
