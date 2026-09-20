"""
Tests for database field-level encryption (AES-256 / Fernet) and decryption.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import unittest
from pathlib import Path
from cryptography.fernet import Fernet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure a test key is available in os.environ for test setup
if not os.getenv("MEDEXPLAIN_DB_KEY"):
    os.environ["MEDEXPLAIN_DB_KEY"] = Fernet.generate_key().decode()

from medexplain.crypto import (
    encrypt_field,
    decrypt_field,
    is_encrypted,
    get_cipher,
    reset_cipher,
)
from medexplain.db import decrypt_report_dict


class TestFieldEncryption(unittest.TestCase):
    def setUp(self) -> None:
        self.original_key = os.getenv("MEDEXPLAIN_DB_KEY")
        if not self.original_key:
            self.original_key = Fernet.generate_key().decode()
            os.environ["MEDEXPLAIN_DB_KEY"] = self.original_key
        reset_cipher()

    def tearDown(self) -> None:
        if self.original_key:
            os.environ["MEDEXPLAIN_DB_KEY"] = self.original_key
        reset_cipher()

    def test_missing_env_key_raises_runtime_error(self) -> None:
        """get_cipher() must raise RuntimeError if MEDEXPLAIN_DB_KEY is missing or empty."""
        if "MEDEXPLAIN_DB_KEY" in os.environ:
            del os.environ["MEDEXPLAIN_DB_KEY"]
        reset_cipher()

        with self.assertRaises(RuntimeError) as ctx:
            get_cipher()
        self.assertIn("MEDEXPLAIN_DB_KEY environment variable is required", str(ctx.exception))

    def test_encrypt_and_decrypt_field(self) -> None:
        plaintext = "Comprehensive Metabolic Panel"
        ciphertext = encrypt_field(plaintext)

        self.assertIsNotNone(ciphertext)
        self.assertNotEqual(plaintext, ciphertext)
        self.assertTrue(ciphertext.startswith("gAAAAA"))
        self.assertTrue(is_encrypted(ciphertext))

        decrypted = decrypt_field(ciphertext)
        self.assertEqual(plaintext, decrypted)

    def test_decrypt_handles_legacy_unencrypted_text(self) -> None:
        legacy = "Plaintext Legacy Result"
        self.assertFalse(is_encrypted(legacy))
        self.assertEqual(decrypt_field(legacy), legacy)

    def test_corrupt_ciphertext_raises_value_error(self) -> None:
        """decrypt_field must raise ValueError when decrypting invalid or tampered gAAAAA ciphertext."""
        corrupted_ciphertext = "gAAAAABinvalid_corrupted_token_data"
        with self.assertRaises(ValueError) as ctx:
            decrypt_field(corrupted_ciphertext)
        self.assertIn("Decryption failed", str(ctx.exception))

    def test_raw_database_ciphertext_assertion(self) -> None:
        """Write encrypted lab report to SQLite, assert raw SQL row is ciphertext, and verify decrypted readback."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE lab_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                trace_id TEXT UNIQUE NOT NULL,
                test_type TEXT NOT NULL,
                raw_results TEXT NOT NULL,
                ai_summary TEXT,
                uploaded_by INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        sensitive_test_type = "Lipid Panel & Fasting Glucose"
        sensitive_raw_results = json.dumps({
            "items": [
                {"test_name": "Glucose", "value": 115.0, "unit": "mg/dL"},
                {"test_name": "LDL Cholesterol", "value": 145.0, "unit": "mg/dL"}
            ]
        })
        sensitive_summary = json.dumps({"summary": "Glucose is slightly elevated at 115 mg/dL."})

        # Encrypt sensitive fields before database insertion
        enc_test_type = encrypt_field(sensitive_test_type)
        enc_raw_results = encrypt_field(sensitive_raw_results)
        enc_summary = encrypt_field(sensitive_summary)

        cursor.execute(
            """
            INSERT INTO lab_reports (patient_id, trace_id, test_type, raw_results, ai_summary, uploaded_by)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (101, "trace_crypto_test_001", enc_test_type, enc_raw_results, enc_summary, 1),
        )
        conn.commit()

        # 1. Assert raw database row content is encrypted ciphertext (NOT plaintext)
        cursor.execute("SELECT test_type, raw_results, ai_summary FROM lab_reports WHERE trace_id = 'trace_crypto_test_001'")
        raw_row = cursor.fetchone()

        self.assertNotIn("Glucose", raw_row["test_type"])
        self.assertNotIn("Lipid Panel", raw_row["test_type"])
        self.assertTrue(raw_row["test_type"].startswith("gAAAAA"))

        self.assertNotIn("Glucose", raw_row["raw_results"])
        self.assertNotIn("115.0", raw_row["raw_results"])
        self.assertTrue(raw_row["raw_results"].startswith("gAAAAA"))

        self.assertNotIn("elevated", raw_row["ai_summary"])
        self.assertTrue(raw_row["ai_summary"].startswith("gAAAAA"))

        # 2. Assert application readback layer correctly decrypts fields
        decrypted_row = decrypt_report_dict(dict(raw_row))

        self.assertEqual(decrypted_row["test_type"], sensitive_test_type)
        self.assertEqual(decrypted_row["raw_results"], sensitive_raw_results)
        self.assertEqual(decrypted_row["ai_summary"], sensitive_summary)

        conn.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
