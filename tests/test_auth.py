"""
MedExplain AI - Unit & Integration Tests for Role-Based Auth & RBAC Middleware.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Use in-memory DB for tests
os.environ["JWT_SECRET"] = "test-jwt-secret-key-12345"

from app import app
from medexplain.auth import _failed_login_attempts
from medexplain.db import get_user_by_email, init_db

TEST_DB_PATH = str(PROJECT_ROOT / "data" / "test_auth_medexplain.db")


class TestRoleBasedAuth(unittest.TestCase):
    def setUp(self) -> None:
        if os.path.exists(TEST_DB_PATH):
            try:
                os.remove(TEST_DB_PATH)
            except Exception:
                pass
        init_db(TEST_DB_PATH)
        _failed_login_attempts.clear()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        if os.path.exists(TEST_DB_PATH):
            try:
                os.remove(TEST_DB_PATH)
            except Exception:
                pass

    def test_user_signup_and_bcrypt_hashing(self) -> None:
        """Test signup creates user with bcrypt hashed password, not plaintext."""
        payload = {
            "email": "patient1@example.com",
            "password": "Password123!",
            "role": "patient",
            "full_name": "Jane Patient",
        }
        res = self.client.post("/api/v1/auth/signup", json=payload)
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertIn("access_token", data)
        self.assertEqual(data["user"]["role"], "patient")

        # Verify DB password is bcrypt hash, not plaintext
        user = get_user_by_email("patient1@example.com")
        self.assertIsNotNone(user)
        self.assertTrue(user["password_hash"].startswith("$2b$"))
        self.assertNotIn("Password123!", user["password_hash"])

    def test_server_side_role_locking(self) -> None:
        """Public signup endpoint must safely lock all self-registrations to 'patient' role."""
        for invalid_role in ["admin", "superuser", "doctor", "root", "lab_assistant"]:
            payload = {
                "email": f"hacker_{invalid_role}@example.com",
                "password": "Password123!",
                "role": invalid_role,
                "full_name": "Hacker User",
            }
            res = self.client.post("/api/v1/auth/signup", json=payload)
            self.assertEqual(res.status_code, 201)
            # Enforce that elevated role was neutralized and user was created as patient
            self.assertEqual(res.json()["user"]["role"], "patient")
            user = get_user_by_email(f"hacker_{invalid_role}@example.com")
            self.assertIsNotNone(user)
            self.assertEqual(user["role"], "patient")

    def test_input_validation_email_and_password_strength(self) -> None:
        """Enforce strict email and password complexity rules."""
        # Invalid email
        res = self.client.post(
            "/api/v1/auth/signup",
            json={"email": "invalid-email", "password": "Password123!", "role": "patient", "full_name": "Test"},
        )
        self.assertEqual(res.status_code, 400)

        # Weak password (no uppercase)
        res = self.client.post(
            "/api/v1/auth/signup",
            json={"email": "weak@example.com", "password": "password123!", "role": "patient", "full_name": "Test"},
        )
        self.assertEqual(res.status_code, 400)

        # Weak password (no special char)
        res = self.client.post(
            "/api/v1/auth/signup",
            json={"email": "weak2@example.com", "password": "Password123", "role": "patient", "full_name": "Test"},
        )
        self.assertEqual(res.status_code, 400)

    def test_generic_login_error_message(self) -> None:
        """Failed login returns generic message without revealing user existence."""
        # Non-existent user
        res1 = self.client.post(
            "/api/v1/auth/login",
            json={"email": "nonexistent@example.com", "password": "Password123!"},
        )
        self.assertEqual(res1.status_code, 401)
        self.assertEqual(res1.json()["detail"], "Invalid email or password.")

        # Signup valid user
        self.client.post(
            "/api/v1/auth/signup",
            json={"email": "user1@example.com", "password": "Password123!", "role": "patient", "full_name": "User One"},
        )

        # Wrong password for existing user
        res2 = self.client.post(
            "/api/v1/auth/login",
            json={"email": "user1@example.com", "password": "WrongPassword123!"},
        )
        self.assertEqual(res2.status_code, 401)
        self.assertEqual(res2.json()["detail"], "Invalid email or password.")

    def test_login_rate_limiting_brute_force_protection(self) -> None:
        """5 failed login attempts trigger 429 Too Many Requests on the 6th attempt."""
        email = "target@example.com"
        # 5 failed attempts
        for i in range(5):
            res = self.client.post(
                "/api/v1/auth/login",
                json={"email": email, "password": f"WrongPwd{i}!"},
            )
            self.assertEqual(res.status_code, 401)

        # 6th attempt should be blocked by rate limiter
        res_blocked = self.client.post(
            "/api/v1/auth/login",
            json={"email": email, "password": "WrongPwd6!"},
        )
        self.assertEqual(res_blocked.status_code, 429)
        self.assertIn("Too many failed login attempts", res_blocked.json()["detail"])

    def test_public_signup_role_override_prevents_privilege_escalation(self) -> None:
        """
        Public self-registration via /api/v1/auth/signup must ALWAYS assign role='patient',
        even if a client submits role='lab_assistant' or any other privilege escalation attempt.
        """
        payload = {
            "email": "attacker@example.com",
            "password": "Password123!",
            "role": "lab_assistant",
            "full_name": "Attacker User",
        }
        res = self.client.post("/api/v1/auth/signup", json=payload)
        self.assertEqual(res.status_code, 201)
        data = res.json()

        # Returned token user claim must be patient
        self.assertEqual(data["user"]["role"], "patient")

        # Database record must be patient
        user = get_user_by_email("attacker@example.com")
        self.assertIsNotNone(user)
        self.assertEqual(user["role"], "patient")

    def test_role_based_access_control_and_isolation(self) -> None:
        """
        Verify RBAC permissions:
        - Patients can access /patient/* routes and only their own records.
        - Lab assistants can access /lab/* routes without access to AI summaries.
        - Patients cannot access /lab/* and Lab Assistants cannot access /patient/*.
        """
        from medexplain.auth import create_access_token, hash_password
        from medexplain.db import create_user

        # Register patient 1
        p1_res = self.client.post(
            "/api/v1/auth/signup",
            json={"email": "p1@example.com", "password": "Password123!", "role": "patient", "full_name": "Patient One"},
        ).json()
        p1_token = p1_res["access_token"]
        p1_headers = {"Authorization": f"Bearer {p1_token}"}

        # Register patient 2
        p2_res = self.client.post(
            "/api/v1/auth/signup",
            json={"email": "p2@example.com", "password": "Password123!", "role": "patient", "full_name": "Patient Two"},
        ).json()
        p2_token = p2_res["access_token"]
        p2_headers = {"Authorization": f"Bearer {p2_token}"}

        # Provision lab assistant via direct DB/admin function (simulating admin script creation)
        lab_user = create_user("lab1@example.com", hash_password("Password123!"), "lab_assistant", "Tech Alex")
        lab_token = create_access_token(lab_user["id"], lab_user["email"], lab_user["role"], lab_user["full_name"])
        lab_headers = {"Authorization": f"Bearer {lab_token}"}

        # 1. Patient accessing /patient/my-reports -> OK
        res = self.client.get("/patient/my-reports", headers=p1_headers)
        self.assertEqual(res.status_code, 200)

        # 2. Patient attempting to access /lab/manage-tests -> Forbidden 403
        res = self.client.get("/lab/manage-tests", headers=p1_headers)
        self.assertEqual(res.status_code, 403)
        self.assertIn("requires 'lab_assistant' role", res.json()["detail"])

        # 3. Lab assistant uploading lab result for Patient 1 -> OK
        upload_payload = {
            "patient_id": p1_res["user"]["id"],
            "test_type": "CBC",
            "raw_results": {"WBC": "11.8", "RBC": "4.5"},
        }
        res_upload = self.client.post("/lab/upload-result", json=upload_payload, headers=lab_headers)
        self.assertEqual(res_upload.status_code, 201)
        report_id = res_upload.json()["report_id"]

        # 4. Patient 1 accessing their own report -> OK
        res = self.client.get(f"/patient/reports/{report_id}", headers=p1_headers)
        self.assertEqual(res.status_code, 200)

        # 5. Patient 2 attempting to access Patient 1's report -> Forbidden 403
        res = self.client.get(f"/patient/reports/{report_id}", headers=p2_headers)
        self.assertEqual(res.status_code, 403)
        self.assertIn("belonging to another patient", res.json()["detail"])

        # 6. Lab assistant managing tests -> OK, but AI summary is excluded
        res = self.client.get("/lab/manage-tests", headers=lab_headers)
        self.assertEqual(res.status_code, 200)
        managed = res.json()["managed_tests"]
        self.assertTrue(len(managed) > 0)
        for item in managed:
            self.assertNotIn("ai_summary", item, "Lab Assistant must not have access to AI patient summaries")

        # 7. Lab assistant attempting to access /patient/my-reports -> Forbidden 403
        res = self.client.get("/patient/my-reports", headers=lab_headers)
        self.assertEqual(res.status_code, 403)


if __name__ == "__main__":
    unittest.main(verbosity=2)
