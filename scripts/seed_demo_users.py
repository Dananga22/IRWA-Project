"""
MedExplain AI - Seed Local Development Demo Users Script.
Module: scripts.seed_demo_users

Creates throwaway demo accounts (one patient, one lab assistant) with randomly
generated passwords printed to console stdout on first run. Never hardcodes or
stores passwords in source control.
"""

from __future__ import annotations

import os
import secrets
import string
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
import bcrypt
from medexplain.db import init_db, get_user_by_email, create_user

load_dotenv()


def _generate_random_password(length: int = 14) -> str:
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def seed_demo_users() -> None:
    init_db()

    demo_users = [
        {
            "email": "demo.patient@local.test",
            "role": "patient",
            "full_name": "Demo Patient Account",
        },
        {
            "email": "demo.lab@local.test",
            "role": "lab_assistant",
            "full_name": "Demo Lab Assistant",
        },
    ]

    print("============================================================")
    print(" MedExplain AI - Local Development Demo User Generator")
    print("============================================================")

    for user_info in demo_users:
        existing = get_user_by_email(user_info["email"])
        if existing:
            print(f"\n[EXISTS] User {user_info['email']} (Role: {user_info['role']}) already seeded in database.")
            continue

        password = _generate_random_password()
        hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

        created = create_user(
            email=user_info["email"],
            password_hash=hashed,
            role=user_info["role"],
            full_name=user_info["full_name"],
        )

        print(f"\n[CREATED] Role: {created['role'].upper()}")
        print(f"  Email:    {created['email']}")
        print(f"  Password: {password}")

    print("\n------------------------------------------------------------")
    print("[SECURITY NOTICE] Demo accounts created above are for local development")
    print("and testing ONLY. Never deploy these credentials to production.")
    print("============================================================\n")


if __name__ == "__main__":
    seed_demo_users()
