"""
MedExplain AI - Admin CLI Tool for Provisioning Lab Assistant Accounts.
Script: scripts/create_lab_assistant.py

Usage:
    python scripts/create_lab_assistant.py --email lab.staff@hospital.org --name "Dr. Sarah Jenkins" [--password CustomPassword123!]

This script runs with local server access to provision lab assistant accounts,
enforcing that lab assistant privileges cannot be self-requested via public signup APIs.
"""

from __future__ import annotations

import argparse
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


def _generate_secure_password(length: int = 16) -> str:
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def create_lab_assistant_account(email: str, full_name: str, password: str | None = None) -> None:
    init_db()

    email_clean = email.strip().lower()
    full_name_clean = full_name.strip()

    if get_user_by_email(email_clean):
        print(f"[ERROR] User with email '{email_clean}' already exists in database.", file=sys.stderr)
        sys.exit(1)

    generated_pass = False
    if not password:
        password = _generate_secure_password()
        generated_pass = True

    if len(password) < 8:
        print("[ERROR] Password must be at least 8 characters long.", file=sys.stderr)
        sys.exit(1)

    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")

    user = create_user(
        email=email_clean,
        password_hash=hashed,
        role="lab_assistant",
        full_name=full_name_clean,
    )

    print("\n============================================================")
    print(" MedExplain AI - Admin Lab Assistant Provisioning Successful")
    print("============================================================")
    print(f"  User ID:   {user['id']}")
    print(f"  Role:      {user['role'].upper()}")
    print(f"  Name:      {user['full_name']}")
    print(f"  Email:     {user['email']}")
    print(f"  Password:  {'[GENERATED] ' + password if generated_pass else '[USER PROVIDED]'}")
    print("============================================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Admin CLI tool to create a Lab Assistant account.")
    parser.add_argument("--email", required=True, help="Lab assistant email address")
    parser.add_argument("--name", required=True, help="Lab assistant full name")
    parser.add_argument("--password", required=False, help="Optional initial password (randomly generated if omitted)")

    args = parser.parse_args()
    create_lab_assistant_account(email=args.email, full_name=args.name, password=args.password)


if __name__ == "__main__":
    main()
