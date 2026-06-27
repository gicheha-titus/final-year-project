from __future__ import annotations

import pytest

from cbc_xai import storage


def test_create_teacher_account_and_authenticate() -> None:
    storage.initialize_database()

    created_user = storage.create_user("teacher2", "Secret12", "Teacher/Counsellor")
    authenticated = storage.authenticate_user("teacher2", "Secret12")
    users = storage.list_users()

    assert created_user["username"] == "teacher2"
    assert created_user["role"] == "Teacher/Counsellor"
    assert created_user["status"] == "Active"
    # authenticate_user returns only {username, role} — not is_active/status.
    assert authenticated == {"username": created_user["username"], "role": created_user["role"]}
    assert any(user["username"] == "teacher2" and user["role"] == "Teacher/Counsellor" for user in users)


def test_duplicate_usernames_are_rejected() -> None:
    storage.initialize_database()
    storage.create_user("teacher3", "Secret12", "Teacher/Counsellor")

    with pytest.raises(ValueError, match="already exists"):
        storage.create_user("teacher3", "Another12", "Teacher/Counsellor")


def test_reset_deactivate_reactivate_and_delete_user() -> None:
    storage.initialize_database()
    storage.create_user("teacher4", "Secret12", "Teacher/Counsellor")

    storage.reset_user_password("teacher4", "Updated34")
    assert storage.authenticate_user("teacher4", "Secret12") is None
    assert storage.authenticate_user("teacher4", "Updated34") == {
        "username": "teacher4",
        "role": "Teacher/Counsellor",
    }

    storage.set_user_active("teacher4", False)
    assert storage.authenticate_user("teacher4", "Updated34") is None
    assert any(
        user["username"] == "teacher4" and user["status"] == "Inactive"
        for user in storage.list_users()
    )

    storage.set_user_active("teacher4", True)
    assert storage.authenticate_user("teacher4", "Updated34") == {
        "username": "teacher4",
        "role": "Teacher/Counsellor",
    }

    storage.delete_user("teacher4")
    assert not any(user["username"] == "teacher4" for user in storage.list_users())


def test_cannot_delete_or_deactivate_last_admin() -> None:
    storage.initialize_database()
    
    # Try to deactivate the default admin (which is the only admin right now)
    with pytest.raises(ValueError, match="last active Admin"):
        storage.set_user_active("admin", False)
        
    # Try to delete the default admin
    with pytest.raises(ValueError, match="last active Admin"):
        storage.delete_user("admin")
        
    # Create a second admin
    storage.create_user("admin2", "Admin@1234", "Admin")
    
    # Now we can deactivate the first admin
    storage.set_user_active("admin", False)
    
    # But we can't deactivate the second admin (now the last active)
    with pytest.raises(ValueError, match="last active Admin"):
        storage.set_user_active("admin2", False)


def test_argon2_migration() -> None:
    storage.initialize_database()
    
    # Manually insert a user with a legacy SHA-256 hash
    import hashlib
    legacy_hash = hashlib.sha256("Legacy@123".encode("utf-8")).hexdigest()
    with storage.get_connection() as connection:
        connection.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            ("legacy_user", legacy_hash, "Teacher/Counsellor")
        )
        connection.commit()
        
    # Authenticate with the legacy password
    assert storage.authenticate_user("legacy_user", "Legacy@123") is not None
    
    # Verify the hash has been migrated to Argon2 in the database
    with storage.get_connection() as connection:
        row = connection.execute(
            "SELECT password_hash FROM users WHERE username = 'legacy_user'"
        ).fetchone()
        new_hash = row["password_hash"]
        
    assert new_hash.startswith("$argon2")
    assert len(new_hash) != 64
