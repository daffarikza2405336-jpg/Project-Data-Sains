from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from modules.auth import (
    authenticate_user,
    change_password,
    register_user,
    reset_password,
)
from modules.workspace import (
    create_project,
    delete_project,
    list_activities,
    list_projects,
    rename_project,
    set_project_favorite,
)


class AccountWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_patch = patch(
            "modules.database.DATABASE_PATH",
            Path(self.temp_dir.name) / "test.sqlite3",
        )
        self.database_patch.start()
        self.recovery_code = register_user("alice", "VeryStrongPassword12")
        register_user("bob", "AnotherStrongPassword12")

    def tearDown(self) -> None:
        self.database_patch.stop()
        self.temp_dir.cleanup()

    def test_password_authentication_and_recovery(self) -> None:
        self.assertEqual(
            authenticate_user("ALICE", "VeryStrongPassword12"),
            "alice",
        )
        self.assertIsNone(authenticate_user("alice", "incorrect-password"))
        new_recovery_code = reset_password(
            "alice",
            self.recovery_code,
            "UpdatedStrongPassword12",
        )
        self.assertIsNone(authenticate_user("alice", "VeryStrongPassword12"))
        self.assertEqual(
            authenticate_user("alice", "UpdatedStrongPassword12"),
            "alice",
        )
        with self.assertRaises(ValueError):
            reset_password("alice", self.recovery_code, "AnotherPassword123")
        self.assertTrue(new_recovery_code)

    def test_password_change_rotates_recovery_code(self) -> None:
        new_recovery_code = change_password(
            "alice",
            "VeryStrongPassword12",
            "ChangedPassword1234",
        )
        self.assertEqual(
            authenticate_user("alice", "ChangedPassword1234"),
            "alice",
        )
        with self.assertRaises(ValueError):
            reset_password("alice", self.recovery_code, "ResetPassword1234")
        self.assertTrue(new_recovery_code)

    def test_project_operations_are_scoped_to_owner(self) -> None:
        project_id = create_project("alice", "Analisis Churn")

        self.assertEqual(len(list_projects("alice")), 1)
        self.assertEqual(list_projects("bob"), [])
        self.assertFalse(rename_project("bob", project_id, "Akses ilegal"))
        self.assertFalse(set_project_favorite("bob", project_id, True))
        self.assertFalse(delete_project("bob", project_id))
        self.assertTrue(rename_project("alice", project_id, "Churn Q4"))
        self.assertTrue(set_project_favorite("alice", project_id, True))
        self.assertEqual(list_projects("alice")[0]["name"], "Churn Q4")
        self.assertEqual(len(list_activities("alice")), 3)
        self.assertTrue(delete_project("alice", project_id))
        self.assertEqual(list_projects("alice"), [])

    def test_invalid_project_name_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            create_project("alice", "  ")

if __name__ == "__main__":
    unittest.main()
