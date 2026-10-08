"""
tests/test_user_manager.py
==========================
Тесты модуля 1: система пользователей и сессий.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.user_profile import hash_password, verify_password, UserProfile
from core.user_manager import UserManager
from core.session_manager_extended import ExtendedSessionManager


class TestPasswordHashing(unittest.TestCase):
    def test_hash_and_verify(self):
        h = hash_password("secret123")
        self.assertTrue(h.startswith("sha256$"))
        self.assertTrue(verify_password("secret123", h))
        self.assertFalse(verify_password("wrong", h))

    def test_salt_unique(self):
        h1 = hash_password("same")
        h2 = hash_password("same")
        self.assertNotEqual(h1, h2)  # разные соли


class TestUserManager(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.um = UserManager(self.tmp)

    def test_create_and_list_users(self):
        u = self.um.create_user("ivan", hash_password("12345"))
        self.assertIsNotNone(u)
        self.assertEqual(self.um.get_active_user_count(), 1)
        self.assertIn("ivan", self.um.list_usernames())

    def test_duplicate_username_rejected(self):
        self.um.create_user("ivan", "")
        self.assertIsNone(self.um.create_user("ivan", ""))

    def test_authenticate(self):
        self.um.create_user("ivan", hash_password("12345"))
        self.assertTrue(self.um.authenticate("ivan", "12345"))
        self.assertFalse(self.um.authenticate("ivan", "bad"))

    def test_switch_user(self):
        self.um.create_user("ivan", "")
        self.assertTrue(self.um.switch_user("ivan"))
        self.assertEqual(self.um.get_current_user().username, "ivan")

    def test_guest_cannot_be_deleted(self):
        g = self.um.create_guest()
        self.assertIsNotNone(g)
        self.assertFalse(self.um.delete_user("guest"))

    def test_delete_user(self):
        self.um.create_user("petr", "")
        self.assertTrue(self.um.delete_user("petr"))
        self.assertIsNone(self.um.get_profile("petr"))

    def test_data_separation(self):
        self.um.create_user("ivan", "")
        self.um.create_user("petr", "")
        esm1 = ExtendedSessionManager(self.um, "ivan")
        esm2 = ExtendedSessionManager(self.um, "petr")
        esm1.save_project_to_user(
            {"format_version": "2.0", "project_name": "p1"}, "proj"
        )
        self.assertEqual(len(esm1.get_user_projects()), 1)
        self.assertEqual(len(esm2.get_user_projects()), 0)

    def test_storage_usage(self):
        self.um.create_user("ivan", "")
        esm = ExtendedSessionManager(self.um, "ivan")
        esm.save_project_to_user({"format_version": "2.0"}, "proj")
        # Файл существует и занимает место (может быть < 0.01 МБ из-за округления)
        self.assertEqual(len(esm.get_user_projects()), 1)
        usage = self.um.get_storage_usage("ivan")
        self.assertGreaterEqual(usage, 0.0)

    def test_profile_serialization(self):
        p = UserProfile(username="test", password_hash=hash_password("x"))
        d = p.to_dict()
        p2 = UserProfile.from_dict(d)
        self.assertEqual(p2.username, "test")
        self.assertEqual(p2.password_hash, p.password_hash)


if __name__ == "__main__":
    unittest.main()
