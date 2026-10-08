"""
tests/test_user_storage_analyzer.py
===================================
Тесты модуля 4: анализ хранилища пользователя.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.user_manager import UserManager
from core.user_storage_analyzer import UserStorageAnalyzer, StorageReport


class TestUserStorageAnalyzer(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.um = UserManager(self.tmp)
        self.um.create_user("ivan", "")
        self.esm = None

    def test_empty_storage(self):
        report = UserStorageAnalyzer(self.um, "ivan").analyze()
        self.assertIsInstance(report, StorageReport)
        self.assertEqual(report.username, "ivan")
        self.assertEqual(report.total_size_mb, 0.0)
        self.assertGreater(len(report.recommendations), 0)

    def test_project_detection(self):
        # Создаём проект через менеджер
        self.um.save_project_to_user("ivan", {"format_version": "2.0"}, "proj")
        report = UserStorageAnalyzer(self.um, "ivan").analyze()
        self.assertEqual(len(report.projects), 1)
        self.assertGreater(report.total_size_mb, 0.0)

    def test_project_meta(self):
        self.um.save_project_to_user(
            "ivan",
            {
                "format_version": "2.0",
                "config": {"model_name": "Test", "type": "mlp"},
                "history": {"val_loss": [0.1]},
            },
            "proj2",
        )
        report = UserStorageAnalyzer(self.um, "ivan").analyze()
        meta = report.projects[0].meta
        self.assertEqual(meta["model_name"], "Test")
        self.assertTrue(meta["trained"])

    def test_format_report_text(self):
        report = UserStorageAnalyzer(self.um, "ivan").analyze()
        text = UserStorageAnalyzer(self.um, "ivan").format_report_text(report)
        self.assertIn("Хранилище", text)
        self.assertIn("Рекомендации", text)

    def test_recommendations_for_many_projects(self):
        for i in range(12):
            self.um.save_project_to_user(
                "ivan", {"format_version": "2.0"}, f"proj{i}"
            )
        report = UserStorageAnalyzer(self.um, "ivan").analyze()
        # Должна быть рекомендация про много проектов
        self.assertTrue(any("проект" in r for r in report.recommendations))


if __name__ == "__main__":
    unittest.main()
