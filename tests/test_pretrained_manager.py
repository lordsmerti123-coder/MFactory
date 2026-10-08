"""
tests/test_pretrained_manager.py
================================
Тесты модуля 2: менеджер предобученных моделей.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pretrained_manager import PretrainedManager
from core.pretrained_model_info import PretrainedModelInfo


class TestPretrainedManager(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.pm = PretrainedManager(
            self.tmp / "shared" / "pretrained_models",
            self.tmp / "models" / "installed_models.json",
        )

    def test_load_manifest(self):
        self.assertGreaterEqual(len(self.pm.get_all_models()), 4)

    def test_model_info_from_dict_accepts_id(self):
        info = PretrainedModelInfo.from_dict(
            {"id": "abc", "name": "Test", "data_type": "text"}
        )
        self.assertEqual(info.model_id, "abc")

    def test_check_model_exists_initially_false(self):
        self.assertFalse(self.pm.check_model_exists("tiny_transformer_math"))

    def test_download_and_load_model(self):
        ok = self.pm.download_model("tiny_transformer_math")
        self.assertTrue(ok)
        self.assertTrue(self.pm.check_model_exists("tiny_transformer_math"))
        self.assertEqual(len(self.pm.get_installed_models()), 1)
        model, config, extra = self.pm.load_model("tiny_transformer_math")
        self.assertIsNotNone(model)
        self.assertEqual(config.model_name, "Маленький Трансформер для математики")

    def test_remove_model(self):
        self.pm.download_model("tiny_rnn_math")
        self.assertTrue(self.pm.remove_model("tiny_rnn_math"))
        self.assertFalse(self.pm.check_model_exists("tiny_rnn_math"))

    def test_integrity_no_hash_returns_true(self):
        # Без sha256 в манифесте проверка всегда True
        self.pm.download_model("tiny_transformer_math")
        self.assertTrue(self.pm.verify_integrity("tiny_transformer_math"))

    def test_unknown_model(self):
        self.assertIsNone(self.pm.get_model("does_not_exist"))
        self.assertFalse(self.pm.download_model("does_not_exist"))
        with self.assertRaises(FileNotFoundError):
            self.pm.load_model("does_not_exist")

    def test_status_summary(self):
        s = self.pm.status_summary()
        self.assertIn("available", s)
        self.assertIn("installed", s)


if __name__ == "__main__":
    unittest.main()
