"""
tests/test_oraculum.py
======================
Тесты модуля 3: ИИ-агент «Оракул».
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.contracts import SessionState, ModelConfig, TrainingHistory
from core.oraculum import (
    OraculumAgent, AgentContext, IntentClassifier, ResourceMonitor,
)
from core.oraculum.response_generator import ResponseGenerator
from core.oraculum.api_adapter import APIAdapter


def _make_state():
    return {
        "session": SessionState(),
        "project": {"name": "P", "scenario": "new", "model_name": "M"},
        "dataset": None,
        "model": None,
        "config": ModelConfig(),
        "history": TrainingHistory(),
    }


class TestIntentClassifier(unittest.TestCase):
    def setUp(self):
        self.clf = IntentClassifier()

    def test_classify_model(self):
        self.assertEqual(self.clf.classify("как создать модель?"), "ask_model")

    def test_classify_explain(self):
        self.assertEqual(self.clf.classify("что такое loss?"), "explain")

    def test_classify_troubleshoot(self):
        self.assertEqual(self.clf.classify("у меня ошибка"), "troubleshoot")

    def test_classify_greeting(self):
        self.assertEqual(self.clf.classify("привет"), "greeting")

    def test_classify_suggest(self):
        self.assertEqual(self.clf.classify("что делать дальше?"), "suggest")

    def test_classify_general(self):
        self.assertEqual(self.clf.classify("какая сегодня погода"), "general")


class TestResponseGenerator(unittest.TestCase):
    def setUp(self):
        self.gen = ResponseGenerator()

    def test_generate_ask_model(self):
        ctx = AgentContext(user_level="beginner")
        r = self.gen.generate("ask_model", ctx, "как создать модель?")
        self.assertIn("Архитектура", r)

    def test_generate_hint(self):
        ctx = AgentContext(user_level="beginner", model_exists=False,
                           dataset_exists=False)
        h = self.gen.generate_hint(ctx)
        self.assertTrue(h.startswith("💡"))


class TestOraculumAgent(unittest.TestCase):
    def setUp(self):
        self.agent = OraculumAgent(_make_state())
        # Изолируем от реальной загрузки CPU/RAM (иначе тест флейкает)
        self.agent.resource_monitor.limits["max_cpu_percent"] = 100

    def test_process_query(self):
        r = self.agent.process_user_query("привет")
        self.assertTrue(len(r) > 0)

    def test_contextual_hint(self):
        h = self.agent.get_contextual_hint()
        self.assertTrue(h.startswith("💡"))

    def test_workflow_hint(self):
        h = self.agent.get_workflow_hint()
        self.assertIn("Следующий шаг", h)

    def test_analyze_screen(self):
        s = self.agent.analyze_screen()
        self.assertEqual(s.scenario, "new")
        self.assertFalse(s.model_exists)

    def test_chat_history(self):
        self.agent.process_user_query("привет")
        self.assertEqual(len(self.agent.get_chat_history()), 2)  # user + assistant
        self.agent.clear_history()
        self.assertEqual(len(self.agent.get_chat_history()), 0)

    def test_api_not_available_by_default(self):
        self.assertFalse(self.agent.is_api_available())

    def test_configure_api_yandex(self):
        self.agent.configure_api("yandex", api_key="k", folder_id="f")
        self.assertEqual(self.agent.api_provider(), "yandex")
        self.assertTrue(self.agent.is_api_available())


class TestResourceMonitor(unittest.TestCase):
    def test_can_process(self):
        rm = ResourceMonitor()
        self.assertTrue(rm.can_process())

    def test_can_use_api_limit(self):
        rm = ResourceMonitor({"api_calls_per_minute": 2})
        self.assertTrue(rm.can_use_api())
        self.assertTrue(rm.can_use_api())
        self.assertFalse(rm.can_use_api())  # лимит 2

    def test_can_load_model_size_limit(self):
        import tempfile
        rm = ResourceMonitor({"max_model_ram_mb": 0})  # ничего не влезет
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            f.write(b"x" * 100)
            path = Path(f.name)
        self.assertFalse(rm.can_load_model(path))
        path.unlink()


class TestAPIAdapter(unittest.TestCase):
    def test_not_available_initially(self):
        a = APIAdapter()
        self.assertFalse(a.is_available())

    def test_send_request_without_provider(self):
        a = APIAdapter()
        self.assertIsNone(a.send_request("test"))

    def test_yandex_config(self):
        a = APIAdapter()
        a.set_yandex_gpt("key", "folder")
        self.assertTrue(a.is_available())
        self.assertEqual(a.provider, "yandex")

    def test_sber_config(self):
        a = APIAdapter()
        a.set_sber_gpt("cid", "csecret")
        self.assertTrue(a.is_available())
        self.assertEqual(a.provider, "sber")


if __name__ == "__main__":
    unittest.main()
