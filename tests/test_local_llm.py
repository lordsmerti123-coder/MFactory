"""
tests/test_local_llm.py
=======================
Тесты локальной GGUF-модели Оракула (llama.cpp).

Тесты НЕ загружают реальную модель: проверяют отказоустойчивость
(неверные пути, отсутствие llama_cpp), контроль ресурсов и логику
выбора «использовать ли локальную модель».
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.contracts import SessionState, ModelConfig, TrainingHistory
from core.oraculum import OraculumAgent, AgentContext, Intent, LocalLLMBackend, ResourceMonitor
from core.oraculum.agent import resolve_local_model_path


def _make_state():
    return {
        "session": SessionState(),
        "project": {"name": "P", "scenario": "new", "model_name": "M"},
        "dataset": None,
        "model": None,
        "config": ModelConfig(),
        "history": TrainingHistory(),
    }


class TestResolveLocalModelPath(unittest.TestCase):
    def test_nonexistent_explicit_returns_none(self):
        self.assertIsNone(resolve_local_model_path("Z:/no_such_model.gguf"))

    def test_existing_file(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            path = f.name
        try:
            self.assertEqual(resolve_local_model_path(path), Path(path))
        finally:
            Path(path).unlink()


class TestLocalLLMBackend(unittest.TestCase):
    def test_load_nonexistent_model_returns_false(self):
        backend = LocalLLMBackend("Z:/no_such_model.gguf")
        self.assertFalse(backend.load())
        self.assertFalse(backend.is_loaded())
        self.assertIsNotNone(backend.get_last_error())

    def test_generate_without_load_returns_none(self):
        backend = LocalLLMBackend("Z:/no_such_model.gguf")
        self.assertIsNone(backend.generate("привет"))
        self.assertFalse(backend.is_hung())

    def test_unload_without_load_is_safe(self):
        backend = LocalLLMBackend("Z:/no_such_model.gguf")
        backend.unload()
        self.assertFalse(backend.is_loaded())


class TestResourceMonitorLLM(unittest.TestCase):
    def test_can_load_llm_nonexistent(self):
        rm = ResourceMonitor()
        self.assertFalse(rm.can_load_llm(Path("Z:/none.gguf")))

    def test_can_load_llm_tiny_file(self):
        rm = ResourceMonitor()
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            f.write(b"x" * 1024)
            path = Path(f.name)
        try:
            self.assertTrue(rm.can_load_llm(path))
        finally:
            path.unlink()

    def test_can_load_llm_size_limit(self):
        rm = ResourceMonitor({"max_llm_model_ram_mb": 0})
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            f.write(b"x" * 1024)
            path = Path(f.name)
        try:
            self.assertFalse(rm.can_load_llm(path))
        finally:
            path.unlink()


class _FakeBackend:
    """Заглушка бэкенда для проверки логики выбора локальной модели."""
    def __init__(self, loaded=True):
        self._loaded = loaded
        self._hung = False
        self._calls = 0

    def is_loaded(self):
        return self._loaded

    def is_hung(self):
        return self._hung

    def generate(self, prompt, system=None, max_tokens=None, temperature=None):
        self._calls += 1
        return "ответ локальной модели"


class TestShouldUseLocalModel(unittest.TestCase):
    def setUp(self):
        self.agent = OraculumAgent(_make_state())

    def _ctx(self, hint_mode):
        return AgentContext(hint_mode=hint_mode)

    def test_no_backend(self):
        self.assertFalse(self.agent._should_use_local_model(
            Intent.GENERAL, self._ctx("ai"), "Хороший вопрос."))

    def test_ai_mode_always_uses(self):
        self.agent._local_backend = _FakeBackend()
        self.assertTrue(self.agent._should_use_local_model(
            Intent.ASK_MODEL, self._ctx("ai"), "ответ шаблона"))

    def test_hybrid_uses_for_general(self):
        self.agent._local_backend = _FakeBackend()
        self.assertTrue(self.agent._should_use_local_model(
            Intent.GENERAL, self._ctx("hybrid"), "Хороший вопрос."))

    def test_hybrid_skips_structured(self):
        self.agent._local_backend = _FakeBackend()
        self.assertFalse(self.agent._should_use_local_model(
            Intent.ASK_MODEL, self._ctx("hybrid"), "ответ шаблона"))

    def test_hybrid_uses_on_fallback(self):
        self.agent._local_backend = _FakeBackend()
        self.assertTrue(self.agent._should_use_local_model(
            Intent.GENERAL, self._ctx("hybrid"), "Я пока не знаю..."))

    def test_static_mode_never_uses(self):
        self.agent._local_backend = _FakeBackend()
        self.assertFalse(self.agent._should_use_local_model(
            Intent.GENERAL, self._ctx("static"), "Хороший вопрос."))


class TestAgentLocalModelGuard(unittest.TestCase):
    def test_load_nonexistent_returns_false(self):
        agent = OraculumAgent(_make_state())
        self.assertFalse(agent.load_local_model("Z:/no_such_model.gguf"))
        self.assertFalse(agent.is_local_model_loaded())

    def test_unload_without_load_is_safe(self):
        agent = OraculumAgent(_make_state())
        agent.unload_local_model()
        self.assertFalse(agent.is_local_model_loaded())


if __name__ == "__main__":
    unittest.main()
