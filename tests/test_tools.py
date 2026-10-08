"""
tests/test_tools.py
===================
Тесты агентских возможностей Оракула: инструменты (tools), разбор tool-call,
агентский цикл ReAct, GUI-исполнитель.
"""

import sys
import json
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.oraculum.tools import (
    ToolRegistry, parse_tool_call, is_reply_object, TOOL_INSTRUCTIONS,
)
from core.oraculum.agent import OraculumAgent
from core.contracts import SessionState, ModelConfig, TrainingHistory


def _make_state():
    return {
        "session": SessionState(),
        "project": {"name": "P", "scenario": "new", "model_name": "M"},
        "dataset": None,
        "model": None,
        "config": ModelConfig(),
        "history": TrainingHistory(),
        "hint_mode": "hybrid",
        "user": {"username": "Admin", "level": "advanced", "is_admin": True},
    }


class TestParseToolCall(unittest.TestCase):
    def test_action_args(self):
        action, args = parse_tool_call(
            '{"action": "switch_tab", "args": {"tab": "Обучение"}}'
        )
        self.assertEqual(action, "switch_tab")
        self.assertEqual(args, {"tab": "Обучение"})

    def test_markdown_fence(self):
        action, args = parse_tool_call(
            '```json\n{"tool": "get_context"}\n```'
        )
        self.assertEqual(action, "get_context")
        self.assertEqual(args, {})

    def test_noise_around_json(self):
        action, args = parse_tool_call(
            'Конечно, вот решение: {"action": "create_model", "args": {}}'
        )
        self.assertEqual(action, "create_model")

    def test_plain_text_is_not_tool_call(self):
        self.assertEqual(parse_tool_call("просто текст"), (None, None))

    def test_truncated_json_repaired(self):
        # Модель Яндекса иногда обрезает JSON (не хватает закрывающих скобок)
        action, args = parse_tool_call(
            ' ```\n{"action": "switch_tab", "args": {"tab": "Обучение"}\n```'
        )
        self.assertEqual(action, "switch_tab")
        self.assertEqual(args, {"tab": "Обучение"})

    def test_noise_with_truncated_json_repaired(self):
        action, args = parse_tool_call(
            'Конечно, вот: {"action": "generate_dataset", '
            '"args": {"task": "addition", "num_samples": 200}'
        )
        self.assertEqual(action, "generate_dataset")
        self.assertEqual(args, {"task": "addition", "num_samples": 200})

    def test_empty(self):
        self.assertEqual(parse_tool_call(""), (None, None))


class TestIsReplyObject(unittest.TestCase):
    def test_reply(self):
        self.assertEqual(is_reply_object({"reply": "привет"}), "привет")

    def test_answer(self):
        self.assertEqual(is_reply_object({"answer": "да"}), "да")

    def test_no_reply(self):
        self.assertIsNone(is_reply_object({"action": "x"}))


class TestToolRegistry(unittest.TestCase):
    def test_register_and_get(self):
        reg = ToolRegistry()
        reg.register("demo", "Демо-инструмент", {"a": "параметр"},
                     handler=lambda a="1": f"ok:{a}")
        self.assertIn("demo", reg.names())
        self.assertEqual(reg.get("demo").name, "demo")
        self.assertIn("demo", reg.describe())

    def test_unknown_returns_none(self):
        reg = ToolRegistry()
        self.assertIsNone(reg.get("nope"))

    def test_instructions_present(self):
        self.assertIn("action", TOOL_INSTRUCTIONS)


class TestOraculumAgentTools(unittest.TestCase):
    def setUp(self):
        self.agent = OraculumAgent(_make_state())
        self.agent.settings["agent_enabled"] = True
        self.agent.settings["agent_max_steps"] = 3

    def test_builtin_tools(self):
        names = self.agent.tools.names()
        for t in ("get_context", "get_status", "list_tools",
                  "model_info", "monitor_training"):
            self.assertIn(t, names)

    def test_execute_builtin(self):
        res = self.agent._execute_tool("get_context", {})
        self.assertIsInstance(res, str)
        self.assertIn("Сценарий", res)

    def test_execute_unknown(self):
        res = self.agent._execute_tool("nope", {})
        self.assertIn("error", res)

    def test_execute_via_dispatcher(self):
        calls = {}

        def dispatcher(name, args):
            calls[name] = args
            return json.dumps({"done": True}, ensure_ascii=False)

        self.agent.set_action_dispatcher(dispatcher)
        self.agent.register_tool("gui_action", "Действие", {"x": "y"})
        res = self.agent._execute_tool("gui_action", {"x": 1})
        self.assertEqual(calls.get("gui_action"), {"x": 1})
        self.assertIn("done", res)

    def test_run_agent_loop_tool_then_reply(self):
        calls = {"n": 0}

        def fake_chat(messages):
            calls["n"] += 1
            if calls["n"] == 1:
                return '{"action": "get_context"}'
            return '{"reply": "Вот контекст."}'

        self.agent._agent_chat = fake_chat
        self.agent._agent_backend_available = lambda: True
        result = self.agent.run_agent_loop("что происходит?")
        self.assertEqual(result, "Вот контекст.")
        self.assertEqual(calls["n"], 2)

    def test_run_agent_loop_no_backend(self):
        self.agent._agent_backend_available = lambda: False
        result = self.agent.run_agent_loop("помоги")
        self.assertIn("недоступен", result)

    def test_run_agent_loop_bounded_steps(self):
        calls = {"n": 0}

        def fake_chat(messages):
            calls["n"] += 1
            return '{"action": "get_context"}'  # бесконечный цикл инструментов

        self.agent._agent_chat = fake_chat
        self.agent._agent_backend_available = lambda: True
        result = self.agent.run_agent_loop("делай", max_steps=2)
        # Цикл должен завершиться, а не зависнуть
        self.assertLessEqual(calls["n"], 2)
        self.assertIsInstance(result, str)

    def test_agent_mode_enabled_flag(self):
        self.agent.settings["agent_enabled"] = False
        self.assertFalse(self.agent.agent_mode_enabled())
        self.agent.settings["agent_enabled"] = True
        self.assertTrue(self.agent.agent_mode_enabled())

    def test_training_monitor_hooks(self):
        self.agent.on_training_started(False)
        self.agent.on_training_epoch(3, {"train_loss": 0.5, "val_loss": 0.4})
        state = json.loads(self.agent._tool_monitor_training())
        self.assertEqual(state["status"], "running")
        self.assertEqual(state["last_epoch"], 3)
        self.agent.on_training_finished({"best_val_loss": 0.2, "epochs_completed": 3})
        state = json.loads(self.agent._tool_monitor_training())
        self.assertEqual(state["status"], "finished")


class TestOraculumDiagnostics(unittest.TestCase):
    """Конкретные ответы о состоянии программы («что происходит»)."""

    def _agent(self, **overrides):
        state = _make_state()
        state.update(overrides)
        agent = OraculumAgent(state)
        # Тест не должен зависеть от глобального файла настроек и загрузки CPU
        agent.settings["agent_enabled"] = False
        agent.settings["provider"] = "local"
        agent.resource_monitor.limits["max_cpu_percent"] = 100
        return agent

    def test_status_intent(self):
        from core.oraculum.intent_classifier import IntentClassifier, Intent
        clf = IntentClassifier()
        self.assertEqual(clf.classify("что происходит"), Intent.STATUS)
        self.assertEqual(clf.classify("что сейчас происходит"), Intent.STATUS)
        self.assertEqual(clf.classify("как создать модель"), Intent.ASK_MODEL)

    def test_diagnose_no_dataset(self):
        agent = self._agent()
        findings = agent.diagnose()
        joined = " ".join(findings)
        self.assertTrue(any("Данные не загружены" in f for f in findings), joined)

    def test_diagnose_training_tab_without_dataset(self):
        # Ключевой сценарий: пользователь на вкладке «Обучение», данных нет
        session = SessionState()
        session.current_tab = "Обучение"
        agent = self._agent(session=session)
        joined = " ".join(agent.diagnose())
        self.assertIn("пытаетесь обучить модель", joined)
        self.assertIn("датасет не загружен", joined)

    def test_diagnose_ready_to_train(self):
        class FakeDataset:
            is_split = True

        agent = self._agent(dataset=FakeDataset(), model=object())
        joined = " ".join(agent.diagnose())
        self.assertTrue(any("готово к обучению" in f for f in agent.diagnose()), joined)

    def test_diagnose_trained_summary(self):
        class FakeDataset:
            is_split = True

        history = TrainingHistory()
        history.val_loss = [1.0, 0.5, 0.3]
        agent = self._agent(dataset=FakeDataset(), model=object(), history=history)
        joined = " ".join(agent.diagnose())
        self.assertIn("Обучение завершено", joined)

    def test_build_status_answer(self):
        agent = self._agent()
        answer = agent.build_status_answer()
        self.assertIn("Вот что сейчас происходит", answer)
        self.assertGreater(len(answer), 30)

    def test_process_query_status_fast_answer(self):
        # STATUS должен отвечать мгновенно, без обращения к бэкенду
        agent = self._agent()
        response = agent.process_user_query("что происходит")
        self.assertIn("Данные не загружены", response)


class TestOracleFixes(unittest.TestCase):
    """Регрессия: Оракул отвечает по делу, а не «кто я такая»."""

    def test_intent_broad_keywords(self):
        from core.oraculum.intent_classifier import IntentClassifier, Intent
        clf = IntentClassifier()
        self.assertEqual(clf.classify("как обучить трансформер"), Intent.ASK_MODEL)
        self.assertEqual(clf.classify("как создать нейросеть"), Intent.ASK_MODEL)
        self.assertEqual(clf.classify("загрузи данные"), Intent.ASK_DATA)
        self.assertEqual(clf.classify("что такое эпоха"), Intent.EXPLAIN)

    def test_identity_answer_detected(self):
        agent = OraculumAgent(_make_state())
        self.assertTrue(agent._is_identity_answer(
            "Я — модель искусственного интеллекта, могу помогать"))
        self.assertTrue(agent._is_identity_answer("Я Оракул, ваш помощник"))
        self.assertFalse(agent._is_identity_answer(
            "Сначала загрузите данные во вкладке «Генератор»"))

    def test_general_answer_is_state_aware(self):
        agent = OraculumAgent(_make_state())
        agent.resource_monitor.limits["max_cpu_percent"] = 100
        resp = agent.process_user_query("какая сегодня погода")
        # общий вопрос → ответ про состояние программы, а не «Я Оракул»
        self.assertIn("Данные не загружены", resp)
        self.assertNotIn("Я Оракул", resp)


if __name__ == "__main__":
    unittest.main()
