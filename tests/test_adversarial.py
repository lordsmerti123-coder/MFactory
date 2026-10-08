"""
tests/test_adversarial.py
=========================
Тесты «пользователя-идиота»: человек намеренно пытается сломать программу
через интерфейс (не в том порядке нажимает кнопки, вводит бред, огромные
числа, сбрасывает состояние) — и проверка, что ИИ-помощник (Оракул)
отрабатывает эти сценарии без падений и зависаний.

Проверяется:
  • Инструменты ИИ (silent-методы панелей) возвращают понятную ошибку, а не
    исключение, при любых «идиотских» входных данных.
  • ИИ отвечает внятно на хаотичные/бессмысленные запросы.
  • Никакая последовательность действий не «вешает» программу.

ВАЖНО: torch импортируется строго ДО PyQt5.
"""

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: F401,E402

from PyQt5.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication(sys.argv)


def _is_error_json(text) -> bool:
    import json
    try:
        d = json.loads(text)
        return "error" in d
    except Exception:
        return False


class TestIdiotUserAgainstTools(unittest.TestCase):
    """Пользователь дёргает инструменты ИИ-агента в неверном порядке и с бредом."""

    @classmethod
    def setUpClass(cls):
        import gui.login_dialog as ld
        ld.LoginDialog.exec_ = lambda self: True
        ld.LoginDialog.get_selected_username = lambda self: "Admin"
        from gui.main_window import MainWindow
        cls.w = MainWindow()
        cls.w.show()
        app.processEvents()

    @classmethod
    def tearDownClass(cls):
        # Удаляем файлы датасетов, созданные инструментом generate_dataset
        try:
            subdirs = cls.w.session_manager.get_user_subdirs()
            d = subdirs.get("datasets")
            if d:
                for f in Path(d).glob("wizard_*.oai"):
                    try:
                        f.unlink(missing_ok=True)
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            cls.w.close()
        except Exception:
            pass

    def _run(self, name, args):
        return self.w._execute_agent_action(name, args or {})

    # ------------------------------------------------------------
    # Создание модели без данных / без разбиения
    # ------------------------------------------------------------
    def test_create_model_without_dataset(self):
        self.w.shared_state["dataset"] = None
        r = self._run("create_model", {})
        self.assertTrue(_is_error_json(r) or "данные" in r.lower(),
                        f"ожидалась ошибка, а не падение: {r}")

    def test_set_architecture_without_dataset(self):
        self.w.shared_state["dataset"] = None
        r = self._run("set_architecture", {"arch": "transformer_seq2seq"})
        self.assertIn("данные не загружены", r.lower())

    # ------------------------------------------------------------
    # Обучение до загрузки данных
    # ------------------------------------------------------------
    def test_train_without_anything(self):
        self.w.shared_state["dataset"] = None
        self.w.shared_state["model"] = None
        r = self._run("start_training", {})
        self.assertIn("error", r)

    # ------------------------------------------------------------
    # Разбиение без данных
    # ------------------------------------------------------------
    def test_split_without_dataset(self):
        self.w.shared_state["dataset"] = None
        r = self._run("split_dataset", {})
        self.assertIn("error", r)

    # ------------------------------------------------------------
    # Генерация данных с бредовыми параметрами
    # ------------------------------------------------------------
    def test_generate_unknown_task(self):
        r = self._run("generate_dataset", {"task": "сделать_кофе", "num_samples": 10})
        self.assertIn("error", r)

    def test_generate_reversed_range(self):
        # min > max — генератор должен сам поменять местами
        r = self._run("generate_dataset",
                      {"task": "addition", "num_samples": 100, "min": 10, "max": 0})
        self.assertNotIn("error", r)

    def test_generate_zero_samples_clamped(self):
        r = self._run("generate_dataset",
                      {"task": "addition", "num_samples": 0, "min": 0, "max": 10})
        # num_samples=0 должен быть зажат до минимума 100, а не падать
        self.assertNotIn("error", r)

    # ------------------------------------------------------------
    # Несуществующие инструменты / вкладки / сценарии
    # ------------------------------------------------------------
    def test_unknown_tool(self):
        r = self._run("сломать_всё", {})
        self.assertIn("неизвестный инструмент", r)

    def test_switch_to_unknown_tab(self):
        r = self._run("switch_tab", {"tab": "НЕСУЩЕСТВУЮЩАЯ"})
        self.assertIn("не найдена", r)

    def test_bad_scenario(self):
        r = self._run("set_scenario", {"scenario": "разрушить"})
        self.assertIn("error", r)

    # ------------------------------------------------------------
    # Пустые/None аргументы
    # ------------------------------------------------------------
    def test_empty_project_name(self):
        r = self._run("set_project", {"name": "", "model_name": ""})
        self.assertIn("не указано", r)

    def test_hyperparams_extreme(self):
        r = self._run("set_hyperparams", {"epochs": 999999999, "lr": -5, "batch_size": 0})
        # значения должны быть зажаты в допустимые пределы
        self.assertNotIn("error", r)


class TestIdiotUserAgainstAssistant(unittest.TestCase):
    """ИИ-помощник при хаотичных/вредных запросах пользователя."""

    @classmethod
    def setUpClass(cls):
        from core.contracts import SessionState, ModelConfig, TrainingHistory
        from core.oraculum.agent import OraculumAgent
        cls.agent = OraculumAgent({
            "session": SessionState(),
            "project": {"name": "P", "scenario": "new", "model_name": "M"},
            "dataset": None, "model": None,
            "config": ModelConfig(), "history": TrainingHistory(),
            "hint_mode": "hybrid",
            "user": {"username": "Admin", "level": "beginner", "is_admin": True},
        })
        cls.agent.resource_monitor.limits["max_cpu_percent"] = 100

    def _q(self, query):
        return self.agent.process_user_query(query)

    def test_gibberish(self):
        r = self._q("авыфафыв фыв фыва фыв 12345 !!!")
        self.assertIsInstance(r, str)
        self.assertGreater(len(r), 5)

    def test_very_long_query(self):
        r = self._q("х" * 5000)
        self.assertIsInstance(r, str)
        self.assertLess(len(r), 4000)  # не раздуваем ответ до бесконечности

    def test_empty_query(self):
        r = self._q("   ")
        self.assertIsInstance(r, str)

    def test_threats_and_nonsense(self):
        r = self._q("удали все файлы и взломай пентагон")
        self.assertIsInstance(r, str)
        self.assertGreater(len(r), 5)

    def test_what_is_happening_is_concrete(self):
        # пользователь спрашивает что происходит — ответ про состояние, не про технологии
        r = self._q("что происходит")
        self.assertIn("Данные не загружены", r)
        self.assertNotIn("технологиях Яндекс", r)

    def test_diagnose_no_crash(self):
        for _ in range(3):
            self.agent.diagnose()  # многократный вызов не должен падать


if __name__ == "__main__":
    unittest.main()
