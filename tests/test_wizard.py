"""
tests/test_wizard.py
====================
Тесты режима «Мастер»: планировщик и плеер шагов (wizard.py).
"""

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ВАЖНО: torch импортируется строго ДО PyQt5 (ограничение проекта, иначе
# на Windows не загрузится c10.dll). Это влияет на весь тестовый процесс.
import torch  # noqa: F401,E402

from PyQt5.QtWidgets import QApplication  # noqa: E402
from PyQt5.QtTest import QTest  # noqa: E402

from core.oraculum.wizard import (
    WizardStep, WizardPlan, WizardPlanner, WizardPlayer,
)

app = QApplication.instance() or QApplication(sys.argv)


def _spin(app, ms):
    """Крутит event loop, пока плеер не завершит работу или не пройдёт таймаут."""
    deadline = time.time() + ms / 1000.0 + 0.5
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.005)


def _tiny_plan(agent=None, wait_step=False):
    steps = [
        WizardStep("s1", "Шаг 1", "Делаю первый шаг", "tool_a", {"x": 1},
                   delay_ms=1),
        WizardStep("s2", "Шаг 2", "Делаю второй шаг", "tool_b", {}, delay_ms=1),
    ]
    if wait_step:
        steps.append(WizardStep("s3", "Шаг 3", "Жду условия", "tool_c", {},
                                delay_ms=1, wait=lambda: True, wait_timeout=5))
    steps.append(WizardStep("done", "Готово", "Финиш", None, {}, delay_ms=1))
    return WizardPlan("test", "Тестовый план", steps)


class TestWizardPlanner(unittest.TestCase):
    def test_demo_plan_structure(self):
        planner = WizardPlanner(agent=None)
        plan = planner.plan_demo_transformer()
        self.assertEqual(plan.scenario, "demo_transformer")
        titles = [s.step_id for s in plan.steps]
        for expected in ("project", "generate", "split", "arch",
                         "hyper", "train", "analyze", "done"):
            self.assertIn(expected, titles)

    def test_demo_plan_has_training_wait(self):
        planner = WizardPlanner(agent=None)
        plan = planner.plan_demo_transformer()
        train = next(s for s in plan.steps if s.step_id == "train")
        self.assertEqual(train.tool, "start_training")
        self.assertIsNotNone(train.wait)

    def test_interactive_plan(self):
        planner = WizardPlanner(agent=None)
        plan = planner.plan_interactive({"name": "P", "model_name": "M"})
        self.assertEqual(plan.scenario, "interactive")
        self.assertTrue(plan.steps[0].args.get("name") == "P")


class TestWizardPlayer(unittest.TestCase):
    def setUp(self):
        self.executed = []

        def execute_tool(name, args):
            self.executed.append((name, dict(args or {})))
            return '{"done": true}'

        self.player = WizardPlayer(execute_tool)

    def test_runs_all_steps(self):
        done = []
        self.player.finished.connect(lambda: done.append(True))
        self.player.start(_tiny_plan())
        _spin(app, 500)
        names = [n for n, _ in self.executed]
        self.assertEqual(names, ["tool_a", "tool_b"])
        self.assertTrue(done)

    def test_wait_step(self):
        done = []
        self.player.finished.connect(lambda: done.append(True))
        self.player.start(_tiny_plan(wait_step=True))
        _spin(app, 800)
        self.assertEqual([n for n, _ in self.executed],
                         ["tool_a", "tool_b", "tool_c"])
        self.assertTrue(done)

    def test_stop_aborts(self):
        aborted = []
        self.player.aborted.connect(lambda: aborted.append(True))
        self.player.start(_tiny_plan())
        # сразу останавливаем — дальше не пойдёт
        self.player.stop()
        _spin(app, 300)
        self.assertTrue(aborted)

    def test_narration_emitted(self):
        narrations = []
        self.player.narration.connect(narrations.append)
        self.player.start(_tiny_plan())
        _spin(app, 500)
        self.assertIn("Делаю первый шаг", narrations)

    def test_error_is_reported(self):
        errors = []

        def bad_tool(name, args):
            raise RuntimeError("boom")

        player = WizardPlayer(bad_tool)
        player.error.connect(lambda step_id, err: errors.append(err))
        player.start(_tiny_plan())
        _spin(app, 500)
        self.assertTrue(any("boom" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
