"""
tests/test_workflow.py
======================
Тесты модуля 5: WorkflowManager (исправление сценариев).
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.contracts import (
    SessionState, ModelConfig, TrainingHistory, DatasetContainer,
)
from core.workflow_manager import WorkflowManager
from core.workflow_step import WorkflowStep


def _base_state(scenario="new"):
    return {
        "session": SessionState(),
        "project": {"name": "", "scenario": scenario, "model_name": ""},
        "dataset": None,
        "model": None,
        "config": ModelConfig(),
        "history": TrainingHistory(),
    }


class TestWorkflowStep(unittest.TestCase):
    def test_project_step(self):
        step = WorkflowStep("project", "Создать проект")
        self.assertFalse(step.is_completed(_base_state()))
        ss = _base_state()
        ss["project"]["name"] = "P"
        self.assertTrue(step.is_completed(ss))

    def test_model_step(self):
        step = WorkflowStep("model", "Создать модель")
        ss = _base_state()
        ss["model"] = object()
        self.assertTrue(step.is_completed(ss))

    def test_split_step(self):
        step = WorkflowStep("split", "Разбиение")
        ds = DatasetContainer()
        ds.train_inputs = [1, 2, 3]
        ss = _base_state()
        ss["dataset"] = ds
        self.assertTrue(step.is_completed(ss))


class TestWorkflowManager(unittest.TestCase):
    def test_new_scenario_order(self):
        wm = WorkflowManager(_base_state("new"))
        self.assertEqual(wm.get_current_step().action, "project")
        self.assertEqual(len(wm.get_all_steps()), 8)

    def test_progression(self):
        ss = _base_state("new")
        wm = WorkflowManager(ss)
        ss["project"]["name"] = "P"
        self.assertEqual(wm.get_current_step().action, "data")
        ds = DatasetContainer()
        ds.raw_inputs = [1, 2, 3]
        ds.raw_outputs = [1, 2, 3]
        ss["dataset"] = ds
        self.assertEqual(wm.get_current_step().action, "split")

    def test_finetune_scenario(self):
        wm = WorkflowManager(_base_state("finetune"))
        self.assertEqual(wm.get_current_step().action, "load_model")

    def test_play_scenario(self):
        wm = WorkflowManager(_base_state("play"))
        self.assertEqual(wm.get_current_step().action, "load_model")
        self.assertEqual(len(wm.get_all_steps()), 3)

    def test_all_done(self):
        ss = _base_state("new")
        ss["project"]["name"] = "P"
        ds = DatasetContainer()
        ds.train_inputs = [1, 2]
        ds.train_outputs = [1, 2]
        ds.raw_inputs = [1, 2]
        ds.raw_outputs = [1, 2]
        ss["dataset"] = ds
        ss["model"] = object()
        h = TrainingHistory()
        h.val_loss = [0.5]
        ss["history"] = h
        wm = WorkflowManager(ss)
        self.assertIsNone(wm.get_current_step())
        self.assertEqual(wm.get_next_action()["action"], "all_done")

    def test_progress_percent(self):
        ss = _base_state("new")
        wm = WorkflowManager(ss)
        # Шаги params/analyze/export изначально считаются завершёнными
        initial = wm.get_progress()["percent"]
        ss["project"]["name"] = "P"
        self.assertGreater(wm.get_progress()["percent"], initial)

    def test_scenario_switch(self):
        ss = _base_state("new")
        wm = WorkflowManager(ss)
        self.assertEqual(wm.get_current_step().action, "project")
        ss["project"]["scenario"] = "play"
        self.assertEqual(wm.get_current_step().action, "load_model")


if __name__ == "__main__":
    unittest.main()
