"""
tests/test_optimizer_factory.py
===============================
Регрессия: планировщики (особенно linear warmup) и валидация настроек.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from core.optimizer_factory import OptimizerFactory


def _opt(lr=0.001):
    return OptimizerFactory.create(
        torch.nn.Linear(4, 1),
        {"optimizer": "adamw", "learning_rate": lr},
    )


class TestSchedulerRobustness(unittest.TestCase):
    def test_warmup_clamps_when_ge_epochs(self):
        """Разогрев >= эпох не должен отключать планировщик (регрессия школы)."""
        opt = _opt()
        sched = OptimizerFactory.create_scheduler(opt, {
            "scheduler": "linear_warmup_linear_decay",
            "epochs": 3,
            "scheduler_params": {"warmup_epochs": 5, "warmup_start_lr": 1e-6},
        })
        self.assertIsNotNone(sched, "планировщик не должен пропадать")
        self.assertIn("LambdaLR", type(sched).__name__)

    def test_warmup_cosine_clamps(self):
        opt = _opt()
        sched = OptimizerFactory.create_scheduler(opt, {
            "scheduler": "linear_warmup_cosine",
            "epochs": 2,
            "scheduler_params": {"warmup_epochs": 10},
        })
        self.assertIsNotNone(sched)

    def test_linear_warmup_gui_name_works(self):
        """Имя «linear_warmup» из GUI должно создавать планировщик (было рассогласование)."""
        opt = _opt()
        sched = OptimizerFactory.create_scheduler(opt, {
            "scheduler": "linear_warmup",
            "epochs": 50,
            "learning_rate": 0.001,
        })
        self.assertIsNotNone(sched)
        self.assertIn("LambdaLR", type(sched).__name__)
        # Кривая LR должна расти (разогрев), а не быть постоянной
        curve = OptimizerFactory.get_lr_curve(
            {"scheduler": "linear_warmup", "epochs": 50, "learning_rate": 0.001},
            total_epochs=50,
        )
        self.assertLess(curve[0], max(curve), "LR должен расти на разогреве")
        self.assertEqual([], OptimizerFactory.validate({"scheduler": "linear_warmup"}))

    def test_all_schedulers_create(self):
        """Каждый планировщик создаётся без падения для разумного конфига."""
        for name in OptimizerFactory.list_schedulers():
            opt = _opt()
            sched = OptimizerFactory.create_scheduler(opt, {
                "scheduler": name,
                "epochs": 10,
                "learning_rate": 0.001,
                "scheduler_params": {"warmup_epochs": 2},
            })
            # none → None, остальные → объект
            if name == "none":
                self.assertIsNone(sched)
            else:
                self.assertIsNotNone(sched, f"планировщик {name} не создался")

    def test_warmup_hint_warns(self):
        hint = OptimizerFactory.get_contextual_hint({
            "scheduler": "linear_warmup_cosine",
            "epochs": 3,
            "learning_rate": 0.001,
            "scheduler_params": {"warmup_epochs": 5},
        })
        self.assertIn("Разогрев", hint)

    def test_validate_catches_bad_lr(self):
        errs = OptimizerFactory.validate({"learning_rate": -1.0})
        self.assertTrue(any("LR" in e for e in errs))


if __name__ == "__main__":
    unittest.main()
