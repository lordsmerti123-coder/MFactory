"""
core/workflow_step.py
=====================
Шаг рабочего процесса.

ОТВЕТСТВЕННОСТЬ:
  • Описание одного шага (действие, вкладка, подсказка).
  • Проверка завершённости шага по состоянию shared_state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass
class WorkflowStep:
    """Один шаг рабочего процесса."""
    action: str
    description: str
    tab: str = ""
    is_optional: bool = False
    hint: str = ""
    # Кастомная проверка завершённости (если нужна)
    check_fn: Optional[Callable[[Dict[str, Any]], bool]] = None

    # ============================================================
    # ПРОВЕРКА ЗАВЕРШЁННОСТИ
    # ============================================================
    def is_completed(self, shared_state: dict) -> bool:
        if self.check_fn is not None:
            try:
                return bool(self.check_fn(shared_state))
            except Exception:
                return False

        project = shared_state.get("project", {}) or {}
        dataset = shared_state.get("dataset")
        model = shared_state.get("model")
        history = shared_state.get("history", {}) or {}

        has_split = False
        if dataset is not None:
            if hasattr(dataset, "is_split"):
                has_split = dataset.is_split
            elif isinstance(dataset, dict):
                has_split = bool(dataset.get("train_inputs"))

        model_trained = False
        if history:
            if hasattr(history, "val_loss"):
                model_trained = bool(history.val_loss)
            elif isinstance(history, dict):
                model_trained = bool(history.get("val_loss"))

        checks = {
            "project": lambda: bool(project.get("name")),
            "data": lambda: dataset is not None,
            "split": lambda: has_split,
            "model": lambda: model is not None,
            "params": lambda: bool(shared_state.get("config")),
            "train": lambda: model_trained,
            "analyze": lambda: True,
            "export": lambda: True,
            "load_model": lambda: model is not None,
            "sandbox": lambda: True,
        }
        return checks.get(self.action, lambda: True)()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "description": self.description,
            "tab": self.tab,
            "is_optional": self.is_optional,
            "hint": self.hint,
        }

    def __repr__(self) -> str:
        return f"<WorkflowStep {self.action} ({'опц.' if self.is_optional else 'обяз.'})>"
