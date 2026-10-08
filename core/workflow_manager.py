"""
core/workflow_manager.py
========================
Гибкий пошаговый помощник (WorkflowManager).

ОТВЕТСТВЕННОСТЬ:
  • Построение рабочего процесса в зависимости от сценария.
  • Определение текущего рекомендуемого шага (НЕ блокирует пользователя).
  • Подсказки «что делать дальше».
  • Отслеживание прогресса.

ПРИНЦИП:
  Пользователь может делать шаги в любом порядке. Менеджер лишь
  подсвечивает первый незавершённый шаг и подсказывает следующее действие.

ЗАВИСИМОСТИ:
  • core/workflow_step.py → WorkflowStep
  • core/contracts.py    → ScenarioType
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.workflow_step import WorkflowStep
from core.logger import get_logger

logger = get_logger(__name__)


class WorkflowManager:
    """Управляет рабочим процессом по сценарию."""

    def __init__(self, shared_state: dict):
        self.shared_state = shared_state
        self.steps: List[WorkflowStep] = []
        self._last_scenario = ""
        self._build_workflow()

    # ============================================================
    # ПОСТРОЕНИЕ ПРОЦЕССА
    # ============================================================
    def _build_workflow(self):
        project = self.shared_state.get("project", {}) or {}
        scenario = project.get("scenario", "new")
        if hasattr(scenario, "value"):
            scenario = scenario.value
        if scenario == self._last_scenario and self.steps:
            return
        self._last_scenario = scenario

        if scenario == "finetune":
            self.steps = [
                WorkflowStep("load_model", "Загрузить модель",
                             tab="Обучение", hint="Загрузите обученную модель (.pth) во вкладке «Обучение»."),
                WorkflowStep("data", "Загрузить новые данные",
                             tab="Данные", hint="Загрузите или сгенерируйте новый датасет."),
                WorkflowStep("split", "Применить разбиение",
                             tab="Данные", hint="Примените разбиение 80/10/10 во вкладке «Данные»."),
                WorkflowStep("train", "Запустить дообучение",
                             tab="Обучение", hint="Модель и данные готовы. Запустите дообучение."),
                WorkflowStep("analyze", "Проанализировать результат",
                             tab="Анализ", is_optional=True,
                             hint="Проверьте результат во вкладке «Анализ»."),
                WorkflowStep("export", "Экспортировать",
                             tab="Экспорт", is_optional=True,
                             hint="Экспортируйте результат на флешку или в HTML."),
            ]
        elif scenario == "play":
            self.steps = [
                WorkflowStep("load_model", "Загрузить модель",
                             tab="Песочница", hint="Загрузите модель или проект во вкладке «Песочница»."),
                WorkflowStep("sandbox", "Играть с моделью",
                             tab="Песочница", hint="Введите пример и посмотрите, что ответит модель."),
                WorkflowStep("export", "Экспортировать",
                             tab="Экспорт", is_optional=True,
                             hint="Сохраните результат, если он вам нравится."),
            ]
        else:  # "new"
            self.steps = [
                WorkflowStep("project", "Создать проект",
                             tab="Проект", hint="Укажите имя проекта и модели во вкладке «Проект»."),
                WorkflowStep("data", "Сгенерировать или загрузить данные",
                             tab="Данные", hint="Сгенерируйте датасет во вкладке «Генератор» или загрузите файл."),
                WorkflowStep("split", "Применить разбиение данных",
                             tab="Данные", hint="Примените разбиение 80/10/10 во вкладке «Данные»."),
                WorkflowStep("model", "Создать модель",
                             tab="Архитектура", hint="Создайте модель во вкладке «Архитектура»."),
                WorkflowStep("params", "Настроить гиперпараметры",
                             tab="Гиперпараметры", is_optional=True,
                             hint="Настройте скорость обучения и число эпох во вкладке «Гиперпараметры»."),
                WorkflowStep("train", "Запустить обучение",
                             tab="Обучение", hint="Запустите обучение во вкладке «Обучение»."),
                WorkflowStep("analyze", "Проанализировать результат",
                             tab="Анализ", is_optional=True,
                             hint="Посмотрите отчёт во вкладке «Анализ»."),
                WorkflowStep("export", "Экспортировать результат",
                             tab="Экспорт", is_optional=True,
                             hint="Экспортируйте модель на флешку или в HTML."),
            ]
        logger.info(f"Рабочий процесс построен для сценария '{scenario}' ({len(self.steps)} шагов)")

    # ============================================================
    # ТЕКУЩИЙ ШАГ
    # ============================================================
    def get_current_step(self) -> Optional[WorkflowStep]:
        """Возвращает первый незавершённый шаг."""
        self._build_workflow()  # обновляем, если сценарий изменился
        for step in self.steps:
            if not step.is_completed(self.shared_state):
                return step
        return None

    def get_current_step_index(self) -> int:
        current = self.get_current_step()
        if current is None:
            return len(self.steps)
        try:
            return self.steps.index(current)
        except ValueError:
            return 0

    def get_next_action(self) -> Dict[str, Any]:
        """Словарь с действием, вкладкой и подсказкой."""
        step = self.get_current_step()
        if step is None:
            return {"action": "all_done", "tab": "", "hint": "Всё готово!",
                    "description": "Все шаги выполнены", "is_optional": True}
        return {
            "action": step.action,
            "tab": step.tab,
            "hint": step.hint,
            "description": step.description,
            "is_optional": step.is_optional,
        }

    def get_progress(self) -> Dict[str, Any]:
        """Прогресс: сколько шагов завершено."""
        self._build_workflow()
        completed = 0
        for step in self.steps:
            if step.is_completed(self.shared_state):
                completed += 1
        total = len(self.steps)
        return {
            "completed": completed,
            "total": total,
            "percent": int(completed / total * 100) if total else 100,
            "all_done": completed >= total,
        }

    def get_all_steps(self) -> List[Dict[str, Any]]:
        """Список всех шагов с их статусом."""
        self._build_workflow()
        result = []
        for step in self.steps:
            result.append({
                **step.to_dict(),
                "completed": step.is_completed(self.shared_state),
            })
        return result

    def get_next_step_description(self) -> str:
        action = self.get_next_action()
        return action.get("description", "")

    def get_suggest_step_text(self) -> str:
        """Текст для Оракула (suggest_step)."""
        action = self.get_next_action()
        if action.get("action") == "all_done":
            return "Всё готово! Проверьте модель в песочнице и экспортируйте."
        return action.get("description", "")

    # ============================================================
    # СЛУЖЕБНОЕ
    # ============================================================
    def reset(self):
        """Сбрасывает построенный процесс (пересоберётся при следующем вызове)."""
        self._last_scenario = ""
        self.steps = []

    def __repr__(self) -> str:
        progress = self.get_progress()
        return (
            f"<WorkflowManager scenario='{self._last_scenario}' "
            f"progress={progress['completed']}/{progress['total']}>"
        )
