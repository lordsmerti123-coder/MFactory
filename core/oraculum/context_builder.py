"""
core/oraculum/context_builder.py
================================
Сбор полного контекста для Оракула из shared_state и логов.

ОТВЕТСТВЕННОСТЬ:
  • Агрегация состояния проекта, датасета, модели, сессии.
  • Извлечение последних действий и ошибок из коллектора логов.
  • Формирование AgentContext — единой структуры для генерации ответов.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from core.logger import curator_collector, LogCategory, get_logger

logger = get_logger(__name__)


@dataclass
class AgentContext:
    """Полный контекст для агента."""
    current_tab: str = ""
    scenario: str = "new"
    model_exists: bool = False
    dataset_exists: bool = False
    has_split: bool = False
    model_trained: bool = False
    recent_actions: List[Dict[str, Any]] = field(default_factory=list)
    mistakes: List[Dict[str, Any]] = field(default_factory=list)
    hint_mode: str = "hybrid"
    user_level: str = "beginner"
    last_model_name: str = ""
    errors_last_hour: int = 0
    current_widget_id: str = ""
    current_value: Any = None
    suggest_step: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_tab": self.current_tab,
            "scenario": self.scenario,
            "model_exists": self.model_exists,
            "dataset_exists": self.dataset_exists,
            "has_split": self.has_split,
            "model_trained": self.model_trained,
            "recent_actions": self.recent_actions[-10:],
            "mistakes": self.mistakes[-5:],
            "hint_mode": self.hint_mode,
            "user_level": self.user_level,
            "last_model_name": self.last_model_name,
            "errors_last_hour": self.errors_last_hour,
            "suggest_step": self.suggest_step,
        }


class ContextBuilder:
    """Собирает AgentContext из shared_state и логов."""

    def __init__(self, shared_state: dict):
        self.shared_state = shared_state

    def build(self) -> AgentContext:
        try:
            return self._build()
        except Exception as e:
            import traceback
            logger.warning(f"Ошибка сборки контекста: {e}\n{traceback.format_exc()}")
            return AgentContext()

    def _build(self) -> AgentContext:
        project = self.shared_state.get("project", {}) or {}
        dataset = self.shared_state.get("dataset")
        model = self.shared_state.get("model")
        history = self.shared_state.get("history", {}) or {}
        session = self.shared_state.get("session", {}) or {}

        # Сессия может быть SessionState-подобным объектом или dict.
        # НЕ используем session.to_dict(): некоторые поля (hint_mode) могут
        # храниться строкой, и сериализация упадёт. Читаем атрибуты напрямую.
        def _get(obj, attr, default=""):
            if obj is None:
                return default
            if isinstance(obj, dict):
                return obj.get(attr, default)
            value = getattr(obj, attr, default)
            if hasattr(value, "value"):
                value = value.value
            return value

        session_dict = {
            "current_tab": _get(session, "current_tab", ""),
            "scenario": _get(session, "scenario", "new"),
            "hint_mode": _get(session, "hint_mode", "hybrid"),
            "user_level": _get(session, "user_level", "beginner"),
            "last_model_name": _get(session, "last_model_name", ""),
            "user_actions_log": _get(session, "user_actions_log", []),
        }

        # Собираем логи из глобального коллектора
        recent_logs = []
        try:
            recent_logs = curator_collector.get_recent(50)
        except Exception:
            pass

        actions = [
            {"panel": e.panel, "action": e.action, "details": e.message}
            for e in recent_logs if e.category == LogCategory.USER_ACTION
        ]
        errors = [e for e in recent_logs if e.category == LogCategory.ERROR]

        now = datetime.now()
        recent_errors = [
            e for e in errors
            if e.timestamp >= (now - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
        ]

        dataset_exists = dataset is not None
        has_split = False
        if dataset_exists:
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

        scenario = project.get("scenario", session_dict.get("scenario", "new"))
        if hasattr(scenario, "value"):
            scenario = scenario.value

        return AgentContext(
            current_tab=session_dict.get("current_tab", ""),
            scenario=scenario,
            model_exists=model is not None,
            dataset_exists=dataset_exists,
            has_split=has_split,
            model_trained=model_trained,
            recent_actions=actions[-10:],
            mistakes=session_dict.get("user_actions_log", [])[-5:],
            hint_mode=session_dict.get("hint_mode", "hybrid"),
            user_level=session_dict.get("user_level", "beginner"),
            last_model_name=project.get("model_name", session_dict.get("last_model_name", "")),
            errors_last_hour=len(recent_errors),
            suggest_step=self._guess_suggest_step(
                model is not None, dataset_exists, has_split, model_trained, scenario
            ),
        )

    @staticmethod
    def _guess_suggest_step(model_exists: bool, dataset_exists: bool,
                            has_split: bool, model_trained: bool,
                            scenario: str) -> str:
        """Угадывает следующий рекомендуемый шаг."""
        if scenario == "play":
            return "Песочница готова — введите пример для модели."
        if not dataset_exists:
            return "Сгенерируйте или загрузите данные."
        if dataset_exists and not has_split:
            return "Примените разбиение данных."
        if not model_exists:
            return "Создайте модель во вкладке «Архитектура»."
        if not model_trained:
            return "Запустите обучение во вкладке «Обучение»."
        return "Обучение завершено — проанализируйте результат и экспортируйте."
