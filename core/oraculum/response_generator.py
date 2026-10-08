"""
core/oraculum/response_generator.py
===================================
Генерация ответов Оракула на основе намерения, контекста и шаблонов.

ОТВЕТСТВЕННОСТЬ:
  • Выбор шаблона с учётом уровня пользователя.
  • Подстановка параметров (имя модели, вкладка, число ошибок).
  • Генерация ненавязчивых подсказок по текущему состоянию.
"""

from __future__ import annotations

import random
from typing import Any, Optional

from core.oraculum.templates import get_template
from core.oraculum.intent_classifier import Intent
from core.logger import get_logger

logger = get_logger(__name__)


class ResponseGenerator:
    """Генерирует ответы по намерению и контексту."""

    def __init__(self):
        self._last_hint = ""

    # ============================================================
    # ОТВЕТ ПО НАМЕРЕНИЮ
    # ============================================================
    def generate(self, intent: str, context, query: str) -> str:
        level = getattr(context, "user_level", "beginner")
        level = level if level in ("beginner", "intermediate", "advanced") else "beginner"

        # Специфичные шаблоны по темам
        topic_key = ""
        if intent == Intent.EXPLAIN:
            topic_key = self._detect_explain_topic(query)

        template_key = intent
        if topic_key:
            template_key = f"{intent}_{topic_key}"
        # Уровень пользователя как суффикс (если есть такой шаблон)
        level_key = f"{intent}_{level}"
        if topic_key:
            level_key = f"{intent}_{topic_key}_{level}"
            # Проверяем, существует ли шаблон с темой и уровнем
            if not get_template(level_key) or level_key == f"{intent}_{topic_key}_{level}":
                pass
        if get_template(level_key):
            template_key = level_key

        template = get_template(template_key)
        if not template:
            template = get_template(intent)
        if not template:
            template = get_template("general")

        params = {
            "model_name": getattr(context, "last_model_name", "") or "безымянная",
            "scenario": self._scenario_label(getattr(context, "scenario", "new")),
            "current_tab": getattr(context, "current_tab", "") or "Проект",
            "errors": getattr(context, "errors_last_hour", 0),
            "query": (query or "").strip()[:80],
            "suggest_step": getattr(context, "suggest_step", "") or "продолжить работу",
        }
        try:
            return template.format(**params)
        except (KeyError, IndexError, ValueError):
            return template

    def _detect_explain_topic(self, query: str) -> str:
        q = (query or "").lower()
        if any(w in q for w in ("трансформер", "attention", "внимание")):
            return "transformer"
        if "loss" in q or "потер" in q:
            return "loss"
        if "эпох" in q:
            return "epoch"
        if "переобуч" in q or "overfit" in q:
            return "overfitting"
        return ""

    @staticmethod
    def _scenario_label(scenario: str) -> str:
        labels = {
            "new": "создаю с нуля",
            "finetune": "дообучаю готовую",
            "play": "играю с моделью",
        }
        return labels.get(scenario, scenario)

    # ============================================================
    # ПОДСКАЗКИ
    # ============================================================
    def generate_hint(self, context) -> str:
        """Ненавязчивая подсказка на основе состояния."""
        hints = []
        if not getattr(context, "dataset_exists", False):
            hints.append("Создайте или загрузите данные во вкладке «Генератор/Данные».")
        elif not getattr(context, "has_split", False):
            hints.append("Примените разбиение данных во вкладке «Данные».")
        if not getattr(context, "model_exists", False):
            hints.append("Создайте модель во вкладке «Архитектура».")
        if (getattr(context, "model_exists", False)
                and getattr(context, "has_split", False)
                and not getattr(context, "model_trained", False)):
            hints.append("Модель готова к обучению! Запустите во вкладке «Обучение».")
        if getattr(context, "model_trained", False):
            hints.append("Обучение завершено! Проверьте результат во вкладке «Анализ».")
        errors = getattr(context, "errors_last_hour", 0)
        if errors > 0:
            hints.append(f"За последний час было {errors} ошибок. Проверьте логи.")

        if not hints:
            return "Всё идёт по плану! Продолжайте экспериментировать."

        hint = random.choice(hints)
        if hint == self._last_hint and len(hints) > 1:
            hint = random.choice([h for h in hints if h != self._last_hint])
        self._last_hint = hint
        return f"💡 {hint}"

    def generate_workflow_hint(self, context) -> str:
        """Подсказка о следующем шаге рабочего процесса."""
        step = getattr(context, "suggest_step", "")
        if not step:
            return ""
        return f"📋 Следующий шаг: {step}"
