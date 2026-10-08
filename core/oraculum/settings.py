"""
core/oraculum/settings.py
=========================
Настройки ИИ-агента «Оракул» (провайдер, ключи, системный промпт, автоконтекст).

ОТВЕТСТВЕННОСТЬ:
  • Загрузка/сохранение настроек в data/system/oraculum_settings.json.
  • Значения по умолчанию, если файла нет или он повреждён.
  • Шаблон системного промпта с поддержкой автоконтекста.

Настройки редактируются администратором из панели администратора
и применяются Оракулом при каждом запросе к внешнему API / локальной модели.

ЗАВИСИМОСТИ:
  • config.py → ORACULUM_SETTINGS_FILE
  • Стандартная библиотека.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from config import ORACULUM_SETTINGS_FILE

# Системный промпт по умолчанию. Поддерживает подстановку блока контекста
# через {context} — блок добавляется, когда включён автоконтекст.
DEFAULT_SYSTEM_PROMPT = (
    "Ты Оракул — ИИ-гуру программы OracleAI Studio. Ты отлично знаешь её "
    "вкладки (Проект, Генератор, Данные, Архитектура, Гиперпараметры, Обучение, "
    "Мониторинг, Анализ, Экспорт, Песочница), сценарии и нейросети. "
    "Ты помогаешь пользователю шаг за шагом: объясняешь, что происходит, "
    "что делать дальше и почему. Отвечай кратко, понятно и по-русски. "
    "Если в контексте есть проблемы (нет данных, не создана модель) — "
    "сначала скажи о них."
)

DEFAULT_SETTINGS: Dict[str, Any] = {
    "provider": "local",           # "local" | "yandex" | "sber"
    "api_key": "",
    "folder_id": "",
    "client_id": "",
    "client_secret": "",
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "auto_context": True,          # подставлять сценарий + данные пользователя
    "model_path": "",              # путь к локальной GGUF-модели (пусто = авто)
    # Агентский режим: Оракул может выполнять действия в программе (tools)
    "agent_enabled": False,
    "agent_max_steps": 8,
    "agent_temperature": 0.4,
}

_lock = threading.Lock()


def _deep_merge(defaults: dict, overrides: dict) -> dict:
    merged = dict(defaults)
    for k, v in (overrides or {}).items():
        if k in merged:
            merged[k] = v
    return merged


def load_settings() -> Dict[str, Any]:
    """Загружает настройки Оракула. Возвращает полный словарь."""
    data: Dict[str, Any] = {}
    try:
        p = Path(ORACULUM_SETTINGS_FILE)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
    except (json.JSONDecodeError, OSError):
        data = {}
    return _deep_merge(DEFAULT_SETTINGS, data)


def save_settings(settings: Dict[str, Any]) -> bool:
    """Сохраняет настройки Оракула атомарно. Возвращает True при успехе."""
    try:
        p = Path(ORACULUM_SETTINGS_FILE)
        p.parent.mkdir(parents=True, exist_ok=True)
        with _lock:
            tmp = p.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(_deep_merge(DEFAULT_SETTINGS, settings),
                          f, ensure_ascii=False, indent=2)
            if p.exists():
                p.unlink()
            tmp.rename(p)
        return True
    except OSError:
        return False


def build_system_prompt(settings: Dict[str, Any],
                        context_text: str = "") -> str:
    """
    Формирует системный промпт из шаблона и (при необходимости) контекста.

    Если в шаблоне есть подстановка {context}, она заменяется на context_text.
    Иначе контекст добавляется отдельным абзацем после шаблона.
    """
    template = (settings or {}).get("system_prompt") or DEFAULT_SYSTEM_PROMPT
    auto = (settings or {}).get("auto_context", True)

    if not auto or not context_text:
        return template

    if "{context}" in template:
        return template.replace("{context}", context_text)
    return f"{template}\n\n{context_text}"
