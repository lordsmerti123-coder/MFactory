"""
core/activity_tracker.py
========================
Глобальный трекер активности пользователя для панели администратора.

ОТВЕТСТВЕННОСТЬ:
  • Кольцевой буфер событий активности: наведение мыши, клики, нажатия
    клавиш, ошибки и действия пользователя.
  • Потокобезопасная запись из любого потока (GUI, обучение, фоновые задачи).
  • Подписка на LogEventBus для автоматического сбора ошибок и действий.

ЗАВИСИМОСТИ:
  • core/logger.py → LogEventBus, LogCategory (для сбора ошибок/действий)
  • НЕ импортирует PyQt5: Qt-специфичный перехват событий живёт в
    gui/activity_filter.py.

ИСПОЛЬЗОВАНИЕ:
    from core.activity_tracker import activity_tracker
    activity_tracker.record("click", "QPushButton", "Создать модель")
    events = activity_tracker.get_recent(200)
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ActivityEvent:
    """Одно событие активности пользователя."""
    timestamp: str
    category: str          # "hover" | "click" | "key" | "error" | "action" | "system"
    target: str            # класс виджета (QPushButton, QLineEdit, ...) или модуль
    name: str              # текст/имя элемента (текст кнопки, objectName)
    details: str           # доп. информация (координаты, сообщение ошибки)
    user: str = ""         # текущий пользователь
    panel: str = ""        # вкладка/панель, если известна

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "category": self.category,
            "target": self.target,
            "name": self.name,
            "details": self.details,
            "user": self.user,
            "panel": self.panel,
        }


class ActivityTracker:
    """Потокобезопасный кольцевой буфер активности."""

    def __init__(self, maxlen: int = 8000):
        self._buffer: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._current_user = ""

    # ============================================================
    # КОНТЕКСТ
    # ============================================================
    def set_user(self, username: str):
        self._current_user = username

    def get_user(self) -> str:
        return self._current_user

    # ============================================================
    # ЗАПИСЬ
    # ============================================================
    def record(self, category: str, target: str = "", name: str = "",
               details: str = "", panel: str = ""):
        event = ActivityEvent(
            timestamp=datetime.now().isoformat(timespec="milliseconds"),
            category=category,
            target=target or "",
            name=name or "",
            details=details or "",
            user=self._current_user,
            panel=panel or "",
        )
        with self._lock:
            self._buffer.append(event)

    # ============================================================
    # ЧТЕНИЕ
    # ============================================================
    def get_recent(self, n: int = 200,
                   category: Optional[str] = None) -> List[ActivityEvent]:
        with self._lock:
            items = list(self._buffer)[-n:]
        if category:
            items = [e for e in items if e.category == category]
        return items

    def get_recent_dicts(self, n: int = 200,
                         category: Optional[str] = None) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.get_recent(n, category)]

    def get_errors(self, n: int = 100) -> List[ActivityEvent]:
        return self.get_recent(n, category="error")

    def get_clicks(self, n: int = 200) -> List[ActivityEvent]:
        return self.get_recent(n, category="click")

    def count(self) -> int:
        with self._lock:
            return len(self._buffer)

    def clear(self):
        with self._lock:
            self._buffer.clear()

    # ============================================================
    # ПОДПИСКА НА ЛОГИ (ошибки + действия)
    # ============================================================
    def attach_to_log_bus(self):
        """Подписывает трекер на LogEventBus для сбора ошибок и действий."""
        try:
            from core.logger import log_event_bus, LogCategory, LogSubscriber

            class _Bridge(LogSubscriber):
                def __init__(self, tracker):
                    self.tracker = tracker

                def on_log_entry(self, entry):
                    cat = entry.category
                    if cat == LogCategory.ERROR:
                        self.tracker.record(
                            "error", entry.module or "system",
                            "", entry.message, panel=entry.panel,
                        )
                    elif cat == LogCategory.USER_ACTION:
                        self.tracker.record(
                            "action", entry.module or "gui",
                            entry.action or "", entry.message, panel=entry.panel,
                        )

            log_event_bus.subscribe(_Bridge(self))
        except Exception:
            pass


# Глобальный экземпляр (единая точка для всего приложения).
activity_tracker = ActivityTracker()
activity_tracker.attach_to_log_bus()
