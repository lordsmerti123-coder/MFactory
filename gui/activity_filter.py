"""
gui/activity_filter.py
======================
Qt-перехватчик событий для глобального логирования активности пользователя.

ОТВЕТСТВЕННОСТЬ:
  • Перехват кликов мышью, наведения курсора и нажатий клавиш.
  • Запись событий в core.activity_tracker (кольцевой буфер).
  • Защита от перегрузки: наведение/клавиши троттлятся по времени.

УСТАНОВКА (в main.py, после создания QApplication):
    from gui.activity_filter import install_activity_filter
    install_activity_filter(app)

ЗАВИСИМОСТИ:
  • PyQt5 (QtCore, QtGui)
  • core/activity_tracker.py
"""

from __future__ import annotations

import time

from PyQt5.QtCore import QObject, QEvent, Qt
from PyQt5.QtGui import QMouseEvent, QKeyEvent

from core.activity_tracker import activity_tracker

# Минимальный интервал (сек) между записями одного и того же типа события,
# чтобы наведение/клавиши не заваливали буфер.
HOVER_THROTTLE = 0.25
KEY_THROTTLE = 0.15


def _widget_label(widget) -> tuple:
    """Возвращает (класс, имя, текст) для виджета."""
    cls = type(widget).__name__ if widget is not None else "Unknown"
    name = ""
    text = ""
    try:
        name = widget.objectName() or ""
    except Exception:
        pass
    try:
        # Пытаемся извлечь читаемый текст кнопок/меток/чекбоксов
        for attr in ("text", "title", "windowTitle"):
            if hasattr(widget, attr):
                try:
                    v = getattr(widget, attr)
                    if callable(v):
                        v = v()
                    if isinstance(v, str) and v.strip():
                        text = v.strip()
                        break
                except Exception:
                    pass
        # text() у QLineEdit
        if not text and hasattr(widget, "text"):
            try:
                v = widget.text()
                if isinstance(v, str) and v.strip():
                    text = v.strip()
            except Exception:
                pass
    except Exception:
        pass
    # Для QTabBar — текущая вкладка
    if cls == "QTabBar":
        try:
            idx = widget.currentIndex()
            if idx >= 0:
                text = widget.tabText(idx)
        except Exception:
            pass
    return cls, name, (text[:60] if text else "")


class ActivityEventFilter(QObject):
    """Глобальный фильтр событий приложения."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_hover = 0.0
        self._last_key = 0.0
        self._last_hover_target = ""

    def eventFilter(self, obj, event):  # noqa: N802 (Qt naming)
        etype = event.type()
        now = time.time()

        try:
            if etype == QEvent.MouseButtonPress:
                self._on_click(obj, event)
            elif etype == QEvent.Enter:
                self._on_hover(obj, now, force=True)
            elif etype == QEvent.MouseMove:
                self._on_hover(obj, now, force=False)
            elif etype == QEvent.KeyPress:
                self._on_key(obj, event, now)
        except Exception:
            pass

        return super().eventFilter(obj, event)

    # ============================================================
    # ОБРАБОТЧИКИ
    # ============================================================
    def _on_click(self, obj, event: QMouseEvent):
        cls, name, text = _widget_label(obj)
        btn_map = {
            Qt.LeftButton: "ЛКМ",
            Qt.RightButton: "ПКМ",
            Qt.MiddleButton: "СКМ",
        }
        btn = btn_map.get(event.button(), "кнопка")
        pos = event.globalPos()
        detail = f"{btn} @ ({pos.x()}, {pos.y()})"
        if name:
            detail += f" | objectName={name}"
        activity_tracker.record(
            "click", cls, text, detail
        )

    def _on_hover(self, obj, now: float, force: bool):
        cls, name, text = _widget_label(obj)
        target = f"{cls}:{name}:{text}"
        if not force and target == self._last_hover_target:
            return
        if not force and (now - self._last_hover) < HOVER_THROTTLE:
            return
        self._last_hover = now
        self._last_hover_target = target
        detail = ""
        if name:
            detail = f"objectName={name}"
        activity_tracker.record("hover", cls, text, detail)

    def _on_key(self, obj, event: QKeyEvent, now: float):
        # Пропускаем парольные поля
        try:
            if hasattr(obj, "echoMode") and obj.echoMode() in (
                getattr(obj, "Password", 2), 2
            ):
                return
        except Exception:
            pass
        if (now - self._last_key) < KEY_THROTTLE:
            return
        self._last_key = now
        cls, name, text = _widget_label(obj)
        try:
            key_text = event.text() or ""
        except Exception:
            key_text = ""
        detail = ""
        if name:
            detail = f"objectName={name}"
        activity_tracker.record(
            "key", cls, text, f"key='{key_text}' {detail}".strip()
        )


_installed_filter = None


def install_activity_filter(app) -> ActivityEventFilter:
    """Устанавливает глобальный фильтр активности на QApplication."""
    global _installed_filter
    if _installed_filter is not None:
        return _installed_filter
    _installed_filter = ActivityEventFilter(app)
    app.installEventFilter(_installed_filter)
    return _installed_filter
