"""
gui/widgets/highlight.py
========================
Наглядная подсветка виджетов и вкладок для режима «Мастер».

ОТВЕТСТВЕННОСТЬ:
  • highlight_widget — свечение вокруг виджета (цветная рамка).
  • highlight_tab / clear_tab_highlight — подсветка вкладки QTabWidget.
  • Безопасные no-op, если эффект не применился.

ЗАВИСИМОСТИ:
  • PyQt5 (QtWidgets.QGraphicsDropShadowEffect, QtGui.QColor)
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtWidgets import QWidget, QTabWidget, QGraphicsDropShadowEffect
from PyQt5.QtGui import QColor

# Цвета подсветки
HIGHLIGHT_COLOR = "#4a88c7"   # синий
DONE_COLOR = "#a6e3a1"        # зелёный
WARN_COLOR = "#e5c07b"        # жёлтый


def highlight_widget(widget: Optional[QWidget],
                     color: str = HIGHLIGHT_COLOR,
                     blur: int = 10) -> bool:
    """Накладывает свечение на виджет. Возвращает True при успехе."""
    if widget is None:
        return False
    try:
        effect = QGraphicsDropShadowEffect(widget)
        effect.setBlurRadius(blur)
        effect.setColor(QColor(color))
        effect.setOffset(0, 0)
        widget.setGraphicsEffect(effect)
        return True
    except Exception:
        return False


def clear_highlight(widget: Optional[QWidget]):
    """Снимает свечение с виджета."""
    if widget is None:
        return
    try:
        widget.setGraphicsEffect(None)
    except Exception:
        pass


def highlight_tab(tab_widget: Optional[QTabWidget],
                  index: int,
                  color: str = HIGHLIGHT_COLOR) -> bool:
    """Подсвечивает вкладку под номером index. Возвращает True при успехе."""
    if tab_widget is None or index < 0 or index >= tab_widget.count():
        return False
    try:
        tab_bar = tab_widget.tabBar()
        tab_bar.setStyleSheet(
            f"QTabBar::tab:selected {{ background-color: {color}; "
            "color: #1e1e1e; font-weight: bold; }"
        )
        return True
    except Exception:
        return False


def clear_tab_highlight(tab_widget: Optional[QTabWidget]):
    """Снимает подсветку с вкладок."""
    if tab_widget is None:
        return
    try:
        tab_widget.tabBar().setStyleSheet("")
    except Exception:
        pass
