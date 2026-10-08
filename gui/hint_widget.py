"""
gui/hint_widget.py
==================
Универсальный хелпер подсказок. НЕ является визуальным виджетом.
Подключается к любому QWidget и перехватывает его enterEvent/leaveEvent.
При наведении запрашивает подсказку у HintEngine и отображает её.

Зависит от: core/hint_engine.py (может быть ещё не готов — обрабатываем).
Зависит от: core/contracts.py (типы).
Зависит от: core/session.py (SessionState).

Использование в любой панели:
    from gui.hint_widget import HintWidget
    HintWidget(self.lr_spin, hint_engine, "lr_spin", "hyperparams_panel")

Поддержка 4 режимов:
    - "static"  → только правило-ориентированные подсказки
    - "ai"      → только ИИ-подсказки (если доступны)
    - "hybrid"  → ИИ если есть, иначе статика (по умолчанию)
    - "none"    → ничего не показывается
"""

import logging
from typing import Any, Optional

from PyQt5.QtWidgets import (
    QWidget, QLabel, QApplication, QToolTip
)
from PyQt5.QtCore import (
    QObject, Qt, QTimer, QPoint, QEvent, QSize
)
from PyQt5.QtGui import QCursor, QFont, QPalette

from core.logger import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────
# Безопасный импорт модулей, которые могут быть ещё не написаны
# ──────────────────────────────────────────────
try:
    from core.hint_engine import HintEngine, HintContext
    HINT_ENGINE_AVAILABLE = True
except ImportError:
    HINT_ENGINE_AVAILABLE = False
    logger.warning("core.hint_engine не найден — подсказки будут пустыми.")

try:
    from core.contracts import SessionState
    CONTRACTS_AVAILABLE = True
except ImportError:
    CONTRACTS_AVAILABLE = False
    SessionState = None
    logger.warning("core.contracts не найден — SessionState недоступен.")


# ──────────────────────────────────────────────
# Константы оформления
# ──────────────────────────────────────────────
HINT_FONT_SIZE = 12
HINT_MAX_WIDTH = 420
HINT_PADDING = 10
HINT_SHOW_DELAY_MS = 350       # задержка перед показом (мс)
HIDE_ON_LEAVE_DELAY_MS = 100   # задержка скрытия при уходе курсора
HINT_BG_COLOR = "#1e2a3a"
HINT_BORDER_COLOR = "#4a88c7"
HINT_TEXT_COLOR = "#d4dae2"
HINT_AI_PREFIX_COLOR = "#7ec8e3"


# ──────────────────────────────────────────────
# Всплывающий виджет подсказки ( QLabel в frameless окне )
# ──────────────────────────────────────────────
class _HintPopup(QWidget):
    """
    Внутренний всплывающий виджет. Полупрозрачное окно без рамки,
    отображающее текст подсказки рядом с курсором.
    """

    def __init__(self):
        super().__init__(
            None,
            Qt.ToolTip | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setFocusPolicy(Qt.NoFocus)

        self._label = QLabel(self)
        self._label.setWordWrap(True)
        self._label.setMaximumWidth(HINT_MAX_WIDTH)
        self._label.setTextFormat(Qt.RichText)

        font = QFont("Segoe UI", HINT_FONT_SIZE)
        self._label.setFont(font)
        self._label.setStyleSheet(
            f"""
            QLabel {{
                background-color: {HINT_BG_COLOR};
                color: {HINT_TEXT_COLOR};
                border: 1px solid {HINT_BORDER_COLOR};
                border-radius: 6px;
                padding: {HINT_PADDING}px;
            }}
            """
        )

        from PyQt5.QtWidgets import QVBoxLayout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)

    def show_hint(self, text: str, global_pos: QPoint):
        """Показать подсказку рядом с указанной глобальной позицией."""
        if not text or not text.strip():
            return

        self._label.setText(text)
        self._label.adjustSize()
        self.adjustSize()

        # Позиционирование: стараемся не выходить за пределы экрана
        screen = QApplication.primaryScreen()
        if screen:
            screen_geo = screen.availableGeometry()
            x = global_pos.x() + 16
            y = global_pos.y() + 20

            if x + self.width() > screen_geo.right():
                x = global_pos.x() - self.width() - 8
            if y + self.height() > screen_geo.bottom():
                y = global_pos.y() - self.height() - 8

            x = max(screen_geo.left(), x)
            y = max(screen_geo.top(), y)
            self.move(x, y)

        self.show()
        self.raise_()

    def hide_hint(self):
        self.hide()


# ──────────────────────────────────────────────
# Основной класс хелпера
# ──────────────────────────────────────────────
class HintWidget(QObject):
    """
    Хелпер, привязываемый к любому QWidget.
    Перехватывает enterEvent / leaveEvent и показывает контекстную подсказку.

    Параметры:
        parent_widget : QWidget
            Виджет, к которому привязываем подсказку.
        hint_engine   : HintEngine | None
            Экземпляр движка подсказок. Может быть None (заглушка).
        widget_id     : str
            Уникальный идентификатор элемента, напр. "lr_spin".
        panel_name    : str
            Имя панели, напр. "hyperparams_panel".
        shared_state  : dict | None
            Ссылка на shared_state главного окна (для контекста).
    """

    _popup: Optional[_HintPopup] = None  # один общий попап на всё приложение

    def __init__(
        self,
        parent_widget: QWidget,
        hint_engine: Any = None,
        widget_id: str = "",
        panel_name: str = "",
        shared_state: Optional[dict] = None,
    ):
        super().__init__(parent_widget)
        self.parent_widget = parent_widget
        self.hint_engine = hint_engine
        self.widget_id = widget_id
        self.panel_name = panel_name
        self.shared_state = shared_state or {}

        self._show_timer = QTimer(self)
        self._show_timer.setSingleShot(True)
        self._show_timer.timeout.connect(self._do_show_hint)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._do_hide_hint)

        self._install_event_filter()

    # ──────────────────────────────────────────
    # Установка фильтра событий
    # ──────────────────────────────────────────
    def _install_event_filter(self):
        """Перехватываем события виджета."""
        if self.parent_widget is not None:
            self.parent_widget.installEventFilter(self)

    def eventFilter(self, obj, event: QEvent) -> bool:
        """Обработка событий виджета-хозяина."""
        if obj is not self.parent_widget:
            return super().eventFilter(obj, event)

        if event.type() == QEvent.Enter:
            self._hide_timer.stop()
            self._show_timer.start(HINT_SHOW_DELAY_MS)
            return False  # не блокируем событие

        elif event.type() == QEvent.Leave:
            self._show_timer.stop()
            self._hide_timer.start(HIDE_ON_LEAVE_DELAY_MS)
            return False

        elif event.type() == QEvent.FocusOut:
            self._show_timer.stop()
            self._do_hide_hint()
            return False

        elif event.type() == QEvent.Hide:
            self._show_timer.stop()
            self._do_hide_hint()
            return False

        return super().eventFilter(obj, event)

    # ──────────────────────────────────────────
    # Показ подсказки
    # ──────────────────────────────────────────
    def _do_show_hint(self):
        """Запрашивает подсказку и показывает попап."""
        hint_text = self._get_hint_text()
        if not hint_text:
            return

        global_pos = QCursor.pos()

        # Используем общий попап
        if HintWidget._popup is None:
            HintWidget._popup = _HintPopup()

        HintWidget._popup.show_hint(hint_text, global_pos)

    def _do_hide_hint(self):
        """Скрывает попап."""
        if HintWidget._popup is not None:
            HintWidget._popup.hide_hint()

    # ──────────────────────────────────────────
    # Получение текста подсказки
    # ──────────────────────────────────────────
    def _get_hint_text(self) -> str:
        """
        Формирует текст подсказки через HintEngine.
        Если движок недоступен — возвращает пустую строку.
        """
        # Режим "none" — ничего не показываем
        hint_mode = self.shared_state.get("hint_mode", "hybrid")
        if hint_mode == "none":
            return ""

        # Если движок не передан или не импортирован
        if self.hint_engine is None or not HINT_ENGINE_AVAILABLE:
            return ""

        try:
            context = self._build_context()
            return self.hint_engine.get_hint(context)
        except Exception as e:
            logger.debug(f"HintWidget ошибка получения подсказки: {e}")
            return ""

    def _build_context(self) -> Any:
        """Собирает HintContext для движка."""
        if not HINT_ENGINE_AVAILABLE:
            return None

        session = self.shared_state.get("session", None)
        model_config = self.shared_state.get("config", None)
        dataset_meta = None
        dataset = self.shared_state.get("dataset", None)
        if dataset and isinstance(dataset, dict):
            dataset_meta = dataset.get("meta", None)

        # Пытаемся получить текущее значение виджета
        current_value = self._get_widget_value()

        return HintContext(
            widget_id=self.widget_id,
            panel_name=self.panel_name,
            model_config=model_config,
            dataset_meta=dataset_meta,
            session=session,
            current_value=current_value,
        )

    def _get_widget_value(self) -> Any:
        """Пытается извлечь текущее значение привязанного виджета."""
        w = self.parent_widget
        if w is None:
            return None
        try:
            # QSpinBox / QDoubleSpinBox
            if hasattr(w, "value"):
                return w.value()
            # QLineEdit
            if hasattr(w, "text"):
                return w.text()
            # QComboBox
            if hasattr(w, "currentText"):
                return w.currentText()
            # QCheckBox
            if hasattr(w, "isChecked"):
                return w.isChecked()
        except Exception:
            pass
        return None

    # ──────────────────────────────────────────
    # Публичное управление
    # ──────────────────────────────────────────
    def show_hint_now(self):
        """Принудительно показать подсказку (например, по кнопке)."""
        self._show_timer.stop()
        self._do_show_hint()

    def hide_hint_now(self):
        """Принудительно скрыть."""
        self._hide_timer.stop()
        self._do_hide_hint()

    def update_shared_state(self, shared_state: dict):
        """Обновить ссылку на shared_state (при смене контекста)."""
        self.shared_state = shared_state

    def detach(self):
        """Отвязать от виджета (при удалении)."""
        self._show_timer.stop()
        self._hide_timer.stop()
        if self.parent_widget is not None:
            self.parent_widget.removeEventFilter(self)


# ──────────────────────────────────────────────
# Утилита массовой привязки
# ──────────────────────────────────────────────
def attach_hints(
    widgets_map: dict,
    hint_engine: Any,
    panel_name: str,
    shared_state: dict,
) -> list:
    """
    Массово привязывает подсказки к виджетам.

    Параметры:
        widgets_map : dict
            Словарь вида { "widget_id": QWidget, ... }
        hint_engine : HintEngine
            Экземпляр движка.
        panel_name  : str
            Имя панели.
        shared_state : dict
            Общее состояние.

    Возвращает:
        Список созданных HintWidget (для возможного открепления).

    Пример:
        attach_hints(
            {"lr_spin": self.lr_spin, "batch_spin": self.batch_spin},
            hint_engine,
            "hyperparams_panel",
            self.shared_state,
        )
    """
    helpers = []
    for widget_id, widget in widgets_map.items():
        if widget is not None:
            h = HintWidget(
                parent_widget=widget,
                hint_engine=hint_engine,
                widget_id=widget_id,
                panel_name=panel_name,
                shared_state=shared_state,
            )
            helpers.append(h)
    return helpers


# ──────────────────────────────────────────────
# Быстрая заглушка-подсказка (если нет движка)
# ──────────────────────────────────────────────
def set_fallback_tooltip(widget: QWidget, text: str):
    """
    Если HintEngine недоступен, можно хотя бы поставить стандартный tooltip.
    Вызывается как запасной вариант.
    """
    if widget is not None and text:
        widget.setToolTip(text)