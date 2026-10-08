"""
gui/panels/logs_panel.py
========================
Панель просмотра логов приложения.

Реализует:
- Перехват всех логов через безопасный QTextEditLogger (через сигнал, без краша QTextCursor)
- Цветовую подсветку уровней: ERROR=красный, WARNING=жёлтый, INFO=серый, DEBUG=синий
- Фильтрацию по уровню
- Сохранение и очистку истории
- Защиту от дублирования handler'ов при повторном создании панели
- Интеграцию с SessionManager и HintEngine
- Контракт: принимает shared_state первым аргументом
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QPushButton, QLabel, QComboBox, QFileDialog,
    QGroupBox, QMessageBox, QCheckBox
)
from PyQt5.QtCore import QObject, pyqtSignal, Qt
from PyQt5.QtGui import QTextCursor

from core.logger import get_logger
from gui.hint_widget import HintWidget

logger = get_logger(__name__)


# ============================================================
# Излучатель сигнала (обёртка над QObject)
# logging.Handler не наследует QObject, поэтому нужен посредник
# ============================================================
class LogEmitter(QObject):
    log_signal = pyqtSignal(str, str)  # (level_name, formatted_message)


# ============================================================
# Безопасный Handler для QTextEdit
# ============================================================
class QTextEditLogger(logging.Handler):
    """
    Безопасный для потоков логгер для PyQt5.
    Отправляет сообщения через сигнал, а не напрямую в QTextEdit
    (предотвращает краш QTextCursor при обновлении из рабочего потока).
    """
    # Класс-атрибут для отслеживания уже подключённых экземпляров
    _active_instances = set()

    def __init__(self, emitter: LogEmitter):
        super().__init__()
        self.emitter = emitter
        self._is_active = True
        QTextEditLogger._active_instances.add(id(self))

    def emit(self, record):
        if not self._is_active:
            return
        try:
            msg = self.format(record)
            level_name = record.levelname  # DEBUG, INFO, WARNING, ERROR, CRITICAL
            self.emitter.log_signal.emit(level_name, msg)
        except Exception:
            self.handleError(record)

    def deactivate(self):
        """Отключает handler (чтобы он не слал сигналы в уничтоженный виджет)."""
        self._is_active = False
        QTextEditLogger._active_instances.discard(id(self))


# ============================================================
# Основная панель
# ============================================================
class LogsPanel(QWidget):
    """
    Панель логов.
    Сигналов нет (пассивный наблюдатель).
    """

    # Идентификатор для поиска существующего handler'а в логгере
    HANDLER_TAG = "LogsPanel_QTextEditHandler"

    def __init__(
        self,
        shared_state: Optional[Dict[str, Any]] = None,
        hint_engine: Optional[Any] = None,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.shared_state = shared_state or {}
        self.hint_engine = hint_engine
        self.log_handler: Optional[QTextEditLogger] = None
        self.emitter: Optional[LogEmitter] = None
        self.auto_scroll = True
        self.current_filter = "ALL"

        # Буфер всех сообщений для фильтрации
        self.all_logs = []  # list of (level, msg, timestamp)

        self.init_ui()
        self._setup_hints()
        self._attach_log_handler()

    # ========================================================
    # UI
    # ========================================================
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # === Верхняя панель: управление ===
        top_bar = QGroupBox("📝 Логи приложения")
        top_layout = QHBoxLayout()

        # Фильтр по уровню
        top_layout.addWidget(QLabel("Уровень:"))
        self.level_filter = QComboBox()
        self.level_filter.addItems(["ALL", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.level_filter.setCurrentText("INFO")
        self.level_filter.setToolTip(
            "Фильтр отображаемых логов.\n"
            "• ALL — все сообщения (много шума)\n"
            "• DEBUG — отладка (для разработчиков)\n"
            "• INFO — обычный ход работы (рекомендуется)\n"
            "• WARNING — предупреждения\n"
            "• ERROR — только ошибки"
        )
        self.level_filter.currentTextChanged.connect(self._on_filter_changed)
        top_layout.addWidget(self.level_filter)

        # Автоскролл
        self.auto_scroll_check = QCheckBox("Автоскролл")
        self.auto_scroll_check.setChecked(True)
        self.auto_scroll_check.setToolTip(
            "Автоматически прокручивать лог вниз при появлении новых записей.\n"
            "Снимите галочку, чтобы прочитать старые сообщения."
        )
        self.auto_scroll_check.stateChanged.connect(self._on_autoscroll_changed)
        top_layout.addWidget(self.auto_scroll_check)

        top_layout.addStretch()

        # Кнопка очистить
        self.btn_clear = QPushButton("🗑 Очистить")
        self.btn_clear.setToolTip("Удаляет всю историю логов из окна (файл на диске не трогается).")
        self.btn_clear.clicked.connect(self._clear_logs)
        top_layout.addWidget(self.btn_clear)

        # Кнопка сохранить
        self.btn_save = QPushButton("💾 Сохранить в файл")
        self.btn_save.setToolTip("Сохраняет текущую историю логов в текстовый файл.")
        self.btn_save.clicked.connect(self._save_logs)
        top_layout.addWidget(self.btn_save)

        # Кнопка открыть файл
        self.btn_open_log = QPushButton("📂 Открыть лог-файл")
        self.btn_open_log.setToolTip(
            "Открывает папку с лог-файлом приложения.\n"
            "Там лежит oracleai_latest.log с полной историей."
        )
        self.btn_open_log.clicked.connect(self._open_log_folder)
        top_layout.addWidget(self.btn_open_log)

        top_bar.setLayout(top_layout)
        layout.addWidget(top_bar)

        # === Статус ===
        self.status_label = QLabel("📊 Записей: 0")
        self.status_label.setStyleSheet("color: #a9b7c6; font-size: 11px;")
        layout.addWidget(self.status_label)

        # === Область логов ===
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setLineWrapMode(QTextEdit.NoWrap)
        self.text_area.setStyleSheet(
            "QTextEdit {"
            "  background-color: #1e1e1e;"
            "  border: 1px solid #555;"
            "  font-family: 'Consolas', 'Courier New', monospace;"
            "  font-size: 12px;"
            "  padding: 4px;"
            "  color: #a9b7c6;"
            "}"
        )
        self.text_area.setHtml(
            '<p style="color: #777; text-align: center; margin-top: 30px;">'
            'Лог пуст. Логи появятся по мере работы приложения.</p>'
        )
        layout.addWidget(self.text_area, stretch=1)

    # ========================================================
    # ПОДСКАЗКИ
    # ========================================================
    def _setup_hints(self):
        """Подключает HintWidget ко всем интерактивным элементам."""
        if not self.hint_engine:
            # Пытаемся достать из shared_state
            self.hint_engine = self.shared_state.get("hint_engine")
        if not self.hint_engine:
            return

        try:
            HintWidget(self.level_filter, self.hint_engine,
                       "logs_level_filter", "logs_panel")
            HintWidget(self.auto_scroll_check, self.hint_engine,
                       "logs_autoscroll", "logs_panel")
            HintWidget(self.btn_clear, self.hint_engine,
                       "logs_clear_btn", "logs_panel")
            HintWidget(self.btn_save, self.hint_engine,
                       "logs_save_btn", "logs_panel")
            HintWidget(self.btn_open_log, self.hint_engine,
                       "logs_open_folder_btn", "logs_panel")
        except Exception as e:
            logger.warning(f"Не удалось подключить подсказки к LogsPanel: {e}")

    # ========================================================
    # ПОДКЛЮЧЕНИЕ HANDLER'А (с защитой от дублирования)
    # ========================================================
    def _attach_log_handler(self):
        """
        Подключает QTextEditLogger к корневому логгеру.
        Проверяет, нет ли уже активного handler'а от LogsPanel.
        """
        root_logger = get_logger()

        # Ищем уже существующий handler с нашим тегом
        existing_handler = None
        for h in root_logger.handlers:
            if isinstance(h, QTextEditLogger) and getattr(h, '_logs_panel_tag', None) == self.HANDLER_TAG:
                existing_handler = h
                break

        if existing_handler is not None:
            # Уже есть — деактивируем старый и создаём новый (для нового виджета)
            existing_handler.deactivate()
            try:
                root_logger.removeHandler(existing_handler)
            except ValueError:
                pass

        # Создаём emitter и handler
        self.emitter = LogEmitter()
        self.log_handler = QTextEditLogger(self.emitter)
        self.log_handler._logs_panel_tag = self.HANDLER_TAG  # Метка для поиска
        self.log_handler.setFormatter(
            logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s',
                              datefmt='%H:%M:%S')
        )

        # Подключаем сигнал к методу отображения
        self.emitter.log_signal.connect(self._append_colored_log)

        # Регистрируем handler в корневом логгере
        root_logger.addHandler(self.log_handler)
        logger.debug("LogsPanel: handler подключён к логгеру.")

    # ========================================================
    # ОТОБРАЖЕНИЕ ЛОГОВ
    # ========================================================
    def _append_colored_log(self, level_name: str, message: str):
        """
        Добавляет сообщение с цветовой подсветкой уровня.
        Проверяет фильтр перед отображением.
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.all_logs.append((level_name, message, timestamp))

        # Проверяем фильтр
        if not self._passes_filter(level_name):
            self._update_status()
            return

        color = self._level_color(level_name)
        icon = self._level_icon(level_name)

        html = (
            f'<p style="margin: 1px 0; color: {color};">'
            f'{icon} {self._escape_html(message)}'
            f'</p>'
        )
        self.text_area.append(html)

        # Автоскролл
        if self.auto_scroll:
            cursor = self.text_area.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.text_area.setTextCursor(cursor)

        self._update_status()

    def _level_color(self, level: str) -> str:
        """Цвет для уровня лога."""
        colors = {
            "CRITICAL": "#ff5555",  # ярко-красный
            "ERROR": "#e06c75",     # красный
            "WARNING": "#e5c07b",   # жёлтый
            "INFO": "#a9b7c6",      # серый (по умолчанию)
            "DEBUG": "#6a9fcf",     # синий
        }
        return colors.get(level, "#a9b7c6")

    def _level_icon(self, level: str) -> str:
        """Эмодзи для уровня лога."""
        icons = {
            "CRITICAL": "💀",
            "ERROR": "❌",
            "WARNING": "⚠️",
            "INFO": "ℹ️",
            "DEBUG": "🔍",
        }
        return icons.get(level, "•")

    @staticmethod
    def _escape_html(text: str) -> str:
        """Экранирует спецсимволы HTML."""
        return (
            text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )

    def _passes_filter(self, level_name: str) -> bool:
        """Проверяет, проходит ли уровень через текущий фильтр."""
        if self.current_filter == "ALL":
            return True
        level_order = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        try:
            filter_idx = level_order.index(self.current_filter)
            msg_idx = level_order.index(level_name)
            return msg_idx >= filter_idx
        except ValueError:
            return True

    # ========================================================
    # ОБРАБОТЧИКИ СОБЫТИЙ UI
    # ========================================================
    def _on_filter_changed(self, new_filter: str):
        """Смена фильтра — перерисовывает всю историю."""
        self.current_filter = new_filter
        self._rerender_logs()
        self._log_action("filter_changed", new_filter)

    def _on_autoscroll_changed(self, state: int):
        """Включение/выключение автоскролла."""
        self.auto_scroll = (state == Qt.Checked)

    def _rerender_logs(self):
        """Перерисовывает всю историю с учётом текущего фильтра."""
        self.text_area.clear()
        shown = 0
        for level, msg, ts in self.all_logs:
            if not self._passes_filter(level):
                continue
            color = self._level_color(level)
            icon = self._level_icon(level)
            html = (
                f'<p style="margin: 1px 0; color: {color};">'
                f'{icon} {self._escape_html(msg)}'
                f'</p>'
            )
            self.text_area.append(html)
            shown += 1
        self._update_status(shown)

    def _update_status(self, shown: Optional[int] = None):
        """Обновляет счётчик записей."""
        total = len(self.all_logs)
        if shown is None:
            # Считаем сколько прошло через фильтр
            shown = sum(1 for lvl, _, _ in self.all_logs if self._passes_filter(lvl))
        self.status_label.setText(
            f"📊 Записей: {shown} из {total} | Фильтр: {self.current_filter}"
        )

    def _clear_logs(self):
        """Очищает историю (но НЕ файл на диске)."""
        if not self.all_logs:
            return
        reply = QMessageBox.question(
            self,
            "Очистить логи?",
            f"Удалить {len(self.all_logs)} записей из окна?\n"
            f"Файл oracleai_latest.log на диске не будет тронут.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        self.all_logs.clear()
        self.text_area.setHtml(
            '<p style="color: #777; text-align: center; margin-top: 30px;">'
            'Лог очищен.</p>'
        )
        self._update_status(0)
        self._log_action("clear_logs", "История очищена")

    def _save_logs(self):
        """Сохраняет историю в текстовый файл."""
        if not self.all_logs:
            QMessageBox.information(self, "Пусто", "Нет записей для сохранения.")
            return

        default_name = f"oracleai_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить логи", default_name,
            "Текст (*.txt);;Все файлы (*)"
        )
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                for level, msg, ts in self.all_logs:
                    f.write(f"[{ts}] [{level}] {msg}\n")
            QMessageBox.information(
                self, "Успех",
                f"Логи сохранены в:\n{path}\n\nЗаписей: {len(self.all_logs)}"
            )
            self._log_action("save_logs", f"Сохранено в {path}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить:\n{e}")

    def _open_log_folder(self):
        """Открывает папку с лог-файлом."""
        try:
            from config import LOGS_DIR
            log_dir = Path(LOGS_DIR)
            if not log_dir.exists():
                QMessageBox.warning(self, "Папка не найдена", f"Папка {log_dir} не существует.")
                return

            import os
            import platform
            import subprocess

            if platform.system() == "Windows":
                os.startfile(str(log_dir))
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", str(log_dir)])
            else:
                subprocess.Popen(["xdg-open", str(log_dir)])

            self._log_action("open_log_folder", str(log_dir))
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось открыть папку:\n{e}")

    # ========================================================
    # СЛУЖЕБНЫЕ
    # ========================================================
    def refresh(self):
        """
        Вызывается из MainWindow при переключении на эту вкладку.
        Перерисовывает логи с учётом фильтра.
        """
        self._rerender_logs()

    def _log_action(self, action: str, details: str = ""):
        """Логирует действие пользователя в SessionManager."""
        try:
            session_mgr = self.shared_state.get("session_manager")
            if session_mgr:
                session_mgr.log_action("logs_panel", action, details)
        except Exception:
            pass

    # ========================================================
    # ЖИЗНЕННЫЙ ЦИКЛ
    # ========================================================
    def showEvent(self, event):
        """При показе вкладки — обновляем отображение."""
        super().showEvent(event)
        self.refresh()

    def closeEvent(self, event):
        """При уничтожении панели — отключаем handler."""
        if self.log_handler:
            self.log_handler.deactivate()
            try:
                get_logger().removeHandler(self.log_handler)
            except ValueError:
                pass
        super().closeEvent(event)