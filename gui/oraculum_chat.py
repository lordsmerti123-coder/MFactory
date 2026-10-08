"""
gui/oraculum_chat.py
====================
Мини-чат ИИ-агента «Оракул».

ОТВЕТСТВЕННОСТЬ:
  • Компактный виджет чата (можно сворачивать).
  • Отправка запросов агенту, отображение истории.
  • Кнопка контекстной подсказки.
  • Настройка API-провайдера (Яндекс / Сбер).

ЗАВИСИМОСТИ:
  • core.oraculum.agent → OraculumAgent
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit,
    QPushButton, QLabel, QDialog, QFormLayout, QComboBox,
    QDialogButtonBox, QMessageBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from core.oraculum.agent import OraculumAgent
from core.logger import get_logger

logger = get_logger(__name__)


class _OracleWorker(QThread):
    """Фоновый поток генерации ответа локальной моделью (не блокирует UI)."""

    finished = pyqtSignal(str)

    def __init__(self, agent: OraculumAgent, query: str, parent=None):
        super().__init__(parent)
        self.agent = agent
        self.query = query

    def run(self):
        try:
            response = self.agent.process_user_query(self.query)
        except Exception as e:
            logger.warning(f"Ошибка обработки запроса в фоне: {e}")
            response = f"⚠️ Не удалось получить ответ: {e}"
        self.finished.emit(response)


class OraculumChatWidget(QWidget):
    """Виджет мини-чата с Оракулом.

    docked=True — встраиваемый режим: виджет помещается внутрь главного окна
    (QSplitter), а не создаётся как отдельное окно, поэтому пользователь
    не может его «потерять» за другими окнами.
    """

    # Испускается при переключении тумблера «Агент» (True = включён).
    # MainWindow по нему подгружает локальную модель в фоне.
    agent_mode_toggled = pyqtSignal(bool)

    def __init__(self, agent: OraculumAgent, parent=None, docked: bool = False):
        super().__init__(parent)
        self.agent = agent
        self._docked = docked
        if not docked:
            self.setWindowFlags(Qt.Window)
            self.setWindowTitle("💬 Оракул — ИИ-помощник")
            self.setMinimumSize(340, 420)
        else:
            self.setMinimumWidth(300)
        self._collapsed = False
        self._saved_geometry = None
        self._busy = False
        self._worker = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # === Шапка ===
        header = QHBoxLayout()
        self.title_label = QLabel("💬 Оракул")
        self.title_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #c678dd;"
        )
        header.addWidget(self.title_label)
        header.addStretch()

        self.btn_hint = QPushButton("💡 Подсказка")
        self.btn_hint.setToolTip("Оракул посмотрит на текущий экран и даст совет.")
        self.btn_hint.clicked.connect(self._on_hint)
        header.addWidget(self.btn_hint)

        self.btn_settings = QPushButton("⚙️")
        self.btn_settings.setToolTip("Настройка внешнего API (Яндекс GPT / Сбер).")
        self.btn_settings.clicked.connect(self._open_settings)
        header.addWidget(self.btn_settings)

        layout.addLayout(header)

        # === Статус модели/провайдера ===
        self.status_label = QLabel("🔒 Локальные шаблоны (без интернета)")
        self.status_label.setStyleSheet(
            "color: #7f9f7f; font-size: 11px; padding: 2px 4px;"
            "background-color: #252526; border-radius: 4px;"
        )
        layout.addWidget(self.status_label)

        # === Режим ИИ-агента (Оракул может выполнять действия в программе) ===
        self.btn_agent = QPushButton("🤖 Агент")
        self.btn_agent.setCheckable(True)
        self.btn_agent.setToolTip(
            "Включить агентский режим: Оракул получает инструменты и может "
            "сам переключать вкладки, создавать модель, запускать обучение."
        )
        self.btn_agent.toggled.connect(self._on_agent_toggle)
        header.addWidget(self.btn_agent)
        self.btn_agent.setEnabled(False)  # активируется после инициализации настроек

        # === Область чата ===
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet(
            "QTextEdit {"
            "  background-color: #1e1e1e;"
            "  border: 1px solid #555;"
            "  color: #a9b7c6;"
            "  font-size: 12px;"
            "  padding: 6px;"
            "}"
        )
        layout.addWidget(self.chat_display, stretch=1)

        # === Ввод ===
        bottom = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Спроси Оракула...")
        self.input_field.returnPressed.connect(self.send_message)
        bottom.addWidget(self.input_field, stretch=1)

        self.btn_send = QPushButton("Отправить")
        self.btn_send.clicked.connect(self.send_message)
        bottom.addWidget(self.btn_send)

        layout.addLayout(bottom)

        self.add_message("Оракул", "Привет! Я твой помощник. Спрашивай о чём угодно.")

    # ============================================================
    # СООБЩЕНИЯ
    # ============================================================
    def send_message(self):
        query = self.input_field.text().strip()
        if not query:
            return
        if self._busy:
            return
        self.add_message("Вы", query)
        self.input_field.clear()

        if self.agent.is_local_model_loaded():
            # Локальная модель работает медленно — генерируем в фоне,
            # чтобы не замораживать интерфейс.
            self._run_async(query)
        else:
            response = self.agent.process_user_query(query)
            self.add_message("Оракул", response)

    def _run_async(self, query: str):
        self._busy = True
        self.btn_send.setEnabled(False)
        self.input_field.setEnabled(False)
        self._previous_status = self.status_label.text()
        self.status_label.setText("⏳ Оракул думает (локальная модель)...")
        self._worker = _OracleWorker(self.agent, query, self)
        self._worker.finished.connect(self._on_worker_done)
        self._worker.start()

    def _on_worker_done(self, text: str):
        self._busy = False
        self.btn_send.setEnabled(True)
        self.input_field.setEnabled(True)
        if hasattr(self, "_previous_status"):
            self.status_label.setText(self._previous_status)
        self.add_message("Оракул", text)

    def set_local_model_status(self, loaded: bool, msg: str = ""):
        """Обновляет строку статуса модели/провайдера."""
        if loaded:
            self.status_label.setText("🟢 Локальная модель загружена")
            self.status_label.setStyleSheet(
                "color: #a6e3a1; font-size: 11px; padding: 2px 4px;"
                "background-color: #252526; border-radius: 4px;"
            )
        else:
            self.status_label.setText(
                f"🔒 Шаблоны" + (f" — {msg}" if msg else " (без интернета)")
            )
            self.status_label.setStyleSheet(
                "color: #7f9f7f; font-size: 11px; padding: 2px 4px;"
                "background-color: #252526; border-radius: 4px;"
            )

    # ============================================================
    # ПРОВАЙДЕР И АГЕНТСКИЙ РЕЖИМ
    # ============================================================
    def init_agent_controls(self):
        """Синхронизирует тумблер «Агент» и статус провайдера с настройками."""
        try:
            agent_mode = self.agent.agent_mode_enabled()
            self.btn_agent.blockSignals(True)
            self.btn_agent.setChecked(agent_mode)
            self.btn_agent.blockSignals(False)
            self.btn_agent.setEnabled(True)
        except Exception:
            self.btn_agent.setEnabled(False)
        self.refresh_provider_status()

    def _on_agent_toggle(self, checked: bool):
        """Включает/выключает агентский режим (инструменты)."""
        try:
            self.agent.apply_settings({"agent_enabled": bool(checked)})
            if checked:
                self.add_message(
                    "Оракул",
                    "🤖 Агентский режим включён. Теперь я могу сам выполнять "
                    "действия: переключать вкладки, создавать модель, запускать "
                    "обучение. Напишите «что происходит» или «создай трансформер».",
                )
                # Подгружаем локальную модель в фоне, чтобы у агента был «мозг»
                # (иначе команды уйдут в статические ответы). MainWindow по этому
                # сигналу запустит асинхронную загрузку.
                self.agent_mode_toggled.emit(True)
            else:
                self.add_message("Оракул", "Агентский режим выключен.")
                self.agent_mode_toggled.emit(False)
        except Exception as e:
            self.add_message("Оракул", f"Не удалось переключить режим: {e}")
        self.refresh_provider_status()

    def refresh_provider_status(self):
        """Показывает, какой «мозг» сейчас отвечает: Яндекс GPT / Сбер / GGUF / шаблоны."""
        try:
            agent = self.agent
            parts = []
            agent_mode = agent.agent_mode_enabled()
            if agent_mode:
                parts.append("🤖 агент")
            if agent.api_adapter.is_available():
                provider = agent.api_provider()
                if provider == "yandex":
                    parts.append("🌐 Яндекс GPT")
                elif provider == "sber":
                    parts.append("🌐 Сбер GigaChat")
                else:
                    parts.append("🌐 Внешний ИИ")
            if agent.is_local_model_loaded():
                parts.append("🟢 GGUF")
            else:
                parts.append("🔒 шаблоны")
            self.status_label.setText("🧠 Оракул: " + " | ".join(parts))
            self.status_label.setStyleSheet(
                "color: #a6e3a1; font-size: 11px; padding: 2px 4px;"
                "background-color: #252526; border-radius: 4px;"
            )
        except Exception:
            pass

    def _on_hint(self):
        hint = self.agent.get_contextual_hint()
        self.add_message("Оракул", hint)

    def add_message(self, sender: str, text: str):
        color = "#4a88c7" if sender == "Вы" else "#a6e3a1"
        if sender == "Оракул":
            color = "#a6e3a1"
        html = (
            f'<p style="margin: 4px 0;">'
            f'<span style="color: {color}; font-weight: bold;">{sender}:</span> '
            f'<span style="color: #d4d4d4;">{self._escape(text)}</span>'
            f'</p>'
        )
        self.chat_display.append(html)
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    @staticmethod
    def _escape(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )

    def clear_chat(self):
        self.chat_display.clear()
        self.agent.clear_history()

    def closeEvent(self, event):
        """Корректно завершает фоновый поток генерации при закрытии."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(3000)
        super().closeEvent(event)

    # ============================================================
    # НАСТРОЙКИ API
    # ============================================================
    def _open_settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Настройки Оракула")
        dialog.setMinimumWidth(420)
        layout = QVBoxLayout(dialog)

        form = QFormLayout()

        self.provider_combo = QComboBox()
        self.provider_combo.addItems([
            "🔒 Локальные шаблоны (без интернета)",
            "🌐 Яндекс GPT",
            "🌐 Сбер GigaChat",
        ])
        self.provider_combo.setCurrentIndex(0)
        form.addRow("Провайдер:", self.provider_combo)

        self.api_key_edit = QLineEdit()
        self.api_key_edit.setPlaceholderText("API-ключ (для Яндекса)")
        form.addRow("API-ключ:", self.api_key_edit)

        self.folder_id_edit = QLineEdit()
        self.folder_id_edit.setPlaceholderText("folder_id (для Яндекса)")
        form.addRow("Folder ID:", self.folder_id_edit)

        self.client_id_edit = QLineEdit()
        self.client_id_edit.setPlaceholderText("client_id (для Сбера)")
        form.addRow("Client ID:", self.client_id_edit)

        self.client_secret_edit = QLineEdit()
        self.client_secret_edit.setPlaceholderText("client_secret (для Сбера)")
        self.client_secret_edit.setEchoMode(QLineEdit.Password)
        form.addRow("Client Secret:", self.client_secret_edit)

        layout.addLayout(form)

        info = QLabel(
            "⚠️ Ключи хранятся локально в сессии и не передаются третьим лицам. "
            "При отсутствии сети или ключей Оракул работает на локальных шаблонах."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #777; font-size: 11px;")
        layout.addWidget(info)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(lambda: self._apply_settings(dialog))
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        dialog.exec_()

    def _apply_settings(self, dialog: QDialog):
        index = self.provider_combo.currentIndex()
        if index == 0:
            self.agent.configure_api("local")
        elif index == 1:
            self.agent.configure_api(
                "yandex",
                api_key=self.api_key_edit.text().strip(),
                folder_id=self.folder_id_edit.text().strip(),
            )
        else:
            self.agent.configure_api(
                "sber",
                client_id=self.client_id_edit.text().strip(),
                client_secret=self.client_secret_edit.text().strip(),
            )
        self.refresh_provider_status()
        self.add_message("Оракул", "Настройки сохранены.")
        dialog.accept()
