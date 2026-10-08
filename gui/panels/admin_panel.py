"""
gui/panels/admin_panel.py
=========================
Панель администратора (дебаг-панель) OracleAI Studio v2.0.

ДОСТУПНА ТОЛЬКО ПОЛЬЗОВАТЕЛЮ-АДМИНИСТРАТОРУ (ник из config.ADMIN_USERNAME).

ОТВЕТСТВЕННОСТЬ:
  • Полные логи приложения (текущие + по пользователям).
  • Логирование действий пользователя (мышь, клики, клавиши, ошибки).
  • Состояние нейросети: модель, параметры, память.
  • Состояние подключений/API: провайдер, ключи (маскированно), локальная модель.
  • Что «думает» Оракул (анализ экрана, последние ответы).
  • Ресурсы: RAM / VRAM / CPU / процесс.
  • Управление пользователями и их хранилищем.
  • Настройка чата Оракула (провайдер, ключи, системный промпт, автоконтекст).

ЗАВИСИМОСТИ:
  • shared_state (model, config, oraculum, user_manager, session_manager, ...)
  • core.activity_tracker → activity_tracker
  • core.oraculum.agent → OraculumAgent
  • PyQt5
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QTextEdit,
    QLabel, QPushButton, QComboBox, QCheckBox, QLineEdit, QGroupBox,
    QFormLayout, QMessageBox, QSpinBox, QDoubleSpinBox,
)
from PyQt5.QtCore import Qt, QTimer, QObject, pyqtSignal
from PyQt5.QtGui import QTextCursor

from core.logger import get_logger, log_event_bus, LogSubscriber
from core.activity_tracker import activity_tracker

logger = get_logger(__name__)

REFRESH_MS = 2000


# ============================================================
# БЕЗОПАСНЫЙ СБОР СИСТЕМНОЙ ИНФОРМАЦИИ
# ============================================================
def gather_system_info() -> Dict[str, Any]:
    """Собирает RAM/CPU/GPU/процесс. Никогда не бросает исключений."""
    info: Dict[str, Any] = {}
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["ram"] = {
            "total_mb": round(vm.total / 1048576, 1),
            "used_mb": round(vm.used / 1048576, 1),
            "available_mb": round(vm.available / 1048576, 1),
            "percent": round(vm.percent, 1),
        }
        info["cpu_percent"] = round(psutil.cpu_percent(interval=None), 1)
        info["cpu_count"] = psutil.cpu_count(logical=True)
        proc = psutil.Process()
        mem = proc.memory_info()
        info["process"] = {
            "rss_mb": round(mem.rss / 1048576, 1),
            "vms_mb": round(getattr(mem, "vms", 0) / 1048576, 1),
            "cpu_percent": round(proc.cpu_percent(interval=None), 1),
            "threads": proc.num_threads(),
        }
    except Exception:
        pass

    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            total = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
            info["gpu_total_mb"] = round(total / 1048576, 1)
            info["gpu_allocated_mb"] = round(torch.cuda.memory_allocated(0) / 1048576, 1)
            info["gpu_reserved_mb"] = round(torch.cuda.memory_reserved(0) / 1048576, 1)
    except Exception:
        info["cuda_available"] = False
    return info


def _html_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )


# ============================================================
# ЖИВОЙ ПРОСМОТР ЛОГОВ (подписка на глобальную шину log_event_bus)
# ============================================================
class _AdminLogEmitter(QObject):
    log_signal = pyqtSignal(str, str)


class _AdminLogBridge(LogSubscriber):
    """Подписчик на log_event_bus, пересылающий записи в панель админа."""

    def __init__(self, emitter: _AdminLogEmitter):
        self.emitter = emitter
        self._active = True

    def on_log_entry(self, entry):
        if not self._active:
            return
        try:
            line = (
                f"{entry.timestamp} [{entry.level}] "
                f"{entry.module} - {entry.message}"
            )
            self.emitter.log_signal.emit(entry.level, line)
        except Exception:
            pass

    def deactivate(self):
        self._active = False


# ============================================================
# ПАНЕЛЬ АДМИНИСТРАТОРА
# ============================================================
class AdminPanel(QWidget):
    """Полноценная панель администратора."""

    HANDLER_TAG = "AdminPanel_LogHandler"

    def __init__(self, shared_state: Dict[str, Any], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.shared_state = shared_state
        self._log_bridge: Optional[_AdminLogBridge] = None
        self._log_emitter: Optional[_AdminLogEmitter] = None
        self._log_lines: List[str] = []

        self.init_ui()
        self._attach_log_handler()

        # Таймер автообновления динамических вкладок
        self._timer = QTimer(self)
        self._timer.setInterval(REFRESH_MS)
        self._timer.timeout.connect(self._auto_refresh)
        self._timer.start()

        self.refresh()

    # ============================================================
    # UI
    # ============================================================
    def init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # --- Сводка сверху ---
        self.summary_label = QLabel("🛡 Панель администратора")
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextFormat(Qt.RichText)
        self.summary_label.setStyleSheet(
            "font-size: 13px; color: #a9b7c6; padding: 6px; "
            "background-color: #2b2b2b; border: 1px solid #555; border-radius: 4px;"
        )
        root.addWidget(self.summary_label)

        # --- Вкладки ---
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        root.addWidget(self.tabs, stretch=1)

        self._build_logs_tab()
        self._build_activity_tab()
        self._build_network_tab()
        self._build_api_tab()
        self._build_resources_tab()
        self._build_users_tab()

    def _build_logs_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Полный лог приложения (DEBUG+)"))
        bar.addStretch()
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["ALL", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.log_level_combo.setCurrentText("ALL")
        self.log_level_combo.currentTextChanged.connect(self._rerender_logs)
        bar.addWidget(self.log_level_combo)
        btn_clear = QPushButton("Очистить")
        btn_clear.clicked.connect(self._clear_logs)
        bar.addWidget(btn_clear)
        lay.addLayout(bar)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QTextEdit.NoWrap)
        self.log_view.setStyleSheet(
            "font-family: 'Consolas', monospace; font-size: 11px; "
            "background-color: #1e1e1e; color: #a9b7c6; border: 1px solid #555;"
        )
        lay.addWidget(self.log_view)
        self.tabs.addTab(w, "📝 Логи")

    def _build_activity_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Действия пользователя (мышь / клики / клавиши / ошибки)"))
        bar.addStretch()
        self.activity_cat_combo = QComboBox()
        self.activity_cat_combo.addItems(
            ["all", "hover", "click", "key", "error", "action"]
        )
        self.activity_cat_combo.currentTextChanged.connect(self._refresh_activity)
        bar.addWidget(self.activity_cat_combo)
        btn_clear_act = QPushButton("Очистить")
        btn_clear_act.clicked.connect(lambda: (activity_tracker.clear(), self._refresh_activity()))
        bar.addWidget(btn_clear_act)
        lay.addLayout(bar)

        self.activity_view = QTextEdit()
        self.activity_view.setReadOnly(True)
        self.activity_view.setLineWrapMode(QTextEdit.NoWrap)
        self.activity_view.setStyleSheet(
            "font-family: 'Consolas', monospace; font-size: 11px; "
            "background-color: #1e1e1e; color: #d4d4d4; border: 1px solid #555;"
        )
        lay.addWidget(self.activity_view)
        self.tabs.addTab(w, "🖱 Действия")

    def _build_network_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.network_view = QTextEdit()
        self.network_view.setReadOnly(True)
        self.network_view.setStyleSheet(
            "font-family: 'Consolas', monospace; font-size: 12px; "
            "background-color: #1e1e1e; color: #a9b7c6; border: 1px solid #555;"
        )
        lay.addWidget(self.network_view)
        self.tabs.addTab(w, "🧠 Нейросеть")

    def _build_api_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        # --- Настройки Оракула ---
        grp = QGroupBox("Настройки чата Оракула (провайдер / ключи / автопромпт)")
        form = QFormLayout(grp)

        self.provider_combo = QComboBox()
        self.provider_combo.addItems([
            "🔒 Локальная модель (GGUF)",
            "🌐 Яндекс GPT",
            "🌐 Сбер GigaChat",
        ])
        form.addRow("Провайдер:", self.provider_combo)

        self.api_key_edit = QLineEdit()
        self.api_key_edit.setPlaceholderText("API-ключ Яндекса")
        form.addRow("API-ключ (Яндекс):", self.api_key_edit)

        self.folder_id_edit = QLineEdit()
        self.folder_id_edit.setPlaceholderText("folder_id Яндекса")
        form.addRow("Folder ID:", self.folder_id_edit)

        self.client_id_edit = QLineEdit()
        self.client_id_edit.setPlaceholderText("client_id Сбера")
        form.addRow("Client ID (Сбер):", self.client_id_edit)

        self.client_secret_edit = QLineEdit()
        self.client_secret_edit.setPlaceholderText("client_secret Сбера")
        self.client_secret_edit.setEchoMode(QLineEdit.Password)
        form.addRow("Client Secret (Сбер):", self.client_secret_edit)

        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Путь к локальной GGUF-модели (пусто = авто)")
        form.addRow("GGUF-модель:", self.model_path_edit)

        self.auto_context_check = QCheckBox("Автопромпт: подставлять сценарий и данные пользователя")
        self.auto_context_check.setChecked(True)
        form.addRow("", self.auto_context_check)

        self.system_prompt_edit = QTextEdit()
        self.system_prompt_edit.setMaximumHeight(110)
        self.system_prompt_edit.setPlaceholderText("Системный промпт Оракула. {context} — место вставки контекста.")
        form.addRow("Системный промпт:", self.system_prompt_edit)

        self.agent_enabled_check = QCheckBox(
            "Агентский режим: Оракул может выполнять действия в программе "
            "(переключать вкладки, создавать модель, обучать)"
        )
        self.agent_enabled_check.setChecked(False)
        self.agent_enabled_check.setToolTip(
            "В этом режиме Оракул получает инструменты (tools) и может "
            "выполнять их сам: например, создать модель или запустить обучение."
        )
        form.addRow("", self.agent_enabled_check)

        self.agent_max_steps_spin = QSpinBox()
        self.agent_max_steps_spin.setRange(1, 10)
        self.agent_max_steps_spin.setValue(5)
        self.agent_max_steps_spin.setToolTip("Максимум шагов агентского цикла за один запрос.")
        form.addRow("Макс. шагов агента:", self.agent_max_steps_spin)

        self.agent_temperature_spin = QDoubleSpinBox()
        self.agent_temperature_spin.setRange(0.0, 1.5)
        self.agent_temperature_spin.setValue(0.4)
        self.agent_temperature_spin.setSingleStep(0.1)
        self.agent_temperature_spin.setDecimals(2)
        self.agent_temperature_spin.setToolTip("Температура генерации в агентском режиме.")
        form.addRow("Температура агента:", self.agent_temperature_spin)

        lay.addWidget(grp)

        btn_row = QHBoxLayout()
        btn_load = QPushButton("⟳ Загрузить из файла")
        btn_load.clicked.connect(self._load_api_settings)
        btn_save = QPushButton("💾 Сохранить настройки")
        btn_save.setObjectName("StartBtn")
        btn_save.clicked.connect(self._save_api_settings)
        btn_row.addWidget(btn_load)
        btn_row.addWidget(btn_save)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        # --- Статус подключений ---
        self.api_status_view = QTextEdit()
        self.api_status_view.setReadOnly(True)
        self.api_status_view.setStyleSheet(
            "font-family: 'Consolas', monospace; font-size: 11px; "
            "background-color: #1e1e1e; color: #a9b7c6; border: 1px solid #555;"
        )
        lay.addWidget(self.api_status_view, stretch=1)

        self.tabs.addTab(w, "🌐 Подключения / API")

    def _build_resources_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.resources_view = QTextEdit()
        self.resources_view.setReadOnly(True)
        self.resources_view.setStyleSheet(
            "font-family: 'Consolas', monospace; font-size: 12px; "
            "background-color: #1e1e1e; color: #a9b7c6; border: 1px solid #555;"
        )
        lay.addWidget(self.resources_view)
        self.tabs.addTab(w, "💾 Ресурсы")

    def _build_users_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.users_view = QTextEdit()
        self.users_view.setReadOnly(True)
        self.users_view.setStyleSheet(
            "font-family: 'Consolas', monospace; font-size: 12px; "
            "background-color: #1e1e1e; color: #a9b7c6; border: 1px solid #555;"
        )
        lay.addWidget(self.users_view, stretch=1)
        self.tabs.addTab(w, "👥 Пользователи")

    # ============================================================
    # ЛОГИ (handler)
    # ============================================================
    def _attach_log_handler(self):
        """Подписывается на глобальную шину логов (все логгеры приложения)."""
        self._log_emitter = _AdminLogEmitter()
        self._log_bridge = _AdminLogBridge(self._log_emitter)
        self._log_emitter.log_signal.connect(self._on_log_line)
        log_event_bus.subscribe(self._log_bridge)

    def _on_log_line(self, level_name: str, message: str):
        self._log_lines.append((level_name, message))
        if len(self._log_lines) > 5000:
            self._log_lines = self._log_lines[-5000:]
        if self._log_passes_filter(level_name):
            self._append_log_line(level_name, message)

    def _log_passes_filter(self, level_name: str) -> bool:
        cur = self.log_level_combo.currentText() if hasattr(self, "log_level_combo") else "ALL"
        if cur == "ALL":
            return True
        order = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        try:
            return order.index(level_name) >= order.index(cur)
        except ValueError:
            return True

    def _append_log_line(self, level_name: str, message: str):
        colors = {
            "CRITICAL": "#ff5555", "ERROR": "#e06c75", "WARNING": "#e5c07b",
            "INFO": "#a9b7c6", "DEBUG": "#6a9fcf",
        }
        color = colors.get(level_name, "#a9b7c6")
        self.log_view.append(
            f'<span style="color:{color};">{_html_escape(message)}</span>'
        )
        if self._auto_scroll:
            cur = self.log_view.textCursor()
            cur.movePosition(QTextCursor.End)
            self.log_view.setTextCursor(cur)

    @property
    def _auto_scroll(self) -> bool:
        return True

    def _rerender_logs(self, *_):
        if not hasattr(self, "log_view"):
            return
        self.log_view.clear()
        for level, msg in self._log_lines:
            if self._log_passes_filter(level):
                self._append_log_line(level, msg)

    def _clear_logs(self):
        self._log_lines.clear()
        self.log_view.clear()

    # ============================================================
    # ДЕЙСТВИЯ ПОЛЬЗОВАТЕЛЯ
    # ============================================================
    def _refresh_activity(self, *_):
        if not hasattr(self, "activity_view"):
            return
        cat = self.activity_cat_combo.currentText()
        cat = None if cat == "all" else cat
        events = activity_tracker.get_recent(400, category=cat)
        colors = {
            "click": "#a6e3a1", "hover": "#6a9fcf", "key": "#e5c07b",
            "error": "#e06c75", "action": "#c678dd", "system": "#a9b7c6",
        }
        self.activity_view.clear()
        self.activity_view.append(
            f"<b>Всего событий в буфере: {activity_tracker.count()}</b>"
        )
        for e in events:
            color = colors.get(e.category, "#a9b7c6")
            line = (
                f"[{e.timestamp[11:19]}] <span style='color:{color};'>"
                f"{e.category.upper()}</span> "
                f"<b>{_html_escape(e.target)}</b>"
            )
            if e.name:
                line += f" «{_html_escape(e.name)}»"
            if e.user:
                line += f" (👤 {_html_escape(e.user)})"
            if e.details:
                line += f" — {_html_escape(e.details)}"
            self.activity_view.append(f'<p style="margin:1px 0;">{line}</p>')

    # ============================================================
    # НЕЙРОСЕТЬ
    # ============================================================
    def _refresh_network(self):
        model = self.shared_state.get("model")
        config = self._config_dict()
        lines = []
        if model is None:
            lines.append("🧠 Модель: <b>не создана</b>")
        else:
            try:
                params = sum(p.numel() for p in model.parameters())
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                arch = config.get("architecture") or config.get("type", "?")
                name = config.get("model_name", "Модель")
                lines.append(f"🧠 Модель: <b>{_html_escape(str(name))}</b>")
                lines.append(f"Архитектура: {_html_escape(str(arch))}")
                lines.append(f"Параметров: {params:,} (обучаемых {trainable:,})")
                try:
                    from core.model_factory import ModelFactory
                    lines.append(f"Память весов: {ModelFactory.estimate_memory(model):.2f} МБ")
                except Exception:
                    pass
                try:
                    dev = next(model.parameters()).device
                    lines.append(f"Устройство: {dev}")
                except Exception:
                    pass
            except Exception as e:
                lines.append(f"Ошибка анализа модели: {e}")

        # Состояние обучения/датасета
        dataset = self.shared_state.get("dataset")
        lines.append(f"Данные: {'<b>загружены</b>' if dataset is not None else 'нет'}")
        history = self.shared_state.get("history")
        if history is not None:
            try:
                val = history.val_loss if hasattr(history, "val_loss") else history.get("val_loss", [])
                lines.append(f"Эпох завершено: {len(val) if val else 0}")
            except Exception:
                pass

        # Что «думает» Оракул
        agent = self.shared_state.get("oraculum")
        if agent is not None:
            try:
                analysis = agent.analyze_screen()
                lines.append("<br><b>💭 Оракул (анализ экрана):</b>")
                lines.append(_html_escape(analysis.to_text()))
            except Exception:
                pass

        self.network_view.setHtml("<br>".join(lines))

    # ============================================================
    # API / ПОДКЛЮЧЕНИЯ
    # ============================================================
    def _config_dict(self) -> Dict[str, Any]:
        from core.contracts import config_to_dict
        return config_to_dict(self.shared_state.get("config"))

    def _load_api_settings(self):
        agent = self.shared_state.get("oraculum")
        if agent is None:
            return
        settings = agent.get_settings()
        provider = settings.get("provider", "local")
        idx = {"local": 0, "yandex": 1, "sber": 2}.get(provider, 0)
        self.provider_combo.setCurrentIndex(idx)
        self.api_key_edit.setText(settings.get("api_key", ""))
        self.folder_id_edit.setText(settings.get("folder_id", ""))
        self.client_id_edit.setText(settings.get("client_id", ""))
        self.client_secret_edit.setText(settings.get("client_secret", ""))
        self.model_path_edit.setText(settings.get("model_path", ""))
        self.auto_context_check.setChecked(bool(settings.get("auto_context", True)))
        self.system_prompt_edit.setPlainText(settings.get("system_prompt", ""))
        self.agent_enabled_check.setChecked(bool(settings.get("agent_enabled", False)))
        self.agent_max_steps_spin.setValue(int(settings.get("agent_max_steps", 5)))
        self.agent_temperature_spin.setValue(float(settings.get("agent_temperature", 0.4)))
        self._refresh_api_status()

    def _save_api_settings(self):
        agent = self.shared_state.get("oraculum")
        if agent is None:
            QMessageBox.warning(self, "Ошибка", "ИИ-агент Оракул недоступен.")
            return
        provider = {0: "local", 1: "yandex", 2: "sber"}[self.provider_combo.currentIndex()]
        settings = {
            "provider": provider,
            "api_key": self.api_key_edit.text().strip(),
            "folder_id": self.folder_id_edit.text().strip(),
            "client_id": self.client_id_edit.text().strip(),
            "client_secret": self.client_secret_edit.text().strip(),
            "model_path": self.model_path_edit.text().strip(),
            "auto_context": self.auto_context_check.isChecked(),
            "system_prompt": self.system_prompt_edit.toPlainText().strip(),
            "agent_enabled": self.agent_enabled_check.isChecked(),
            "agent_max_steps": self.agent_max_steps_spin.value(),
            "agent_temperature": self.agent_temperature_spin.value(),
        }
        try:
            agent.apply_settings(settings)
            agent.configure_api(
                provider,
                api_key=settings["api_key"],
                folder_id=settings["folder_id"],
                client_id=settings["client_id"],
                client_secret=settings["client_secret"],
            )
            QMessageBox.information(self, "Готово", "Настройки Оракула сохранены.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить:\n{e}")
        self._refresh_api_status()

    def _refresh_api_status(self):
        agent = self.shared_state.get("oraculum")
        lines = []
        if agent is None:
            self.api_status_view.setHtml("Оракул недоступен.")
            return
        st = agent.get_status()
        lines.append(f"Провайдер: <b>{_html_escape(str(st['provider']))}</b>")
        lines.append(f"API доступен: {'✅' if st['api_available'] else '❌'}")
        if st["api_key_masked"]:
            lines.append(f"API-ключ: {_html_escape(st['api_key_masked'])}")
        if st["api_error"]:
            lines.append(f"Последняя ошибка API: <span style='color:#e06c75;'>{_html_escape(str(st['api_error']))}</span>")
        lines.append(f"Локальная GGUF-модель: {'🟢 загружена' if st['local_model_loaded'] else '⚪ не загружена'}")
        if st["local_model_path"]:
            lines.append(f"Путь модели: {_html_escape(st['local_model_path'])}")
        if st["local_model_error"]:
            lines.append(f"Ошибка модели: <span style='color:#e06c75;'>{_html_escape(str(st['local_model_error']))}</span>")
        lines.append(f"Автоконтекст: {'✅ включён' if st['auto_context'] else '❌ выключен'}")
        lines.append(f"Агентский режим: {'🟢 включён' if st.get('agent_enabled') else '⚪ выключен'}")
        lines.append(f"Инструментов: {st.get('tools_count', 0)} | Обучение: {st.get('training_status', 'idle')}")
        lines.append(f"Сообщений в истории чата: {st['history_len']}")
        usage = st.get("usage", {})
        if usage:
            lines.append(
                f"Ресурсы Оракула: RAM {usage.get('ram_used_mb', 0)}/{usage.get('ram_total_mb', 0)} МБ, "
                f"CPU {usage.get('cpu_percent', 0)}%, модель {usage.get('model_size_mb', 0)} МБ"
            )
        self.api_status_view.setHtml("<br>".join(lines))

    # ============================================================
    # РЕСУРСЫ
    # ============================================================
    def _refresh_resources(self):
        info = gather_system_info()
        lines = []
        if "ram" in info:
            r = info["ram"]
            lines.append("<b>Оперативная память (RAM):</b>")
            lines.append(
                f"  Всего: {r['total_mb']} МБ | Использовано: {r['used_mb']} МБ "
                f"({r['percent']}%) | Свободно: {r['available_mb']} МБ"
            )
        lines.append(f"<b>CPU:</b> {info.get('cpu_percent', '?')}% "
                     f"(ядер: {info.get('cpu_count', '?')})")
        if "process" in info:
            p = info["process"]
            lines.append(
                f"<b>Процесс OracleAI:</b> RSS {p['rss_mb']} МБ, "
                f"CPU {p['cpu_percent']}%, потоков {p['threads']}"
            )
        if info.get("cuda_available"):
            lines.append("<b>GPU (CUDA):</b>")
            lines.append(f"  Устройство: {_html_escape(str(info.get('gpu_name', '?')))}")
            lines.append(
                f"  VRAM всего: {info.get('gpu_total_mb', 0)} МБ | "
                f"Выделено: {info.get('gpu_allocated_mb', 0)} МБ | "
                f"Зарезервировано: {info.get('gpu_reserved_mb', 0)} МБ"
            )
        else:
            lines.append("<b>GPU:</b> недоступен (CUDA выключена)")
        self.resources_view.setHtml("<br>".join(lines))

    # ============================================================
    # ПОЛЬЗОВАТЕЛИ
    # ============================================================
    def _refresh_users(self):
        um = self.shared_state.get("user_manager")
        lines = []
        if um is None:
            self.users_view.setHtml("Система пользователей недоступна.")
            return
        lines.append("<b>Пользователи:</b>")
        for profile in um.get_all_users():
            is_admin = um.is_admin(profile.user_id)
            counts = um.count_user_objects(profile.username)
            storage = um.get_storage_usage(profile.username)
            role = "👑 админ" if is_admin else profile.user_level.value
            lines.append(
                f"• <b>{_html_escape(profile.username)}</b> ({role})"
                f" — проектов {counts.get('projects', 0)}, моделей {counts.get('models', 0)}, "
                f"датасетов {counts.get('datasets', 0)}, хранилище {storage} МБ, "
                f"гость: {'да' if profile.is_guest else 'нет'}"
            )
            if profile.last_login:
                lines.append(f"  последний вход: {_html_escape(profile.last_login)}")
        self.users_view.setHtml("<br>".join(lines))

    # ============================================================
    # СВОДКА + АВТООБНОВЛЕНИЕ
    # ============================================================
    def _refresh_summary(self):
        info = gather_system_info()
        parts = ["🛡 <b>Панель администратора</b>"]
        if "ram" in info:
            parts.append(f"RAM {info['ram']['percent']}%")
        parts.append(f"CPU {info.get('cpu_percent', '?')}%")
        if info.get("cuda_available"):
            parts.append(f"VRAM {info.get('gpu_allocated_mb', 0)}/{info.get('gpu_total_mb', 0)} МБ")
        agent = self.shared_state.get("oraculum")
        if agent is not None:
            parts.append("API: " + ("✅" if agent.is_api_available() else "❌"))
            parts.append("GGUF: " + ("🟢" if agent.is_local_model_loaded() else "⚪"))
        parts.append(f"Событий активности: {activity_tracker.count()}")
        self.summary_label.setText(" | ".join(parts))

    def _auto_refresh(self):
        self._refresh_summary()
        self._refresh_resources()
        self._refresh_network()
        self._refresh_api_status()
        # Действия и пользователи — реже, чтобы не перегружать UI
        if int(datetime.now().timestamp()) % 3 == 0:
            self._refresh_activity()
            self._refresh_users()

    def refresh(self):
        """Полное обновление всех вкладок."""
        self._load_api_settings()
        self._refresh_summary()
        self._refresh_resources()
        self._refresh_network()
        self._refresh_activity()
        self._refresh_users()

    # ============================================================
    # ЖИЗНЕННЫЙ ЦИКЛ
    # ============================================================
    def showEvent(self, event):
        super().showEvent(event)
        self.refresh()

    def closeEvent(self, event):
        if self._log_bridge:
            self._log_bridge.deactivate()
            try:
                log_event_bus.unsubscribe(self._log_bridge)
            except Exception:
                pass
        super().closeEvent(event)
