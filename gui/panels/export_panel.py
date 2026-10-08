"""
gui/panels/export_panel.py
==========================
Панель экспорта и портативности. Все сценарии сохранения и выгрузки
результатов обучения за пределы программы.

Зависимости:
  - core/contracts.py  → ModelConfig, DatasetContainer, TrainingHistory, FORMAT_VERSION
  - core/exporter.py   → Exporter (статические методы экспорта)
  - core/session.py    → SessionManager, SessionState
  - core/hint_engine.py → HintEngine, HintContext
  - gui/hint_widget.py → HintWidget

Сигналы:
  - export_done(str)   → путь к созданному файлу/папке
  - navigate_requested(str) → запрос на переход в другую панель ("sandbox", "analysis")

Входы (из shared_state):
  - "session"   : SessionState
  - "project"   : dict  {"name", "scenario", "model_name", "description"}
  - "dataset"   : DatasetContainer | None
  - "model"     : nn.Module | None
  - "config"    : ModelConfig
  - "history"   : TrainingHistory
  - "hint_mode" : str  ("static"|"ai"|"hybrid"|"none")

КЛЮЧЕВОЕ ПРАВИЛО: не создаём новые ключи в shared_state.
"""

import logging
from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QLabel, QFileDialog, QMessageBox, QTextEdit, QFrame,
    QScrollArea, QSizePolicy, QApplication
)
from PyQt5.QtCore import pyqtSignal, Qt, QTimer

from core.contracts import (
    ModelConfig, DatasetContainer, TrainingHistory, FORMAT_VERSION
)
from core.exporter import Exporter
from core.hint_engine import HintEngine, HintContext
from gui.hint_widget import HintWidget
from core.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────
#  КОНСТАНТЫ ПАНЕЛИ
# ─────────────────────────────────────────────────────────────────────
PANEL_NAME = "export_panel"

# Описание каждого сценария экспорта для информационной панели
EXPORT_SCENARIOS = {
    "project": {
        "icon": "💾",
        "title": "Сохранить проект (.oai)",
        "desc": (
            "Полный файл проекта: датасет + конфигурация модели + веса + "
            "история обучения + словарь. Можно открыть в этой программе "
            "и продолжить работу."
        ),
        "requires_model": True,
        "requires_dataset": True,
        "requires_history": False,
        "hint_id": "export_project_btn",
    },
    "model_pth": {
        "icon": "🧠",
        "title": "Сохранить модель (.pth)",
        "desc": (
            "Только веса и конфигурация модели. Для продвинутых: можно "
            "загрузить в Python-скрипт или дообучить позже."
        ),
        "requires_model": True,
        "requires_dataset": False,
        "requires_history": False,
        "hint_id": "export_model_btn",
    },
    "flash_chat": {
        "icon": "📦",
        "title": "Экспорт на флешку (чат-режим)",
        "desc": (
            "Создаёт папку с автономным чатом: run.bat + chat.py + модель + "
            "инструкция. Запускается на любом ПК без установки программы."
        ),
        "requires_model": True,
        "requires_dataset": False,
        "requires_history": False,
        "hint_id": "export_flash_btn",
    },
    "html_sandbox": {
        "icon": "🌐",
        "title": "Экспорт в HTML (песочница)",
        "desc": (
            "Один HTML-файл с моделью внутри (через ONNX + JS). Открывается "
            "в браузере без Python. Идеально для демонстрации."
        ),
        "requires_model": True,
        "requires_dataset": False,
        "requires_history": False,
        "hint_id": "export_html_btn",
    },
    "onnx": {
        "icon": "🔄",
        "title": "Экспорт в ONNX",
        "desc": (
            "Стандартный формат для запуска модели в других программах и "
            "на других устройствах. Для продвинутых пользователей."
        ),
        "requires_model": True,
        "requires_dataset": False,
        "requires_history": False,
        "hint_id": "export_onnx_btn",
    },
    "report": {
        "icon": "📄",
        "title": "Сгенерировать отчёт (.md)",
        "desc": (
            "Документ с описанием проекта, графиков обучения и результатов. "
            "Можно показать учителю или вставить в школьную работу."
        ),
        "requires_model": False,
        "requires_dataset": False,
        "requires_history": True,
        "hint_id": "export_report_btn",
    },
}


class ExportPanel(QWidget):
    """
    Панель экспорта и портативности.

    Позволяет пользователю сохранить результаты обучения в различных
    форматах: от полного проекта до автономного чата на флешке.
    """

    # Сигнал: экспорт завершён, передаёт путь к файлу/папке
    export_done = pyqtSignal(str)
    # Сигнал: пользователь хочет перейти в другую панель
    navigate_requested = pyqtSignal(str)

    def __init__(self, shared_state: dict, hint_engine: HintEngine, parent=None):
        super().__init__(parent)
        self.shared_state = shared_state
        self.hint_engine = hint_engine
        self._buttons: dict = {}
        self._status_widgets: dict = {}
        self.init_ui()
        self._install_hints()

    # ================================================================
    #  ИНИЦИАЛИЗАЦИЯ ИНТЕРФЕЙСА
    # ================================================================

    def init_ui(self):
        """Создаёт все виджеты панели."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # ── Заголовок ──
        header = QLabel("📤 Экспорт и Портативность")
        header.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #cc7832; padding: 5px;"
        )
        layout.addWidget(header)

        subtitle = QLabel(
            "Сохраните результаты обучения, чтобы использовать их дома, "
            "показать друзьям или продолжить работу позже."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #a9b7c6; font-size: 13px; padding: 2px;")
        layout.addWidget(subtitle)

        # ── Информационная карточка текущего состояния ──
        self.state_card = self._create_state_card()
        layout.addWidget(self.state_card)

        # ── Кнопки экспорта в скролле ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        buttons_widget = QWidget()
        buttons_layout = QVBoxLayout(buttons_widget)
        buttons_layout.setSpacing(8)

        for key, scenario in EXPORT_SCENARIOS.items():
            card = self._create_export_card(key, scenario)
            buttons_layout.addWidget(card)

        buttons_layout.addStretch()
        scroll.setWidget(buttons_widget)
        layout.addWidget(scroll, stretch=1)

        # ── Статусная строка ──
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(
            "color: #a6e3a1; font-size: 12px; padding: 6px; "
            "background-color: #313335; border-left: 3px solid #385a3a;"
        )
        self.status_label.setVisible(False)
        layout.addWidget(self.status_label)

    def _create_state_card(self) -> QGroupBox:
        """Карточка: что сейчас есть в памяти для экспорта."""
        group = QGroupBox("📋 Что сейчас готово к экспорту")
        layout = QVBoxLayout()

        self.state_model_label = QLabel("—")
        self.state_data_label = QLabel("—")
        self.state_history_label = QLabel("—")

        for lbl in (self.state_model_label, self.state_data_label,
                    self.state_history_label):
            lbl.setStyleSheet("color: #a9b7c6; font-size: 13px; padding: 2px;")
            layout.addWidget(lbl)

        group.setLayout(layout)
        return group

    def _create_export_card(self, key: str, scenario: dict) -> QGroupBox:
        """Создаёт карточку одного сценария экспорта."""
        group = QGroupBox(f"{scenario['icon']} {scenario['title']}")
        layout = QVBoxLayout()

        # Описание
        desc_label = QLabel(scenario["desc"])
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #a9b7c6; font-size: 12px;")
        layout.addWidget(desc_label)

        # Требования
        req_parts = []
        if scenario["requires_model"]:
            req_parts.append("модель")
        if scenario["requires_dataset"]:
            req_parts.append("данные")
        if scenario["requires_history"]:
            req_parts.append("история обучения")
        if req_parts:
            req_label = QLabel(f"Требуется: {', '.join(req_parts)}")
            req_label.setStyleSheet("color: #777; font-size: 11px; font-style: italic;")
            layout.addWidget(req_label)

        # Кнопка
        btn_row = QHBoxLayout()
        btn = QPushButton(f"{scenario['icon']} Экспортировать")
        btn.setMinimumHeight(36)
        btn.clicked.connect(lambda checked, k=key: self._on_export_clicked(k))
        btn_row.addWidget(btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Статус для этой кнопки
        status_lbl = QLabel("")
        status_lbl.setStyleSheet("color: #e5c07b; font-size: 11px; padding: 2px;")
        status_lbl.setVisible(False)
        layout.addWidget(status_lbl)

        group.setLayout(layout)
        self._buttons[key] = btn
        self._status_widgets[key] = status_lbl
        return group

    # ================================================================
    #  ПОДСКАЗКИ (тотальные, на каждый элемент)
    # ================================================================

    def _install_hints(self):
        """Подключает HintWidget к каждой кнопке и значимому элементу."""
        for key, scenario in EXPORT_SCENARIOS.items():
            if key in self._buttons:
                HintWidget(
                    parent_widget=self._buttons[key],
                    hint_engine=self.hint_engine,
                    widget_id=scenario["hint_id"],
                    panel_name=PANEL_NAME,
                )

        # Подсказки на карточку состояния
        HintWidget(
            parent_widget=self.state_card,
            hint_engine=self.hint_engine,
            widget_id="export_state_card",
            panel_name=PANEL_NAME,
        )

    # ================================================================
    #  ОБНОВЛЕНИЕ СОСТОЯНИЯ
    # ================================================================

    def _get_config(self) -> ModelConfig:
        """
        Возвращает конфиг как ModelConfig. После режима «Мастер»/агента
        shared_state["config"] может быть dict (результат config.to_dict()),
        поэтому приводим его обратно к ModelConfig, иначе экспорт падает с
        AttributeError: 'dict' object has no attribute 'model_name'.
        """
        cfg = self.shared_state.get("config")
        if cfg is None:
            return ModelConfig()
        if isinstance(cfg, ModelConfig):
            return cfg
        if isinstance(cfg, dict):
            try:
                return ModelConfig.from_dict(cfg)
            except Exception:
                return ModelConfig()
        return ModelConfig()

    def save_model_silent(self, name: str = "") -> str:
        """
        Сохраняет модель в .pth без диалога (для ИИ-агента).
        Возвращает JSON-строку {"done": ..., "path": ...} или {"error": ...}.
        """
        model = self.shared_state.get("model")
        if model is None:
            return '{"error": "модель не создана — сначала создайте и обучите модель"}'
        config = self._get_config()
        safe_name = "".join(
            c for c in ((name or "").strip() or config.model_name or "модель")
            if c.isalnum() or c in (" ", "-", "_")
        ).strip() or "модель"

        # Папка моделей пользователя (или ./models как фолбэк)
        models_dir = Path("models")
        session_mgr = self.shared_state.get("session_manager")
        try:
            subdirs = session_mgr.get_user_subdirs()
            models_dir = Path(subdirs.get("models", models_dir))
        except Exception:
            pass
        models_dir.mkdir(parents=True, exist_ok=True)
        path = models_dir / f"{safe_name}.pth"

        # Словарь для текстовых моделей
        vocab = None
        dataset = self.shared_state.get("dataset")
        if dataset is not None:
            try:
                if getattr(getattr(dataset, "meta", None), "data_type", None) is not None \
                        and dataset.meta.data_type.value == "text":
                    vocab = dataset.vocab or {}
            except Exception:
                vocab = None

        try:
            Exporter.export_model_only(model=model, config=config,
                                       path=path, vocab=vocab)
            return f'{{"done": true, "path": "{path}", "name": "{safe_name}"}}'
        except Exception as e:
            return f'{{"error": "{type(e).__name__}: {e}"}}'

    def refresh(self):
        """
        Вызывается при переключении вкладок или изменении данных.
        Обновляет карточку состояния и доступность кнопок.
        """
        model = self.shared_state.get("model")
        dataset = self.shared_state.get("dataset")
        config: ModelConfig = self._get_config()
        history: TrainingHistory = self.shared_state.get("history", TrainingHistory())
        project = self.shared_state.get("project", {})

        # ── Карточка состояния ──
        if model:
            name = config.model_name or project.get("model_name", "Без имени")
            arch = config.architecture.value if hasattr(config.architecture, 'value') else str(config.architecture)
            self.state_model_label.setText(
                f"🧠 Модель: «{name}» ({arch})"
            )
            self.state_model_label.setStyleSheet("color: #a6e3a1; font-size: 13px; padding: 2px;")
        else:
            self.state_model_label.setText("🧠 Модель: не создана")
            self.state_model_label.setStyleSheet("color: #e06c75; font-size: 13px; padding: 2px;")

        if dataset:
            n = dataset.meta.num_samples if dataset.meta else 0
            self.state_data_label.setText(f"📊 Данные: {n} примеров")
            self.state_data_label.setStyleSheet("color: #a6e3a1; font-size: 13px; padding: 2px;")
        else:
            self.state_data_label.setText("📊 Данные: не загружены")
            self.state_data_label.setStyleSheet("color: #e06c75; font-size: 13px; padding: 2px;")

        if history and history.epochs_completed > 0:
            self.state_history_label.setText(
                f"📈 История: {history.epochs_completed} эпох, "
                f"лучший Val Loss = {history.best_val_loss:.4f}"
            )
            self.state_history_label.setStyleSheet("color: #a6e3a1; font-size: 13px; padding: 2px;")
        else:
            self.state_history_label.setText("📈 История: обучение не проводилось")
            self.state_history_label.setStyleSheet("color: #777; font-size: 13px; padding: 2px;")

        # ── Доступность кнопок ──
        has_model = model is not None
        has_dataset = dataset is not None
        has_history = (history is not None and history.epochs_completed > 0)

        for key, scenario in EXPORT_SCENARIOS.items():
            available = True
            if scenario["requires_model"] and not has_model:
                available = False
            if scenario["requires_dataset"] and not has_dataset:
                available = False
            if scenario["requires_history"] and not has_history:
                available = False

            if key in self._buttons:
                self._buttons[key].setEnabled(available)

            # Статус под кнопкой
            if key in self._status_widgets:
                status_lbl = self._status_widgets[key]
                if not available:
                    missing = []
                    if scenario["requires_model"] and not has_model:
                        missing.append("модель")
                    if scenario["requires_dataset"] and not has_dataset:
                        missing.append("данные")
                    if scenario["requires_history"] and not has_history:
                        missing.append("история")
                    status_lbl.setText(f"⚠️ Не хватает: {', '.join(missing)}")
                    status_lbl.setStyleSheet("color: #e5c07b; font-size: 11px;")
                    status_lbl.setVisible(True)
                else:
                    status_lbl.setVisible(False)

    def showEvent(self, event):
        """При показе панели — обновляем состояние."""
        super().showEvent(event)
        self.refresh()

    # ================================================================
    #  ОБРАБОТКА КЛИКОВ
    # ================================================================

    def _on_export_clicked(self, key: str):
        """Диспетчер: определяет тип экспорта и вызывает нужный метод."""
        handlers = {
            "project": self._export_project,
            "model_pth": self._export_model_pth,
            "flash_chat": self._export_flash_chat,
            "html_sandbox": self._export_html_sandbox,
            "onnx": self._export_onnx,
            "report": self._export_report,
        }
        handler = handlers.get(key)
        if handler:
            try:
                QApplication.setOverrideCursor(Qt.WaitCursor)
                handler()
            except Exception as e:
                logger.error(f"Ошибка экспорта ({key}): {e}")
                QMessageBox.critical(
                    self, "Ошибка экспорта",
                    f"Не удалось выполнить экспорт:\n{e}"
                )
            finally:
                QApplication.restoreOverrideCursor()

    # ================================================================
    #  СЦЕНАРИЙ 1: Сохранить проект (.oai)
    # ================================================================

    def _export_project(self):
        """Полный проект: датасет + модель + конфиг + история."""
        default_name = self._get_default_filename("project", ".oai")
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить проект",
            default_name,
            "OracleAI Project (*.oai);;Все файлы (*)"
        )
        if not path:
            return

        dataset: DatasetContainer = self.shared_state.get("dataset")
        model = self.shared_state.get("model")
        config: ModelConfig = self._get_config()
        history: TrainingHistory = self.shared_state.get("history", TrainingHistory())

        Exporter.export_project(
            dataset=dataset,
            model=model,
            config=config,
            history=history,
            path=Path(path),
        )

        self._show_success(path, "Проект сохранён")
        self.export_done.emit(path)

    # ================================================================
    #  СЦЕНАРИЙ 2: Сохранить модель (.pth)
    # ================================================================

    def _export_model_pth(self):
        """Только веса и конфигурация модели."""
        default_name = self._get_default_filename("model", ".pth")
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить модель",
            default_name,
            "PyTorch Model (*.pth);;Все файлы (*)"
        )
        if not path:
            return

        model = self.shared_state.get("model")
        config: ModelConfig = self._get_config()

        Exporter.export_model_only(
            model=model,
            config=config,
            path=Path(path),
        )

        self._show_success(path, "Модель сохранена")
        self.export_done.emit(path)

    # ================================================================
    #  СЦЕНАРИЙ 3: Экспорт на флешку (чат-режим)
    # ================================================================

    def _export_flash_chat(self):
        """
        Создаёт папку с автономным чатом.
        Пользователь выбирает директорию (например, корень флешки).
        """
        model_name = self._get_config().model_name
        if not model_name:
            model_name = self.shared_state.get("project", {}).get("model_name", "модель")

        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку для экспорта (например, корень флешки)",
            str(Path.home()),
        )
        if not dir_path:
            return

        model = self.shared_state.get("model")
        config: ModelConfig = self._get_config()
        dataset: DatasetContainer = self.shared_state.get("dataset")

        # Извлекаем словарь из датасета, если он текстовый
        vocab = {}
        if dataset and dataset.meta and dataset.meta.data_type.value == "text":
            vocab = dataset.vocab or {}

        export_dir = Path(dir_path) / f"{model_name}_на_флешке"
        Exporter.export_flash_chat(
            model=model,
            config=config,
            vocab=vocab,
            path_dir=export_dir,
        )

        self._show_success(str(export_dir), "Папка для флешки создана")
        self.export_done.emit(str(export_dir))

        # Дополнительная инструкция
        QMessageBox.information(
            self, "📦 Готово!",
            f"Папка создана:\n{export_dir}\n\n"
            "Содержимое:\n"
            "• run_chat.bat — запустить чат (двойной клик)\n"
            "• chat.py — скрипт чата\n"
            "• модель (.pth) — веса нейросети\n"
            "• README.txt — инструкция\n\n"
            "Просто скопируйте папку на флешку и запускайте на любом ПК!"
        )

    # ================================================================
    #  СЦЕНАРИЙ 4: Экспорт в HTML (песочница)
    # ================================================================

    def _export_html_sandbox(self):
        """Один HTML-файл с моделью внутри."""
        default_name = self._get_default_filename("sandbox", ".html")
        path, _ = QFileDialog.getSaveFileName(
            self, "Экспорт в HTML",
            default_name,
            "HTML файл (*.html);;Все файлы (*)"
        )
        if not path:
            return

        model = self.shared_state.get("model")
        config: ModelConfig = self._get_config()
        dataset: DatasetContainer = self.shared_state.get("dataset")

        vocab = {}
        if dataset and dataset.meta and dataset.meta.data_type.value == "text":
            vocab = dataset.vocab or {}

        Exporter.export_html_sandbox(
            model=model,
            config=config,
            vocab=vocab,
            path=Path(path),
        )

        self._show_success(path, "HTML-песочница создана")
        self.export_done.emit(path)

    # ================================================================
    #  СЦЕНАРИЙ 5: Экспорт в ONNX
    # ================================================================

    def _export_onnx(self):
        """Экспорт модели в формат ONNX."""
        default_name = self._get_default_filename("model", ".onnx")
        path, _ = QFileDialog.getSaveFileName(
            self, "Экспорт в ONNX",
            default_name,
            "ONNX Model (*.onnx);;Все файлы (*)"
        )
        if not path:
            return

        model = self.shared_state.get("model")
        config: ModelConfig = self._get_config()

        Exporter.export_onnx(
            model=model,
            config=config,
            path=Path(path),
        )

        self._show_success(path, "ONNX-модель создана")
        self.export_done.emit(path)

    # ================================================================
    #  СЦЕНАРИЙ 6: Сгенерировать отчёт (.md)
    # ================================================================

    def _export_report(self):
        """Генерирует текстовый отчёт об обучении."""
        default_name = self._get_default_filename("report", ".md")
        path, _ = QFileDialog.getSaveFileName(
            self, "Сгенерировать отчёт",
            default_name,
            "Markdown (*.md);;HTML (*.html);;Все файлы (*)"
        )
        if not path:
            return

        config: ModelConfig = self._get_config()
        history: TrainingHistory = self.shared_state.get("history", TrainingHistory())

        Exporter.export_report(
            history=history,
            config=config,
            path=Path(path),
        )

        self._show_success(path, "Отчёт сгенерирован")
        self.export_done.emit(path)

    # ================================================================
    #  ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ================================================================

    def _get_default_filename(self, prefix: str, ext: str) -> str:
        """Формирует имя файла по умолчанию на основе проекта."""
        project = self.shared_state.get("project", {})
        config: ModelConfig = self._get_config()
        name = config.model_name or project.get("model_name", prefix)
        # Очищаем от недопустимых символов
        safe_name = "".join(c for c in name if c.isalnum() or c in (" ", "-", "_")).strip()
        if not safe_name:
            safe_name = prefix
        return f"{safe_name}{ext}"

    def _show_success(self, path: str, title: str):
        """Показывает зелёную статусную строку."""
        self.status_label.setText(f"✅ {title}: {path}")
        self.status_label.setStyleSheet(
            "color: #a6e3a1; font-size: 12px; padding: 6px; "
            "background-color: #313335; border-left: 3px solid #385a3a;"
        )
        self.status_label.setVisible(True)
        logger.info(f"{title}: {path}")

        # Автоскрытие через 10 секунд
        QTimer.singleShot(10_000, self._hide_status)

    def _hide_status(self):
        """Скрывает статусную строку."""
        self.status_label.setVisible(False)

    # ================================================================
    #  ПУБЛИЧНЫЙ ИНТЕРФЕЙС ДЛЯ ДРУГИХ ПАНЕЛЕЙ
    # ================================================================

    def enable_flash_export(self):
        """Внешний вызов: разрешить экспорт на флешку (после успешного обучения)."""
        if "flash_chat" in self._buttons:
            self._buttons["flash_chat"].setEnabled(True)

    def get_export_summary(self) -> str:
        """Возвращает текстовое резюме для отчёта или анализа."""
        model = self.shared_state.get("model")
        dataset = self.shared_state.get("dataset")
        config: ModelConfig = self._get_config()
        parts = []
        if model:
            parts.append(f"Модель: {config.model_name}")
        if dataset:
            parts.append(f"Данные: {dataset.meta.num_samples} примеров")
        return " | ".join(parts) if parts else "Нечего экспортировать"