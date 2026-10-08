"""
gui/panels/sandbox_panel.py
============================
Панель «Песочница» — чат с готовой обученной моделью.
Сценарий В: пользователь загружает модель с флешки или после обучения
и общается с ней в интерактивном режиме.

Поддерживает:
  - Текстовые модели (генерация через model.generate)
  - Числовые модели (регрессия / классификация)
  - Безопасный ввод: мусор не роняет программу

Контракты:
  - Использует shared_state["model"], shared_state["config"],
    shared_state["dataset"], shared_state["session"]
  - Импортирует типы из core.contracts
  - Логирует действия через SessionManager

Зависимости: core.contracts, core.model_factory, core.logger, gui.hint_widget
"""

import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QLineEdit, QTextEdit,
    QFileDialog, QMessageBox, QFrame, QSplitter,
    QScrollArea, QSizePolicy, QDialog
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont

from core.contracts import (
    ModelConfig, DataType, TaskType, FORMAT_VERSION,
    config_to_dict, dataset_to_legacy_dict,
)
from core.model_factory import ModelFactory
from core.logger import get_logger
from gui.hint_widget import HintWidget

logger = get_logger()


# ============================================================
#  ВСПОМОГАТЕЛЬНЫЙ КЛАСС: сообщение чата
# ============================================================
class ChatMessage:
    """Одно сообщение в истории чата."""

    ROLE_USER = "user"
    ROLE_MODEL = "model"
    ROLE_SYSTEM = "system"
    ROLE_ERROR = "error"

    def __init__(self, role: str, text: str, timestamp: str = None):
        self.role = role
        self.text = text
        self.timestamp = timestamp or datetime.now().strftime("%H:%M:%S")

    def to_html(self) -> str:
        """Форматирование в HTML для отображения в QTextEdit."""
        colors = {
            self.ROLE_USER: "#4a88c7",
            self.ROLE_MODEL: "#a6e3a1",
            self.ROLE_SYSTEM: "#e5c07b",
            self.ROLE_ERROR: "#e06c75",
        }
        icons = {
            self.ROLE_USER: "👤",
            self.ROLE_MODEL: "🤖",
            self.ROLE_SYSTEM: "⚙️",
            self.ROLE_ERROR: "❌",
        }
        color = colors.get(self.role, "#a9b7c6")
        icon = icons.get(self.role, "•")
        return (
            f'<p style="margin: 4px 0;">'
            f'<span style="color: #777; font-size: 11px;">[{self.timestamp}]</span> '
            f'<span style="color: {color}; font-weight: bold;">{icon} {self.role}:</span> '
            f'<span style="color: #d4d4d4;">{self.text}</span>'
            f'</p>'
        )


# ============================================================
#  ОСНОВНАЯ ПАНЕЛЬ
# ============================================================
class SandboxPanel(QWidget):
    """
    Панель песочницы: интерактивный чат с обученной моделью.

    Сигналы:
        model_loaded: Модель загружена и готова к работе
        export_requested: Пользователь хочет экспортировать результат
    """

    model_loaded = pyqtSignal(object, dict)  # (model, config)
    export_requested = pyqtSignal()

    def __init__(self, shared_state: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.shared_state = shared_state
        self.chat_history: List[ChatMessage] = []
        self.init_ui()

    # ========================================================
    #  UI
    # ========================================================
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # === 1. Информация о модели ===
        self.model_info_group = QGroupBox("🧠 Модель в песочнице")
        model_info_layout = QVBoxLayout()

        self.model_name_label = QLabel("Модель не загружена")
        self.model_name_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #e06c75;"
        )
        model_info_layout.addWidget(self.model_name_label)

        self.model_meta_label = QLabel("")
        self.model_meta_label.setWordWrap(True)
        self.model_meta_label.setStyleSheet("color: #a9b7c6; font-size: 12px;")
        model_info_layout.addWidget(self.model_meta_label)

        btn_row = QHBoxLayout()
        self.btn_load_model = QPushButton("📂 Загрузить модель (.pth / .oai)")
        self.btn_load_model.setToolTip(
            "Загрузите обученную модель из файла, чтобы начать общение."
        )
        self.btn_load_model.clicked.connect(self._load_model_dialog)
        btn_row.addWidget(self.btn_load_model)

        self.btn_load_project = QPushButton("🗂 Открыть проект (.oai)")
        self.btn_load_project.setToolTip(
            "Открыть полный проект: модель + данные + словарь."
        )
        self.btn_load_project.clicked.connect(self._load_project_dialog)
        btn_row.addWidget(self.btn_load_project)

        self.btn_pretrained = QPushButton("🎁 Готовые модели")
        self.btn_pretrained.setToolTip(
            "Загрузить предобученную мини-модель из библиотеки готовых сетей "
            "(текст, изображения, музыка, перевод)."
        )
        self.btn_pretrained.clicked.connect(self._load_pretrained_dialog)
        btn_row.addWidget(self.btn_pretrained)

        btn_row.addStretch()
        model_info_layout.addLayout(btn_row)

        self.model_info_group.setLayout(model_info_layout)
        layout.addWidget(self.model_info_group)

        # === 2. Область чата ===
        self.chat_group = QGroupBox("💬 Чат с нейросетью")
        chat_layout = QVBoxLayout()

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet(
            "QTextEdit {"
            "  background-color: #1e1e1e;"
            "  border: 1px solid #555;"
            "  font-family: 'Consolas', 'Courier New', monospace;"
            "  font-size: 13px;"
            "  padding: 8px;"
            "}"
        )
        self.chat_display.setHtml(
            '<p style="color: #777; text-align: center; margin-top: 30px;">'
            'Загрузите модель и начните общение.<br>'
            'Для текстовых моделей — вводите текст.<br>'
            'Для числовых — числа через пробел.</p>'
        )
        chat_layout.addWidget(self.chat_display)

        # Кнопки управления чатом
        chat_btn_row = QHBoxLayout()
        self.btn_clear_chat = QPushButton("🗑 Очистить чат")
        self.btn_clear_chat.setToolTip("Удаляет всю историю сообщений.")
        self.btn_clear_chat.clicked.connect(self._clear_chat)
        chat_btn_row.addWidget(self.btn_clear_chat)

        self.btn_export_chat = QPushButton("💾 Сохранить историю чата")
        self.btn_export_chat.setToolTip(
            "Сохраняет историю общения с моделью в текстовый файл."
        )
        self.btn_export_chat.clicked.connect(self._export_chat)
        chat_btn_row.addWidget(self.btn_export_chat)

        chat_btn_row.addStretch()
        chat_layout.addLayout(chat_btn_row)

        self.chat_group.setLayout(chat_layout)
        layout.addWidget(self.chat_group, stretch=1)

        # === 3. Поле ввода ===
        self.input_group = QGroupBox("✏️ Ввод")
        input_layout = QVBoxLayout()

        self.input_hint_label = QLabel("")
        self.input_hint_label.setWordWrap(True)
        self.input_hint_label.setStyleSheet(
            "color: #e5c07b; font-size: 12px; font-style: italic;"
        )
        input_layout.addWidget(self.input_hint_label)

        input_row = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Введите вопрос или данные для модели...")
        self.input_field.setStyleSheet(
            "QLineEdit {"
            "  font-size: 16px;"
            "  padding: 8px;"
            "  background-color: #313335;"
            "  border: 1px solid #555;"
            "  color: #a9b7c6;"
            "}"
            "QLineEdit:focus { border: 1px solid #4a88c7; }"
        )
        self.input_field.returnPressed.connect(self._on_send)
        input_row.addWidget(self.input_field, stretch=1)

        self.btn_send = QPushButton("🚀 Отправить")
        self.btn_send.setStyleSheet(
            "QPushButton {"
            "  font-size: 14px; font-weight: bold;"
            "  padding: 8px 20px;"
            "  background-color: #385a3a; border-color: #4a7a4c;"
            "}"
            "QPushButton:hover { background-color: #4a7a4c; }"
            "QPushButton:disabled { background-color: #313335; color: #777; }"
        )
        self.btn_send.clicked.connect(self._on_send)
        input_row.addWidget(self.btn_send)

        input_layout.addLayout(input_row)
        self.input_group.setLayout(input_layout)
        layout.addWidget(self.input_group)

        # === 4. Подсказки на каждый элемент ===
        self._setup_hints()

    def _setup_hints(self):
        """Подключает подсказки к каждому интерактивному элементу."""
        try:
            hint_engine = self.shared_state.get("hint_engine")
            if hint_engine:
                HintWidget(self.btn_load_model, hint_engine,
                           "sandbox_load_model", "sandbox_panel")
                HintWidget(self.btn_load_project, hint_engine,
                           "sandbox_load_project", "sandbox_panel")
                HintWidget(self.btn_clear_chat, hint_engine,
                           "sandbox_clear_chat", "sandbox_panel")
                HintWidget(self.btn_export_chat, hint_engine,
                           "sandbox_export_chat", "sandbox_panel")
                HintWidget(self.input_field, hint_engine,
                           "sandbox_input_field", "sandbox_panel")
                HintWidget(self.btn_send, hint_engine,
                           "sandbox_send_btn", "sandbox_panel")
        except Exception as e:
            logger.warning(f"Не удалось подключить подсказки: {e}")

    # ========================================================
    #  ЗАГРУЗКА МОДЕЛИ
    # ========================================================
    def _load_model_dialog(self):
        """Диалог загрузки модели из .pth файла."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Загрузить модель",
            "", "Модели (*.pth *.pt);;Все файлы (*)"
        )
        if not path:
            return
        self._load_model_from_path(Path(path))

    def _load_project_dialog(self):
        """Диалог загрузки полного проекта .oai."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Открыть проект",
            "", "Проекты OracleAI (*.oai);;Все файлы (*)"
        )
        if not path:
            return
        self._load_project_from_path(Path(path))

    def _load_pretrained_dialog(self):
        """Диалог выбора предобученной модели из библиотеки готовых сетей."""
        pm = self.shared_state.get("pretrained_manager")
        if pm is None:
            QMessageBox.warning(
                self, "Нет библиотеки",
                "Менеджер готовых моделей недоступен."
            )
            return

        models = pm.get_all_models()
        if not models:
            QMessageBox.information(self, "Пусто", "Библиотека готовых моделей пуста.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("🎁 Библиотека готовых моделей")
        dialog.setMinimumWidth(560)
        layout = QVBoxLayout(dialog)

        info = QLabel(
            "Выберите готовую мини-модель. Если она не установлена — "
            "она будет создана автоматически (без интернета)."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #a9b7c6;")
        layout.addWidget(info)

        from PyQt5.QtWidgets import QListWidget, QDialogButtonBox
        self.pretrained_list = QListWidget()
        self.pretrained_models = models
        for m in models:
            status = "✅ установлена" if m.installed else "⬇️ скачать"
            self.pretrained_list.addItem(
                f"{m.name}\n    [{m.category}] {m.architecture.value} · "
                f"{m.size_mb} МБ · {status}\n    {m.description}"
            )
        self.pretrained_list.setFixedHeight(220)
        self.pretrained_list.setCurrentRow(0)
        layout.addWidget(self.pretrained_list)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(lambda: self._on_pretrained_selected(dialog))
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec_()

    def _on_pretrained_selected(self, dialog: QDialog):
        row = self.pretrained_list.currentRow()
        if row < 0 or row >= len(self.pretrained_models):
            dialog.reject()
            return
        info = self.pretrained_models[row]
        pm = self.shared_state.get("pretrained_manager")

        if not info.installed:
            ok = pm.download_model(info.model_id)
            if not ok:
                QMessageBox.critical(
                    self, "Ошибка",
                    f"Не удалось подготовить модель «{info.name}»."
                )
                return

        try:
            model, config, extra = pm.load_model(info.model_id)
            self.shared_state["model"] = model
            self.shared_state["config"] = config
            vocab = extra.get("vocab", {})
            if vocab:
                dataset = self.shared_state.get("dataset")
                if dataset is None:
                    self.shared_state["dataset"] = {"vocab": vocab}
                elif isinstance(dataset, dict):
                    dataset["vocab"] = vocab
                else:
                    dataset.vocab = vocab
            self._on_model_ready(model, config_to_dict(config))
            self._log_action("load_pretrained", f"Модель «{info.name}»")
            dialog.accept()
        except Exception as e:
            logger.error(f"Ошибка загрузки готовой модели: {e}")
            QMessageBox.critical(
                self, "Ошибка", f"Не удалось загрузить модель:\n{e}"
            )

    def _load_model_from_path(self, path: Path):
        """Загружает модель из .pth и обновляет интерфейс."""
        try:
            model, config, _extra = ModelFactory.load_model(path)
            self.shared_state["model"] = model
            self.shared_state["config"] = config
            self._on_model_ready(model, config_to_dict(config))
            self._log_action("load_model", f"Загружена модель из {path.name}")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            self._add_message(
                ChatMessage.ROLE_ERROR,
                f"Не удалось загрузить модель: {e}"
            )
            QMessageBox.critical(self, "Ошибка", f"Файл повреждён или несовместим:\n{e}")

    def _load_project_from_path(self, path: Path):
        """Загружает проект .oai: модель + словарь + конфиг."""
        try:
            import json
            with open(path, 'r', encoding='utf-8') as f:
                project_data = json.load(f)

            # Извлекаем конфиг модели
            config_dict = project_data.get("config", {})
            vocab = project_data.get("vocab", {})
            normalization = project_data.get("normalization")

            # Если есть веса модели — загружаем
            if "model_state_dict" in project_data:
                model = ModelFactory.create_model(config_dict)
                model.load_state_dict(project_data["model_state_dict"])
                self.shared_state["model"] = model
                self.shared_state["config"] = ModelConfig.from_dict(config_dict)
                # Сохраняем словарь в dataset-подобную структуру
                if vocab:
                    dataset = self.shared_state.get("dataset")
                    if dataset is None:
                        self.shared_state["dataset"] = {"vocab": vocab}
                    elif isinstance(dataset, dict):
                        dataset["vocab"] = vocab
                    else:
                        # DatasetContainer
                        dataset.vocab = vocab
                self._on_model_ready(model, config_to_dict(self.shared_state["config"]))
                self._log_action("load_project", f"Открыт проект {path.name}")
            else:
                QMessageBox.warning(
                    self, "Внимание",
                    "Проект загружен, но веса модели не найдены.\n"
                    "Возможно, модель ещё не была обучена."
                )
        except Exception as e:
            logger.error(f"Ошибка открытия проекта: {e}")
            self._add_message(
                ChatMessage.ROLE_ERROR,
                f"Не удалось открыть проект: {e}"
            )
            QMessageBox.critical(self, "Ошибка", f"Не удалось открыть проект:\n{e}")

    def refresh(self):
        """
        Вызывается извне при изменении shared_state.
        Обновляет информацию о модели.
        """
        model = self.shared_state.get("model")
        config = config_to_dict(self.shared_state.get("config"))
        if model:
            self._on_model_ready(model, config)

    def _on_model_ready(self, model, config_dict: Dict):
        """Обновляет UI когда модель готова к работе."""
        config_dict = config_to_dict(config_dict)
        self.btn_load_model.setStyleSheet(
            "background-color: #385a3a; border-color: #4a7a4c;"
        )

        # Имя модели
        model_name = config_dict.get("model_name", "Безымянная модель")
        arch = config_dict.get("type", config_dict.get("architecture", "unknown"))
        if isinstance(arch, str):
            arch_display = arch.upper()
        else:
            arch_display = str(arch)

        params = sum(p.numel() for p in model.parameters())
        data_type = config_dict.get("data_type", "numeric")

        self.model_name_label.setText(f"🧠 {model_name}")
        self.model_name_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #a6e3a1;"
        )
        self.model_meta_label.setText(
            f"Архитектура: {arch_display} | "
            f"Параметры: {params:,} | "
            f"Тип данных: {data_type} | "
            f"Загружена: {datetime.now().strftime('%H:%M')}"
        )

        # Обновляем подсказку ввода
        self._update_input_hint(config_dict)

        # Приветственное сообщение
        self._add_message(
            ChatMessage.ROLE_SYSTEM,
            f"Модель «{model_name}» готова к работе! "
            f"Архитектура: {arch_display}. Введите данные для предсказания."
        )

        # Сигнал для других панелей
        self.model_loaded.emit(model, config_dict)

    def _update_input_hint(self, config_dict: Dict):
        """Меняет подсказку в зависимости от типа модели."""
        data_type = config_dict.get("data_type", "numeric")
        if data_type == "text":
            self.input_hint_label.setText(
                "📝 Текстовая модель: введите вопрос или выражение "
                "(например, «2+2» или зашифрованное слово)."
            )
            self.input_field.setPlaceholderText("Введите текст для модели...")
        elif data_type == "image":
            self.input_hint_label.setText(
                "🖼 Модель для изображений: используйте кнопку загрузки изображения."
            )
            self.input_field.setPlaceholderText("Изображения загружаются кнопкой выше...")
        else:
            input_dim = config_dict.get("input_dim", 1)
            self.input_hint_label.setText(
                f"🔢 Числовая модель: введите ровно {input_dim} чисел(а) через пробел."
            )
            self.input_field.setPlaceholderText(f"Например: {'1.0 ' * int(input_dim)}")

    # ========================================================
    #  ЧАТ
    # ========================================================
    def _add_message(self, role: str, text: str):
        """Добавляет сообщение в чат и обновляет отображение."""
        msg = ChatMessage(role, text)
        self.chat_history.append(msg)
        self.chat_display.append(msg.to_html())
        # Автоскролл вниз
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _clear_chat(self):
        """Очищает историю чата."""
        self.chat_history.clear()
        self.chat_display.clear()
        self.chat_display.setHtml(
            '<p style="color: #777; text-align: center;">Чат очищен.</p>'
        )
        self._log_action("clear_chat", "Чат очищен")

    def _export_chat(self):
        """Сохраняет историю чата в файл."""
        if not self.chat_history:
            QMessageBox.information(self, "Пусто", "Нечего сохранять — чат пуст.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить историю чата",
            "chat_history.txt", "Текст (*.txt);;HTML (*.html)"
        )
        if not path:
            return

        try:
            if path.endswith(".html"):
                content = "<html><body>\n"
                content += "<h2>История чата с нейросетью</h2>\n"
                for msg in self.chat_history:
                    content += msg.to_html() + "\n"
                content += "</body></html>"
            else:
                lines = []
                for msg in self.chat_history:
                    lines.append(f"[{msg.timestamp}] {msg.role}: {msg.text}")
                content = "\n".join(lines)

            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)

            self._add_message(ChatMessage.ROLE_SYSTEM, f"История сохранена в {path}")
            self._log_action("export_chat", f"Чат сохранён в {path}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить:\n{e}")

    # ========================================================
    #  ОТПРАВКА И ПРЕДСКАЗАНИЕ
    # ========================================================
    def _on_send(self):
        """Обработка нажатия кнопки отправки или Enter."""
        model = self.shared_state.get("model")
        if not model:
            self._add_message(
                ChatMessage.ROLE_SYSTEM,
                "⚠️ Модель не загружена. Нажмите «Загрузить модель» выше."
            )
            return

        user_input = self.input_field.text().strip()
        if not user_input:
            return

        # Показываем ввод пользователя
        self._add_message(ChatMessage.ROLE_USER, user_input)
        self.input_field.clear()

        # Определяем тип и делаем предсказание
        config = config_to_dict(self.shared_state.get("config"))
        data_type = config.get("data_type", "numeric")

        try:
            if data_type == "text":
                result, details = self._predict_text(model, user_input, config)
            elif data_type == "image":
                result, details = self._predict_image_stub(config)
            else:
                result, details = self._predict_numeric(model, user_input, config)

            self._add_message(ChatMessage.ROLE_MODEL, result)
            if details:
                self._add_message(ChatMessage.ROLE_SYSTEM, details)

        except ValueError as ve:
            self._add_message(ChatMessage.ROLE_ERROR, str(ve))
        except Exception as e:
            logger.error(f"Сбой предсказания: {e}")
            self._add_message(
                ChatMessage.ROLE_ERROR,
                f"Внутренняя ошибка модели: {e}"
            )

        self._log_action("predict", f"Ввод: {user_input[:50]}")

    def _predict_text(
        self, model, input_text: str, config: Dict
    ) -> Tuple[str, str]:
        """
        Предсказание для текстовой модели.
        Токенизация → generate → декодирование.
        """
        dataset = dataset_to_legacy_dict(self.shared_state.get("dataset"))
        vocab = dataset.get("vocab", config.get("vocab", {}))

        if not vocab:
            return (
                "Словарь не найден. Модель не может обработать текст.",
                "Загрузите проект (.oai) со словарём или создайте модель заново."
            )

        inv_vocab = {v: k for k, v in vocab.items()}
        pad_idx = vocab.get("<PAD>", 0)
        unk_idx = vocab.get("<UNK>", 1)
        bos_idx = vocab.get("<BOS>", 2)
        eos_idx = vocab.get("<EOS>", 3)

        # Токенизация
        x_seq = [vocab.get(char, unk_idx) for char in input_text]
        unknowns = sum(1 for x in x_seq if x == unk_idx)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        x_tensor = torch.tensor([x_seq], dtype=torch.long).to(device)

        with torch.no_grad():
            if hasattr(model, "generate"):
                max_len = config.get("max_len", 50)
                output_indices = model.generate(x_tensor, max_len=max_len)
                generated = output_indices[0].cpu().tolist()

                chars = []
                for idx in generated:
                    if idx == eos_idx:
                        break
                    if idx in (pad_idx, bos_idx):
                        continue
                    chars.append(inv_vocab.get(idx, "?"))
                result = "".join(chars)

                details = f"🧠 Обработано символов: {len(x_seq)} → сгенерировано: {len(chars)}"
                if unknowns > 0:
                    details += f" | ⚠️ {unknowns} неизвестных символов заменены на <UNK>"

                return result if result else "(пустой ответ)", details
            else:
                return (
                    "Эта архитектура не поддерживает генерацию текста.",
                    "Для текстовых ответов используйте Transformer или RNN."
                )

    def _predict_numeric(
        self, model, input_text: str, config: Dict
    ) -> Tuple[str, str]:
        """
        Предсказание для числовой модели.
        Парсинг чисел → тензор → forward.
        """
        try:
            # Разрешаем запятые как разделители
            cleaned = input_text.replace(",", " ").replace(";", " ")
            x_vals = [float(x) for x in cleaned.split()]
        except ValueError:
            raise ValueError(
                "Я понимаю только числа, разделённые пробелом.\n"
                "Например: «1.5 2.0 3.14»"
            )

        expected_dim = config.get("input_dim", 1)
        if isinstance(expected_dim, (list, tuple)):
            expected_dim = expected_dim[0] if expected_dim else 1

        if len(x_vals) != expected_dim:
            raise ValueError(
                f"Модель ожидает ровно {expected_dim} чисел(а) на вход, "
                f"а вы ввели {len(x_vals)}. "
                f"Добавьте или уберите числа."
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        x_tensor = torch.tensor([x_vals], dtype=torch.float32).to(device)

        with torch.no_grad():
            output_tensor = model(x_tensor)

        # Интерпретация результата
        output_np = output_tensor[0].cpu().numpy()
        output_dim = config.get("output_dim", 1)
        num_classes = config.get("num_classes", 0)

        if num_classes > 1 and output_np.shape[0] == num_classes:
            # Классификация: argmax + вероятности
            probs = np.exp(output_np) / np.exp(output_np).sum()
            predicted_class = int(np.argmax(probs))
            confidence = probs[predicted_class]
            result = f"Класс: {predicted_class} (уверенность: {confidence:.1%})"
            details = f"🧠 Распределение: {', '.join(f'{i}: {p:.1%}' for i, p in enumerate(probs))}"
            return result, details
        else:
            # Регрессия
            result = str(np.round(output_np, 6))
            details = f"🧠 Обработано {len(x_vals)} числовых признаков."
            return result, details

    def _predict_image_stub(self, config: Dict) -> Tuple[str, str]:
        """Заглушка для моделей на изображениях."""
        return (
            "🖼 Работа с изображениями в песочнице появится в следующей версии.",
            "Попробуйте загрузить изображение через вкладку «Данные»."
        )

    # ========================================================
    #  ВСПОМОГАТЕЛЬНЫЕ
    # ========================================================
    def _log_action(self, action: str, details: str = ""):
        """Логирует действие пользователя в историю сеанса."""
        try:
            session_mgr = self.shared_state.get("session_manager")
            if session_mgr:
                session_mgr.log_action("sandbox_panel", action, details)
        except Exception:
            pass

    def showEvent(self, event):
        """При показе вкладки обновляем информацию о модели."""
        super().showEvent(event)
        self.refresh()