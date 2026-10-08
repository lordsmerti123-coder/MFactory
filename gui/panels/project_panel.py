"""
gui/panels/project_panel.py
============================
Панель проекта: первый экран, который видит пользователь.
Отвечает за выбор сценария работы, именование проекта и модели.

Зависимости: core/curator.py (может отсутствовать на момент сборки — предусмотрен fallback).
Сигналы:
    - scenario_selected(str): при выборе сценария ("new" | "finetune" | "play")
    - project_created(dict): при создании/загрузке проекта
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QFormLayout, QLineEdit, QPushButton, QLabel,
    QRadioButton, QButtonGroup, QTextEdit, QFileDialog,
    QMessageBox, QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import pyqtSignal, Qt
from pathlib import Path

from core.logger import get_logger

logger = get_logger()

# ============================================================
# ПОПЫТКА ИМПОРТА КУРАТОРА (может быть ещё не написан)
# ============================================================
try:
    from core.curator import Curator
    _CURATOR_AVAILABLE = True
except ImportError:
    _CURATOR_AVAILABLE = False
    logger.warning(
        "core/curator.py не найден. "
        "Генератор имён будет использовать локальный список."
    )

# Локальный fallback-список весёлых имён
_FALLBACK_NAMES = [
    "Мозгожуй-3000", "Умник-2000", "НейроВася", "Гендальф Серый",
    "Скайнет-младший", "Пиксель", "Байтик", "Тензорчик", "Градиентик",
    "Эпоха-1", "Лоссик", "Свёрточка", "Рекуррентик", "Трансформерчик",
    "Капсулка", "Диффузик", "Автокодировщик", "Шпион-Сеть",
    "Квантовый Хомяк", "НейроКот", "Обучашка", "Векторок",
    "Эмбеддингушка", "Сину-Сеть", "Активатор", "Макспулер",
    "Батч-Нормик", "Адамушка", "Дропаутик", "Пиксель-Мозг"
]

# ============================================================
# ПОПЫТКА ИМПОРТА ДВИЖКА ПОДСКАЗОК (может быть ещё не написан)
# ============================================================
try:
    from core.hint_engine import HintEngine
    _HINT_ENGINE_AVAILABLE = True
except ImportError:
    _HINT_ENGINE_AVAILABLE = False


class ProjectPanel(QWidget):
    """
    Первый экран приложения. Пользователь:
    1. Вводит имя проекта.
    2. Генерирует или вводит имя модели.
    3. Выбирает сценарий: «С нуля» / «Дообучить» / «Играть».
    4. (Опционально) описывает задачу.
    5. Нажимает «Создать проект» или «Загрузить существующий».
    """

    # Сигнал выбора сценария. Аргумент: "new" | "finetune" | "play"
    scenario_selected = pyqtSignal(str)
    # Сигнал создания/загрузки проекта. Аргумент: словарь с данными проекта.
    project_created = pyqtSignal(dict)

    def __init__(self, shared_state: dict, parent=None):
        super().__init__(parent)
        self.shared_state = shared_state

        # Инициализируем куратора (или None если модуль недоступен)
        self._curator = Curator() if _CURATOR_AVAILABLE else None

        # Убеждаемся, что ключ "project" существует в shared_state
        if "project" not in self.shared_state:
            self.shared_state["project"] = {
                "name": "",
                "scenario": "new",
                "model_name": "",
                "description": "",
            }

        self._project_ready = False  # Флаг: проект создан/загружен

        self.init_ui()
        self._apply_hints()

    # ============================================================
    # ИНИЦИАЛИЗАЦИЯ ИНТЕРФЕЙСА
    # ============================================================
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # ---------- Заголовок ----------
        header = QLabel("🏠 Добро пожаловать в OracleAI Studio!")
        header.setStyleSheet(
            "font-size: 22px; font-weight: bold; color: #cc7832; "
            "padding: 8px; margin-bottom: 4px;"
        )
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

        subtitle = QLabel(
            "Создай проект, дай модели имя и выбери, чем будем заниматься."
        )
        subtitle.setStyleSheet("font-size: 14px; color: #a9b7c6; margin-bottom: 8px;")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)

        # ---------- 1. Имя проекта и описание ----------
        project_group = QGroupBox("📋 Проект")
        project_form = QFormLayout()

        self.project_name_edit = QLineEdit()
        self.project_name_edit.setPlaceholderText("Например: Математика-7Б, Фигуры, Шифровка...")
        self.project_name_edit.setMaximumWidth(400)
        self.project_name_edit.setToolTip(
            "Имя проекта — как название папки для всех твоих наработок.\n"
            "Потом сможешь сохранить проект на флешку и продолжить дома."
        )
        project_form.addRow("Имя проекта:", self.project_name_edit)

        self.description_edit = QTextEdit()
        self.description_edit.setPlaceholderText(
            "Необязательно. Например: «Учим нейросеть складывать числа до 100»"
        )
        self.description_edit.setMaximumHeight(60)
        self.description_edit.setToolTip(
            "Краткое описание того, чему ты хочешь научить нейросеть.\n"
            "Это поможет в отчёте и если вернёшься к проекту через неделю."
        )
        project_form.addRow("Описание задачи:", self.description_edit)

        project_group.setLayout(project_form)
        layout.addWidget(project_group)

        # ---------- 2. Имя модели ----------
        model_group = QGroupBox("🧠 Модель")
        model_layout = QHBoxLayout()

        model_label = QLabel("Имя модели:")
        model_label.setToolTip(
            "Каждая нейросеть получает имя. Можно придумать самому,\n"
            "а можно нажать 🎲 и программа предложит весёлое имя."
        )
        model_layout.addWidget(model_label)

        self.model_name_edit = QLineEdit()
        self.model_name_edit.setPlaceholderText("НейроВася-7")
        self.model_name_edit.setMaximumWidth(300)
        self.model_name_edit.setToolTip(
            "Имя модели. Отображается на вкладках «Обучение», «Мониторинг», «Анализ».\n"
            "Можно использовать весёлое имя или дать своё."
        )
        model_layout.addWidget(self.model_name_edit)

        self.btn_random_name = QPushButton("🎲 Другое имя")
        self.btn_random_name.setToolTip(
            "Нажми, чтобы сгенерировать случайное весёлое имя для модели.\n"
            "Например: «Мозгожуй-3000» или «НейроКот»."
        )
        self.btn_random_name.clicked.connect(self._generate_funny_name)
        model_layout.addWidget(self.btn_random_name)

        model_layout.addStretch()
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # ---------- 3. Выбор сценария ----------
        scenario_group = QGroupBox("🎯 Что будем делать?")
        scenario_layout = QVBoxLayout()

        self.scenario_button_group = QButtonGroup(self)

        # Сценарий А: С нуля
        self.radio_new = QRadioButton("🆕 Создаю нейросеть с нуля")
        self.radio_new.setChecked(True)
        self.radio_new.setToolTip(
            "Полный путь: генерируешь данные → строишь архитектуру → "
            "настраиваешь параметры → обучаешь → анализируешь.\n"
            "Идеально для первого раза!"
        )
        self.scenario_button_group.addButton(self.radio_new, 0)

        desc_new = QLabel(
            "   Генератор данных → Архитектура → Гиперпараметры → Обучение → Анализ"
        )
        desc_new.setStyleSheet("color: #777; font-size: 12px; margin-left: 24px;")

        # Сценарий Б: Дообучение
        self.radio_finetune = QRadioButton("🔄 Дообучаю готовую модель")
        self.radio_finetune.setToolTip(
            "У тебя уже есть обученная модель (.oai или .pth), и ты хочешь "
            "продолжить её обучение на новых данных.\n"
            "Например: модель выучила сложение, теперь учим вычитание."
        )
        self.scenario_button_group.addButton(self.radio_finetune, 1)

        desc_finetune = QLabel(
            "   Загрузка модели → Новые данные → Обучение → Анализ"
        )
        desc_finetune.setStyleSheet("color: #777; font-size: 12px; margin-left: 24px;")

        # Сценарий В: Игра / Тест
        self.radio_play = QRadioButton("🎮 Хочу поиграть с готовой моделью")
        self.radio_play.setToolTip(
            "Модель уже обучена (твоя или с флешки). Ты хочешь просто "
            "попробовать её в деле: задать вопрос, нарисовать картинку.\n"
            "Без обучения — только тестирование и чат."
        )
        self.scenario_button_group.addButton(self.radio_play, 2)

        desc_play = QLabel(
            "   Загрузка модели → Песочница / Чат → Экспорт"
        )
        desc_play.setStyleSheet("color: #777; font-size: 12px; margin-left: 24px;")

        scenario_layout.addWidget(self.radio_new)
        scenario_layout.addWidget(desc_new)
        scenario_layout.addSpacing(6)
        scenario_layout.addWidget(self.radio_finetune)
        scenario_layout.addWidget(desc_finetune)
        scenario_layout.addSpacing(6)
        scenario_layout.addWidget(self.radio_play)
        scenario_layout.addWidget(desc_play)

        # Подключаем переключение сценариев
        self.scenario_button_group.buttonClicked.connect(self._on_scenario_changed)

        scenario_group.setLayout(scenario_layout)
        layout.addWidget(scenario_group)

        # ---------- 4. Кнопки действий ----------
        actions_layout = QHBoxLayout()

        self.btn_create = QPushButton("🚀 Создать проект и продолжить")
        self.btn_create.setObjectName("StartBtn")
        self.btn_create.setMinimumHeight(40)
        self.btn_create.setToolTip(
            "Создаёт новый пустой проект и открывает следующую вкладку.\n"
            "Не забудь ввести имя проекта!"
        )
        self.btn_create.clicked.connect(self._create_project)
        actions_layout.addWidget(self.btn_create)

        self.btn_load = QPushButton("📂 Загрузить существующий проект (.oai)")
        self.btn_load.setMinimumHeight(40)
        self.btn_load.setToolTip(
            "Открывает ранее сохранённый проект.\n"
            "Файл .oai содержит: датасет + модель + настройки + историю обучения."
        )
        self.btn_load.clicked.connect(self._load_project)
        actions_layout.addWidget(self.btn_load)

        layout.addLayout(actions_layout)

        # ---------- 5. Статусная строка ----------
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(
            "color: #a6e3a1; font-size: 12px; padding: 4px; "
            "background-color: #313335; border-left: 3px solid #385a3a;"
        )
        self.status_label.setVisible(False)
        layout.addWidget(self.status_label)

        layout.addStretch()

        # Генерируем имя модели по умолчанию
        self._generate_funny_name()

        # ---------- 6. Информация о пользователе и хранилище ----------
        self.storage_group = QGroupBox("👤 Пользователь и хранилище")
        storage_layout = QVBoxLayout()

        self.user_info_label = QLabel("")
        self.user_info_label.setWordWrap(True)
        self.user_info_label.setStyleSheet("color: #a9b7c6; font-size: 12px;")
        storage_layout.addWidget(self.user_info_label)

        self.btn_storage_analysis = QPushButton("📊 Анализ хранилища")
        self.btn_storage_analysis.setToolTip(
            "Показывает отчёт: проекты, модели, датасеты и рекомендации."
        )
        self.btn_storage_analysis.clicked.connect(self._show_storage_analysis)
        storage_layout.addWidget(self.btn_storage_analysis)

        self.storage_report_label = QLabel("")
        self.storage_report_label.setWordWrap(True)
        self.storage_report_label.setStyleSheet(
            "color: #e5c07b; font-size: 11px;"
        )
        storage_layout.addWidget(self.storage_report_label)

        self.storage_group.setLayout(storage_layout)
        self.storage_group.setVisible(False)
        layout.addWidget(self.storage_group)

    # ============================================================
    # ПОЛЬЗОВАТЕЛЬ И ХРАНИЛИЩЕ
    # ============================================================
    def _refresh_user_storage(self):
        """Обновляет блок пользователя и хранилища."""
        try:
            user_summary = self.shared_state.get("user") or {}
            username = user_summary.get("username", "гость")
            if not username:
                return
            self.storage_group.setVisible(True)
            level = user_summary.get("level", "beginner")
            self.user_info_label.setText(
                f"👤 Пользователь: <b>{username}</b> (уровень: {level}) | "
                f"Проектов: {user_summary.get('projects', 0)} | "
                f"Моделей: {user_summary.get('models', 0)} | "
                f"Датасетов: {user_summary.get('datasets', 0)} | "
                f"Хранилище: {user_summary.get('storage_usage_mb', 0)} МБ"
            )
        except Exception as e:
            logger.warning(f"Ошибка обновления информации о пользователе: {e}")

    def _show_storage_analysis(self):
        """Показывает полный анализ хранилища пользователя."""
        try:
            from core.user_storage_analyzer import UserStorageAnalyzer
            user_manager = self.shared_state.get("user_manager")
            if user_manager is None:
                QMessageBox.information(self, "Нет данных", "Система пользователей недоступна.")
                return
            username = (self.shared_state.get("user") or {}).get("username", "guest")
            analyzer = UserStorageAnalyzer(user_manager, username)
            report = analyzer.analyze()
            text = analyzer.format_report_text(report)
            self.storage_report_label.setText(text)
            self._log_action("storage_analysis", f"{len(report.projects)} проектов")
        except Exception as e:
            logger.error(f"Ошибка анализа хранилища: {e}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось проанализировать хранилище:\n{e}")

    def showEvent(self, event):
        """При показе вкладки обновляем информацию о пользователе."""
        super().showEvent(event)
        self._refresh_user_storage()

    # ============================================================
    # ПОДСКАЗКИ
    # ============================================================
    def _apply_hints(self):
        """
        Применяет контекстные подсказки.
        Если HintEngine доступен — использует его.
        Иначе — полагается на setToolTip (уже установлен в init_ui).
        """
        if not _HINT_ENGINE_AVAILABLE:
            return

        # Будущее: подключение HintWidget к каждому элементу.
        # Пока движок не реализован — тултипы уже заданы в init_ui.
        # Пример будущего использования:
        # HintWidget(self.project_name_edit, hint_engine, "project_name", "project_panel")
        pass

    # ============================================================
    # ГЕНЕРАЦИЯ ВЕСЁЛОГО ИМЕНИ
    # ============================================================
    def _generate_funny_name(self):
        """Генерирует случайное весёлое имя для модели."""
        if self._curator is not None:
            try:
                name = self._curator.generate_model_name()
            except Exception as e:
                logger.warning(f"Curator.generate_model_name() упал: {e}. Fallback.")
                import random
                name = random.choice(_FALLBACK_NAMES)
        else:
            import random
            name = random.choice(_FALLBACK_NAMES)

        self.model_name_edit.setText(name)
        logger.info(f"Сгенерировано имя модели: {name}")

    # ============================================================
    # СМЕНА СЦЕНАРИЯ
    # ============================================================
    def _on_scenario_changed(self, button):
        """Вызывается при переключении радио-кнопок сценария."""
        scenario_map = {0: "new", 1: "finetune", 2: "play"}
        scenario = scenario_map.get(
            self.scenario_button_group.id(button), "new"
        )

        # Обновляем shared_state
        self.shared_state["project"]["scenario"] = scenario

        # Логируем действие пользователя для истории сеансов
        self._log_action("scenario_changed", f"Сценарий: {scenario}")

        # Изменяем текст кнопки создания в зависимости от сценария
        if scenario == "new":
            self.btn_create.setText("🚀 Создать проект и продолжить")
            self.btn_create.setEnabled(True)
            self.btn_load.setVisible(True)
        elif scenario == "finetune":
            self.btn_create.setText("🔄 Загрузить модель для дообучения")
            self.btn_create.setEnabled(False)
            self.btn_load.setVisible(True)
            self._show_status(
                "ℹ️ Для дообучения нажмите «Загрузить существующий проект» "
                "и выберите файл .oai с ранее обученной моделью."
            )
        elif scenario == "play":
            self.btn_create.setText("🎮 Загрузить модель для игры")
            self.btn_create.setEnabled(False)
            self.btn_load.setVisible(True)
            self._show_status(
                "ℹ️ Для игры нажмите «Загрузить существующий проект» "
                "и выберите файл .oai или модель с флешки."
            )

        logger.info(f"Сценарий проекта изменён: {scenario}")

    # ============================================================
    # СОЗДАНИЕ НОВОГО ПРОЕКТА
    # ============================================================
    def create_project_silent(self, name: str, model_name: str,
                              description: str = "") -> str:
        """Создаёт проект без модальных диалогов (для ИИ-агента)."""
        name = (name or "").strip()
        model_name = (model_name or "").strip()
        if not name:
            return "имя проекта не указано"
        if not model_name:
            model_name = f"{name}-модель"
        self.project_name_edit.setText(name)
        self.model_name_edit.setText(model_name)
        if hasattr(self, "description_edit"):
            self.description_edit.setPlainText(description or "")
        self._create_project()
        return f"проект «{name}» создан, модель «{model_name}»"

    def _create_project(self):
        """Создаёт новый проект и эмитит сигнал."""
        project_name = self.project_name_edit.text().strip()
        model_name = self.model_name_edit.text().strip()
        description = self.description_edit.toPlainText().strip()
        scenario = self._get_current_scenario()

        # --- Валидация ---
        if not project_name:
            QMessageBox.warning(
                self, "Внимание",
                "Введите имя проекта! Например: «Математика-7Б».\n"
                "Это имя будет отображаться на всех вкладках."
            )
            self.project_name_edit.setFocus()
            return

        if not model_name:
            # Если имя модели пустое — генерируем автоматически
            self._generate_funny_name()
            model_name = self.model_name_edit.text().strip()

        # --- Сохраняем в shared_state ---
        project_data = {
            "name": project_name,
            "scenario": scenario,
            "model_name": model_name,
            "description": description,
        }
        self.shared_state["project"].update(project_data)
        self._project_ready = True

        # --- Логирование ---
        self._log_action("project_created", f"Проект: {project_name}, сценарий: {scenario}")
        logger.info(
            f"Проект создан: '{project_name}' | Модель: '{model_name}' | Сценарий: {scenario}"
        )

        # --- Статус ---
        self._show_status(
            f"✅ Проект «{project_name}» создан! Модель: «{model_name}». "
            f"Сценарий: {scenario}."
        )

        # --- Эмитим сигналы ---
        self.project_created.emit(project_data)
        self.scenario_selected.emit(scenario)

    # ============================================================
    # ЗАГРУЗКА СУЩЕСТВУЮЩЕГО ПРОЕКТА
    # ============================================================
    def _load_project(self):
        """Открывает диалог загрузки и читает .oai файл."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Загрузить проект или модель",
            str(Path(".")),
            "OracleAI Project (*.oai);;PyTorch Model (*.pth);;All Files (*)"
        )
        if not path:
            return

        file_path = Path(path)
        if not file_path.exists():
            QMessageBox.critical(self, "Ошибка", f"Файл не найден:\n{path}")
            return

        suffix = file_path.suffix.lower()

        try:
            if suffix == ".oai":
                self._load_oai_project(file_path)
            elif suffix == ".pth":
                self._load_pth_model(file_path)
            else:
                QMessageBox.warning(
                    self, "Неподдерживаемый формат",
                    f"Формат '{suffix}' не поддерживается.\n"
                    "Ожидаются файлы .oai (проект) или .pth (модель)."
                )
                return
        except Exception as e:
            logger.error(f"Ошибка загрузки проекта: {e}")
            QMessageBox.critical(
                self, "Ошибка загрузки",
                f"Не удалось прочитать файл:\n{e}\n\n"
                "Возможно, файл повреждён или создан в другой версии программы."
            )

    def _load_oai_project(self, path: Path):
        """Загружает полный проект из .oai файла."""
        import json

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Проверяем версию формата
        format_version = data.get("format_version", "1.0")
        if format_version.startswith("1."):
            self._show_status(
                f"⚠️ Файл создан в старой версии (формат {format_version}). "
                "Пытаемся открыть..."
            )

        # Извлекаем данные проекта
        project_name = data.get("metadata", {}).get("project_name", path.stem)
        model_name = data.get("metadata", {}).get("model_name", "Модель")
        description = data.get("metadata", {}).get("description", "")
        scenario = self._get_current_scenario()

        # Сохраняем в shared_state
        project_data = {
            "name": project_name,
            "scenario": scenario,
            "model_name": model_name,
            "description": description,
            "loaded_file": str(path),
            "raw_data": data,  # полные данные для других модулей
        }
        self.shared_state["project"].update(project_data)

        # Обновляем UI
        self.project_name_edit.setText(project_name)
        self.model_name_edit.setText(model_name)
        self.description_edit.setPlainText(description)
        self._project_ready = True

        # Логируем
        self._log_action("project_loaded", f"Загружен: {project_name} из {path.name}")
        logger.info(f"Проект загружен: '{project_name}' из {path}")

        self._show_status(
            f"✅ Проект «{project_name}» загружен из {path.name}.\n"
            f"Модель: «{model_name}». Формат: v{format_version}."
        )

        # Эмитим сигналы
        self.project_created.emit(project_data)
        self.scenario_selected.emit(scenario)

    def _load_pth_model(self, path: Path):
        """Загружает только модель из .pth файла."""
        import torch

        checkpoint = torch.load(path, map_location="cpu")

        # Извлекаем конфиг модели
        config = checkpoint.get("config", {})
        model_name = config.get("model_name", path.stem)
        model_type = config.get("type", "unknown")

        # Сохраняем в shared_state
        project_data = {
            "name": path.stem,
            "scenario": self._get_current_scenario(),
            "model_name": model_name,
            "description": f"Загружена из {path.name}",
            "loaded_file": str(path),
            "checkpoint": checkpoint,  # полные данные для других модулей
        }
        self.shared_state["project"].update(project_data)

        # Обновляем UI
        self.project_name_edit.setText(path.stem)
        self.model_name_edit.setText(model_name)
        self._project_ready = True

        # Логируем
        self._log_action("model_loaded", f"Модель: {model_name} ({model_type}) из {path.name}")
        logger.info(f"Модель загружена: '{model_name}' ({model_type}) из {path}")

        self._show_status(
            f"✅ Модель «{model_name}» ({model_type.upper()}) загружена из {path.name}.\n"
            "Для дообучения перейдите на вкладку «Данные». Для теста — «Песочница»."
        )

        # Эмитим сигналы
        self.project_created.emit(project_data)
        self.scenario_selected.emit(self._get_current_scenario())

    # ============================================================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ============================================================
    def _get_current_scenario(self) -> str:
        """Возвращает текущий выбранный сценарий."""
        checked_id = self.scenario_button_group.checkedId()
        scenario_map = {0: "new", 1: "finetune", 2: "play"}
        return scenario_map.get(checked_id, "new")

    def _show_status(self, text: str):
        """Показывает статусную строку."""
        self.status_label.setText(text)
        self.status_label.setVisible(True)

    def _log_action(self, action: str, details: str = ""):
        """
        Логирует действие пользователя для истории сеансов.
        Если SessionManager доступен — записывает туда.
        """
        # Попытка записать в сессию
        session = self.shared_state.get("session")
        if session is not None and hasattr(session, "log_action"):
            try:
                session.log_action("project_panel", action, details)
            except Exception as e:
                logger.debug(f"Не удалось записать в сессию: {e}")

        # Всегда пишем в лог
        logger.info(f"[Проект] {action}: {details}")

    # ============================================================
    # ПУБЛИЧНЫЕ МЕТОДЫ ДЛЯ ВНЕШНЕГО ДОСТУПА
    # ============================================================
    def is_project_ready(self) -> bool:
        """Возвращает True, если проект создан или загружен."""
        return self._project_ready

    def get_project_data(self) -> dict:
        """Возвращает текущие данные проекта из shared_state."""
        return self.shared_state.get("project", {})

    def set_scenario(self, scenario: str):
        """Программно устанавливает сценарий (для внешнего управления)."""
        scenario_to_id = {"new": 0, "finetune": 1, "play": 2}
        btn_id = scenario_to_id.get(scenario, 0)
        button = self.scenario_button_group.button(btn_id)
        if button:
            button.setChecked(True)
            self._on_scenario_changed(button)