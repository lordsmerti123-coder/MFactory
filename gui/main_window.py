"""
gui/main_window.py
==================
Главное окно OracleAI Studio v2.0.

Реализует:
- Три сценария входа: "new" (с нуля), "finetune" (дообучение), "play" (песочница)
- Единый shared_state согласно контрактам
- Интеграцию SessionManager, HintEngine, Curator
- Все критические связи сигналов между панелями

ПРИМЕЧАНИЯ ДЛЯ СБОРЩИКА:
1. Логи создаются как LogsPanel() (без shared_state, если панель не принимает его)
2. Все панели получают shared_state первым аргументом
3. Панели, требующие hint_engine отдельно (например, ExportPanel), получают его вторым
4. Обязательные методы Мониторинга, которые должны быть реализованы в панели №26:
   - on_training_started(is_text: bool) — сброс графиков и установка режима
   - on_training_finished() — деактивация кнопки экстренного стопа
   - update_plots(epoch: int, logs: dict) — обновление графиков
   - set_model_name(name: str) — отображение имени модели
   - сигнал stop_requested = pyqtSignal() — для экстренной остановки
5. Имя модели передаётся в Мониторинг при создании (из Архитектуры)
   и при создании проекта (из Панели Проекта).
"""

import sys
from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QVBoxLayout, QWidget,
    QHBoxLayout, QLabel, QComboBox, QSplitter, QPushButton,
)
from PyQt5.QtCore import Qt, pyqtSignal

# ============================================================
# GUI
# ============================================================
from gui.styles import MODERN_THEME
from gui.panels.project_panel import ProjectPanel
from gui.panels.generator_panel import GeneratorPanel
from gui.panels.dataset_panel import DatasetPanel
from gui.panels.architecture_panel import ArchitecturePanel
from gui.panels.hyperparams_panel import HyperparamsPanel
from gui.panels.training_panel import TrainingPanel
from gui.panels.monitoring_panel import MonitoringPanel
from gui.panels.analysis_panel import AnalysisPanel
from gui.panels.export_panel import ExportPanel
from gui.panels.sandbox_panel import SandboxPanel
from gui.panels.logs_panel import LogsPanel
from gui.panels.admin_panel import AdminPanel
from gui.panels.wizard_panel import WizardPanel

# ============================================================
# ЯДРО
# ============================================================
from core.contracts import SessionState, ModelConfig, TrainingHistory
from core.session import SessionManager
from core.session_manager_extended import ExtendedSessionManager
from core.user_manager import UserManager
from core.pretrained_manager import PretrainedManager
from core.workflow_manager import WorkflowManager
from core.oraculum.agent import OraculumAgent
from core.hint_engine import HintEngine
from core.curator import Curator
from core.logger import get_logger
from core.activity_tracker import activity_tracker

# ============================================================
# GUI-МОДУЛИ (новые)
# ============================================================
from gui.login_dialog import LoginDialog
from gui.oraculum_chat import OraculumChatWidget
from gui.workflow_widget import WorkflowWidget
from gui.agent_bridge import AgentBridge

# ============================================================
# КОНФИГ
# ============================================================
from config import (
    APP_NAME,
    __version__,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    WINDOW_MIN_WIDTH,
    WINDOW_MIN_HEIGHT,
    WINDOW_SCREEN_FRACTION,
)

logger = get_logger(__name__)


def _safe_create_panel(panel_class, shared_state, hint_engine=None, extra_args=None):
    """
    Безопасный конструктор панели. Пробует разные сигнатуры __init__:
    1. (shared_state)
    2. (shared_state, hint_engine)
    3. (shared_state, hint_engine, **extra_args)
    4. () — для панелей без аргументов (например, LogsPanel)
    Это защищает от падений при несовпадении сигнатур между панелями.
    """
    extra_args = extra_args or {}

    # Попытка 1: с hint_engine и extras
    if hint_engine is not None:
        try:
            return panel_class(shared_state, hint_engine, **extra_args)
        except TypeError:
            pass

    # Попытка 2: только shared_state
    try:
        return panel_class(shared_state, **extra_args)
    except TypeError:
        pass

    # Попытка 3: только hint_engine (на случай странной сигнатуры)
    if hint_engine is not None:
        try:
            return panel_class(hint_engine, **extra_args)
        except TypeError:
            pass

    # Попытка 4: без аргументов
    try:
        return panel_class(**extra_args)
    except TypeError as e:
        logger.error(
            f"Не удалось создать панель {panel_class.__name__}: {e}. "
            f"Проверьте сигнатуру __init__."
        )
        # Fallback — создаём пустой QWidget, чтобы приложение не упало
        placeholder = QWidget()
        placeholder.setStyleSheet("background-color: #6a3636;")
        return placeholder


class MainWindow(QMainWindow):
    """Главное окно OracleAI Studio v2.0."""

    # Сигнал статуса локальной GGUF-модели (вызывается из фонового потока).
    local_model_status = pyqtSignal(bool, str)

    # Соответствие индексов выпадающего списка режимам подсказок
    HINT_MODE_MAP = {
        0: "static",   # 💬 Подсказки (статические)
        1: "ai",       # 🤖 ИИ (локальная модель)
        2: "hybrid",   # 🔀 Смешанный (рекомендуется)
        3: "none",     # 🚫 Без подсказок
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{__version__} — Образовательная среда ИИ")
        self._apply_window_size()
        self.setStyleSheet(MODERN_THEME)

        # ================================================================
        # 1. ИНИЦИАЛИЗАЦИЯ ЯДРА (до создания панелей)
        # ================================================================
        # --- 1.1 Система пользователей ---
        self.user_manager = UserManager()
        self._login_user()

        # --- 1.2 Сессии (расширенные, под пользователя) ---
        self.session_manager = ExtendedSessionManager(
            self.user_manager, self.current_username
        )
        self.session_manager.increment_sessions()

        self.hint_engine = HintEngine(self.session_manager)
        self.curator = Curator(self.session_manager.state)

        # --- 1.3 Предобученные модели ---
        self.pretrained_manager = PretrainedManager()

        # ================================================================
        # 2. SHARED_STATE — ЕДИНЫЙ ИСТОЧНИК СОСТОЯНИЯ
        # ================================================================
        self.shared_state = {
            "session": self.session_manager.state,
            "project": {
                "name": "",
                "scenario": "new",
                "model_name": "",
                "description": "",
            },
            "dataset": None,
            "model": None,
            "config": ModelConfig(),
            "history": TrainingHistory(),
            "hint_mode": "hybrid",
            "gguf_backend": None,
            # Служебные объекты для панелей и виджетов подсказок
            "hint_engine": self.hint_engine,
            "curator": self.curator,
            "session_manager": self.session_manager,
            # Новые ключи v2.0: пользователи, модели, Оракул, рабочий процесс
            "user": self.session_manager.get_user_summary(),
            "user_manager": self.user_manager,
            "pretrained_manager": self.pretrained_manager,
            "pretrained_models": self.pretrained_manager.get_all_models(),
        }

        # --- 1.4 ИИ-агент «Оракул» (после shared_state) ---
        self.oraculum_agent = OraculumAgent(self.shared_state)
        self.shared_state["oraculum"] = self.oraculum_agent
        # Подключаем агента к движку подсказок, чтобы ИИ-подсказки
        # (режим «ИИ») генерировались локальной моделью / API.
        HintEngine.set_default_agent(self.oraculum_agent)

        # Агентские возможности: мост в GUI (потокобезопасное выполнение
        # инструментов в главном потоке) + реестр GUI-инструментов.
        self._agent_handlers = {}
        self.agent_bridge = AgentBridge(self._execute_agent_action)
        self.oraculum_agent.set_action_dispatcher(self.agent_bridge.dispatch)
        self._register_agent_tools()

        # --- 1.5 Рабочий процесс (после shared_state) ---
        self.workflow_manager = WorkflowManager(self.shared_state)
        self.shared_state["workflow"] = self.workflow_manager

        # ================================================================
        # 3. ИНИЦИАЛИЗАЦИЯ ИНТЕРФЕЙСА
        # ================================================================
        # Чат создаётся ДО init_ui, чтобы init_ui мог встроить его в сплиттер.
        self._setup_oraculum_chat()
        self.init_ui()

        # ================================================================
        # 4. ПОДКЛЮЧЕНИЕ СИГНАЛОВ
        # ================================================================
        self.connect_signals()

        # ================================================================
        # 5. СТАНДАРТНЫЙ СЦЕНАРИЙ ПРИ ЗАПУСКЕ
        # ================================================================
        self._setup_scenario("new")

        # Статус локальной модели (из фонового потока) → чат Оракула
        self.local_model_status.connect(self._on_local_model_status)

        logger.info(
            f"MainWindow инициализирован. Пользователь: "
            f"'{self.current_username}', уровень: "
            f"{self.session_manager.profile.user_level.value}"
        )

    # ====================================================================
    # АВТОРИЗАЦИЯ
    # ====================================================================
    def _apply_window_size(self):
        """
        Адаптивный размер окна: занимает ~85% доступной области экрана.
        На 2K-экранах это даёт полноразмерное окно, а не фиксированные
        1200×800 (которые выглядят маленькими и тесными).
        Координаты логические — при QT_SCALE_FACTOR они масштабируются
        автоматически.
        """
        try:
            from PyQt5.QtWidgets import QApplication
            screen = QApplication.primaryScreen()
            if screen is not None:
                avail = screen.availableGeometry()
                w = int(avail.width() * WINDOW_SCREEN_FRACTION)
                h = int(avail.height() * WINDOW_SCREEN_FRACTION)
                w = max(w, WINDOW_MIN_WIDTH)
                h = max(h, WINDOW_MIN_HEIGHT)
            else:
                w, h = WINDOW_WIDTH, WINDOW_HEIGHT
        except Exception:
            w, h = WINDOW_WIDTH, WINDOW_HEIGHT
        self.resize(w, h)

    def _login_user(self):
        """Показывает окно входа и выбирает текущего пользователя."""
        dialog = LoginDialog(self.user_manager)
        if dialog.exec_():
            username = dialog.get_selected_username()
        else:
            # Пользователь закрыл диалог — входим как гость
            username = None
        # Если имя не выбрано или пользователь не найден — гость
        if not username or self.user_manager.get_profile(username) is None:
            profile = self.user_manager.create_guest()
            username = profile.username
        self.user_manager.switch_user(username)
        self.current_username = username
        self.is_admin = self.user_manager.is_admin(username)
        activity_tracker.set_user(username)

    # ====================================================================
    # ОРАКУЛ (мини-чат)
    # ====================================================================
    def _setup_oraculum_chat(self):
        """Создаёт чат Оракула как встроенную панель (не отдельное окно)."""
        try:
            self.oraculum_chat = OraculumChatWidget(
                self.oraculum_agent, self, docked=True
            )
            # Приветственное сообщение от Оракула с учётом текущего шага
            hint = self.oraculum_agent.get_workflow_hint()
            if hint:
                self.oraculum_chat.add_message("Оракул", hint)
            # Тумблер «Агент» и статус провайдера (Яндекс GPT / GGUF / шаблоны)
            self.oraculum_chat.init_agent_controls()
            # Включение агента → подгружаем локальную модель в фоне
            if hasattr(self.oraculum_chat, "agent_mode_toggled"):
                self.oraculum_chat.agent_mode_toggled.connect(
                    self._on_agent_mode_toggled
                )
            logger.info("Чат Оракула создан (встроенная панель).")
        except Exception as e:
            logger.warning(f"Не удалось создать чат Оракула: {e}")
            self.oraculum_chat = None

    def _on_chat_toggle(self, checked: bool):
        """Показывает/скрывает встроенный чат Оракула."""
        if self.oraculum_chat is None:
            return
        self.oraculum_chat.setVisible(checked)
        if checked and hasattr(self, "splitter"):
            self.splitter.setSizes([max(self.width() - 380, 640), 360])

    # ====================================================================
    # ИНИЦИАЛИЗАЦИЯ ИНТЕРФЕЙСА
    # ====================================================================
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # ------------------------------------------------
        # Верхняя панель: выбор режима подсказок
        # ------------------------------------------------
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Режим подсказок:"))

        self.hint_mode_combo = QComboBox()
        self.hint_mode_combo.addItems([
            "💬 Подсказки (статические)",
            "🤖 ИИ (локальная модель)",
            "🔀 Смешанный (рекомендуется)",
            "🚫 Без подсказок",
        ])
        self.hint_mode_combo.setCurrentIndex(2)  # hybrid по умолчанию
        self.hint_mode_combo.setToolTip(
            "Режим работы системы помощи:\n"
            "• Подсказки — только правила, без ИИ. Работает всегда.\n"
            "• ИИ — только локальная модель (если подключена).\n"
            "• Смешанный — ИИ, если доступен; иначе правила (рекомендуется).\n"
            "• Без подсказок — для продвинутых пользователей."
        )
        self.hint_mode_combo.currentIndexChanged.connect(self._on_hint_mode_changed)
        top_bar.addWidget(self.hint_mode_combo)

        # Пользователь
        top_bar.addSpacing(16)
        top_bar.addWidget(QLabel("👤:"))
        self.user_label = QLabel(self.current_username)
        self.user_label.setToolTip(
            "Текущий пользователь. Данные (проекты, модели) хранятся отдельно "
            "для каждого пользователя."
        )
        self.user_label.setStyleSheet(
            "color: #a6e3a1; font-weight: bold; padding: 2px 8px;"
            "background-color: #313335; border-radius: 8px;"
        )
        top_bar.addWidget(self.user_label)

        # Кнопка «Оракул» — показать/скрыть встроенный чат помощника
        self.btn_chat_toggle = QPushButton("🤖 Оракул")
        self.btn_chat_toggle.setCheckable(True)
        self.btn_chat_toggle.setChecked(True)
        self.btn_chat_toggle.setToolTip(
            "Показать/скрыть чат ИИ-помощника. В чате можно спросить, "
            "что сейчас происходит в программе и что делать дальше."
        )
        self.btn_chat_toggle.toggled.connect(self._on_chat_toggle)
        top_bar.addWidget(self.btn_chat_toggle)

        # Кнопка «Мастер» — автономный режим создания нейросети
        self.btn_wizard = QPushButton("🚀 Мастер")
        self.btn_wizard.setToolTip(
            "Автономный режим: Оракул создаст и обучит нейросеть наглядно, "
            "шаг за шагом, с комментариями."
        )
        self.btn_wizard.clicked.connect(self._open_wizard)
        top_bar.addWidget(self.btn_wizard)

        top_bar.addStretch()

        # Индикатор рабочего процесса (WorkflowManager)
        self.workflow_widget = WorkflowWidget(self.workflow_manager)
        top_bar.addWidget(self.workflow_widget)

        layout.addLayout(top_bar)

        # ------------------------------------------------
        # Вкладки + встроенный чат Оракула (сплиттер)
        # ------------------------------------------------
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.tabs)
        self.splitter.setStretchFactor(0, 1)
        if self.oraculum_chat is not None:
            self.splitter.addWidget(self.oraculum_chat)
            self.splitter.setStretchFactor(1, 0)
            self.splitter.setSizes([max(self.width() - 380, 640), 360])
        layout.addWidget(self.splitter, stretch=1)

        # ------------------------------------------------
        # Создание панелей (безопасно через _safe_create_panel)
        # ------------------------------------------------
        # Базовые панели — принимают только shared_state
        self.project_panel = _safe_create_panel(
            ProjectPanel, self.shared_state, self.hint_engine
        )
        self.generator_panel = _safe_create_panel(
            GeneratorPanel, self.shared_state, self.hint_engine
        )
        self.dataset_panel = _safe_create_panel(
            DatasetPanel, self.shared_state, self.hint_engine
        )
        self.arch_panel = _safe_create_panel(
            ArchitecturePanel, self.shared_state, self.hint_engine
        )
        self.params_panel = _safe_create_panel(
            HyperparamsPanel, self.shared_state, self.hint_engine
        )
        self.training_panel = _safe_create_panel(
            TrainingPanel, self.shared_state, self.hint_engine
        )
        self.monitoring_panel = _safe_create_panel(
            MonitoringPanel, self.shared_state, self.hint_engine
        )
        self.analysis_panel = _safe_create_panel(
            AnalysisPanel, self.shared_state, self.hint_engine
        )

        # ExportPanel — требует hint_engine вторым аргументом (сигнатура панели)
        self.export_panel = _safe_create_panel(
            ExportPanel, self.shared_state, self.hint_engine
        )

        # SandboxPanel — принимает shared_state, hint_engine берёт из shared_state.get()
        self.sandbox_panel = _safe_create_panel(
            SandboxPanel, self.shared_state, self.hint_engine
        )

        # LogsPanel — в текущей реализации НЕ принимает shared_state,
        # только parent. Создаём без shared_state.
        self.logs_panel = _safe_create_panel(
            LogsPanel, self.shared_state, self.hint_engine
        )

        # Панель администратора — только для пользователя-админа.
        self.admin_panel = None
        if self.is_admin:
            self.admin_panel = _safe_create_panel(
                AdminPanel, self.shared_state, self.hint_engine
            )
            if isinstance(self.admin_panel, QWidget) and not hasattr(self.admin_panel, "refresh"):
                self.admin_panel = None

        # Панель «Мастер» — автономное создание нейросети (доступна всем)
        self.wizard_panel = WizardPanel(
            self.oraculum_agent, self._execute_agent_action, self
        )
        self.wizard_panel.bind_tabs(self.tabs)

        # Переключение вкладок → логирование и обновление контекста
        self.tabs.currentChanged.connect(self._on_tab_changed)

    # ====================================================================
    # ПОДКЛЮЧЕНИЕ СИГНАЛОВ
    # ====================================================================
    def connect_signals(self):
        # ================================================================
        # ПРОЕКТ
        # ================================================================
        if hasattr(self.project_panel, "scenario_selected"):
            self.project_panel.scenario_selected.connect(self._on_scenario_selected)
        if hasattr(self.project_panel, "model_name_changed"):
            self.project_panel.model_name_changed.connect(self._on_model_name_changed)
        if hasattr(self.project_panel, "project_created"):
            self.project_panel.project_created.connect(self._on_project_created)

        # ================================================================
        # ДАННЫЕ → АРХИТЕКТУРА / ГИПЕРПАРАМЕТРЫ
        # ================================================================
        if hasattr(self.dataset_panel, "dataset_updated") and \
           hasattr(self.arch_panel, "refresh") and \
           hasattr(self.params_panel, "refresh"):
            self.dataset_panel.dataset_updated.connect(self.arch_panel.refresh)
            self.dataset_panel.dataset_updated.connect(self.params_panel.refresh)

        # ================================================================
        # АРХИТЕКТУРА → ОБЩЕЕ СОСТОЯНИЕ + ИМЯ МОДЕЛИ
        # ================================================================
        if hasattr(self.arch_panel, "model_created"):
            self.arch_panel.model_created.connect(self._on_model_created)

        # ================================================================
        # ОБУЧЕНИЕ ↔ МОНИТОРИНГ (ОБЯЗАТЕЛЬНЫЕ СВЯЗИ)
        # ================================================================
        # 1. Обучение → Мониторинг (графики каждую эпоху)
        if hasattr(self.training_panel, "epoch_finished") and \
           hasattr(self.monitoring_panel, "update_plots"):
            self.training_panel.epoch_finished.connect(
                self.monitoring_panel.update_plots
            )

        # 2. Обучение началось → Мониторинг (сброс + режим)
        if hasattr(self.training_panel, "training_started") and \
           hasattr(self.monitoring_panel, "on_training_started"):
            self.training_panel.training_started.connect(
                self.monitoring_panel.on_training_started
            )

        # 3. Обучение завершилось → Мониторинг (деактивация стопа)
        if hasattr(self.training_panel, "training_finished") and \
           hasattr(self.monitoring_panel, "on_training_finished"):
            self.training_panel.training_finished.connect(
                lambda h: self.monitoring_panel.on_training_finished()
            )

        # 4. Мониторинг → Обучение (экстренный стоп)
        if hasattr(self.monitoring_panel, "stop_requested") and \
           hasattr(self.training_panel, "stop_training"):
            self.monitoring_panel.stop_requested.connect(
                self.training_panel.stop_training
            )

        # ================================================================
        # ОБУЧЕНИЕ → АНАЛИЗ
        # ================================================================
        if hasattr(self.training_panel, "training_finished") and \
           hasattr(self.analysis_panel, "run_analysis"):
            self.training_panel.training_finished.connect(
                self.analysis_panel.run_analysis
            )

        # ================================================================
        # АНАЛИЗ → ПОВТОР / ЭКСПОРТ
        # ================================================================
        if hasattr(self.analysis_panel, "retry_requested"):
            self.analysis_panel.retry_requested.connect(self._on_retry_requested)
        if hasattr(self.analysis_panel, "export_requested"):
            self.analysis_panel.export_requested.connect(self._on_export_requested)

        # ================================================================
        # ЭКСПОРТ / ПЕСОЧНИЦА (опциональные сигналы)
        # ================================================================
        if hasattr(self.export_panel, "export_done"):
            self.export_panel.export_done.connect(
                lambda path: logger.info(f"Экспорт завершён: {path}")
            )

        if hasattr(self.sandbox_panel, "model_loaded"):
            self.sandbox_panel.model_loaded.connect(
                lambda m, c: logger.info(f"Модель загружена в песочницу: {c.get('model_name', '?')}")
            )

        # ================================================================
        # АГЕНТСКИЕ ВОЗМОЖНОСТИ ОРАКУЛА
        # ================================================================
        # Обработчики GUI-инструментов (панели уже созданы в init_ui)
        self._populate_agent_handlers()

        # Оракул следит за обучением (агентский режим)
        if hasattr(self.training_panel, "training_started"):
            self.training_panel.training_started.connect(
                self.oraculum_agent.on_training_started
            )
        if hasattr(self.training_panel, "epoch_finished"):
            self.training_panel.epoch_finished.connect(
                self.oraculum_agent.on_training_epoch
            )
        if hasattr(self.training_panel, "training_finished"):
            self.training_panel.training_finished.connect(
                self.oraculum_agent.on_training_finished
            )

        # Режим «Мастер»: комментарии Оракула → чат
        self.wizard_panel.panel_narration.connect(self._on_wizard_narration)

        logger.info("Все сигналы подключены.")

    # ====================================================================
    # СЦЕНАРИИ ВХОДА
    # ====================================================================
    def _setup_scenario(self, scenario: str):
        """Перестраивает вкладки под выбранный сценарий."""
        scenario = scenario.lower().strip()
        if scenario not in ("new", "finetune", "play"):
            scenario = "new"

        self.shared_state["project"]["scenario"] = scenario
        self.tabs.clear()

        # Всегда: проект первым
        self.tabs.addTab(self.project_panel, "🏠 Проект")

        if scenario == "new":
            self.tabs.addTab(self.generator_panel, "✨ Генератор")
            self.tabs.addTab(self.dataset_panel, "📁 Данные")
            self.tabs.addTab(self.arch_panel, "🧠 Архитектура")
            self.tabs.addTab(self.params_panel, "⚙️ Гиперпараметры")
            self.tabs.addTab(self.training_panel, "🚀 Обучение")
            self.tabs.addTab(self.monitoring_panel, "📈 Мониторинг")
            self.tabs.addTab(self.analysis_panel, "🔍 Анализ")
            self.tabs.addTab(self.export_panel, "💾 Экспорт")

        elif scenario == "finetune":
            self.tabs.addTab(self.dataset_panel, "📁 Данные")
            self.tabs.addTab(self.training_panel, "🚀 Обучение")
            self.tabs.addTab(self.monitoring_panel, "📈 Мониторинг")
            self.tabs.addTab(self.analysis_panel, "🔍 Анализ")
            self.tabs.addTab(self.export_panel, "💾 Экспорт")

        elif scenario == "play":
            self.tabs.addTab(self.sandbox_panel, "🎮 Песочница")
            self.tabs.addTab(self.export_panel, "💾 Экспорт")

        # Всегда: логи в конце
        self.tabs.addTab(self.logs_panel, "📝 Логи")

        # Для администратора — панель дебага/администрирования
        if self.is_admin and self.admin_panel is not None:
            self.tabs.addTab(self.admin_panel, "🛡 Админ")

        # «Мастер» — автономный режим создания нейросети
        if hasattr(self, "wizard_panel") and self.wizard_panel is not None:
            self.tabs.addTab(self.wizard_panel, "🚀 Мастер")

        logger.info(f"Сценарий '{scenario}' активирован.")
        self.session_manager.log_action("main_window", "scenario_changed", scenario)

        # Обновляем рабочий процесс и подсказки Оракула
        self._refresh_workflow()

    # ====================================================================
    # РАБОЧИЙ ПРОЦЕСС / ОРАКУЛ
    # ====================================================================
    def _refresh_workflow(self):
        """Обновляет индикатор шагов и контекст Оракула."""
        if hasattr(self, "workflow_widget"):
            try:
                self.workflow_widget.refresh()
            except Exception as e:
                logger.warning(f"Ошибка обновления рабочего процесса: {e}")
        if hasattr(self, "oraculum_agent"):
            try:
                self.shared_state["workflow"] = self.workflow_manager
                self.shared_state["user"] = self.session_manager.get_user_summary()
            except Exception:
                pass

    # ====================================================================
    # ОБРАБОТЧИКИ СОБЫТИЙ
    # ====================================================================
    def _on_scenario_selected(self, scenario: str):
        """Вызывается из ProjectPanel при выборе сценария."""
        self._setup_scenario(scenario)

    def _on_project_created(self, project_info: dict):
        """Вызывается из ProjectPanel при создании проекта."""
        self.shared_state["project"].update(project_info)
        name = project_info.get("model_name", "")
        if name:
            self._on_model_name_changed(name)
        self._refresh_workflow()

    def _on_model_created(self, model, config):
        """Вызывается из ArchitecturePanel при создании модели."""
        self.shared_state["model"] = model
        self.shared_state["config"] = config

        # Извлекаем имя модели из config (ModelConfig или dict)
        if hasattr(config, "model_name"):
            name = config.model_name
        elif isinstance(config, dict):
            name = config.get("model_name", "Модель")
        else:
            name = "Модель"

        self._on_model_name_changed(name)
        self._refresh_workflow()

    def _on_model_name_changed(self, name: str):
        """
        Имя модели (из project_panel или architecture_panel) → Мониторинг.
        При создании/загрузке модели вызывается этот метод.
        """
        self.shared_state["project"]["model_name"] = name
        if hasattr(self.monitoring_panel, "set_model_name"):
            self.monitoring_panel.set_model_name(name)
        logger.info(f"Имя модели обновлено: {name}")
        self._refresh_workflow()

    def _on_hint_mode_changed(self, index: int):
        """Переключает режим подсказок."""
        mode = self.HINT_MODE_MAP.get(index, "hybrid")
        self.shared_state["hint_mode"] = mode
        if hasattr(self.session_manager, "state"):
            self.session_manager.state.hint_mode = mode
        logger.info(f"Режим подсказок изменён на: {mode}")

        # Локальная GGUF-модель: «ИИ» → ленивая загрузка, «Без подсказок» → выгрузка
        if mode == "ai":
            self._load_local_model_async()
        elif mode in ("static", "none"):
            self._unload_local_model()

    def _load_local_model_async(self):
        """Загружает локальную GGUF-модель в фоновом потоке (без заморозки UI)."""
        if self.oraculum_agent.is_local_model_loaded():
            return
        import threading

        def _work():
            ok = self.oraculum_agent.ensure_local_model()
            msg = "локальная модель загружена" if ok else (
                self.oraculum_agent.local_model_error()
                or "локальная модель недоступна"
            )
            self.local_model_status.emit(ok, msg)

        threading.Thread(target=_work, daemon=True).start()

    def _on_agent_mode_toggled(self, checked: bool):
        """Включение агентского режима → подгружаем локальную модель в фоне."""
        if checked:
            self._load_local_model_async()

    def _unload_local_model(self):
        self.oraculum_agent.unload_local_model()
        self.local_model_status.emit(False, "локальная модель выгружена")

    def _on_local_model_status(self, ok: bool, msg: str):
        """Принимает статус локальной модели и обновляет чат."""
        if hasattr(self, "oraculum_chat") and self.oraculum_chat is not None:
            self.oraculum_chat.set_local_model_status(ok, msg)
            try:
                self.oraculum_chat.refresh_provider_status()
            except Exception:
                pass

    # ====================================================================
    # АГЕНТСКИЕ ВОЗМОЖНОСТИ ОРАКУЛА (tools)
    # ====================================================================
    def _register_agent_tools(self):
        """Регистрирует GUI-инструменты (метаданные). Обработчики — в _agent_handlers."""
        reg = self.oraculum_agent.register_tool
        reg("switch_tab",
            "Переключить вкладку программы. tab — название вкладки "
            "(Проект, Генератор, Данные, Архитектура, Гиперпараметры, "
            "Обучение, Мониторинг, Анализ, Экспорт, Песочница, Логи).",
            {"tab": "название вкладки"})
        reg("set_scenario",
            "Выбрать сценарий работы (new/finetune/play).",
            {"scenario": "new | finetune | play"})
        reg("create_model",
            "Создать модель по текущим настройкам архитектуры (без подтверждений).")
        reg("split_dataset",
            "Применить разбиение данных на train/val/test.")
        reg("start_training",
            "Запустить обучение модели (без модальных диалогов).")
        reg("stop_training",
            "Остановить текущее обучение.")
        reg("analyze",
            "Запустить анализ результатов обучения.")
        reg("save_model",
            "Сохранить текущую модель в файл (.pth). name — имя файла (без расширения).",
            {"name": "имя файла"})
        # --- Инструменты режима «Мастер» ---
        reg("set_project",
            "Создать проект (параметры: name — имя проекта, model_name — имя модели).",
            {"name": "имя проекта", "model_name": "имя модели"})
        reg("generate_dataset",
            "Сгенерировать датасет (task: addition/subtraction/multiplication/division/"
            "mixed_math/linear_equation/quadratic/cipher_caesar/cipher_atbash; "
            "num_samples — количество примеров; min/max — диапазон чисел, "
            "например сложение 0–10: min=0, max=10).",
            {"task": "тип задачи", "num_samples": "количество примеров",
             "min": "нижняя граница", "max": "верхняя граница"})
        reg("set_architecture",
            "Создать модель заданной архитектуры (arch: transformer_seq2seq, mlp и др.; "
            "arch_params — параметры).",
            {"arch": "архитектура", "arch_params": "словарь параметров"})
        reg("set_hyperparams",
            "Задать гиперпараметры обучения (epochs, lr, batch_size).",
            {"epochs": "число", "lr": "число", "batch_size": "число"})
        reg("get_state",
            "Получить текущее состояние программы (JSON): данные, разбиение, модель, обучение.")

    def _populate_agent_handlers(self):
        """Привязывает обработчики GUI-инструментов к панелям (после их создания)."""
        self._agent_handlers = {
            "switch_tab": self._action_switch_tab,
            "set_scenario": self._action_set_scenario,
            "create_model": self._action_create_model,
            "split_dataset": self._action_split_dataset,
            "start_training": self._action_start_training,
            "stop_training": self._action_stop_training,
            "analyze": self._action_analyze,
            "save_model": self._action_save_model,
            "set_project": self._action_set_project,
            "generate_dataset": self._action_generate_dataset,
            "set_architecture": self._action_set_architecture,
            "set_hyperparams": self._action_set_hyperparams,
            "get_state": self._action_get_state,
        }

    def _execute_agent_action(self, name: str, args: dict) -> str:
        """Выполняет GUI-инструмент Оракула (в главном потоке)."""
        handler = self._agent_handlers.get(name)
        if handler is None:
            return f'{{"error": "неизвестный инструмент: {name}"}}'
        try:
            result = handler(args or {})
            if isinstance(result, str):
                return result
            return str(result)
        except Exception as e:
            logger.error(f"Ошибка агентского инструмента {name}: {e}")
            return f'{{"error": "{type(e).__name__}: {e}"}}'

    def _action_switch_tab(self, args: dict) -> str:
        import json as _json
        wanted = str(args.get("tab", "")).strip()
        tabs = [self.tabs.tabText(i) for i in range(self.tabs.count())]
        for i, t in enumerate(tabs):
            if wanted.lower() in t.lower() or t.lower().startswith(wanted.lower()):
                self.tabs.setCurrentIndex(i)
                return _json.dumps({"done": True, "tab": t}, ensure_ascii=False)
        return _json.dumps({"error": f"вкладка не найдена. Доступны: {tabs}"},
                           ensure_ascii=False)

    def _action_set_scenario(self, args: dict) -> str:
        import json as _json
        scenario = str(args.get("scenario", "")).strip().lower()
        if scenario not in ("new", "finetune", "play"):
            return _json.dumps({"error": "сценарий должен быть new/finetune/play"},
                               ensure_ascii=False)
        self.project_panel.set_scenario(scenario)
        return _json.dumps({"done": True, "scenario": scenario}, ensure_ascii=False)

    def _action_create_model(self, args: dict) -> str:
        import json as _json
        panel = self.arch_panel
        if not hasattr(panel, "create_btn") or not panel.create_btn.isEnabled():
            return _json.dumps({"error": "модель нельзя создать сейчас: загрузите "
                                          "и разбейте данные"}, ensure_ascii=False)
        # Экспериментальные архитектуры — не запускаем без подтверждения
        exp = getattr(panel, "EXPERIMENTAL_ARCHS", set())
        if getattr(panel, "_current_arch", None) in exp:
            return _json.dumps({"error": "экспериментальная архитектура — "
                                          "создайте модель вручную"}, ensure_ascii=False)
        # Валидация без диалога
        err = panel._validate_before_create()
        if err:
            return _json.dumps({"error": err}, ensure_ascii=False)
        panel.create_model()
        report = ""
        if hasattr(panel, "report_text"):
            report = panel.report_text.toPlainText()
        return _json.dumps({"done": True, "report": report[:500]}, ensure_ascii=False)

    def _action_split_dataset(self, args: dict) -> str:
        import json as _json
        if self.shared_state.get("dataset") is None:
            return _json.dumps({"error": "данные не загружены"}, ensure_ascii=False)
        self.dataset_panel.split_dataset()
        return _json.dumps({"done": True, "message": "разбиение применено"},
                           ensure_ascii=False)

    def _action_start_training(self, args: dict) -> str:
        import json as _json
        result = self.training_panel.start_training(silent=True)
        if isinstance(result, str) and result:
            return _json.dumps({"error": result}, ensure_ascii=False)
        return _json.dumps({"done": True, "message": "обучение запущено"},
                           ensure_ascii=False)

    def _action_stop_training(self, args: dict) -> str:
        import json as _json
        self.training_panel.stop_training(silent=True)
        return _json.dumps({"done": True, "message": "запрошена остановка обучения"},
                           ensure_ascii=False)

    def _action_analyze(self, args: dict) -> str:
        import json as _json
        history = self.shared_state.get("history")
        if history is None:
            return _json.dumps({"error": "нет истории обучения"}, ensure_ascii=False)
        try:
            self.analysis_panel.run_analysis(history)
            return _json.dumps({"done": True, "message": "анализ запущен"},
                               ensure_ascii=False)
        except Exception as e:
            return _json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)

    def _action_save_model(self, args: dict) -> str:
        """Сохраняет модель в .pth без диалога (для ИИ-агента)."""
        if not hasattr(self, "export_panel") or self.export_panel is None:
            return '{"error": "панель экспорта недоступна"}'
        return self.export_panel.save_model_silent(str(args.get("name", "")))

    # ====================================================================
    # ИНСТРУМЕНТЫ РЕЖИМА «МАСТЕР»
    # ====================================================================
    def _action_set_project(self, args: dict) -> str:
        return self.project_panel.create_project_silent(
            str(args.get("name", "")),
            str(args.get("model_name", "")),
            str(args.get("description", "")),
        )

    # Синонимы задач, которыми модель Яндекса называет генерацию данных
    # («caesar_cipher» вместо «cipher_caesar» и т.п.).
    _DATASET_TASK_ALIASES = {
        "caesar_cipher": "cipher_caesar",
        "cezar": "cipher_caesar",
        "цезарь": "cipher_caesar",
        "atbash": "cipher_atbash",
        "сложение": "addition",
        "сумма": "addition",
        "вычитание": "subtraction",
        "умножение": "multiplication",
        "деление": "division",
        "смешанная": "mixed_math",
        "уравнение": "linear_equation",
        "квадратное": "quadratic",
    }

    @classmethod
    def _normalize_dataset_task(cls, task: str) -> str:
        """Приводит имя задачи от модели к каноническому виду генератора."""
        t = (task or "").strip().lower()
        return cls._DATASET_TASK_ALIASES.get(t, t)

    def _action_generate_dataset(self, args: dict) -> str:
        import json as _json, time as _time
        from pathlib import Path
        task = self._normalize_dataset_task(str(args.get("task", "addition")))
        num = int(args.get("num_samples", 5000))
        try:
            subdirs = self.session_manager.get_user_subdirs()
            save_dir = subdirs.get("datasets", Path("."))
            save_path = save_dir / f"wizard_{task}_{int(_time.time())}.oai"
            num_range = None
            if "min" in args or "max" in args:
                lo = int(args.get("min", 0))
                hi = int(args.get("max", 10))
                if lo > hi:
                    lo, hi = hi, lo
                num_range = (lo, hi)
            msg = self.generator_panel.generate_dataset_silent(
                task, num, save_path, num_range=num_range
            )
            if "создан" not in msg:
                return _json.dumps({"error": msg}, ensure_ascii=False)
            load_msg = self.dataset_panel.load_dataset_silent(save_path)
            return _json.dumps({"done": True, "message": f"{msg}; {load_msg}"},
                               ensure_ascii=False)
        except Exception as e:
            return _json.dumps({"error": f"{type(e).__name__}: {e}"},
                               ensure_ascii=False)

    def _action_set_architecture(self, args: dict) -> str:
        import json as _json
        arch = str(args.get("arch", "transformer_seq2seq")).strip()
        # Модель может назвать архитектуру коротко («transformer») —
        # подставляем полное имя seq2seq-трансформера.
        if arch.lower() in ("transformer", "transformers"):
            arch = "transformer_seq2seq"
        arch_params = args.get("arch_params", {}) or {}
        result = self.arch_panel.create_model_preset(arch, arch_params)
        if "✅" in result:
            return _json.dumps({"done": True, "report": result}, ensure_ascii=False)
        return _json.dumps({"error": result}, ensure_ascii=False)

    def _action_set_hyperparams(self, args: dict) -> str:
        import json as _json
        msg = self.params_panel.set_hyperparams_silent(
            epochs=args.get("epochs"),
            lr=args.get("lr"),
            batch_size=args.get("batch_size"),
        )
        return _json.dumps({"done": True, "message": msg}, ensure_ascii=False)

    def _action_get_state(self, args: dict) -> str:
        import json as _json
        context = self.oraculum_agent.context_builder.build()
        return _json.dumps({
            "current_tab": context.current_tab,
            "scenario": context.scenario,
            "dataset_exists": context.dataset_exists,
            "has_split": context.has_split,
            "model_exists": context.model_exists,
            "model_trained": context.model_trained,
            "training_status": self.oraculum_agent._training_state.get("status", "idle"),
        }, ensure_ascii=False)

    # ====================================================================
    # РЕЖИМ «МАСТЕР»: КНОПКА И СВЯЗЬ С ЧАТОМ
    # ====================================================================
    def _open_wizard(self):
        """Переключает на вкладку «Мастер»."""
        for i in range(self.tabs.count()):
            if "Мастер" in self.tabs.tabText(i):
                self.tabs.setCurrentIndex(i)
                return

    def _on_wizard_narration(self, text: str):
        """Комментарии Оракула из режима «Мастер» — в чат."""
        if getattr(self, "oraculum_chat", None) is not None:
            self.oraculum_chat.add_message("Оракул", text)

    def _on_tab_changed(self, index: int):
        """Логирует переключение вкладок и обновляет контекст."""
        if index < 0:
            return
        tab_name = self.tabs.tabText(index)
        self.session_manager.log_action("main_window", "tab_changed", tab_name)
        try:
            self.session_manager.log_tab_change(tab_name)
        except Exception:
            pass

        # Обновляем рабочий процесс при переключении вкладок
        self._refresh_workflow()

        # Обновляем панели при переключении
        widget = self.tabs.widget(index)
        if widget is self.arch_panel and hasattr(self.arch_panel, "refresh"):
            try:
                self.arch_panel.refresh()
            except Exception as e:
                logger.warning(f"Ошибка обновления arch_panel: {e}")
        elif widget is self.params_panel and hasattr(self.params_panel, "refresh"):
            try:
                self.params_panel.refresh()
            except Exception as e:
                logger.warning(f"Ошибка обновления params_panel: {e}")
        elif widget is self.sandbox_panel and hasattr(self.sandbox_panel, "refresh"):
            try:
                self.sandbox_panel.refresh()
            except Exception as e:
                logger.warning(f"Ошибка обновления sandbox_panel: {e}")
        elif widget is self.admin_panel and hasattr(self.admin_panel, "refresh"):
            try:
                self.admin_panel.refresh()
            except Exception as e:
                logger.warning(f"Ошибка обновления admin_panel: {e}")

    def _on_retry_requested(self, new_config):
        """Пользователь хочет повторить обучение с новыми параметрами."""
        # analysis_panel передаёт dict — приводим к ModelConfig для консистентности
        if not isinstance(new_config, ModelConfig):
            try:
                new_config = ModelConfig.from_dict(new_config or {})
            except Exception as e:
                logger.warning(f"Не удалось нормализовать конфиг повтора: {e}")
        self.shared_state["config"] = new_config
        # Переключаемся на вкладку обучения
        for i in range(self.tabs.count()):
            if self.tabs.widget(i) is self.training_panel:
                self.tabs.setCurrentIndex(i)
                break

    def _on_export_requested(self):
        """Пользователь запросил экспорт."""
        for i in range(self.tabs.count()):
            if self.tabs.widget(i) is self.export_panel:
                self.tabs.setCurrentIndex(i)
                break

    # ====================================================================
    # ЗАКРЫТИЕ ОКНА
    # ====================================================================
    def closeEvent(self, event):
        """Сохраняет состояние сеанса при закрытии."""
        try:
            # Обновляем счётчики профиля пользователя
            profile = self.user_manager.get_profile(self.current_username)
            if profile is not None:
                counts = self.user_manager.count_user_objects(self.current_username)
                profile.total_projects = counts["projects"]
                profile.total_models = counts["models"]
                profile.total_datasets = counts["datasets"]
                profile.storage_usage_mb = self.user_manager.get_storage_usage(
                    self.current_username
                )
                self.user_manager.update_profile(
                    self.current_username,
                    total_projects=profile.total_projects,
                    total_models=profile.total_models,
                    total_datasets=profile.total_datasets,
                    storage_usage_mb=profile.storage_usage_mb,
                )
            self.session_manager.save()
            # Закрываем чат Оракула
            if hasattr(self, "oraculum_chat"):
                self.oraculum_chat.close()
            logger.info("Сессия сохранена при выходе.")
        except Exception as e:
            logger.error(f"Не удалось сохранить сессию: {e}")
        super().closeEvent(event)