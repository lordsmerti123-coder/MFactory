"""
gui/panels/monitoring_panel.py
==============================
Панель визуализации процесса обучения в реальном времени.

Зависимости по контрактам:
  - core.contracts: TrainingHistory, SessionState, ModelConfig
  - core.explainer: Explainer
  - core.curator: Curator
  - core.hint_engine: HintEngine, HintContext
  - gui.hint_widget: HintWidget
  - core.logger: get_logger
  - pyqtgraph (графики)
  - PyQt5

Сигналы (выходы):
  - stop_requested: pyqtSignal()

Входы (слоты, вызываются из main_window):
  - update_plots(epoch: int, logs: dict)
  - reset()
  - set_mode(is_text: bool)
  - set_model_name(name: str)
  - show_sample_predictions(predictions: List[Dict])
  - set_curator_comment(comment: str)

Контракт с training_panel (через main_window):
  training_panel.epoch_finished  → self.update_plots
  training_panel.training_started → self._on_training_started

Контракт с trainer (через callback):
  Trainer вызывает cb.on_epoch_end → training_panel → сюда
  Каждые 5 эпох: trainer генерирует sample_predictions → сюда

Контракт с curator:
  Curator.analyze_realtime(history) → str (комментарий)
  Curator.comment_prediction(inp, exp, pred) → str
  Curator.get_random_fact() → str
  Curator.generate_model_name() → str (не используется здесь, но в карточке)

Режимы подсказок (из shared_state["hint_mode"]):
  "static" | "ai" | "hybrid" | "none"
"""

import random
import time
from typing import Dict, List, Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QGroupBox, QPushButton, QTextEdit,
    QSplitter, QFrame, QSizePolicy, QScrollArea
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor

import pyqtgraph as pg

from core.logger import get_logger
from core.explainer import Explainer

# Безопасные импорты новых модулей (могут быть ещё не готовы при сборке)
try:
    from core.contracts import (
        TrainingHistory, SessionState, ModelConfig, config_to_dict,
    )
except ImportError:
    TrainingHistory = None
    SessionState = None
    ModelConfig = None

    def config_to_dict(config):
        if isinstance(config, dict):
            return config
        if hasattr(config, "to_dict"):
            return config.to_dict()
        return {}

try:
    from core.curator import Curator
except ImportError:
    Curator = None

try:
    from core.hint_engine import HintEngine, HintContext
except ImportError:
    HintEngine = None
    HintContext = None

try:
    from gui.hint_widget import HintWidget
except ImportError:
    HintWidget = None

logger = get_logger(__name__)

# ============================================================
#  КОНСТАНТЫ ПАНЕЛИ
# ============================================================
PREDICTION_INTERVAL = 5       # Показывать живые примеры каждые N эпох
FACT_INTERVAL = 15            # Показывать факт каждые N эпох (если обучение долгое)
MAX_PREDICTIONS_SHOWN = 3     # Максимум примеров в блоке за раз
WATCHER_COLOR_DANGER = "#e06c75"
WATCHER_COLOR_WARNING = "#e5c07b"
WATCHER_COLOR_OK = "#a6e3a1"
WATCHER_COLOR_INFO = "#4a88c7"
WATCHER_COLOR_DEFAULT = "#a9b7c6"


class MonitoringPanel(QWidget):
    """
    Панель мониторинга обучения.

    Отображает:
    1. Графики Loss и Metric (pyqtgraph)
    2. ИИ-Смотритель (комментарии куратора/эксплейнера)
    3. Живые примеры предсказаний (каждые 5 эпох)
    4. Интересные факты (при долгом обучении)
    5. Карточку модели (имя, архитектура, параметры)
    6. Кнопку экстренной остановки

    Все элементы имеют подсказки через HintWidget.
    """

    # === СИГНАЛЫ ===
    stop_requested = pyqtSignal()

    def __init__(self, shared_state: dict, parent=None):
        super().__init__(parent)
        self.shared_state = shared_state

        # Внутреннее состояние
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_metric": [],
            "val_metric": [],
        }
        self.is_text_mode: bool = False
        self.is_training: bool = False
        self.model_name: str = "Модель"
        self._epochs_since_last_fact: int = 0
        self._training_start_time: float = 0.0

        # Куратор (может быть None если модуль ещё не готов)
        self.curator = None
        if Curator is not None:
            try:
                session = shared_state.get("session")
                self.curator = Curator(session)
            except Exception as e:
                logger.warning(f"Curator не инициализирован: {e}")
                self.curator = None

        self.init_ui()
        self._install_hints()

    # ============================================================
    #  UI
    # ============================================================
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # === ВЕРХНЯЯ СТРОКА: Карточка модели + Кнопка стоп ===
        top_bar = QHBoxLayout()
        top_bar.setSpacing(10)

        # Карточка модели
        self.model_card_label = QLabel("🧠 Модель: — | Архитектура: — | Параметры: —")
        self.model_card_label.setWordWrap(True)
        self.model_card_label.setStyleSheet(
            "font-size: 13px; color: #a9b7c6; padding: 4px 8px; "
            "background-color: #313335; border: 1px solid #555; border-radius: 3px;"
        )
        top_bar.addWidget(self.model_card_label, stretch=1)

        # Кнопка экстренной остановки
        self.btn_emergency_stop = QPushButton("🛑 ЭКСТРЕННЫЙ СТОП")
        self.btn_emergency_stop.setObjectName("StopBtn")
        self.btn_emergency_stop.setFixedHeight(36)
        self.btn_emergency_stop.setStyleSheet(
            "QPushButton { background-color: #6a3636; border: 2px solid #8f4646; "
            "color: white; font-weight: bold; font-size: 14px; padding: 4px 16px; }"
            "QPushButton:hover { background-color: #8f4646; }"
            "QPushButton:disabled { background-color: #313335; color: #777; }"
        )
        self.btn_emergency_stop.setEnabled(False)
        self.btn_emergency_stop.clicked.connect(self._on_stop_clicked)
        top_bar.addWidget(self.btn_emergency_stop)

        layout.addLayout(top_bar)

        # === ОСНОВНАЯ ОБЛАСТЬ: Splitter (графики слева, панели справа) ===
        splitter = QSplitter(Qt.Horizontal)

        # --- ЛЕВАЯ ЧАСТЬ: Графики ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        pg.setConfigOptions(antialias=True)
        pg.setConfigOption("background", "#2b2b2b")
        pg.setConfigOption("foreground", "#a9b7c6")

        # График Loss
        self.plot_loss = pg.PlotWidget(title="📉 Ошибка (Loss) — чем ниже, тем лучше")
        self.plot_loss.setLabel("left", "Значение ошибки")
        self.plot_loss.setLabel("bottom", "Эпоха обучения")
        self.plot_loss.addLegend(offset=(10, 10))
        self.plot_loss.showGrid(x=True, y=True, alpha=0.3)
        self.plot_loss.setMinimumHeight(180)
        self.curve_train_loss = self.plot_loss.plot(
            pen=pg.mkPen("#4a88c7", width=2), name="Train (Учеба)"
        )
        self.curve_val_loss = self.plot_loss.plot(
            pen=pg.mkPen("#cc7832", width=2), name="Val (Экзамен)"
        )

        # График Metric
        self.plot_metric = pg.PlotWidget(title="🎯 Качество (Метрика)")
        self.plot_metric.setLabel("left", "Значение метрики")
        self.plot_metric.setLabel("bottom", "Эпоха обучения")
        self.plot_metric.addLegend(offset=(10, 10))
        self.plot_metric.showGrid(x=True, y=True, alpha=0.3)
        self.plot_metric.setMinimumHeight(180)
        self.curve_train_metric = self.plot_metric.plot(
            pen=pg.mkPen("#385a3a", width=2), name="Train Metric"
        )
        self.curve_val_metric = self.plot_metric.plot(
            pen=pg.mkPen("#8f4646", width=2), name="Val Metric"
        )

        left_layout.addWidget(self.plot_loss)
        left_layout.addWidget(self.plot_metric)

        # Режим
        self.mode_label = QLabel("")
        self.mode_label.setStyleSheet("color: #777; font-size: 11px;")
        self.mode_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.mode_label)

        splitter.addWidget(left_widget)

        # --- ПРАВАЯ ЧАСТЬ: Смотритель + Примеры + Факты ---
        right_widget = QWidget()
        right_widget.setMinimumWidth(320)
        right_widget.setMaximumWidth(420)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        # ИИ-Смотритель
        watcher_group = QGroupBox("🤖 ИИ-Смотритель (анализ в реальном времени)")
        watcher_layout = QVBoxLayout()
        self.watcher_label = QLabel("👀 Жду запуска обучения...")
        self.watcher_label.setWordWrap(True)
        self.watcher_label.setTextFormat(Qt.RichText)
        self.watcher_label.setStyleSheet(
            f"font-size: 14px; color: {WATCHER_COLOR_DEFAULT}; "
            "font-weight: bold; padding: 5px;"
        )
        self.watcher_label.setMinimumHeight(60)
        watcher_layout.addWidget(self.watcher_label)
        watcher_group.setLayout(watcher_layout)
        right_layout.addWidget(watcher_group)

        # Живые примеры
        predictions_group = QGroupBox("🧪 Живые примеры (каждые 5 эпох)")
        predictions_layout = QVBoxLayout()
        self.predictions_text = QTextEdit()
        self.predictions_text.setReadOnly(True)
        self.predictions_text.setMaximumHeight(180)
        self.predictions_text.setStyleSheet(
            "font-family: 'Consolas', monospace; font-size: 12px; "
            "background-color: #1e1e1e; color: #a9b7c6; border: 1px solid #555;"
        )
        self.predictions_text.setPlaceholderText(
            "Примеры предсказаний появятся здесь после 5-й эпохи..."
        )
        predictions_layout.addWidget(self.predictions_text)
        predictions_group.setLayout(predictions_layout)
        right_layout.addWidget(predictions_group)

        # Интересный факт
        fact_group = QGroupBox("💡 Интересный факт")
        fact_layout = QVBoxLayout()
        self.fact_label = QLabel(
            "Начните обучение — и я расскажу что-нибудь интересное о нейросетях!"
        )
        self.fact_label.setWordWrap(True)
        self.fact_label.setStyleSheet(
            "font-size: 12px; color: #4a88c7; font-style: italic; padding: 4px;"
        )
        fact_layout.addWidget(self.fact_label)
        fact_group.setLayout(fact_layout)
        right_layout.addWidget(fact_group)

        # Прогресс-инфо
        self.progress_info_label = QLabel("")
        self.progress_info_label.setStyleSheet(
            "color: #777; font-size: 11px; padding: 2px;"
        )
        self.progress_info_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.progress_info_label)

        right_layout.addStretch()
        splitter.addWidget(right_widget)

        # Пропорции сплиттера
        splitter.setSizes([600, 350])
        layout.addWidget(splitter, stretch=1)

    # ============================================================
    #  ПОДСКАЗКИ (установка на каждый элемент)
    # ============================================================
    def _install_hints(self):
        """Устанавливает HintWidget на все интерактивные элементы."""
        if HintWidget is None:
            return
        hint_engine = self.shared_state.get("hint_engine")
        if hint_engine is None:
            return

        HintWidget(
            self.plot_loss, hint_engine,
            widget_id="monitoring.loss_plot",
            panel_name="monitoring_panel"
        )
        HintWidget(
            self.plot_metric, hint_engine,
            widget_id="monitoring.metric_plot",
            panel_name="monitoring_panel"
        )
        HintWidget(
            self.btn_emergency_stop, hint_engine,
            widget_id="monitoring.stop_btn",
            panel_name="monitoring_panel"
        )
        HintWidget(
            self.watcher_label, hint_engine,
            widget_id="monitoring.watcher_area",
            panel_name="monitoring_panel"
        )
        HintWidget(
            self.predictions_text, hint_engine,
            widget_id="monitoring.predictions",
            panel_name="monitoring_panel"
        )
        HintWidget(
            self.fact_label, hint_engine,
            widget_id="monitoring.fact",
            panel_name="monitoring_panel"
        )

    # ============================================================
    #  ПУБЛИЧНЫЕ СЛОТЫ (вызываются из main_window / training_panel)
    # ============================================================
    def set_mode(self, is_text: bool):
        """Устанавливает режим отображения метрик (текст / числа)."""
        self.is_text_mode = is_text
        if is_text:
            self.plot_metric.setTitle("🎯 Точность токенов (выше = лучше)")
            self.plot_metric.setLabel("left", "Точность (0.0 – 1.0)")
            self.mode_label.setText(
                "📝 Текстовый режим: метрика = доля правильно предсказанных символов"
            )
        else:
            self.plot_metric.setTitle("🎯 Ошибка (метрика) — чем ниже, тем лучше")
            self.plot_metric.setLabel("left", "Значение ошибки")
            self.mode_label.setText(
                "🔢 Числовой режим: метрика = значение функции потерь"
            )
        # Обновляем легенду
        self.plot_metric.addLegend(offset=(10, 10))

    def set_model_name(self, name: str):
        """Устанавливает имя модели для карточки."""
        self.model_name = name
        self._update_model_card()

    def reset(self):
        """Полный сброс при начале нового обучения."""
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_metric": [], "val_metric": [],
        }
        self.is_training = True
        self._epochs_since_last_fact = 0
        self._training_start_time = time.time()

        # Очищаем графики
        self.curve_train_loss.setData([], [])
        self.curve_val_loss.setData([], [])
        self.curve_train_metric.setData([], [])
        self.curve_val_metric.setData([], [])

        # Сбрасываем тексты
        self.watcher_label.setText("👀 Наблюдаю за первыми шагами нейросети...")
        self.watcher_label.setStyleSheet(
            f"font-size: 14px; color: {WATCHER_COLOR_WARNING}; "
            "font-weight: bold; padding: 5px;"
        )
        self.predictions_text.clear()
        self.fact_label.setText("Начните обучение — и я расскажу что-нибудь интересное!")
        self.progress_info_label.setText("")

        # Активируем кнопку стоп
        self.btn_emergency_stop.setEnabled(True)

        # Обновляем карточку
        self._update_model_card()

        # Логируем в сеанс
        self._log_session_action("training_started")

        logger.info("Мониторинг: графики сброшены, обучение начинается.")

    def update_plots(self, epoch: int, logs: dict):
        """
        Вызывается каждую эпоху из training_panel через сигнал.

        Параметры:
            epoch: номер завершённой эпохи
            logs: {"train_loss": float, "val_loss": float,
                   "train_metric": float, "val_metric": float}
        """
        # Сохраняем в историю
        self.history["train_loss"].append(logs.get("train_loss", 0.0))
        self.history["val_loss"].append(logs.get("val_loss", 0.0))
        self.history["train_metric"].append(logs.get("train_metric", 0.0))
        self.history["val_metric"].append(logs.get("val_metric", 0.0))

        # Обновляем графики
        epochs = list(range(1, len(self.history["train_loss"]) + 1))
        self.curve_train_loss.setData(epochs, self.history["train_loss"])
        self.curve_val_loss.setData(epochs, self.history["val_loss"])
        self.curve_train_metric.setData(epochs, self.history["train_metric"])
        self.curve_val_metric.setData(epochs, self.history["val_metric"])

        # Анализ от Explainer
        advice = Explainer.analyze_realtime(self.history)
        text = f"{advice.icon} {advice.message}"
        self.watcher_label.setText(text)
        self._set_watcher_color(text)

        # Обновляем прогресс-инфо
        elapsed = time.time() - self._training_start_time
        self.progress_info_label.setText(
            f"Эпоха {epoch} | Прошло: {self._format_time(elapsed)}"
        )

        # Каждые N эпох — показываем факт
        self._epochs_since_last_fact += 1
        if self._epochs_since_last_fact >= FACT_INTERVAL:
            self._show_random_fact()
            self._epochs_since_last_fact = 0

        # Логируем
        self._log_session_action(f"epoch_{epoch}_completed")

    def show_sample_predictions(self, predictions: List[Dict]):
        """
        Показывает живые примеры предсказаний.

        Формат каждого элемента:
        {
            "input": str,
            "expected": str,
            "predicted": str,
            "correct": bool
        }
        """
        if not predictions:
            return

        html_parts = []
        for i, pred in enumerate(predictions[:MAX_PREDICTIONS_SHOWN]):
            inp = pred.get("input", "?")
            exp = pred.get("expected", "?")
            got = pred.get("predicted", "?")
            correct = pred.get("correct", False)

            if correct:
                icon = "✅"
                color = WATCHER_COLOR_OK
            else:
                icon = "❌"
                color = WATCHER_COLOR_DANGER

            html_parts.append(
                f'<div style="margin-bottom:6px; padding:4px; '
                f'border-left:3px solid {color}; background:#2a2a2a;">'
                f'{icon} <b>Вход:</b> <code>{inp}</code> | '
                f'<b>Ожидалось:</b> <code>{exp}</code> | '
                f'<b>Модель:</b> <code>{got}</code>'
                f'</div>'
            )

        # Комментарий куратора к предсказаниям
        if self.curator is not None:
            try:
                # Берём первый неверный пример для комментария
                wrong = next((p for p in predictions if not p.get("correct")), None)
                if wrong:
                    comment = self.curator.comment_prediction(
                        wrong.get("input", ""),
                        wrong.get("expected", ""),
                        wrong.get("predicted", "")
                    )
                    html_parts.append(
                        f'<div style="margin-top:8px; color:{WATCHER_COLOR_WARNING}; '
                        f'font-style:italic;">💬 {comment}</div>'
                    )
            except Exception:
                pass

        self.predictions_text.setHtml("".join(html_parts))

    def set_curator_comment(self, comment: str):
        """Прямая установка комментария куратора (для внешнего вызова)."""
        self.watcher_label.setText(comment)
        self._set_watcher_color(comment)

    def on_training_finished(self):
        """Вызывается когда обучение полностью завершено."""
        self.is_training = False
        self.btn_emergency_stop.setEnabled(False)
        elapsed = time.time() - self._training_start_time
        self.progress_info_label.setText(
            f"✅ Обучение завершено | Общее время: {self._format_time(elapsed)}"
        )
        self._log_session_action("training_finished")

    # ============================================================
    #  ВНУТРЕННИЕ МЕТОДЫ
    # ============================================================
    def _on_stop_clicked(self):
        """Обработка нажатия экстренной остановки."""
        self.btn_emergency_stop.setEnabled(False)
        self.btn_emergency_stop.setText("⏹ Останавливаю...")
        self.stop_requested.emit()
        self.watcher_label.setText(
            "⏹ Пользователь запросил остановку. Ждём завершения текущей эпохи..."
        )
        self.watcher_label.setStyleSheet(
            f"font-size: 14px; color: {WATCHER_COLOR_WARNING}; "
            "font-weight: bold; padding: 5px;"
        )
        logger.info("Мониторинг: пользователь нажал экстренный стоп.")
        self._log_session_action("stop_requested")

    def _set_watcher_color(self, advice: str):
        """Меняет цвет текста Смотрителя в зависимости от ситуации."""
        if any(kw in advice for kw in ["Авария", "🚨", "взрыв", "NaN", "Infinity"]):
            color = WATCHER_COLOR_DANGER
        elif any(kw in advice for kw in ["Внимание", "⚠️", "зубрит", "переобуч", "пересек"]):
            color = WATCHER_COLOR_WARNING
        elif any(kw in advice for kw in ["нормальный", "✅", "Отлично", "Блестящий"]):
            color = WATCHER_COLOR_OK
        elif any(kw in advice for kw in ["👀", "Ждем", "начала"]):
            color = WATCHER_COLOR_INFO
        else:
            color = WATCHER_COLOR_DEFAULT

        self.watcher_label.setStyleSheet(
            f"font-size: 14px; color: {color}; font-weight: bold; padding: 5px;"
        )

    def _show_random_fact(self):
        """Показывает интересный факт."""
        if self.curator is not None:
            try:
                fact = self.curator.get_random_fact()
                self.fact_label.setText(fact)
                return
            except Exception:
                pass

        # Fallback: локальный список фактов
        fallback_facts = [
            "Знаете ли вы? Первый перцептрон Розенблатт (1957) умел только различать фигуры.",
            "Статья 'Attention Is All You Need' (2017) породила все современные LLM.",
            "Один трансформер на 175B параметров при обучении потребляет энергию как 5 автомобилей.",
            "Слово 'нейрон' в контексте ИИ впервые использовал Маккалок и Питтс в 1943 году.",
            "Dropout был вдохновлён идеей: 'не дай одному нейрону стать слишком уверенным'.",
            "Автоматическое дифференцирование (autograd) — сердце обучения нейросетей.",
            "В 2012 году AlexNet выиграл ImageNet с ошибкой 15.3%. Сейчас лучшие модели — менее 1%.",
            "Функция активации ReLU была предложена ещё в 1969 году, но стала популярной только в 2010-х.",
            "Метод обратного распространения ошибки был описан в 1974 году, но применён к сетям в 1986.",
            "Слово 'эпоха' в обучении означает один полный проход по всем данным.",
        ]
        self.fact_label.setText(random.choice(fallback_facts))

    def _update_model_card(self):
        """Обновляет карточку модели вверху панели."""
        config = config_to_dict(self.shared_state.get("config"))
        model = self.shared_state.get("model")

        if model is None:
            self.model_card_label.setText("🧠 Модель: — | Ожидание создания...")
            return

        arch_type = config.get("type", "unknown").upper()
        params_count = sum(p.numel() for p in model.parameters())
        name = self.model_name or config.get("model_name", "Модель")

        self.model_card_label.setText(
            f"🧠 Модель: «{name}» | Архитектура: {arch_type} | "
            f"Параметры: {params_count:,} | Статус: "
            + ("🚀 Обучение" if self.is_training else "⏸ Ожидание")
        )

    def _format_time(self, seconds: float) -> str:
        """Форматирует секунды в ЧЧ:ММ:СС."""
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _log_session_action(self, action: str):
        """Логирует действие в историю сеанса (если доступна)."""
        session = self.shared_state.get("session")
        if session is None:
            return
        try:
            if hasattr(session, "log_action"):
                session.log_action("monitoring_panel", action)
            elif isinstance(session, dict):
                log_list = session.get("user_actions_log", [])
                log_list.append({
                    "panel": "monitoring_panel",
                    "action": action,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
        except Exception:
            pass

    # ============================================================
    #  СОВМЕСТИМОСТЬ СО СТАРЫМ ИНТЕРФЕЙСОМ
    # ============================================================
    # Эти методы обеспечивают обратную совместимость с main_window,
    # который может вызывать их напрямую.

    def on_training_started(self, is_text: bool):
        """Совместимость: вызывается из main_window._on_training_started."""
        self.reset()
        self.set_mode(is_text)