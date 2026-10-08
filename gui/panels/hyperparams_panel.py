"""
gui/panels/hyperparams_panel.py
================================
Панель гиперпараметров OracleAI Studio v2.0.

Ответственность:
  • Настройка оптимизатора, скорости обучения, штрафов
  • Настройка цикла обучения (эпохи, батч, функция потерь)
  • Планировщик скорости обучения с визуализацией кривой
  • Ранняя остановка (Early Stopping)
  • Текстовые параметры (Teacher Forcing)
  • Продвинутые настройки (перемешивание, градиентное накопление)
  • Динамические контекстные подсказки на КАЖДЫЙ элемент
  • 4 режима подсказок (static / ai / hybrid / none)

Зависимости:
  • core.contracts  — ModelConfig, SessionState, DataType, ArchitectureType, TaskType
  • gui.hint_widget — HintWidget (безопасный импорт с заглушкой)
  • core.hint_engine — HintEngine (безопасный импорт)

Сигналы:
  • params_applied(dict) — параметры применены в shared_state["config"]
"""

import math
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QFormLayout, QComboBox, QSpinBox, QDoubleSpinBox,
    QPushButton, QLabel, QTextEdit, QCheckBox,
    QScrollArea, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
import pyqtgraph as pg

from core.logger import get_logger

# === Безопасный импорт контрактов ===
try:
    from core.contracts import (
        ModelConfig, SessionState, DataType,
        ArchitectureType, TaskType, FORMAT_VERSION,
        config_to_dict, dataset_to_legacy_dict,
    )
    HAS_CONTRACTS = True
except ImportError:
    HAS_CONTRACTS = False

    def config_to_dict(config):
        if isinstance(config, dict):
            return config
        if hasattr(config, "to_dict"):
            return config.to_dict()
        return {}

    def dataset_to_legacy_dict(dataset):
        if isinstance(dataset, dict):
            return dataset
        if hasattr(dataset, "to_dict"):
            d = dataset.to_dict()
            meta = d.get("meta")
            if isinstance(meta, dict) and "type" not in meta and "data_type" in meta:
                meta["type"] = meta["data_type"]
            return d
        return {}

# === Безопасный импорт системы подсказок ===
try:
    from gui.hint_widget import HintWidget
    HAS_HINT_WIDGET = True
except ImportError:
    HAS_HINT_WIDGET = False

try:
    from core.hint_engine import HintEngine
    HAS_HINT_ENGINE = True
except ImportError:
    HAS_HINT_ENGINE = False

logger = get_logger(__name__)


# ============================================================
#  СТАТИЧЕСКИЕ ПОДСКАЗКИ ДЛЯ КАЖДОГО ЭЛЕМЕНТА
# ============================================================
# Ключ = widget_id. Значение = словарь с "default" и контекстными.

PARAM_HINTS = {
    "hyperparams.opt_combo": {
        "default": (
            "Оптимизатор — алгоритм, который решает, КАК именно сеть "
            "исправляет свои ошибки после каждого шага."
        ),
        "when_transformer": (
            "Для Трансформеров рекомендуется AdamW — он лучше работает "
            "с механизмом внимания и регуляризацией."
        ),
        "when_rnn": (
            "Для рекуррентных сетей (RNN/LSTM/GRU) хорошо работает RMSprop. "
            "Adam тоже подходит."
        ),
        "when_mlp": (
            "Для MLP подойдёт любой оптимизатор. Adam — самый простой старт."
        ),
        "when_beginner": (
            "Это как выбор учителя для нейросети. Adam — добрый и понятный учитель. "
            "Начните с него!"
        ),
    },

    "hyperparams.lr_spin": {
        "default": (
            "Скорость обучения (LR) — размер шага, который сеть делает при исправлении "
            "ошибок. Слишком большой — перепрыгнет ответ. Слишком маленький — будет "
            "учиться вечность."
        ),
        "when_transformer_and_high": (
            "⚠️ Слишком высоко для Трансформера! Сеть может разойтись (ошибка улетит "
            "в бесконечность). Рекомендуется 0.0001 – 0.001."
        ),
        "when_transformer_and_ok": (
            "✅ Хороший диапазон для Трансформера. 0.0001–0.001 — стандарт."
        ),
        "when_mlp_and_low": (
            "🐢 Очень медленно для MLP. Попробуйте 0.001–0.01, чтобы ускорить обучение."
        ),
        "when_mlp_and_ok": (
            "✅ Нормальный диапазон для MLP."
        ),
        "when_high_any": (
            "⚠️ Высокое значение! Если ошибка начнёт расти — уменьшите в 10 раз."
        ),
        "when_very_low": (
            "🐢 Очень маленькое значение. Обучение может занять очень много времени. "
            "Убедитесь, что это намеренно."
        ),
        "when_beginner": (
            "Представьте, что нейросеть спускается с горы в тумане. Скорость обучения — "
            "размер её шага. Слишком большой шаг — упадёт в пропасть. Слишком маленький — "
            "замёрзнет на месте. 0.001 — хороший первый шаг."
        ),
    },

    "hyperparams.wd_spin": {
        "default": (
            "Штраф за зубрёжку (L2 / Weight Decay). Заставляет сеть НЕ заучивать "
            "примеры наизусть, а понимать общие закономерности. Чем больше штраф, "
            "тем сильнее сеть сопротивляется запоминанию."
        ),
        "when_overfitting_risk": (
            "💡 У вас мало данных или большая модель — высокий риск зубрёжки. "
            "Поставьте 0.01–0.1, чтобы сеть обобщала, а не заучивала."
        ),
        "when_large_dataset": (
            "При большом датасете зубрёжка маловероятна. Можно оставить 0.0–0.001."
        ),
        "when_beginner": (
            "Представь ученика, который зубрит ответы. Этот штраф говорит: "
            "'Не зубри, а думай!' Если сеть переобучается — увеличьте штраф."
        ),
    },

    "hyperparams.clip_spin": {
        "default": (
            "Защита от сбоев (Градиентный клиппинг). Если градиент (направление "
            "исправления) вдруг стал огромным — он обрезается до этого значения. "
            "Защищает от 'взрыва' ошибки."
        ),
        "when_transformer": (
            "Для Трансформеров стандарт: 1.0. Не отключайте, это защищает от расходимости."
        ),
        "when_beginner": (
            "Это как страховочный трос для альпиниста. Если сеть сделает слишком "
            "резкий шаг — трос её удержит. 1.0 — хорошая длина троса."
        ),
    },

    "hyperparams.epochs_spin": {
        "default": (
            "Количество эпох — сколько раз сеть просмотрит ВЕСЬ датасет от начала "
            "до конца. Больше эпох = больше тренировок, но дольше обучение."
        ),
        "when_few_data": (
            "⚠️ У вас мало данных. Много эпох приведёт к зубрёжке. "
            "Попробуйте 10–30 эпох и включите раннюю остановку."
        ),
        "when_many_data": (
            "Большой датасет — можно ограничиться 10–30 эпохами, сеть и так успеет выучить."
        ),
        "when_beginner": (
            "Одна эпоха = один полный проход по всем примерам. Как прочитать учебник "
            "от корки до корки. 50 эпох = прочитать 50 раз. Иногда достаточно 10."
        ),
    },

    "hyperparams.batch_spin": {
        "default": (
            "Размер партии (батч) — сколько примеров сеть обрабатывает за один раз "
            "перед тем, как исправить ошибку. Больше = быстрее, но нужно больше памяти."
        ),
        "when_gpu_low_memory": (
            "⚠️ Если видеопамяти мало — уменьшите батч до 16–32."
        ),
        "when_transformer": (
            "Для Трансформеров стандарт: 32–64. Если не влезает в память — 16."
        ),
        "when_beginner": (
            "Представьте, что учитель проверяет тетради. Батч — сколько тетрадей "
            "он берёт за раз. 64 — стандартная стопка."
        ),
    },

    "hyperparams.loss_combo": {
        "default": (
            "Функция потерь — 'линейка', которой измеряется, насколько сеть ошиблась. "
            "Чем меньше потеря, тем лучше сеть работает."
        ),
        "when_text": (
            "Для текста используется только Cross Entropy — сеть угадывает следующий "
            "символ из словаря."
        ),
        "when_numeric_regression": (
            "Для числовых задач: MSE наказывает за большие ошибки сильнее, "
            "MAE — одинаково за любые отклонения."
        ),
        "when_beginner": (
            "Это как школьная линейка: чем больше сеть ошиблась, тем больше цифра "
            "на линейке. Мы хотим, чтобы цифра была как можно меньше."
        ),
    },

    "hyperparams.tf_ratio_spin": {
        "default": (
            "Подсказки учителя при генерации. Когда модель учится генерировать текст, "
            "она может использовать ПРАВИЛЬНЫЙ ответ как подсказку (а не своё "
            "предсказание). 1.0 = всегда правильный ответ (легче). "
            "0.0 = всегда свой ответ (сложнее, но надёжнее)."
        ),
        "when_high": (
            "Высокий Teacher Forcing: сеть привыкает к подсказкам. "
            "Рекомендуется начать с 0.7 и снижать к 0.3."
        ),
        "when_low": (
            "Низкий Teacher Forcing: сеть полагается на себя. Сложнее учиться, "
            "но результат будет надёжнее."
        ),
        "when_beginner": (
            "Представьте, что ученик решает задачу. С подсказками учителя (1.0) — "
            "легче. Без подсказок (0.0) — сложнее, но запоминает лучше. "
            "Начните с 0.5–0.7."
        ),
    },

    "hyperparams.scheduler_combo": {
        "default": (
            "Планировщик скорости обучения — как меняется 'размер шага' во время "
            "обучения. Можно начать с большого шага и постепенно уменьшать."
        ),
        "when_beginner": (
            "Как водитель: сначала разгоняется, потом тормозит перед финишем. "
            "'Нет' — едет с одной скоростью всю дорогу."
        ),
    },

    "hyperparams.scheduler_step_spin": {
        "default": (
            "Для ступенчатого планировщика: каждые сколько эпох уменьшать "
            "скорость обучения."
        ),
    },

    "hyperparams.scheduler_gamma_spin": {
        "default": (
            "Во сколько раз уменьшать скорость обучения на каждой ступени. "
            "0.1 = уменьшить в 10 раз. 0.5 = в 2 раза."
        ),
    },

    "hyperparams.early_stop_check": {
        "default": (
            "Ранняя остановка: если ошибка на новых данных (Val) не падает "
            "несколько эпох подряд — останавливаем обучение. Защита от зубрёжки."
        ),
        "when_beginner": (
            "Как учитель, который видит, что ученик уже не усваивает новый материал, "
            "и говорит: 'Хватит, иди отдыхай'. Включите, чтобы не тратить время зря."
        ),
    },

    "hyperparams.patience_spin": {
        "default": (
            "Терпение: сколько эпох ждать улучшения, прежде чем остановиться. "
            "5–10 — хороший выбор. Слишком мало — остановимся слишком рано."
        ),
    },

    "hyperparams.min_delta_spin": {
        "default": (
            "Минимальное улучшение: если ошибка уменьшилась меньше, чем на это "
            "значение — считаем, что улучшения нет. 0.0001 — стандарт."
        ),
    },

    "hyperparams.shuffle_check": {
        "default": (
            "Перемешивание данных: перед каждой эпохой примеры перемешиваются "
            "в случайном порядке. Помогает сети не запоминать порядок примеров. "
            "Рекомендуется включить."
        ),
    },

    "hyperparams.grad_accum_spin": {
        "default": (
            "Градиентное накопление: если батч не влезает в память, можно "
            "накапливать градиенты за несколько маленьких батчей и делать "
            "один большой шаг. 1 = отключено."
        ),
        "when_beginner": (
            "Если у вас слабая видеокарта — поставьте 2 или 4. Это как 'виртуально' "
            "увеличить батч без дополнительной памяти."
        ),
    },
}

# ============================================================
#  ОБЪЯСНЕНИЯ ОПТИМИЗАТОРОВ И ФУНКЦИЙ ПОТЕРЬ
# ============================================================

OPTIMIZER_EXPLANATIONS = {
    "adam": (
        "<b>Adam</b> — самый популярный и «умный» алгоритм. Сам понимает, где "
        "нужно сделать большой шаг при обучении, а где притормозить. "
        "Идеален для новичков."
    ),
    "adamw": (
        "<b>AdamW</b> — Adam с улучшенной системой штрафов за «зубрёжку». "
        "Отлично работает с архитектурой Transformer. Рекомендован для внимания."
    ),
    "sgd": (
        "<b>SGD (Стохастический градиентный спуск)</b> — классика. Работает "
        "медленнее, требует точной настройки «Скорости обучения» (LR), "
        "но иногда позволяет достичь лучшего качества."
    ),
    "rmsprop": (
        "<b>RMSprop</b> — алгоритм, хорошо справляющийся с Рекуррентными "
        "сетями (RNN / LSTM / GRU)."
    ),
}

LOSS_EXPLANATIONS = {
    "mse": (
        "<b>MSE (Среднеквадратичная ошибка)</b> — строго наказывает ИИ за любые "
        "сильные отклонения от правильного числа. Используется для регрессии "
        "и табличных данных."
    ),
    "mae": (
        "<b>MAE (Абсолютная ошибка)</b> — линейно измеряет дистанцию до ответа. "
        "Меньше боится выбросов и аномалий в данных."
    ),
    "cross_entropy": (
        "<b>Cross Entropy (Перекрёстная энтропия)</b> — используется для текста "
        "и классификации. Сеть пытается угадать правильный символ/класс из словаря. "
        "Чем увереннее угадала — тем меньше ошибка."
    ),
}

SCHEDULER_EXPLANATIONS = {
    "none": "Скорость обучения постоянна на протяжении всего обучения.",
    "steplr": "Каждые N эпох скорость обучения уменьшается в G раз (ступеньки вниз).",
    "cosine": "Скорость плавно снижается по косинусной кривой от начальной до почти нуля.",
    "plateau": "Скорость уменьшается, если ошибка на новых данных перестала падать.",
    "linear_warmup": "Сначала скорость растёт (разогрев), потом линейно падает.",
}


# ============================================================
#  ВСПОМОГАТЕЛЬНЫЙ КЛАСС: ЗАГЛУШКА ДЛЯ HINT_WIDGET
# ============================================================

class _NoopHintWidget:
    """Заглушка, если gui.hint_widget ещё не подключён."""
    def __init__(self, parent_widget, hint_engine, widget_id, panel_name):
        pass


# ============================================================
#  ОСНОВНОЙ КЛАСС ПАНЕЛИ
# ============================================================

class HyperparamsPanel(QWidget):
    """
    Панель гиперпараметров.

    Сигналы:
        params_applied(dict) — параметры применены в shared_state["config"].
                               Словарь содержит обновлённые ключи конфига.
    """

    params_applied = pyqtSignal(dict)

    def __init__(self, shared_state: dict, parent=None):
        super().__init__(parent)
        self.shared_state = shared_state

        # --- Система подсказок ---
        self._hint_engine = None
        if HAS_HINT_ENGINE:
            try:
                session_mgr = shared_state.get("session_manager", None)
                if session_mgr is not None:
                    self._hint_engine = HintEngine(session_mgr)
                else:
                    self._hint_engine = None
            except Exception:
                self._hint_engine = None

        self._HintW = HintWidget if HAS_HINT_WIDGET else _NoopHintWidget
        self._hint_registry = []  # чтобы хранить ссылки и не потерять

        self.init_ui()
        self._update_explanation()

    # ============================================================
    #  ПОДСКАЗКИ
    # ============================================================

    def _attach_hint(self, widget, widget_id: str):
        """Подключает подсказку к виджету. Безопасно, если система подсказок отсутствует."""
        hw = self._HintW(widget, self._hint_engine, widget_id, "hyperparams_panel")
        self._hint_registry.append(hw)

    def _get_dynamic_hint(self, widget_id: str) -> str:
        """
        Возвращает контекстную подсказку с учётом текущего состояния.
        Вызывается при наведении (если hint_engine доступен) или
        может быть использована для статусной строки.
        """
        hints = PARAM_HINTS.get(widget_id, {})
        base = hints.get("default", "")

        # Определяем контекст
        config = self.shared_state.get("config", {})
        dataset = self.shared_state.get("dataset", None)
        session = self.shared_state.get("session", None)

        arch = ""
        if HAS_CONTRACTS and isinstance(config, ModelConfig):
            arch = config.architecture.value if hasattr(config.architecture, 'value') else str(config.architecture)
        elif isinstance(config, dict):
            arch = config.get("type", config.get("architecture", ""))

        data_type = ""
        if dataset and isinstance(dataset, dict):
            meta = dataset.get("meta", {})
            if isinstance(meta, dict):
                data_type = meta.get("type", meta.get("data_type", ""))

        lr = self.lr_spin.value() if hasattr(self, 'lr_spin') else 0.001
        epochs = self.epochs_spin.value() if hasattr(self, 'epochs_spin') else 50
        num_samples = 0
        if dataset and isinstance(dataset, dict):
            meta = dataset.get("meta", {})
            if isinstance(meta, dict):
                num_samples = meta.get("num_samples", 0)

        # === Контекстные модификаторы ===

        # Скорость обучения
        if widget_id == "hyperparams.lr_spin":
            is_transformer = "transformer" in str(arch).lower()
            is_mlp = "mlp" in str(arch).lower()
            if is_transformer and lr > 0.01:
                return hints.get("when_transformer_and_high", base)
            elif is_transformer and 0.0001 <= lr <= 0.001:
                return hints.get("when_transformer_and_ok", base)
            elif is_mlp and lr < 0.0001:
                return hints.get("when_mlp_and_low", base)
            elif lr > 0.1:
                return hints.get("when_high_any", base)
            elif lr < 0.00001:
                return hints.get("when_very_low", base)

        # Штраф
        if widget_id == "hyperparams.wd_spin":
            if num_samples > 0 and num_samples < 5000:
                return hints.get("when_overfitting_risk", base)
            elif num_samples > 50000:
                return hints.get("when_large_dataset", base)

        # Эпохи
        if widget_id == "hyperparams.epochs_spin":
            if 0 < num_samples < 2000:
                return hints.get("when_few_data", base)
            elif num_samples > 50000:
                return hints.get("when_many_data", base)

        # Teacher Forcing
        if widget_id == "hyperparams.tf_ratio_spin":
            val = self.tf_ratio_spin.value() if hasattr(self, 'tf_ratio_spin') else 0.5
            if val > 0.7:
                return hints.get("when_high", base)
            elif val < 0.3:
                return hints.get("when_low", base)

        # Уровень пользователя
        if session and hasattr(session, 'total_sessions'):
            if session.total_sessions <= 2:
                beginner_key = "when_beginner"
                if beginner_key in hints:
                    return hints[beginner_key]

        return base

    def _show_hint_inline(self, widget_id: str):
        """Показывает подсказку в строке состояния (обновляет explanation_box)."""
        hint = self._get_dynamic_hint(widget_id)
        if hint:
            # Обновляем нижнюю строку подсказки без замены основного объяснения
            self._inline_hint_label.setText(f"💡 {hint}")

    # ============================================================
    #  UI
    # ============================================================

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # === Скролл-область (панель может быть высокой) ===
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll_content = QWidget()
        self.layout = QVBoxLayout(scroll_content)
        scroll_content.setLayout(self.layout)
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

        # === 0. Интерактивное объяснение ===
        self.explanation_box = QTextEdit()
        self.explanation_box.setReadOnly(True)
        self.explanation_box.setMaximumHeight(130)
        self.explanation_box.setStyleSheet(
            "background-color: #242424; color: #a6e3a1; "
            "font-size: 14px; padding: 10px; border: 1px solid #555;"
        )
        self.layout.addWidget(self.explanation_box)

        # === 1. Оптимизация ===
        self._create_optimization_group()

        # === 2. Цикл обучения ===
        self._create_training_cycle_group()

        # === 3. Планировщик скорости обучения ===
        self._create_scheduler_group()

        # === 4. Ранняя остановка ===
        self._create_early_stopping_group()

        # === 5. Текстовые параметры ===
        self._create_text_params_group()

        # === 6. Продвинутые настройки ===
        self._create_advanced_group()

        # === 7. Кнопка применения ===
        btn_layout = QHBoxLayout()
        self.apply_btn = QPushButton("💾 Применить настройки")
        self.apply_btn.setToolTip("Сохраняет все параметры в конфиг модели.")
        self.apply_btn.clicked.connect(self.apply_params)
        self._attach_hint(self.apply_btn, "hyperparams.apply_btn")
        btn_layout.addStretch()
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addStretch()
        self.layout.addLayout(btn_layout)

        # === 8. Строка динамических подсказок ===
        self._inline_hint_label = QLabel("")
        self._inline_hint_label.setWordWrap(True)
        self._inline_hint_label.setStyleSheet(
            "color: #4a88c7; font-size: 12px; padding: 4px; "
            "background-color: #1e2a3a; border-left: 3px solid #4a88c7;"
        )
        self._inline_hint_label.setVisible(False)
        self.layout.addWidget(self._inline_hint_label)

        self.layout.addStretch()

    # ------------------------------------------------------------
    #  1. ОПТИМИЗАЦИЯ
    # ------------------------------------------------------------

    def _create_optimization_group(self):
        group = QGroupBox("Оптимизация (Как сеть исправляет ошибки)")
        form = QFormLayout()

        # Оптимизатор
        self.opt_combo = QComboBox()
        self.opt_combo.addItems(["adam", "adamw", "sgd", "rmsprop"])
        self.opt_combo.currentIndexChanged.connect(self._update_explanation)
        self._attach_hint(self.opt_combo, "hyperparams.opt_combo")
        self.opt_combo.currentIndexChanged.connect(
            lambda: self._show_hint_inline("hyperparams.opt_combo")
        )
        form.addRow("Оптимизатор:", self.opt_combo)

        # Скорость обучения
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(5)
        self.lr_spin.setRange(0.00001, 1.0)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setSingleStep(0.001)
        self.lr_spin.valueChanged.connect(
            lambda: self._show_hint_inline("hyperparams.lr_spin")
        )
        self._attach_hint(self.lr_spin, "hyperparams.lr_spin")
        form.addRow("Скорость обучения (LR):", self.lr_spin)

        # Штраф за зубрёжку
        self.wd_spin = QDoubleSpinBox()
        self.wd_spin.setDecimals(5)
        self.wd_spin.setRange(0.0, 0.5)
        self.wd_spin.setValue(0.0)
        self.wd_spin.setSingleStep(0.001)
        self.wd_spin.valueChanged.connect(
            lambda: self._show_hint_inline("hyperparams.wd_spin")
        )
        self._attach_hint(self.wd_spin, "hyperparams.wd_spin")
        form.addRow("Штраф за зубрёжку (L2):", self.wd_spin)

        # Градиентный клиппинг
        self.clip_spin = QDoubleSpinBox()
        self.clip_spin.setDecimals(1)
        self.clip_spin.setRange(0.0, 100.0)
        self.clip_spin.setValue(1.0)
        self.clip_spin.setSingleStep(0.5)
        self.clip_spin.valueChanged.connect(
            lambda: self._show_hint_inline("hyperparams.clip_spin")
        )
        self._attach_hint(self.clip_spin, "hyperparams.clip_spin")
        form.addRow("Защита от сбоев (Clip):", self.clip_spin)

        group.setLayout(form)
        self.layout.addWidget(group)

    # ------------------------------------------------------------
    #  2. ЦИКЛ ОБУЧЕНИЯ
    # ------------------------------------------------------------

    def _create_training_cycle_group(self):
        group = QGroupBox("Цикл обучения (Тренировка)")
        form = QFormLayout()

        # Эпохи
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(50)
        self.epochs_spin.valueChanged.connect(
            lambda: self._show_hint_inline("hyperparams.epochs_spin")
        )
        self._attach_hint(self.epochs_spin, "hyperparams.epochs_spin")
        form.addRow("Количество Эпох:", self.epochs_spin)

        # Батч
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1024)
        self.batch_spin.setValue(64)
        self.batch_spin.valueChanged.connect(
            lambda: self._show_hint_inline("hyperparams.batch_spin")
        )
        self._attach_hint(self.batch_spin, "hyperparams.batch_spin")
        form.addRow("Размер партии (Batch):", self.batch_spin)

        # Функция потерь
        self.loss_combo = QComboBox()
        self.loss_combo.addItems(["mse", "mae", "cross_entropy"])
        self.loss_combo.currentIndexChanged.connect(self._update_explanation)
        self.loss_combo.currentIndexChanged.connect(
            lambda: self._show_hint_inline("hyperparams.loss_combo")
        )
        self._attach_hint(self.loss_combo, "hyperparams.loss_combo")
        form.addRow("Функция потерь (Линейка):", self.loss_combo)

        group.setLayout(form)
        self.layout.addWidget(group)

    # ------------------------------------------------------------
    #  3. ПЛАНИРОВЩИК СКОРОСТИ ОБУЧЕНИЯ
    # ------------------------------------------------------------

    def _create_scheduler_group(self):
        group = QGroupBox("Планировщик скорости обучения (Как меняется шаг)")
        layout = QVBoxLayout()

        # Выбор типа
        form = QFormLayout()
        self.scheduler_combo = QComboBox()
        self.scheduler_combo.addItems([
            "none", "steplr", "cosine", "plateau", "linear_warmup"
        ])
        self.scheduler_combo.currentIndexChanged.connect(self._on_scheduler_changed)
        self._attach_hint(self.scheduler_combo, "hyperparams.scheduler_combo")
        form.addRow("Тип планировщика:", self.scheduler_combo)

        # Параметры ступенчатого
        self.scheduler_step_spin = QSpinBox()
        self.scheduler_step_spin.setRange(1, 500)
        self.scheduler_step_spin.setValue(10)
        self.scheduler_step_spin.setToolTip("Каждые сколько эпох уменьшать скорость.")
        self._attach_hint(self.scheduler_step_spin, "hyperparams.scheduler_step_spin")
        form.addRow("Ступень (эпох):", self.scheduler_step_spin)

        self.scheduler_gamma_spin = QDoubleSpinBox()
        self.scheduler_gamma_spin.setRange(0.01, 0.99)
        self.scheduler_gamma_spin.setValue(0.1)
        self.scheduler_gamma_spin.setSingleStep(0.05)
        self.scheduler_gamma_spin.setDecimals(2)
        self.scheduler_gamma_spin.setToolTip("Во сколько раз уменьшать на каждой ступени.")
        self._attach_hint(self.scheduler_gamma_spin, "hyperparams.scheduler_gamma_spin")
        form.addRow("Множитель (gamma):", self.scheduler_gamma_spin)

        layout.addLayout(form)

        # Объяснение планировщика
        self.scheduler_explanation = QLabel("Скорость обучения постоянна.")
        self.scheduler_explanation.setWordWrap(True)
        self.scheduler_explanation.setStyleSheet(
            "color: #a9b7c6; font-size: 12px; padding: 4px;"
        )
        layout.addWidget(self.scheduler_explanation)

        # Визуализация кривой
        pg.setConfigOptions(antialias=True)
        pg.setConfigOption("background", "#242424")
        pg.setConfigOption("foreground", "#a9b7c6")
        self.scheduler_plot = pg.PlotWidget(title="Кривая скорости обучения по эпохам")
        self.scheduler_plot.setMaximumHeight(150)
        self.scheduler_plot.setLabel("left", "LR")
        self.scheduler_plot.setLabel("bottom", "Эпоха")
        self.scheduler_plot.showGrid(x=True, y=True, alpha=0.2)
        self.scheduler_curve = self.scheduler_plot.plot(
            pen=pg.mkPen("#4a88c7", width=2)
        )
        layout.addWidget(self.scheduler_plot)

        # Скрываем параметры ступенчатого по умолчанию
        self.scheduler_step_spin.setVisible(False)
        self.scheduler_gamma_spin.setVisible(False)

        group.setLayout(layout)
        self.layout.addWidget(group)

        # Начальная визуализация
        self._update_scheduler_preview()

    # ------------------------------------------------------------
    #  4. РАННЯЯ ОСТАНОВКА
    # ------------------------------------------------------------

    def _create_early_stopping_group(self):
        group = QGroupBox("Ранняя остановка (Защита от зубрёжки)")
        layout = QVBoxLayout()

        self.early_stop_check = QCheckBox("Включить раннюю остановку")
        self.early_stop_check.setChecked(False)
        self.early_stop_check.toggled.connect(self._on_early_stop_toggled)
        self._attach_hint(self.early_stop_check, "hyperparams.early_stop_check")
        self.early_stop_check.toggled.connect(
            lambda: self._show_hint_inline("hyperparams.early_stop_check")
        )
        layout.addWidget(self.early_stop_check)

        form = QFormLayout()

        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 100)
        self.patience_spin.setValue(7)
        self.patience_spin.setEnabled(False)
        self._attach_hint(self.patience_spin, "hyperparams.patience_spin")
        form.addRow("Терпение (эпох без улучшения):", self.patience_spin)

        self.min_delta_spin = QDoubleSpinBox()
        self.min_delta_spin.setRange(0.0, 0.1)
        self.min_delta_spin.setValue(0.0001)
        self.min_delta_spin.setDecimals(5)
        self.min_delta_spin.setSingleStep(0.0001)
        self.min_delta_spin.setEnabled(False)
        self._attach_hint(self.min_delta_spin, "hyperparams.min_delta_spin")
        form.addRow("Мин. улучшение (min_delta):", self.min_delta_spin)

        layout.addLayout(form)
        group.setLayout(layout)
        self.layout.addWidget(group)

    # ------------------------------------------------------------
    #  5. ТЕКСТОВЫЕ ПАРАМЕТРЫ
    # ------------------------------------------------------------

    def _create_text_params_group(self):
        self.text_group = QGroupBox("Параметры текстового обучения")
        form = QFormLayout()

        self.tf_ratio_spin = QDoubleSpinBox()
        self.tf_ratio_spin.setRange(0.0, 1.0)
        self.tf_ratio_spin.setValue(0.5)
        self.tf_ratio_spin.setSingleStep(0.1)
        self.tf_ratio_spin.setDecimals(2)
        self.tf_ratio_spin.valueChanged.connect(
            lambda: self._show_hint_inline("hyperparams.tf_ratio_spin")
        )
        self._attach_hint(self.tf_ratio_spin, "hyperparams.tf_ratio_spin")
        form.addRow("Подсказки учителя при генерации (Teacher Forcing):", self.tf_ratio_spin)

        self.text_group.setLayout(form)
        self.text_group.setVisible(False)
        self.layout.addWidget(self.text_group)

    # ------------------------------------------------------------
    #  6. ПРОДВИНУТЫЕ НАСТРОЙКИ
    # ------------------------------------------------------------

    def _create_advanced_group(self):
        group = QGroupBox("Продвинутые настройки")
        form = QFormLayout()

        # Перемешивание
        self.shuffle_check = QCheckBox("Перемешивать данные каждую эпоху")
        self.shuffle_check.setChecked(True)
        self._attach_hint(self.shuffle_check, "hyperparams.shuffle_check")
        form.addRow("", self.shuffle_check)

        # Градиентное накопление
        self.grad_accum_spin = QSpinBox()
        self.grad_accum_spin.setRange(1, 64)
        self.grad_accum_spin.setValue(1)
        self.grad_accum_spin.setToolTip("1 = отключено. 2–4 для слабых GPU.")
        self._attach_hint(self.grad_accum_spin, "hyperparams.grad_accum_spin")
        form.addRow("Градиентное накопление (шагов):", self.grad_accum_spin)

        group.setLayout(form)
        self.layout.addWidget(group)

    # ============================================================
    #  ЛОГИКА И ОБНОВЛЕНИЕ
    # ============================================================

    def _on_scheduler_changed(self, index: int):
        """При смене планировщика: показ/скрытие параметров + объяснение."""
        scheduler = self.scheduler_combo.currentText()

        is_steplr = scheduler == "steplr"
        self.scheduler_step_spin.setVisible(is_steplr)
        self.scheduler_gamma_spin.setVisible(is_steplr)

        self.scheduler_explanation.setText(SCHEDULER_EXPLANATIONS.get(scheduler, ""))
        self._update_scheduler_preview()

    def _update_scheduler_preview(self):
        """Рисует кривую изменения скорости обучения по эпохам."""
        scheduler = self.scheduler_combo.currentText()
        epochs = self.epochs_spin.value() if hasattr(self, 'epochs_spin') else 50
        lr0 = self.lr_spin.value() if hasattr(self, 'lr_spin') else 0.001
        step = self.scheduler_step_spin.value() if hasattr(self, 'scheduler_step_spin') else 10
        gamma = self.scheduler_gamma_spin.value() if hasattr(self, 'scheduler_gamma_spin') else 0.1

        x = list(range(1, epochs + 1))
        y = []

        for ep in x:
            if scheduler == "none":
                y.append(lr0)
            elif scheduler == "steplr":
                num_drops = (ep - 1) // step
                y.append(lr0 * (gamma ** num_drops))
            elif scheduler == "cosine":
                y.append(lr0 * 0.5 * (1 + math.cos(math.pi * (ep - 1) / max(epochs - 1, 1))))
            elif scheduler == "plateau":
                # Аппроксимация: плавное снижение во второй половине
                if ep <= epochs // 2:
                    y.append(lr0)
                else:
                    progress = (ep - epochs // 2) / max(epochs // 2, 1)
                    y.append(lr0 * (1 - 0.5 * progress))
            elif scheduler == "linear_warmup":
                warmup_epochs = max(epochs // 10, 1)
                if ep <= warmup_epochs:
                    y.append(lr0 * ep / warmup_epochs)
                else:
                    progress = (ep - warmup_epochs) / max(epochs - warmup_epochs, 1)
                    y.append(lr0 * (1 - progress))
            else:
                y.append(lr0)

        self.scheduler_curve.setData(x, y)

    def _on_early_stop_toggled(self, checked: bool):
        """Включение/выключение полей ранней остановки."""
        self.patience_spin.setEnabled(checked)
        self.min_delta_spin.setEnabled(checked)

    def _update_explanation(self):
        """Обновляет основной блок объяснения (оптимизатор + функция потерь)."""
        opt = self.opt_combo.currentText()
        loss = self.loss_combo.currentText()
        opt_text = OPTIMIZER_EXPLANATIONS.get(opt, "")
        loss_text = LOSS_EXPLANATIONS.get(loss, "")
        self.explanation_box.setHtml(f"{opt_text}<br><br>{loss_text}")

    def refresh(self):
        """
        Вызывается из main_window при обновлении данных или архитектуры.
        Адаптирует интерфейс под текущий тип данных и модель.
        """
        dataset = self.shared_state.get("dataset")
        if not dataset:
            return

        meta = dataset_to_legacy_dict(dataset).get("meta", {})
        data_type = meta.get("type", "numeric")

        # --- Функция потерь ---
        self.loss_combo.blockSignals(True)
        self.loss_combo.clear()
        if data_type == "text":
            self.loss_combo.addItem("cross_entropy")
            self.text_group.setVisible(True)
        else:
            self.loss_combo.addItems(["mse", "mae"])
            self.text_group.setVisible(False)
        self.loss_combo.blockSignals(False)

        # --- Обновляем объяснение ---
        self._update_explanation()

        # --- Обновляем планировщик ---
        self._update_scheduler_preview()

        # --- Обновляем контекстные подсказки ---
        # При наведении подсказки будут взяты динамически через _get_dynamic_hint

        logger.info(f"HyperparamsPanel.refresh: data_type={data_type}")

    # ============================================================
    #  ПРИМЕНЕНИЕ ПАРАМЕТРОВ
    # ============================================================

    def set_hyperparams_silent(self, epochs=None, lr=None, batch_size=None) -> str:
        """Устанавливает гиперпараметры без диалогов (для ИИ-агента)."""
        if epochs is not None:
            self.epochs_spin.setValue(max(1, min(int(epochs), 10000)))
        if lr is not None:
            self.lr_spin.setValue(max(0.00001, min(float(lr), 1.0)))
        if batch_size is not None:
            self.batch_spin.setValue(max(1, min(int(batch_size), 1024)))
        self.apply_params()
        return (f"гиперпараметры применены: эпох={self.epochs_spin.value()}, "
                f"lr={self.lr_spin.value()}, batch={self.batch_spin.value()}")

    def apply_params(self):
        """
        Собирает все значения и записывает в shared_state["config"].
        Эмитит сигнал params_applied с обновлённым словарём.

        Контракт: ключи словаря совпадают с полями ModelConfig.
        """
        dataset = self.shared_state.get("dataset")
        if dataset:
            data_type = dataset_to_legacy_dict(dataset).get("meta", {}).get("type", "numeric")
        else:
            data_type = "numeric"

        # Собираем обновления
        config_update = {
            "optimizer": self.opt_combo.currentText(),
            "learning_rate": self.lr_spin.value(),
            "weight_decay": self.wd_spin.value(),
            "gradient_clipping": self.clip_spin.value(),
            "epochs": self.epochs_spin.value(),
            "batch_size": self.batch_spin.value(),
            "loss_function": self.loss_combo.currentText(),
            "data_type": data_type,

            # Планировщик
            "scheduler": self.scheduler_combo.currentText(),
            "scheduler_step": self.scheduler_step_spin.value(),
            "scheduler_gamma": self.scheduler_gamma_spin.value(),

            # Ранняя остановка
            "early_stopping_enabled": self.early_stop_check.isChecked(),
            "early_stopping_patience": self.patience_spin.value() if self.early_stop_check.isChecked() else 0,
            "early_stopping_min_delta": self.min_delta_spin.value() if self.early_stop_check.isChecked() else 0.0,

            # Продвинутые
            "shuffle_data": self.shuffle_check.isChecked(),
            "gradient_accumulation_steps": self.grad_accum_spin.value(),
        }

        # Текстовые параметры
        if data_type == "text":
            config_update["teacher_forcing_ratio"] = self.tf_ratio_spin.value()

        # === Запись в shared_state ===
        config = self.shared_state.get("config", {})

        if HAS_CONTRACTS and isinstance(config, ModelConfig):
            # Обновляем dataclass
            config.optimizer = config_update["optimizer"]
            config.learning_rate = config_update["learning_rate"]
            config.weight_decay = config_update["weight_decay"]
            config.gradient_clipping = config_update["gradient_clipping"]
            config.epochs = config_update["epochs"]
            config.batch_size = config_update["batch_size"]
            config.loss_function = config_update["loss_function"]
            config.scheduler = config_update["scheduler"]
            config.teacher_forcing_ratio = config_update.get(
                "teacher_forcing_ratio", config.teacher_forcing_ratio
            )
            config.early_stopping_patience = config_update["early_stopping_patience"]
            # Дополнительные параметры в arch_params или extra
            if not hasattr(config, 'extra_params'):
                config.arch_params["scheduler_step"] = config_update["scheduler_step"]
                config.arch_params["scheduler_gamma"] = config_update["scheduler_gamma"]
                config.arch_params["early_stopping_enabled"] = config_update["early_stopping_enabled"]
                config.arch_params["early_stopping_min_delta"] = config_update["early_stopping_min_delta"]
                config.arch_params["shuffle_data"] = config_update["shuffle_data"]
                config.arch_params["gradient_accumulation_steps"] = config_update["gradient_accumulation_steps"]
        else:
            # dict-режим (обратная совместимость)
            if not isinstance(config, dict):
                config = {}
            config.update(config_update)
            self.shared_state["config"] = config

        # === Сигнал ===
        self.params_applied.emit(config_update)

        # === Визуальная обратная связь ===
        self.apply_btn.setText("✅ Успешно сохранено")
        self.apply_btn.setStyleSheet(
            "background-color: #385a3a; border-color: #4a7a4c; font-weight: bold;"
        )
        QTimer.singleShot(2000, self._reset_apply_button)

        logger.info(
            f"Гиперпараметры применены: LR={config_update['learning_rate']}, "
            f"Opt={config_update['optimizer']}, Epochs={config_update['epochs']}, "
            f"Scheduler={config_update['scheduler']}"
        )

    def _reset_apply_button(self):
        """Возвращает кнопку применения в исходное состояние."""
        self.apply_btn.setText("💾 Применить настройки")
        self.apply_btn.setStyleSheet("")

    # ============================================================
    #  ОБРАТНАЯ СОВМЕСТИМОСТЬ: загрузка из старого конфига
    # ============================================================

    def load_from_config(self, config: dict):
        """
        Загружает параметры из словаря (например, при загрузке модели из файла).
        Вызывается из main_window или training_panel.
        """
        if not isinstance(config, dict):
            return

        if "optimizer" in config:
            idx = self.opt_combo.findText(config["optimizer"])
            if idx >= 0:
                self.opt_combo.setCurrentIndex(idx)

        if "learning_rate" in config:
            self.lr_spin.setValue(config["learning_rate"])

        if "weight_decay" in config:
            self.wd_spin.setValue(config["weight_decay"])

        if "gradient_clipping" in config:
            self.clip_spin.setValue(config["gradient_clipping"])

        if "epochs" in config:
            self.epochs_spin.setValue(config["epochs"])

        if "batch_size" in config:
            self.batch_spin.setValue(config["batch_size"])

        if "loss_function" in config:
            idx = self.loss_combo.findText(config["loss_function"])
            if idx >= 0:
                self.loss_combo.setCurrentIndex(idx)

        if "teacher_forcing_ratio" in config:
            self.tf_ratio_spin.setValue(config["teacher_forcing_ratio"])

        if "scheduler" in config:
            idx = self.scheduler_combo.findText(config["scheduler"])
            if idx >= 0:
                self.scheduler_combo.setCurrentIndex(idx)

        if "early_stopping_patience" in config:
            patience = config["early_stopping_patience"]
            if patience > 0:
                self.early_stop_check.setChecked(True)
                self.patience_spin.setValue(patience)
            else:
                self.early_stop_check.setChecked(False)

        self._update_explanation()
        self._update_scheduler_preview()
        logger.info("Гиперпараметры загружены из конфига.")