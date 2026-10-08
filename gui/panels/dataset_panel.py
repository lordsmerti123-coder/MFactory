"""
gui/panels/dataset_panel.py
===========================
Панель загрузки, просмотра и разбиения датасетов.

Поддерживаемые типы данных:
  • Текст (текстовые последовательности)
  • Числа (табличные данные)
  • Изображения (классификация, генерация, автоэнкодеры)
  • Звук (классификация тонов, очистка шума)
  • Графы (связность, классификация узлов)
  • Временные ряды (предсказание, аномалии)

Тотальные подсказки на КАЖДЫЙ элемент интерфейса.
Поддержка 4 режимов подсказок: static / ai / hybrid / none.
Адаптация подсказок под историю сеансов пользователя.

Сигналы:
  • dataset_updated (object) — передаёт DatasetContainer или совместимый dict
Зависимости:
  • core.contracts   — DatasetContainer, DatasetMeta, DataType
  • core.data_loader — DatasetLoader, DatasetWizard
  • core.hint_engine — HintEngine, HintContext (опционально)
  • core.session     — SessionManager (опционально)
  • gui.hint_widget  — HintWidget (опционально)
"""

import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSpinBox, QFileDialog, QMessageBox, QGroupBox, QFormLayout,
    QProgressBar, QHeaderView, QCheckBox, QTableWidget,
    QTableWidgetItem, QScrollArea, QDialog, QTextEdit,
    QSizePolicy, QFrame, QGridLayout, QApplication
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QSize
from PyQt5.QtGui import QPixmap, QImage, QFont

from core.logger import get_logger

logger = get_logger()

# ============================================================
#  ИМПОРТ КОНТРАКТОВ (обязателен по ТЗ)
# ============================================================
try:
    from core.contracts import (
        DatasetContainer, DatasetMeta, DataType, TaskType, FORMAT_VERSION
    )
    CONTRACTS_AVAILABLE = True
except ImportError:
    CONTRACTS_AVAILABLE = False
    FORMAT_VERSION = "2.0"

    # Заглушки для работы без контрактов
    class DataType:
        TEXT = "text"
        NUMERIC = "numeric"
        IMAGE = "image"
        AUDIO = "audio"
        GRAPH = "graph"
        TIME_SERIES = "time_series"
        MULTIMODAL = "multimodal"

    class DatasetMeta:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class DatasetContainer:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

# ============================================================
#  ИМПОРТ ДВИЖКА ПОДСКАЗОК (опционален)
# ============================================================
try:
    from core.hint_engine import HintEngine, HintContext
    HINT_ENGINE_AVAILABLE = True
except ImportError:
    HINT_ENGINE_AVAILABLE = False
    HintEngine = None
    HintContext = None

# ============================================================
#  ИМПОРТ МЕНЕДЖЕРА СЕАНСОВ (опционален)
# ============================================================
try:
    from core.session import SessionManager
    SESSION_AVAILABLE = True
except ImportError:
    SESSION_AVAILABLE = False
    SessionManager = None

# ============================================================
#  ИМПОРТ ЗАГРУЗЧИКА ДАННЫХ
# ============================================================
try:
    from core.data_loader import DatasetLoader, DatasetWizard
    DATA_LOADER_AVAILABLE = True
except ImportError:
    DATA_LOADER_AVAILABLE = False
    DatasetLoader = None
    DatasetWizard = None

# ============================================================
#  ИМПОРТ ТОКЕНИЗАТОРА
# ============================================================
try:
    from core.text_tokenizer import TextTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    TextTokenizer = None

# ============================================================
#  ИМПОРТ ВИДЖЕТА ПОДСКАЗОК (опционален)
# ============================================================
try:
    from gui.hint_widget import HintWidget
    HINT_WIDGET_AVAILABLE = True
except ImportError:
    HINT_WIDGET_AVAILABLE = False
    HintWidget = None

# ============================================================
#  ИМПОРТ РЕДАКТОРА СЛОВАРЯ (опционален)
# ============================================================
try:
    from gui.widgets.vocab_editor import VocabEditorDialog as ExternalVocabEditor
    EXTERNAL_VOCAB_EDITOR = True
except ImportError:
    EXTERNAL_VOCAB_EDITOR = False
    ExternalVocabEditor = None


# ============================================================
#  СТАТИЧЕСКИЕ ПОДСКАЗКИ ДЛЯ КАЖДОГО ЭЛЕМЕНТА
#  ТЗ: "Тотальные подсказки (tooltip на каждый элемент)"
# ============================================================
STATIC_HINTS: Dict[str, Dict[str, str]] = {
    "load_btn": {
        "default": (
            "📁 Нажмите чтобы загрузить датасет.\n"
            "Поддерживаются: .oai, .csv, .json, .pt, папка с изображениями, .wav файлы.\n"
            "Программа определит тип данных и проверит качество."
        ),
        "beginner": (
            "📁 Здесь загружают данные, на которых будет учиться нейросеть.\n"
            "Это как учебник для ученика: чем больше примеров — тем лучше.\n"
            "Сначала сгенерируйте данные во вкладке «Генератор», потом загрузите их здесь."
        ),
    },
    "type_label": {
        "default": (
            "Тип данных определяет какие архитектуры моделей доступны.\n"
            "• Текст → Transformer, RNN, LSTM\n"
            "• Изображения → CNN, GAN, VAE\n"
            "• Числа → MLP, RBFN\n"
            "• Звук → CNN (спектрограммы), RNN"
        ),
    },
    "samples_label": {
        "default": (
            "Общее количество примеров в датасете.\n"
            "Для простых задач хватает 1 000 примеров.\n"
            "Для сложных (Трансформер на математике) нужно 20 000+."
        ),
    },
    "vocab_label": {
        "default": (
            "Словарь — набор уникальных символов, которые модель умеет распознавать.\n"
            "Всё, что не в словаре, станет <UNK> (неизвестно).\n"
            "Как азбука для модели: если буквы нет в азбуке — модель её не прочитает."
        ),
        "beginner": (
            "Словарь — это алфавит, который знает модель.\n"
            "Например: '2', '+', '5', '=' — это 4 символа.\n"
            "Если в данных встретится буква 'ё', а её нет в словаре — модель увидит '?'."
        ),
    },
    "input_dim_label": {
        "default": (
            "Размерность входных данных.\n"
            "Для текста: длина последовательности.\n"
            "Для изображений: высота × ширина × каналы.\n"
            "Для чисел: количество признаков (столбцов)."
        ),
    },
    "train_spin": {
        "default": (
            "Доля данных для обучения. 80% — стандарт.\n"
            "Это примеры, на которых модель тренируется.\n"
            "Если данных мало (<1000), лучше 90/5/5."
        ),
        "beginner": (
            "Это как домашнее задание: модель решает примеры из этой части.\n"
            "Чем больше примеров для тренировки — тем лучше.\n"
            "Но нельзя отдавать 100% — нужно оставить что-то для проверки."
        ),
    },
    "val_spin": {
        "default": (
            "Доля данных для проверки во время обучения.\n"
            "Модель НЕ видит эти примеры при тренировке.\n"
            "После каждой эпохи модель проверяется на этих данных."
        ),
        "beginner": (
            "Это как контрольная работа во время учёбы.\n"
            "Модель тренируется на одних примерах, а проверяется на других.\n"
            "Если на контрольной ошибки растут — модель переобучилась."
        ),
    },
    "test_spin": {
        "default": (
            "Доля данных для финального экзамена.\n"
            "Используется только ОДИН раз — в самом конце.\n"
            "Показывает, насколько хорошо модель обобщает."
        ),
        "beginner": (
            "Это как выпускной экзамен.\n"
            "Модель видит эти примеры впервые.\n"
            "Результат на тесте — честная оценка, чему она научилась."
        ),
    },
    "vocab_limit_check": {
        "default": (
            "Оставить только N самых частых символов.\n"
            "Полезно для экономии памяти, но редкие символы модель не выучит.\n"
            "Все редкие символы станут <UNK> (неизвестно)."
        ),
    },
    "vocab_limit_spin": {
        "default": (
            "Максимальный размер словаря (включая 4 спецтокена).\n"
            "Всё, что сверх этого лимита — станет <UNK>.\n"
            "Для математики обычно хватает 20-50 символов.\n"
            "Для текста на русском языке — 100-200."
        ),
    },
    "split_btn": {
        "default": (
            "✂️ Разрезать данные на 3 части:\n"
            "• Train — на чём учится модель\n"
            "• Val — на чём проверяем во время учёбы\n"
            "• Test — финальный экзамен\n"
            "Без этого шага обучение НЕ запустится!"
        ),
        "beginner": (
            "Это как разделить учебник на три стопки:\n"
            "1. Домашние задания (большая стопка)\n"
            "2. Контрольные (поменьше)\n"
            "3. Экзамен (совсем маленькая, но самая важная)\n"
            "Нажмите кнопку — и данные разделятся!"
        ),
    },
    "vocab_btn": {
        "default": (
            "👁 Показать и отредактировать словарь.\n"
            "Можно увидеть все символы, их индексы и частоту.\n"
            "Можно удалить символ или добавить новый.\n"
            "Изменения повлияют на токенизацию данных."
        ),
    },
    "quality_label": {
        "default": (
            "Отчёт о качестве загруженных данных.\n"
            "Показывает: сколько строк удалено, какие аномалии найдены,\n"
            "есть ли пропуски или некорректные значения."
        ),
    },
    "preview_table": {
        "default": (
            "Предпросмотр первых примеров датасета.\n"
            "Слева — вход (что подаём модели).\n"
            "Справа — ожидаемый выход (правильный ответ)."
        ),
    },
    "image_grid": {
        "default": (
            "Предпросмотр изображений из датасета.\n"
            "Показана сетка 5×5 случайных примеров.\n"
            "Под каждым изображением — его метка (класс)."
        ),
    },
    "audio_btn": {
        "default": (
            "🔊 Воспроизвести пример звуковых данных.\n"
            "Позволяет услышать, что именно модель будет анализировать."
        ),
    },
    "split_info_label": {
        "default": (
            "Результат применённого разбиения.\n"
            "Показывает сколько примеров в каждой части и размер словаря."
        ),
    },
    "meta_group": {
        "default": (
            "Метаданные — паспорт датасета.\n"
            "Тип данных, количество примеров, размерность, словарь.\n"
            "Эта информация определяет какие модели можно использовать."
        ),
    },
    "split_group": {
        "default": (
            "Настройка разбиения данных на обучающую, проверочную и тестовую выборки.\n"
            "Стандартное соотношение: 80% / 10% / 10%.\n"
            "Словарь строится ТОЛЬКО на обучающей выборке (защита от утечки данных)."
        ),
    },
}


# ============================================================
#  ПОТОК ЗАГРУЗКИ ДАННЫХ (не блокирует GUI)
# ============================================================
class DatasetLoaderThread(QThread):
    """Загружает датасет в фоновом потоке."""
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(str)

    def __init__(self, path: Path):
        super().__init__()
        self.path = path

    def run(self):
        try:
            self.progress_signal.emit(f"Читаю файл: {self.path.name}...")

            if not DATA_LOADER_AVAILABLE:
                self.error_signal.emit(
                    "Модуль core.data_loader недоступен. "
                    "Убедитесь что файл существует и проект собран корректно."
                )
                return

            dataset = DatasetLoader.load_dataset(self.path)
            self.finished_signal.emit(dataset)
        except Exception as e:
            logger.error(f"Ошибка загрузки датасета: {e}\n{traceback.format_exc()}")
            self.error_signal.emit(str(e))


# ============================================================
#  ДИАЛОГ РЕДАКТОРА СЛОВАРЯ
# ============================================================
class VocabEditorDialog(QDialog):
    """
    Модальное окно для просмотра и редактирования словаря.
    ТЗ: "Кнопка '👁 Показать словарь' → таблица: символ | индекс | частота"
    ТЗ: "Возможность вручную удалить символ или добавить"
    ТЗ: "Предупреждение при удалении"
    """

    def __init__(self, vocab: Dict[str, int], texts: List[str] = None, parent=None):
        super().__init__(parent)
        self.vocab = dict(vocab)  # копия чтобы не менять оригинал
        self.original_vocab = dict(vocab)
        self.texts = texts or []
        self.char_frequencies = self._compute_frequencies()
        self.setWindowTitle("🔤 Редактор словаря")
        self.setMinimumSize(600, 500)
        self.init_ui()

    def _compute_frequencies(self) -> Dict[str, int]:
        """Считает частоту каждого символа в текстах."""
        freq = {}
        for text in self.texts:
            for char in str(text):
                freq[char] = freq.get(char, 0) + 1
        return freq

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Информация
        info_label = QLabel(
            f"Всего токенов в словаре: {len(self.vocab)}\n"
            f"Из них 4 специальных: <PAD>, <UNK>, <BOS>, <EOS>\n"
            f"Пользовательских символов: {len(self.vocab) - 4}\n\n"
            "⚠️ Удаление символа приведёт к замене всех его вхождений на <UNK>."
        )
        info_label.setStyleSheet("color: #e5c07b; padding: 5px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Таблица словаря
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Символ", "Индекс", "Частота", "Удалить"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._populate_table()
        layout.addWidget(self.table)

        # Кнопки
        btn_layout = QHBoxLayout()

        self.btn_add = QPushButton("➕ Добавить символ")
        self.btn_add.clicked.connect(self._add_symbol)
        btn_layout.addWidget(self.btn_add)

        btn_layout.addStretch()

        self.btn_cancel = QPushButton("Отмена")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)

        self.btn_apply = QPushButton("✅ Применить изменения")
        self.btn_apply.setStyleSheet("font-weight: bold;")
        self.btn_apply.clicked.connect(self._apply_changes)
        btn_layout.addWidget(self.btn_apply)

        layout.addLayout(btn_layout)

    def _populate_table(self):
        """Заполняет таблицу данными словаря."""
        sorted_vocab = sorted(self.vocab.items(), key=lambda x: x[1])
        self.table.setRowCount(len(sorted_vocab))

        for row, (char, idx) in enumerate(sorted_vocab):
            # Символ
            char_item = QTableWidgetItem(self._display_char(char))
            char_item.setFlags(char_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, char_item)

            # Индекс
            idx_item = QTableWidgetItem(str(idx))
            idx_item.setFlags(idx_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 1, idx_item)

            # Частота
            freq = self.char_frequencies.get(char, "—")
            freq_item = QTableWidgetItem(str(freq))
            freq_item.setFlags(freq_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 2, freq_item)

            # Кнопка удаления (только для не-специальных токенов)
            is_special = char in ("<PAD>", "<UNK>", "<BOS>", "<EOS>")
            del_btn = QPushButton("🗑")
            del_btn.setMaximumWidth(40)
            del_btn.setEnabled(not is_special)
            del_btn.setToolTip(
                "Удалить этот символ из словаря.\n"
                "Все его вхождения в данных станут <UNK>."
                if not is_special else
                "Специальный токен. Нельзя удалить."
            )
            if not is_special:
                del_btn.clicked.connect(lambda checked, c=char: self._remove_symbol(c))
            self.table.setCellWidget(row, 3, del_btn)

    @staticmethod
    def _display_char(char: str) -> str:
        """Отображает символ в человекочитаемом виде."""
        specials = {
            "<PAD>": "<PAD> (заполнитель)",
            "<UNK>": "<UNK> (неизвестно)",
            "<BOS>": "<BOS> (начало)",
            "<EOS>": "<EOS> (конец)",
        }
        if char in specials:
            return specials[char]
        if char == " ":
            return "␣ (пробел)"
        if char == "\n":
            return "↵ (перенос строки)"
        if char == "\t":
            return "⇥ (табуляция)"
        return char

    def _remove_symbol(self, char: str):
        """Удаляет символ из словаря с предупреждением."""
        # Считаем сколько вхождений будет затронуто
        affected_count = 0
        total_chars = 0
        for text in self.texts:
            total_chars += len(str(text))
            affected_count += str(text).count(char)

        pct = (affected_count / max(total_chars, 1)) * 100
        reply = QMessageBox.warning(
            self, "Удаление символа",
            f"Вы удаляете символ '{self._display_char(char)}'.\n\n"
            f"Все его вхождения станут <UNK>.\n"
            f"Это затронет {affected_count} символов ({pct:.1f}% данных).\n\n"
            f"Продолжить?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            if char in self.vocab:
                del self.vocab[char]
            self._populate_table()

    def _add_symbol(self):
        """Добавляет новый символ в словарь."""
        from PyQt5.QtWidgets import QInputDialog
        char, ok = QInputDialog.getText(
            self, "Добавить символ",
            "Введите символ для добавления в словарь:"
        )
        if ok and char:
            char = char[0]  # берём только первый символ
            if char in self.vocab:
                QMessageBox.information(self, "Внимание", "Этот символ уже есть в словаре.")
            else:
                new_idx = max(self.vocab.values()) + 1 if self.vocab else 4
                self.vocab[char] = new_idx
                self._populate_table()

    def _apply_changes(self):
        """Применяет изменения и закрывает диалог."""
        self.accept()

    def get_modified_vocab(self) -> Dict[str, int]:
        """Возвращает изменённый словарь."""
        return dict(self.vocab)


# ============================================================
#  ОСНОВНАЯ ПАНЕЛЬ ДАННЫХ
# ============================================================
class DatasetPanel(QWidget):
    """
    Панель загрузки, просмотра и разбиения датасетов.

    ТЗ (раздел 3.2):
      • Ответственность: Загрузка, просмотр, разбиение
      • Сигналы: dataset_updated
      • Слушает: generator_panel, project_panel

    ТЗ (раздел 13.13):
      • Кнопка загрузки
      • Метаданные
      • Разбиение (train/val/test %)
      • Ограничение словаря
      • Кнопка "Применить разбиение"
      • Кнопка "Показать словарь" → открывает vocab_editor
      • Таблица предпросмотра
      • Для изображений: сетка предпросмотра
      • Для звука: кнопка воспроизведения
    """

    # Сигнал: передаёт DatasetContainer или совместимый dict
    # Используем object т.к. PyQt не параметризует dataclass напрямую
    dataset_updated = pyqtSignal(object)

    def __init__(self, shared_state: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.shared_state = shared_state
        self.dataset = None          # Текущий загруженный датасет
        self._load_thread = None     # Фоновый поток загрузки
        self._hint_engine = None     # Движок подсказок
        self._init_hint_engine()
        self.init_ui()
        self._setup_hints()

    # ============================================================
    #  ИНИЦИАЛИЗАЦИЯ ДВИЖКА ПОДСКАЗОК
    # ============================================================
    def _init_hint_engine(self):
        """Инициализирует движок подсказок если доступен."""
        if HINT_ENGINE_AVAILABLE:
            try:
                session_mgr = self.shared_state.get("session_manager")
                if session_mgr:
                    self._hint_engine = HintEngine(session_mgr)
                else:
                    self._hint_engine = HintEngine(None)
            except Exception as e:
                logger.warning(f"HintEngine недоступен: {e}")
                self._hint_engine = None

    def _get_hint(self, widget_id: str) -> str:
        """
        Возвращает подсказку для элемента с учётом:
        • Режима подсказок (static / ai / hybrid / none)
        • Истории сеансов пользователя
        • Контекста (тип данных, модель)
        """
        # Определяем режим
        hint_mode = self.shared_state.get("hint_mode", "hybrid")
        if hint_mode == "none":
            return ""

        # Пытаемся получить подсказку от движка
        if self._hint_engine and HINT_ENGINE_AVAILABLE:
            try:
                session = self.shared_state.get("session")
                dataset_meta = None
                if self.dataset:
                    meta = self._get_meta(self.dataset)
                    dataset_meta = meta

                context = HintContext(
                    widget_id=widget_id,
                    panel_name="dataset_panel",
                    model_config=self.shared_state.get("config"),
                    dataset_meta=dataset_meta,
                    session=session,
                    current_value=None,
                )
                hint = self._hint_engine.get_hint(context)
                if hint:
                    return hint
            except Exception:
                pass

        # Статическая подсказка (fallback)
        if hint_mode == "ai":
            return "🤖 ИИ-подсказка недоступна. Автоматическая подсказка не сгенерирована."

        return self._static_hint(widget_id)

    def _static_hint(self, widget_id: str) -> str:
        """Возвращает статическую подсказку с учётом уровня пользователя."""
        hints = STATIC_HINTS.get(widget_id, {})
        if not hints:
            return ""

        # Определяем уровень пользователя
        session = self.shared_state.get("session")
        if session and hasattr(session, "total_sessions"):
            total = getattr(session, "total_sessions", 0)
            if total <= 2 and "beginner" in hints:
                return hints["beginner"]

        return hints.get("default", "")

    # ============================================================
    #  ПОСТРОЕНИЕ ИНТЕРФЕЙСА
    # ============================================================
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # === 1. КНОПКА ЗАГРУЗКИ ===
        load_layout = QHBoxLayout()

        self.load_btn = QPushButton("📁 Загрузить датасет (.oai, .csv, .json, .pt, 🖼, 🎵)")
        self.load_btn.setMinimumHeight(40)
        self.load_btn.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.load_btn.clicked.connect(self.load_dataset)
        load_layout.addWidget(self.load_btn)

        self.load_folder_btn = QPushButton("📂 Загрузить папку изображений")
        self.load_folder_btn.setMinimumHeight(40)
        self.load_folder_btn.clicked.connect(self.load_image_folder)
        load_layout.addWidget(self.load_folder_btn)

        load_layout.addStretch()
        layout.addLayout(load_layout)

        # Прогресс загрузки
        self.loading_progress = QProgressBar()
        self.loading_progress.setRange(0, 0)  # неопределённый прогресс
        self.loading_progress.setVisible(False)
        self.loading_progress.setMinimumHeight(6)
        layout.addWidget(self.loading_progress)

        self.loading_status = QLabel("")
        self.loading_status.setStyleSheet("color: #4a88c7; font-size: 12px;")
        self.loading_status.setVisible(False)
        layout.addWidget(self.loading_status)

        # === 2. ОТЧЁТ О КАЧЕСТВЕ ДАННЫХ ===
        self.quality_group = QGroupBox("📋 Отчёт о качестве данных")
        quality_layout = QVBoxLayout()
        self.quality_label = QLabel(
            "Данные ещё не загружены. Нажмите «Загрузить датасет» чтобы начать."
        )
        self.quality_label.setWordWrap(True)
        self.quality_label.setStyleSheet(
            "color: #a9b7c6; font-size: 13px; padding: 5px; "
            "background-color: #242424; border-left: 3px solid #555;"
        )
        quality_layout.addWidget(self.quality_label)
        self.quality_group.setLayout(quality_layout)
        self.quality_group.setVisible(False)
        layout.addWidget(self.quality_group)

        # === 3. МЕТАДАННЫЕ ===
        self.meta_group = QGroupBox("📊 Метаданные (Паспорт датасета)")
        meta_layout = QFormLayout()
        meta_layout.setSpacing(6)

        self.type_label = QLabel("—")
        self.type_label.setStyleSheet("color: #4a88c7; font-weight: bold;")
        meta_layout.addRow("Тип данных:", self.type_label)

        self.task_type_label = QLabel("—")
        meta_layout.addRow("Тип задачи:", self.task_type_label)

        self.samples_label = QLabel("—")
        meta_layout.addRow("Всего примеров:", self.samples_label)

        self.input_dim_label = QLabel("—")
        meta_layout.addRow("Размерность входа:", self.input_dim_label)

        self.vocab_label = QLabel("—")
        self.vocab_label.setStyleSheet("color: #4a88c7;")
        meta_layout.addRow("Словарь (для текста):", self.vocab_label)

        self.image_shape_label = QLabel("—")
        meta_layout.addRow("Размер изображений:", self.image_shape_label)

        self.audio_info_label = QLabel("—")
        meta_layout.addRow("Аудио (частота):", self.audio_info_label)

        self.meta_group.setLayout(meta_layout)
        layout.addWidget(self.meta_group)

        # === 4. РАЗБИЕНИЕ И ТОКЕНИЗАЦИЯ ===
        self.split_group = QGroupBox("✂️ Разбиение данных (Train / Validation / Test)")
        split_layout = QVBoxLayout()

        # Проценты
        ratios_layout = QHBoxLayout()

        self.train_spin = QSpinBox()
        self.train_spin.setRange(0, 100)
        self.train_spin.setValue(80)
        self.train_spin.setSuffix(" %")
        ratios_layout.addWidget(QLabel("Train:"))
        ratios_layout.addWidget(self.train_spin)

        self.val_spin = QSpinBox()
        self.val_spin.setRange(0, 100)
        self.val_spin.setValue(10)
        self.val_spin.setSuffix(" %")
        ratios_layout.addWidget(QLabel("Val:"))
        ratios_layout.addWidget(self.val_spin)

        self.test_spin = QSpinBox()
        self.test_spin.setRange(0, 100)
        self.test_spin.setValue(10)
        self.test_spin.setSuffix(" %")
        ratios_layout.addWidget(QLabel("Test:"))
        ratios_layout.addWidget(self.test_spin)

        ratios_layout.addStretch()
        split_layout.addLayout(ratios_layout)

        # Ограничение словаря
        vocab_layout = QHBoxLayout()

        self.vocab_limit_check = QCheckBox("Ограничить словарь (Top-K)")
        self.vocab_limit_check.toggled.connect(self._on_vocab_limit_toggled)
        vocab_layout.addWidget(self.vocab_limit_check)

        self.vocab_limit_spin = QSpinBox()
        self.vocab_limit_spin.setRange(10, 50000)
        self.vocab_limit_spin.setValue(100)
        self.vocab_limit_spin.setEnabled(False)
        vocab_layout.addWidget(self.vocab_limit_spin)

        vocab_layout.addStretch()
        split_layout.addLayout(vocab_layout)

        # Кнопки
        split_btn_layout = QHBoxLayout()

        self.split_btn = QPushButton("✂️ Применить разбиение и собрать словарь")
        self.split_btn.setMinimumHeight(36)
        self.split_btn.setStyleSheet("font-weight: bold;")
        self.split_btn.clicked.connect(self.split_dataset)
        split_btn_layout.addWidget(self.split_btn)

        self.vocab_btn = QPushButton("👁 Показать словарь")
        self.vocab_btn.setMinimumHeight(36)
        self.vocab_btn.clicked.connect(self._show_vocab_editor)
        self.vocab_btn.setEnabled(False)
        split_btn_layout.addWidget(self.vocab_btn)

        split_layout.addLayout(split_btn_layout)

        self.split_group.setLayout(split_layout)
        layout.addWidget(self.split_group)

        # === 5. ИНФОРМАЦИЯ О РАЗБИЕНИИ ===
        self.split_info_label = QLabel("")
        self.split_info_label.setWordWrap(True)
        self.split_info_label.setStyleSheet(
            "color: #a6e3a1; font-size: 13px; padding: 6px; "
            "background-color: #313335; border-left: 3px solid #385a3a;"
        )
        self.split_info_label.setVisible(False)
        layout.addWidget(self.split_info_label)

        # === 6. ПРЕДПРОСМОТР ДАННЫХ ===
        preview_group = QGroupBox("👁 Предпросмотр данных")
        preview_layout = QVBoxLayout()

        # Таблица для текста/чисел
        self.preview_table = QTableWidget()
        self.preview_table.setColumnCount(2)
        self.preview_table.setHorizontalHeaderLabels(
            ["Вход (X — что подаём модели)", "Выход (Y — правильный ответ)"]
        )
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.preview_table.setMaximumHeight(200)
        preview_layout.addWidget(self.preview_table)

        # Сетка для изображений
        self.image_grid_widget = QWidget()
        self.image_grid_layout = QGridLayout(self.image_grid_widget)
        self.image_grid_widget.setVisible(False)
        preview_layout.addWidget(self.image_grid_widget)

        # Кнопка для звука
        self.audio_preview_layout = QHBoxLayout()
        self.audio_play_btn = QPushButton("🔊 Воспроизвести пример звука")
        self.audio_play_btn.clicked.connect(self._play_audio_sample)
        self.audio_play_btn.setVisible(False)
        self.audio_preview_layout.addWidget(self.audio_play_btn)
        self.audio_preview_layout.addStretch()
        preview_layout.addLayout(self.audio_preview_layout)

        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        layout.addStretch()

    # ============================================================
    #  ПОДСКАЗКИ: ПОДКЛЮЧЕНИЕ К КАЖДОМУ ЭЛЕМЕНТУ
    # ============================================================
    def _setup_hints(self):
        """
        ТЗ: "Тотальные подсказки (tooltip на каждый элемент)"
        Устанавливает подсказки на каждый интерактивный элемент.
        Если доступен HintWidget — подключает динамические подсказки.
        """
        # Словарь: элемент → (виджет, widget_id)
        elements = [
            (self.load_btn, "load_btn"),
            (self.load_folder_btn, "load_btn"),
            (self.type_label, "type_label"),
            (self.samples_label, "samples_label"),
            (self.vocab_label, "vocab_label"),
            (self.input_dim_label, "input_dim_label"),
            (self.train_spin, "train_spin"),
            (self.val_spin, "val_spin"),
            (self.test_spin, "test_spin"),
            (self.vocab_limit_check, "vocab_limit_check"),
            (self.vocab_limit_spin, "vocab_limit_spin"),
            (self.split_btn, "split_btn"),
            (self.vocab_btn, "vocab_btn"),
            (self.quality_label, "quality_label"),
            (self.preview_table, "preview_table"),
            (self.split_info_label, "split_info_label"),
            (self.audio_play_btn, "audio_btn"),
            (self.meta_group, "meta_group"),
            (self.split_group, "split_group"),
        ]

        for widget, widget_id in elements:
            # Если доступен HintWidget — единый источник подсказок (popup).
            # Передаём префиксованный id («dataset.*»), чтобы движок отдал
            # подсказку именно этой панели, а не другой (например, project.load_btn).
            if HINT_WIDGET_AVAILABLE and HintWidget and self._hint_engine is not None:
                try:
                    HintWidget(widget, self._hint_engine,
                               f"dataset.{widget_id}", "dataset_panel")
                    continue
                except Exception:
                    pass
            # Фолбэк без HintWidget — обычный tooltip из локальной базы.
            hint = self._get_hint(widget_id)
            if hint:
                widget.setToolTip(hint)

    def refresh_hints(self):
        """Обновляет подсказки (вызывается при смене режима)."""
        self._setup_hints()

    # ============================================================
    #  ЗАГРУЗКА ДАННЫХ
    # ============================================================
    def load_dataset_silent(self, path) -> str:
        """Загружает датасет из файла без диалога (для ИИ-агента)."""
        try:
            if not DATA_LOADER_AVAILABLE:
                return "модуль core.data_loader недоступен"
            dataset = DatasetLoader.load_dataset(Path(path))
            self.on_dataset_loaded(dataset)
            return f"датасет загружен: {Path(path).name}"
        except Exception as e:
            logger.error(f"Ошибка загрузки датасета: {e}")
            return f"ошибка загрузки датасета: {e}"

    def load_dataset(self):
        """Загружает датасет из файла."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите датасет",
            str(Path(".")),
            "Данные OracleAI (*.oai);;"
            "JSON (*.json);;"
            "CSV (*.csv);;"
            "PyTorch (*.pt);;"
            "Все файлы (*)"
        )
        if not path:
            return
        self._start_loading(Path(path))

    def load_image_folder(self):
        """Загружает папку с изображениями."""
        folder = QFileDialog.getExistingDirectory(
            self, "Выберите папку с изображениями",
            str(Path("."))
        )
        if not folder:
            return
        self._start_loading(Path(folder), is_folder=True)

    def _start_loading(self, path: Path, is_folder: bool = False):
        """Запускает фоновую загрузку данных."""
        self.loading_progress.setVisible(True)
        self.loading_status.setVisible(True)
        self.loading_status.setText(f"Загружаю: {path.name}...")
        self.load_btn.setEnabled(False)
        self.load_folder_btn.setEnabled(False)

        if is_folder:
            # Загрузка папки изображений
            self._load_thread = DatasetLoaderThread(path)
        else:
            self._load_thread = DatasetLoaderThread(path)

        self._load_thread.finished_signal.connect(self.on_dataset_loaded)
        self._load_thread.error_signal.connect(self.on_dataset_error)
        self._load_thread.progress_signal.connect(self._on_load_progress)
        self._load_thread.start()

    def _on_load_progress(self, msg: str):
        """Обновляет статус загрузки."""
        self.loading_status.setText(msg)

    def on_dataset_loaded(self, dataset):
        """Вызывается при успешной загрузке датасета."""
        self.loading_progress.setVisible(False)
        self.loading_status.setVisible(False)
        self.load_btn.setEnabled(True)
        self.load_folder_btn.setEnabled(True)

        self.dataset = dataset
        self.shared_state["dataset"] = dataset

        # Обновляем метаданные
        meta = self._get_meta(dataset)
        self._update_meta_display(meta)

        # Отчёт о качестве
        self._show_quality_report(dataset, meta)

        # Предпросмотр
        self._update_preview(dataset, meta)

        # Сбрасываем информацию о разбиении
        self.split_info_label.setVisible(False)

        # Активируем кнопку словаря для текста
        data_type = self._get_data_type(meta)
        vocab = self._get_vocab(dataset)
        self.vocab_btn.setEnabled(data_type == "text" and bool(vocab))

        # Обновляем подсказки
        self.refresh_hints()

        # Отправляем сигнал
        self.dataset_updated.emit(dataset)

        logger.info(f"Датасет загружен: {meta.get('num_samples', '?')} примеров, тип: {data_type}")

    def on_dataset_error(self, error_msg: str):
        """Вызывается при ошибке загрузки."""
        self.loading_progress.setVisible(False)
        self.loading_status.setVisible(False)
        self.load_btn.setEnabled(True)
        self.load_folder_btn.setEnabled(True)
        QMessageBox.critical(self, "Ошибка загрузки", f"Не удалось прочитать файл:\n{error_msg}")

    # ============================================================
    #  РАЗБИЕНИЕ ДАННЫХ
    # ============================================================
    def split_dataset(self):
        """
        Применяет разбиение на train/val/test и строит словарь.
        ТЗ: "Словарь строится ТОЛЬКО на обучающей выборке"
        """
        if self.dataset is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите датасет.")
            return

        train_ratio = self.train_spin.value() / 100.0
        val_ratio = self.val_spin.value() / 100.0
        test_ratio = self.test_spin.value() / 100.0

        # Проверка суммы
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 0.001:
            QMessageBox.warning(
                self, "Ошибка логики",
                f"Сумма процентов должна равняться 100%.\n"
                f"Сейчас: {self.train_spin.value()} + {self.val_spin.value()} + {self.test_spin.value()} = {int(total*100)}%"
            )
            return

        # Проверка что есть хотя бы по одному примеру
        meta = self._get_meta(self.dataset)
        num_samples = meta.get("num_samples", 0)
        if num_samples < 3:
            QMessageBox.warning(
                self, "Слишком мало данных",
                f"В датасете всего {num_samples} примеров.\n"
                f"Для разбиения на 3 части нужно минимум 3 примера."
            )
            return

        try:
            max_vocab = (
                self.vocab_limit_spin.value()
                if self.vocab_limit_check.isChecked()
                else None
            )

            # Используем DatasetLoader если доступен
            if DATA_LOADER_AVAILABLE and DatasetLoader:
                self.dataset = DatasetLoader.split_dataset(
                    self.dataset, train_ratio, val_ratio, test_ratio,
                    max_vocab_size=max_vocab
                )
            else:
                # Fallback: простое разбиение
                self.dataset = self._fallback_split(
                    self.dataset, train_ratio, val_ratio, test_ratio
                )

            self.shared_state["dataset"] = self.dataset

            # Обновляем отображение
            train_n = len(self._get_split_data("train_inputs"))
            val_n = len(self._get_split_data("val_inputs"))
            test_n = len(self._get_split_data("test_inputs"))
            vocab = self._get_vocab(self.dataset)
            data_type = self._get_data_type(meta)

            if data_type == "text" and vocab:
                vocab_note = f" | Словарь: {len(vocab)} токенов"
            else:
                vocab_note = ""

            self.split_info_label.setText(
                f"✅ Разбиение применено:\n"
                f"  • Train = {train_n} примеров (для обучения)\n"
                f"  • Val = {val_n} примеров (для проверки во время учёбы)\n"
                f"  • Test = {test_n} примеров (для финального экзамена)\n"
                f"{vocab_note}\n"
                f"Словарь построен только на обучающей выборке."
            )
            self.split_info_label.setVisible(True)

            # Обновляем метаданные
            self._update_meta_display(self._get_meta(self.dataset))
            self.vocab_btn.setEnabled(data_type == "text" and bool(vocab))

            # Отправляем сигнал
            self.dataset_updated.emit(self.dataset)

            logger.info(
                f"Разбиение применено: {train_n}/{val_n}/{test_n}. "
                f"Словарь: {len(vocab) if vocab else 'N/A'} токенов"
            )

        except Exception as e:
            logger.error(f"Ошибка разбиения: {e}\n{traceback.format_exc()}")
            QMessageBox.critical(self, "Ошибка разбиения", f"Не удалось разбить датасет:\n{e}")

    def _fallback_split(self, dataset, train_r, val_r, test_r):
        """Простое разбиение если DatasetLoader недоступен."""
        import numpy as np

        inputs = dataset.get("raw_inputs", [])
        outputs = dataset.get("raw_outputs", [])
        if not inputs or not outputs:
            raise ValueError("Нет данных для разбиения.")

        total = len(inputs)
        indices = np.random.permutation(total)
        train_end = int(total * train_r)
        val_end = train_end + int(total * val_r)

        dataset["train_inputs"] = [inputs[i] for i in indices[:train_end]]
        dataset["train_outputs"] = [outputs[i] for i in indices[:train_end]]
        dataset["val_inputs"] = [inputs[i] for i in indices[train_end:val_end]]
        dataset["val_outputs"] = [outputs[i] for i in indices[train_end:val_end]]
        dataset["test_inputs"] = [inputs[i] for i in indices[val_end:]]
        dataset["test_outputs"] = [outputs[i] for i in indices[val_end:]]

        # Строим словарь для текста
        meta = dataset.get("meta", {})
        if meta.get("type") == "text":
            if TOKENIZER_AVAILABLE and TextTokenizer:
                tokenizer = TextTokenizer()
                max_vocab = (
                    self.vocab_limit_spin.value()
                    if self.vocab_limit_check.isChecked()
                    else None
                )
                train_texts = dataset["train_inputs"] + dataset["train_outputs"]
                tokenizer.build_vocab(train_texts, max_vocab_size=max_vocab)
                dataset["vocab"] = tokenizer.vocab
                meta["vocab_size"] = len(tokenizer.vocab)
            else:
                # Простой словарь
                all_chars = set()
                for text in dataset["train_inputs"] + dataset["train_outputs"]:
                    all_chars.update(str(text))
                vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
                for i, ch in enumerate(sorted(all_chars), start=4):
                    vocab[ch] = i
                dataset["vocab"] = vocab
                meta["vocab_size"] = len(vocab)

        return dataset

    def _on_vocab_limit_toggled(self, checked: bool):
        """Включает/выключает спин ограничения словаря."""
        self.vocab_limit_spin.setEnabled(checked)

    # ============================================================
    #  РЕДАКТОР СЛОВАРЯ
    # ============================================================
    def _show_vocab_editor(self):
        """
        Открывает модальное окно редактора словаря.
        ТЗ: "Кнопка '👁 Показать словарь' → таблица: символ | индекс | частота"
        """
        vocab = self._get_vocab(self.dataset)
        if not vocab:
            QMessageBox.information(self, "Словарь пуст", "Словарь ещё не построен. Примените разбиение.")
            return

        # Собираем тексты для подсчёта частот
        texts = []
        if self.dataset:
            texts.extend(self._get_split_data("train_inputs") or [])
            texts.extend(self._get_split_data("train_outputs") or [])
            if not texts:
                texts.extend(self.dataset.get("raw_inputs", []))
                texts.extend(self.dataset.get("raw_outputs", []))

        # Используем внешний редактор если доступен, иначе встроенный
        if EXTERNAL_VOCAB_EDITOR and ExternalVocabEditor:
            dialog = ExternalVocabEditor(vocab, texts, self)
        else:
            dialog = VocabEditorDialog(vocab, texts, self)

        if dialog.exec_() == QDialog.Accepted:
            new_vocab = dialog.get_modified_vocab()
            self._apply_vocab_changes(new_vocab)

    def _apply_vocab_changes(self, new_vocab: Dict[str, int]):
        """Применяет изменения словаря к датасету."""
        if not self.dataset:
            return

        old_vocab = self._get_vocab(self.dataset)
        removed = set(old_vocab.keys()) - set(new_vocab.keys())
        added = set(new_vocab.keys()) - set(old_vocab.keys())

        if removed:
            logger.info(f"Удалены символы из словаря: {removed}")
        if added:
            logger.info(f"Добавлены символы в словарь: {added}")

        # Обновляем словарь в датасете
        if isinstance(self.dataset, dict):
            self.dataset["vocab"] = new_vocab
            if "meta" in self.dataset:
                self.dataset["meta"]["vocab_size"] = len(new_vocab)
        elif CONTRACTS_AVAILABLE and isinstance(self.dataset, DatasetContainer):
            self.dataset.vocab = new_vocab
            self.dataset.meta.vocab_size = len(new_vocab)

        # Обновляем отображение
        self.vocab_label.setText(f"{len(new_vocab)} токенов")
        self.dataset_updated.emit(self.dataset)

        QMessageBox.information(
            self, "Словарь обновлён",
            f"Словарь обновлён: {len(new_vocab)} токенов.\n"
            f"Удалено: {len(removed)}, Добавлено: {len(added)}.\n"
            f"⚠️ Пересоберите модель во вкладке «Архитектура» если словарь изменился."
        )

    # ============================================================
    #  ОТОБРАЖЕНИЕ МЕТАДАННЫХ
    # ============================================================
    def _update_meta_display(self, meta: Dict):
        """Обновляет отображение метаданных."""
        if not meta:
            return

        data_type = meta.get("type", meta.get("data_type", "unknown"))
        task_type = meta.get("task_type", "—")
        num_samples = meta.get("num_samples", 0)
        input_dim = meta.get("input_dim", "—")
        vocab_size = meta.get("vocab_size", 0)
        image_shape = meta.get("image_shape")
        audio_rate = meta.get("audio_sample_rate")

        # Тип данных — красивое отображение
        type_names = {
            "text": "📝 Текст",
            "numeric": "🔢 Числа",
            "image": "🖼 Изображения",
            "audio": "🎵 Звук",
            "graph": "🔗 Графы",
            "time_series": "📈 Временные ряды",
            "multimodal": "🌐 Мультимодальные",
        }
        self.type_label.setText(type_names.get(str(data_type), str(data_type)))
        self.task_type_label.setText(str(task_type))
        self.samples_label.setText(f"{num_samples:,}")

        # Размерность входа
        if isinstance(input_dim, (tuple, list)):
            self.input_dim_label.setText(" × ".join(str(d) for d in input_dim))
        else:
            self.input_dim_label.setText(str(input_dim))

        # Словарь
        if data_type == "text" and vocab_size:
            self.vocab_label.setText(f"{vocab_size} токенов")
        elif data_type == "text":
            self.vocab_label.setText("Будет построен при разбиении")
        else:
            self.vocab_label.setText("Не применимо")

        # Изображения
        if image_shape:
            if isinstance(image_shape, (tuple, list)) and len(image_shape) >= 2:
                self.image_shape_label.setText(f"{image_shape[0]}×{image_shape[1]} px")
            else:
                self.image_shape_label.setText(str(image_shape))
        else:
            self.image_shape_label.setText("—")

        # Аудио
        if audio_rate:
            channels = meta.get("audio_channels", 1)
            self.audio_info_label.setText(f"{audio_rate} Гц, {channels} канал(а)")
        else:
            self.audio_info_label.setText("—")

    # ============================================================
    #  ОТЧЁТ О КАЧЕСТВЕ ДАННЫХ
    # ============================================================
    def _show_quality_report(self, dataset, meta: Dict):
        """
        ТЗ: "Отчёт о качестве данных"
        Показывает: сколько строк удалено, какие аномалии найдены.
        """
        report_lines = []
        data_type = self._get_data_type(meta)
        num_samples = meta.get("num_samples", 0)

        report_lines.append(f"📊 Загружено примеров: {num_samples:,}")

        # Проверяем на аномалии
        raw_inputs = self.dataset.get("raw_inputs", []) if isinstance(dataset, dict) else []
        raw_outputs = self.dataset.get("raw_outputs", []) if isinstance(dataset, dict) else []

        if raw_inputs:
            empty_count = sum(1 for x in raw_inputs if x is None or str(x).strip() == "")
            if empty_count > 0:
                report_lines.append(f"⚠️ Обнаружено {empty_count} пустых записей")

            # Проверяем длины строк для текста
            if data_type == "text":
                lengths = [len(str(x)) for x in raw_inputs if x]
                if lengths:
                    max_len = max(lengths)
                    avg_len = sum(lengths) / len(lengths)
                    report_lines.append(f"📏 Длина входа: мин {min(lengths)}, макс {max_len}, средн {avg_len:.0f}")
                    if max_len > 500:
                        report_lines.append(f"⚠️ Очень длинные строки (>500 символов). Рекомендуется увеличить макс. длину.")

            # Проверяем на NaN для чисел
            if data_type == "numeric":
                try:
                    import numpy as np
                    arr = np.array(raw_inputs[:100], dtype=float)
                    nan_count = int(np.isnan(arr).sum())
                    inf_count = int(np.isinf(arr).sum())
                    if nan_count > 0:
                        report_lines.append(f"🚨 Обнаружено {nan_count} значений NaN")
                    if inf_count > 0:
                        report_lines.append(f"🚨 Обнаружено {inf_count} значений Infinity")
                except (ValueError, TypeError):
                    report_lines.append("⚠️ Числовые данные содержат нечисловые значения")

        # Проверяем баланс классов для изображений
        if data_type == "image":
            report_lines.append("🖼 Тип данных: изображения. Проверьте баланс классов.")

        # Проверяем наличие выходов
        if raw_inputs and raw_outputs and len(raw_inputs) != len(raw_outputs):
            report_lines.append(
                f"🚨 ВНИМАНИЕ: Количество входов ({len(raw_inputs)}) "
                f"не совпадает с количеством выходов ({len(raw_outputs)})!"
            )

        report_text = "\n".join(report_lines)
        self.quality_label.setText(report_text)
        self.quality_group.setVisible(True)

    # ============================================================
    #  ПРЕДПРОСМОТР ДАННЫХ
    # ============================================================
    def _update_preview(self, dataset, meta: Dict):
        """Обновляет предпросмотр в зависимости от типа данных."""
        data_type = self._get_data_type(meta)

        if data_type == "image":
            self.preview_table.setVisible(False)
            self.audio_play_btn.setVisible(False)
            self._update_preview_for_images(dataset, meta)
        elif data_type == "audio":
            self.preview_table.setVisible(False)
            self.image_grid_widget.setVisible(False)
            self.audio_play_btn.setVisible(True)
        else:
            # Текст или числа — таблица
            self.image_grid_widget.setVisible(False)
            self.audio_play_btn.setVisible(False)
            self.preview_table.setVisible(True)
            self._update_preview_table(dataset, meta)

    def _update_preview_table(self, dataset, meta: Dict):
        """Предпросмотр для текста и чисел: таблица вход/выход."""
        raw_inputs = self._get_raw_data("raw_inputs", dataset)
        raw_outputs = self._get_raw_data("raw_outputs", dataset)

        n = min(len(raw_inputs), 10)
        self.preview_table.setRowCount(n)

        for i in range(n):
            x_val = str(raw_inputs[i]) if i < len(raw_inputs) else ""
            y_val = str(raw_outputs[i]) if i < len(raw_outputs) else ""

            # Обрезаем очень длинные строки
            if len(x_val) > 100:
                x_val = x_val[:100] + "..."
            if len(y_val) > 100:
                y_val = y_val[:100] + "..."

            self.preview_table.setItem(i, 0, QTableWidgetItem(x_val))
            self.preview_table.setItem(i, 1, QTableWidgetItem(y_val))

    def _update_preview_for_images(self, dataset, meta: Dict):
        """
        ТЗ: "Для изображений: сетка предпросмотра"
        Показывает сетку 5×5 изображений.
        """
        # Очищаем старую сетку
        while self.image_grid_layout.count():
            item = self.image_grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        raw_inputs = self._get_raw_data("raw_inputs", dataset)
        raw_outputs = self._get_raw_data("raw_outputs", dataset)

        n = min(len(raw_inputs), 25)  # 5×5 = 25
        cols = 5

        for i in range(n):
            row = i // cols
            col = i % cols

            cell_widget = QWidget()
            cell_layout = QVBoxLayout(cell_widget)
            cell_layout.setContentsMargins(2, 2, 2, 2)

            # Пытаемся отобразить изображение
            try:
                img_data = raw_inputs[i]
                if isinstance(img_data, str):
                    # Путь к файлу
                    pixmap = QPixmap(str(img_data))
                    if not pixmap.isNull():
                        pixmap = pixmap.scaled(60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        img_label = QLabel()
                        img_label.setPixmap(pixmap)
                        img_label.setAlignment(Qt.AlignCenter)
                        cell_layout.addWidget(img_label)
                elif hasattr(img_data, 'numpy'):
                    # Тензор numpy → QImage
                    import numpy as np
                    arr = img_data.numpy() if callable(getattr(img_data, 'numpy', None)) else np.array(img_data)
                    if arr.ndim == 3 and arr.shape[0] <= 4:
                        arr = arr.transpose(1, 2, 0)
                    if arr.ndim == 2:
                        arr = np.stack([arr] * 3, axis=-1)
                    arr = ((arr - arr.min()) / max(arr.max() - arr.min(), 1e-8) * 255).astype(np.uint8)
                    h, w = arr.shape[:2]
                    channels = arr.shape[2] if arr.ndim == 3 else 1
                    if channels == 1:
                        img = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
                    elif channels == 3:
                        img = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888)
                    else:
                        img = QImage(arr.data, w, h, w * 4, QImage.Format_RGBA8888)
                    pixmap = QPixmap.fromImage(img.scaled(60, 60, Qt.KeepAspectRatio))
                    img_label = QLabel()
                    img_label.setPixmap(pixmap)
                    img_label.setAlignment(Qt.AlignCenter)
                    cell_layout.addWidget(img_label)
            except Exception:
                placeholder = QLabel("🖼")
                placeholder.setAlignment(Qt.AlignCenter)
                placeholder.setStyleSheet("font-size: 24px;")
                cell_layout.addWidget(placeholder)

            # Метка (класс)
            if i < len(raw_outputs):
                label_text = str(raw_outputs[i])
                if len(label_text) > 15:
                    label_text = label_text[:15] + "..."
                lbl = QLabel(label_text)
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setStyleSheet("font-size: 10px; color: #4a88c7;")
                cell_layout.addWidget(lbl)

            self.image_grid_layout.addWidget(cell_widget, row, col)

        self.image_grid_widget.setVisible(True)

    def _play_audio_sample(self):
        """
        ТЗ: "Для звука: кнопка воспроизведения"
        Воспроизводит первый аудио-пример.
        """
        raw_inputs = self._get_raw_data("raw_inputs", self.dataset)
        if not raw_inputs:
            QMessageBox.information(self, "Нет данных", "Аудио данные не загружены.")
            return

        try:
            import numpy as np
            audio_data = raw_inputs[0]

            if isinstance(audio_data, (list, np.ndarray)):
                # Воспроизводим через простой бип (заглушка)
                meta = self._get_meta(self.dataset)
                sample_rate = meta.get("audio_sample_rate", 16000)
                duration = len(audio_data) / max(sample_rate, 1)
                QMessageBox.information(
                    self, "🎵 Аудио пример",
                    f"Длительность: {duration:.2f} сек\n"
                    f"Частота: {sample_rate} Гц\n"
                    f"Сэмплов: {len(audio_data)}\n\n"
                    f"(Воспроизведение будет доступно в следующей версии)"
                )
            elif isinstance(audio_data, str):
                # Путь к файлу
                QMessageBox.information(
                    self, "🎵 Аудио файл",
                    f"Файл: {audio_data}\n"
                    f"(Воспроизведение будет доступно в следующей версии)"
                )
        except Exception as e:
            logger.error(f"Ошибка воспроизведения аудио: {e}")
            QMessageBox.warning(self, "Ошибка", f"Не удалось воспроизвести аудио:\n{e}")

    # ============================================================
    #  ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ============================================================
    def _get_meta(self, dataset) -> Dict:
        """Извлекает метаданные из датасета (dict или DatasetContainer)."""
        if dataset is None:
            return {}
        if isinstance(dataset, dict):
            return dataset.get("meta", {})
        elif CONTRACTS_AVAILABLE and isinstance(dataset, DatasetContainer):
            meta = dataset.meta
            if isinstance(meta, DatasetMeta):
                return meta.__dict__ if hasattr(meta, '__dict__') else {}
            return {}
        return {}

    def _get_data_type(self, meta: Dict) -> str:
        """Извлекает тип данных из метаданных."""
        if not meta:
            return "unknown"
        dt = meta.get("type", meta.get("data_type", "unknown"))
        # DataType enum → строка
        if hasattr(dt, 'value'):
            return dt.value
        return str(dt)

    def _get_vocab(self, dataset) -> Dict[str, int]:
        """Извлекает словарь из датасета."""
        if dataset is None:
            return {}
        if isinstance(dataset, dict):
            return dataset.get("vocab", {})
        elif CONTRACTS_AVAILABLE and isinstance(dataset, DatasetContainer):
            return dataset.vocab or {}
        return {}

    def _get_raw_data(self, key: str, dataset=None) -> List:
        """Извлекает сырые данные по ключу."""
        ds = dataset if dataset is not None else self.dataset
        if ds is None:
            return []
        if isinstance(ds, dict):
            return ds.get(key, [])
        elif CONTRACTS_AVAILABLE and isinstance(ds, DatasetContainer):
            return getattr(ds, key, []) or []
        return []

    def _get_split_data(self, key: str) -> List:
        """Извлекает данные разбиения по ключу."""
        return self._get_raw_data(key, self.dataset)

    # ============================================================
    #  ПУБЛИЧНЫЕ МЕТОДЫ ДЛЯ ДРУГИХ ПАНЕЛЕЙ
    # ============================================================
    def refresh(self):
        """Вызывается при переключении вкладок для обновления состояния."""
        dataset = self.shared_state.get("dataset")
        if dataset is not None and dataset is not self.dataset:
            self.dataset = dataset
            meta = self._get_meta(dataset)
            self._update_meta_display(meta)
            self._update_preview(dataset, meta)
        self.refresh_hints()

    def get_dataset(self):
        """Возвращает текущий датасет."""
        return self.dataset

    def is_split_applied(self) -> bool:
        """Проверяет, применено ли разбиение."""
        if self.dataset is None:
            return False
        if isinstance(self.dataset, dict):
            return "train_inputs" in self.dataset and len(self.dataset.get("train_inputs", [])) > 0
        elif CONTRACTS_AVAILABLE and isinstance(self.dataset, DatasetContainer):
            return bool(self.dataset.train_inputs)
        return False