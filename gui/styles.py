"""
gui/styles.py
=============
Единая тема оформления OracleAI Studio v2.0.
Все стили проекта собраны в одном файле.

Версия формата: 2.0
Зависимости: нет (только строковые константы)

ПРАВИЛА:
  1. Никаких импортов из других модулей проекта.
  2. Все цвета — через константы в начале файла.
  3. Панели применяют MODERN_THEME в main_window.
  4. Дополнительные стили (COMPONENT_*) применяются точечно через
     widget.setStyleSheet(COMPONENT_...) при необходимости.
  5. Все новые виджеты получают стили автоматически из MODERN_THEME,
     если используют стандартные классы PyQt5.
"""

__version__ = "2.0.0"

# ============================================================
#  ЦВЕТОВАЯ ПАЛИТРА (единая для всего проекта)
# ============================================================
class Colors:
    """Централизованная палитра. Использовать в f-строках стилей."""
    # Фоны
    BG_MAIN       = "#2b2b2b"   # Основной фон окна
    BG_PANEL      = "#313335"   # Фон панелей и полей ввода
    BG_DARK       = "#1e1e1e"   # Фон текстовых областей, консоли
    BG_HOVER      = "#3a3c3e"   # Наведение
    BG_ACTIVE     = "#36414f"   # Активные кнопки
    BG_DISABLED   = "#2d2d2d"   # Неактивные элементы

    # Текст
    TEXT_MAIN     = "#a9b7c6"   # Основной текст
    TEXT_DIM      = "#777777"   # Приглушённый текст
    TEXT_BRIGHT   = "#d4d4d4"   # Яркий текст
    TEXT_ACCENT   = "#cc7832"   # Оранжевый акцент (заголовки, вкладки)

    # Акценты
    BLUE          = "#4a88c7"   # Синий: ссылки, фокус, информация
    GREEN         = "#a6e3a1"   # Зелёный: успех, прогресс
    GREEN_DARK    = "#385a3a"   # Тёмно-зелёный: кнопка Старт
    YELLOW        = "#e5c07b"   # Жёлтый: предупреждение, плато
    YELLOW_DARK   = "#635030"   # Тёмно-жёлтый: кнопка Пауза
    RED           = "#e06c75"   # Красный: ошибка, авария
    RED_DARK      = "#6a3636"   # Тёмно-красный: кнопка Стоп
    ORANGE        = "#cc7832"   # Оранжевый: акцент
    PURPLE        = "#c678dd"   # Фиолетовый: эксперименты, заглушки

    # Границы
    BORDER        = "#555555"
    BORDER_LIGHT  = "#4c5052"
    BORDER_FOCUS  = "#4a88c7"

    # Индикатор сложности (светофор)
    COMPLEXITY_EASY    = "#a6e3a1"  # 🟢
    COMPLEXITY_MEDIUM  = "#e5c07b"  # 🟡
    COMPLEXITY_HARD    = "#e06c75"  # 🔴
    COMPLEXITY_EXTREME = "#c678dd"  # 💀

    # Куратор
    CURATOR_INFO    = "#4a88c7"  # Информация
    CURATOR_SUCCESS = "#a6e3a1"  # Хорошо
    CURATOR_WARNING = "#e5c07b"  # Плато, ожидание
    CURATOR_DANGER  = "#e06c75"  # Авария, переобучение
    CURATOR_FUN     = "#c678dd"  # Факты, шутки

    # Подсказки
    HINT_STATIC = "#888888"  # Статическая подсказка
    HINT_AI     = "#c678dd"  # ИИ-подсказка
    HINT_WARN   = "#e5c07b"  # Подсказка-предупреждение

    # Живые примеры
    EXAMPLE_CORRECT = "#a6e3a1"
    EXAMPLE_WRONG   = "#e06c75"


# ============================================================
#  ОСНОВНАЯ ТЕМА (применяется ко всему приложению через
#  app.setStyleSheet(MODERN_THEME) в main_window.py)
# ============================================================
MODERN_THEME = f"""
/* ==================== ОБЩИЕ ==================== */
QWidget {{
    background-color: {Colors.BG_MAIN};
    color: {Colors.TEXT_MAIN};
    font-family: 'Segoe UI', 'Consolas', Arial, sans-serif;
    font-size: 13px;
}}

QMainWindow {{
    background-color: {Colors.BG_MAIN};
}}

QToolTip {{
    background-color: {Colors.BG_DARK};
    color: {Colors.TEXT_BRIGHT};
    border: 1px solid {Colors.BORDER};
    padding: 6px 10px;
    font-size: 12px;
    max-width: 400px;
}}

/* ==================== ГРУППЫ ==================== */
QGroupBox {{
    border: 1px solid {Colors.BORDER};
    margin-top: 1.5ex;
    font-weight: bold;
    border-radius: 0px;
    padding-top: 10px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
    color: {Colors.ORANGE};
}}

/* ==================== КНОПКИ ==================== */
QPushButton {{
    background-color: {Colors.BG_ACTIVE};
    color: {Colors.TEXT_MAIN};
    border: 1px solid {Colors.BORDER_LIGHT};
    padding: 6px 12px;
    border-radius: 0px;
    min-height: 24px;
}}
QPushButton:hover {{
    background-color: {Colors.BORDER_LIGHT};
    border: 1px solid #5c6062;
}}
QPushButton:pressed {{
    background-color: #2d3540;
}}
QPushButton:disabled {{
    background-color: {Colors.BG_DISABLED};
    color: {Colors.TEXT_DIM};
}}

/* Кнопки управления обучением */
QPushButton#StartBtn {{
    background-color: {Colors.GREEN_DARK};
    border-color: #4a7a4c;
    font-weight: bold;
}}
QPushButton#StartBtn:hover {{ background-color: #4a7a4c; }}
QPushButton#StopBtn {{
    background-color: {Colors.RED_DARK};
    border-color: #8f4646;
    font-weight: bold;
}}
QPushButton#StopBtn:hover {{ background-color: #8f4646; }}
QPushButton#PauseBtn {{
    background-color: {Colors.YELLOW_DARK};
    border-color: #856a3e;
}}
QPushButton#PauseBtn:hover {{ background-color: #856a3e; }}

/* Кнопки-акценты */
QPushButton#AccentBtn {{
    background-color: #2a4a6b;
    border-color: {Colors.BLUE};
    color: {Colors.TEXT_BRIGHT};
    font-weight: bold;
}}
QPushButton#AccentBtn:hover {{ background-color: #35608a; }}

QPushButton#DangerBtn {{
    background-color: {Colors.RED_DARK};
    border-color: #8f4646;
}}
QPushButton#DangerBtn:hover {{ background-color: #8f4646; }}

QPushButton#SuccessBtn {{
    background-color: {Colors.GREEN_DARK};
    border-color: #4a7a4c;
}}
QPushButton#SuccessBtn:hover {{ background-color: #4a7a4c; }}

/* Кнопка генерации случайного имени */
QPushButton#DiceBtn {{
    background-color: #3d3552;
    border-color: #5a4a7a;
    font-size: 16px;
}}
QPushButton#DiceBtn:hover {{ background-color: #5a4a7a; }}

/* ==================== ПОЛЯ ВВОДА ==================== */
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background-color: {Colors.BG_PANEL};
    border: 1px solid {Colors.BORDER};
    padding: 4px 8px;
    border-radius: 0px;
    color: {Colors.TEXT_MAIN};
    min-height: 22px;
}}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{
    border: 1px solid {Colors.BORDER_FOCUS};
}}
QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QLineEdit:disabled {{
    background-color: {Colors.BG_DISABLED};
    color: {Colors.TEXT_DIM};
}}
QComboBox::drop-down {{
    border-left: 1px solid {Colors.BORDER};
    width: 20px;
}}
QComboBox::down-arrow {{
    width: 10px;
    height: 10px;
}}
QComboBox QAbstractItemView {{
    background-color: {Colors.BG_PANEL};
    color: {Colors.TEXT_MAIN};
    border: 1px solid {Colors.BORDER};
    selection-background-color: {Colors.BG_ACTIVE};
}}

/* Поле ввода имени модели / проекта */
QLineEdit#NameInput {{
    font-size: 16px;
    padding: 8px;
    font-weight: bold;
    color: {Colors.TEXT_BRIGHT};
}}

/* ==================== ТЕКСТОВЫЕ ОБЛАСТИ ==================== */
QTextEdit, QPlainTextEdit {{
    background-color: {Colors.BG_DARK};
    border: 1px solid {Colors.BORDER};
    color: {Colors.TEXT_MAIN};
    padding: 4px;
}}
QTextEdit:read-only {{
    background-color: #242424;
}}

/* ==================== ТАБЛИЦЫ ==================== */
QTableWidget {{
    background-color: {Colors.BG_MAIN};
    alternate-background-color: {Colors.BG_PANEL};
    gridline-color: {Colors.BORDER};
    border: 1px solid {Colors.BORDER};
    border-radius: 0px;
}}
QTableWidget::item:selected {{
    background-color: #2a4a6b;
    color: {Colors.TEXT_BRIGHT};
}}
QHeaderView::section {{
    background-color: {Colors.BG_PANEL};
    padding: 4px;
    border: 1px solid {Colors.BORDER};
    color: {Colors.ORANGE};
    font-weight: bold;
}}

/* ==================== ВКЛАДКИ ==================== */
QTabWidget::pane {{
    border: 1px solid {Colors.BORDER};
    background-color: {Colors.BG_MAIN};
    border-radius: 0px;
}}
QTabBar::tab {{
    background-color: {Colors.BG_PANEL};
    color: {Colors.TEXT_DIM};
    padding: 8px 16px;
    margin-right: 1px;
    border: 1px solid {Colors.BORDER};
    border-bottom: none;
    border-radius: 0px;
}}
QTabBar::tab:selected {{
    background-color: {Colors.BG_MAIN};
    color: {Colors.ORANGE};
    border-top: 2px solid {Colors.ORANGE};
}}
QTabBar::tab:hover:!selected {{
    background-color: {Colors.BG_HOVER};
    color: {Colors.TEXT_MAIN};
}}
QTabBar::tab:disabled {{
    color: #555555;
}}

/* ==================== ПРОГРЕСС-БАРЫ ==================== */
QProgressBar {{
    border: 1px solid {Colors.BORDER};
    border-radius: 0px;
    text-align: center;
    color: {Colors.TEXT_MAIN};
    font-weight: bold;
    background-color: {Colors.BG_PANEL};
    min-height: 20px;
}}
QProgressBar::chunk {{
    background-color: {Colors.BLUE};
}}

/* ==================== СКРОЛЛ ==================== */
QScrollArea {{
    border: none;
    background-color: transparent;
}}
QScrollBar:vertical {{
    background-color: {Colors.BG_MAIN};
    width: 14px;
    margin: 0px;
}}
QScrollBar::handle:vertical {{
    background-color: {Colors.BORDER};
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background-color: #666666;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar:horizontal {{
    background-color: {Colors.BG_MAIN};
    height: 14px;
    margin: 0px;
}}
QScrollBar::handle:horizontal {{
    background-color: {Colors.BORDER};
    min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{
    background-color: #666666;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}

/* ==================== ЧЕКБОКСЫ И РАДИО ==================== */
QCheckBox {{
    spacing: 6px;
    color: {Colors.TEXT_MAIN};
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {Colors.BORDER};
    background-color: {Colors.BG_PANEL};
}}
QCheckBox::indicator:checked {{
    background-color: {Colors.BLUE};
    border-color: {Colors.BORDER_FOCUS};
}}
QCheckBox::indicator:hover {{
    border-color: {Colors.BORDER_FOCUS};
}}
QCheckBox:disabled {{
    color: {Colors.TEXT_DIM};
}}
QCheckBox::indicator:disabled {{
    background-color: {Colors.BG_DISABLED};
}}

QRadioButton {{
    spacing: 6px;
    color: {Colors.TEXT_MAIN};
}}
QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {Colors.BORDER};
    border-radius: 8px;
    background-color: {Colors.BG_PANEL};
}}
QRadioButton::indicator:checked {{
    background-color: {Colors.BLUE};
    border-color: {Colors.BORDER_FOCUS};
}}
QRadioButton:disabled {{
    color: {Colors.TEXT_DIM};
}}

/* ==================== СЛАЙДЕРЫ ==================== */
QSlider::groove:horizontal {{
    height: 6px;
    background-color: {Colors.BG_PANEL};
    border: 1px solid {Colors.BORDER};
}}
QSlider::handle:horizontal {{
    background-color: {Colors.BLUE};
    width: 14px;
    margin: -4px 0;
}}
QSlider::handle:horizontal:hover {{
    background-color: #6aa3d7;
}}

/* ==================== СПИСОК (QListWidget) ==================== */
QListWidget {{
    background-color: {Colors.BG_PANEL};
    border: 1px solid {Colors.BORDER};
    color: {Colors.TEXT_MAIN};
}}
QListWidget::item {{
    padding: 4px;
}}
QListWidget::item:selected {{
    background-color: #2a4a6b;
    color: {Colors.TEXT_BRIGHT};
}}
QListWidget::item:hover {{
    background-color: {Colors.BG_HOVER};
}}

/* ==================== ДЕРЕВО (QTreeWidget) ==================== */
QTreeWidget {{
    background-color: {Colors.BG_PANEL};
    border: 1px solid {Colors.BORDER};
    color: {Colors.TEXT_MAIN};
}}
QTreeWidget::item:selected {{
    background-color: #2a4a6b;
}}
QHeaderView::section {{
    background-color: {Colors.BG_PANEL};
    color: {Colors.ORANGE};
    padding: 4px;
    border: 1px solid {Colors.BORDER};
    font-weight: bold;
}}

/* ==================== МЕНЮ ==================== */
QMenuBar {{
    background-color: {Colors.BG_MAIN};
    color: {Colors.TEXT_MAIN};
    border-bottom: 1px solid {Colors.BORDER};
}}
QMenuBar::item:selected {{
    background-color: {Colors.BG_ACTIVE};
}}
QMenu {{
    background-color: {Colors.BG_PANEL};
    border: 1px solid {Colors.BORDER};
    color: {Colors.TEXT_MAIN};
}}
QMenu::item:selected {{
    background-color: {Colors.BG_ACTIVE};
}}
QMenu::separator {{
    height: 1px;
    background-color: {Colors.BORDER};
    margin: 4px 0;
}}

/* ==================== СТАТУС-БАР ==================== */
QStatusBar {{
    background-color: {Colors.BG_MAIN};
    color: {Colors.TEXT_DIM};
    border-top: 1px solid {Colors.BORDER};
}}

/* ==================== РАЗДЕЛИТЕЛИ ==================== */
QSplitter::handle {{
    background-color: {Colors.BORDER};
    width: 2px;
    height: 2px;
}}

/* ==================== ФРЕЙМЫ ==================== */
QFrame {{
    border: none;
}}
QFrame[frameShape="4"], QFrame[frameShape="5"] {{
    color: {Colors.BORDER};
    max-height: 1px;
}}

/* ==================== ДИАЛОГИ ==================== */
QMessageBox {{
    background-color: {Colors.BG_MAIN};
}}
QFileDialog {{
    background-color: {Colors.BG_MAIN};
}}
"""


# ============================================================
#  ДОПОЛНИТЕЛЬНЫЕ СТИЛИ ДЛЯ СПЕЦИФИЧНЫХ КОМПОНЕНТОВ
#  Применяются точечно: widget.setStyleSheet(STYLE)
# ============================================================

# --- Карточка модели (вкладка Обучение / Проект) ---
MODEL_CARD_STYLE = f"""
QGroupBox {{
    border: 2px solid {Colors.BLUE};
    border-radius: 4px;
    padding: 10px;
    background-color: #242830;
}}
QGroupBox::title {{
    color: {Colors.BLUE};
    font-size: 14px;
}}
QLabel {{
    font-size: 13px;
    padding: 2px 0;
}}
"""

# --- Блок куратора / ИИ-Смотрителя ---
CURATOR_BOX_STYLE = f"""
QGroupBox {{
    border: 1px solid {Colors.BORDER};
    border-left: 4px solid {Colors.CURATOR_WARNING};
    background-color: #2a2a2a;
    padding: 8px;
}}
QLabel {{
    font-size: 14px;
    font-weight: bold;
    padding: 4px;
}}
"""

# Варианты окраски куратора в зависимости от ситуации
CURATOR_STATES = {
    "info":    f"border-left: 4px solid {Colors.CURATOR_INFO};    color: {Colors.CURATOR_INFO};",
    "success": f"border-left: 4px solid {Colors.CURATOR_SUCCESS}; color: {Colors.CURATOR_SUCCESS};",
    "warning": f"border-left: 4px solid {Colors.CURATOR_WARNING}; color: {Colors.CURATOR_WARNING};",
    "danger":  f"border-left: 4px solid {Colors.CURATOR_DANGER};  color: {Colors.CURATOR_DANGER};",
    "fun":     f"border-left: 4px solid {Colors.CURATOR_FUN};     color: {Colors.CURATOR_FUN};",
}

def get_curator_style(state: str) -> str:
    """Возвращает стиль для блока куратора по состоянию."""
    base = f"""
    QGroupBox {{
        border: 1px solid {Colors.BORDER};
        {CURATOR_STATES.get(state, CURATOR_STATES['warning'])}
        background-color: #2a2a2a;
        padding: 8px;
    }}
    QLabel {{
        font-size: 14px;
        font-weight: bold;
        padding: 4px;
        {CURATOR_STATES.get(state, CURATOR_STATES['warning'])}
    }}
    """
    return base


# --- Живые примеры (мониторинг: 45-68=67? и т.д.) ---
EXAMPLE_CORRECT_STYLE = f"""
QLabel {{
    color: {Colors.EXAMPLE_CORRECT};
    font-size: 13px;
    font-family: 'Consolas', monospace;
    padding: 3px 6px;
    background-color: #1e2b1e;
    border-left: 3px solid {Colors.EXAMPLE_CORRECT};
}}
"""

EXAMPLE_WRONG_STYLE = f"""
QLabel {{
    color: {Colors.EXAMPLE_WRONG};
    font-size: 13px;
    font-family: 'Consolas', monospace;
    padding: 3px 6px;
    background-color: #2b1e1e;
    border-left: 3px solid {Colors.EXAMPLE_WRONG};
}}
"""


# --- Индикатор сложности (светофор) ---
COMPLEXITY_STYLES = {
    "easy":    f"color: {Colors.COMPLEXITY_EASY};    font-size: 14px; font-weight: bold; padding: 5px;",
    "medium":  f"color: {Colors.COMPLEXITY_MEDIUM};  font-size: 14px; font-weight: bold; padding: 5px;",
    "hard":    f"color: {Colors.COMPLEXITY_HARD};    font-size: 14px; font-weight: bold; padding: 5px;",
    "extreme": f"color: {Colors.COMPLEXITY_EXTREME}; font-size: 14px; font-weight: bold; padding: 5px;",
}


# --- Панель подсказок (виджет подсказки / hint_widget) ---
HINT_STATIC_STYLE = f"""
QLabel {{
    color: {Colors.HINT_STATIC};
    font-size: 12px;
    font-style: italic;
    padding: 4px;
    background-color: #262626;
    border: 1px dashed {Colors.BORDER};
    border-radius: 3px;
}}
"""

HINT_AI_STYLE = f"""
QLabel {{
    color: {Colors.HINT_AI};
    font-size: 12px;
    font-style: italic;
    padding: 4px;
    background-color: #2a2433;
    border: 1px dashed #5a4a7a;
    border-radius: 3px;
}}
"""

HINT_WARNING_STYLE = f"""
QLabel {{
    color: {Colors.HINT_WARN};
    font-size: 12px;
    font-weight: bold;
    padding: 4px;
    background-color: #2b2a1e;
    border: 1px solid {Colors.HINT_WARN};
    border-radius: 3px;
}}
"""


# --- Чат / Песочница (sandbox_panel) ---
CHAT_USER_STYLE = f"""
QLabel {{
    color: {Colors.TEXT_BRIGHT};
    background-color: #2a4a6b;
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 14px;
    margin: 2px 40px 2px 2px;
}}
"""

CHAT_MODEL_STYLE = f"""
QLabel {{
    color: {Colors.TEXT_MAIN};
    background-color: {Colors.BG_PANEL};
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 14px;
    margin: 2px 2px 2px 40px;
    border: 1px solid {Colors.BORDER};
}}
"""

CHAT_INPUT_STYLE = f"""
QLineEdit {{
    font-size: 16px;
    padding: 10px;
    background-color: {Colors.BG_PANEL};
    border: 1px solid {Colors.BORDER};
    border-radius: 4px;
    color: {Colors.TEXT_BRIGHT};
}}
QLineEdit:focus {{
    border: 1px solid {Colors.BORDER_FOCUS};
}}
"""


# --- Панель проекта (экран выбора сценария) ---
SCENARIO_CARD_STYLE = f"""
QFrame {{
    background-color: {Colors.BG_PANEL};
    border: 2px solid {Colors.BORDER};
    border-radius: 6px;
    padding: 12px;
}}
QFrame:hover {{
    border-color: {Colors.BLUE};
    background-color: {Colors.BG_HOVER};
}}
QLabel {{
    color: {Colors.TEXT_MAIN};
}}
"""

SCENARIO_CARD_SELECTED_STYLE = f"""
QFrame {{
    background-color: #2a3548;
    border: 2px solid {Colors.BLUE};
    border-radius: 6px;
    padding: 12px;
}}
QLabel {{
    color: {Colors.TEXT_BRIGHT};
}}
"""


# --- Панель экспорта ---
EXPORT_BTN_STYLE = f"""
QPushButton {{
    text-align: left;
    padding: 10px 16px;
    font-size: 14px;
    border: 1px solid {Colors.BORDER};
    background-color: {Colors.BG_PANEL};
    border-radius: 4px;
}}
QPushButton:hover {{
    background-color: {Colors.BG_HOVER};
    border-color: {Colors.BLUE};
}}
"""


# --- Визуализация архитектуры ---
ARCH_VIZ_STYLE = f"""
QFrame {{
    background-color: {Colors.BG_DARK};
    border: 1px solid {Colors.BORDER};
}}
QLabel {{
    font-family: 'Consolas', monospace;
    font-size: 12px;
    color: {Colors.TEXT_MAIN};
}}
"""


# --- Редактор словаря (модальное окно) ---
VOCAB_EDITOR_STYLE = f"""
QTableWidget {{
    font-family: 'Consolas', monospace;
    font-size: 13px;
}}
QHeaderView::section {{
    color: {Colors.ORANGE};
    font-weight: bold;
}}
"""


# --- Аудиоплеер ---
AUDIO_PLAYER_STYLE = f"""
QFrame {{
    background-color: {Colors.BG_PANEL};
    border: 1px solid {Colors.BORDER};
    padding: 6px;
}}
QPushButton {{
    background-color: {Colors.BG_ACTIVE};
    border: 1px solid {Colors.BORDER_LIGHT};
    padding: 4px 10px;
    font-size: 16px;
}}
QPushButton:hover {{
    background-color: {Colors.BORDER_LIGHT};
}}
QSlider::groove:horizontal {{
    height: 4px;
    background-color: {Colors.BG_DARK};
}}
QSlider::handle:horizontal {{
    background-color: {Colors.BLUE};
    width: 12px;
    margin: -4px 0;
    border-radius: 6px;
}}
"""


# --- Генератор весёлых имён ---
NAME_GENERATOR_STYLE = f"""
QLineEdit {{
    font-size: 18px;
    font-weight: bold;
    color: {Colors.ORANGE};
    padding: 8px;
    background-color: {Colors.BG_DARK};
    border: 1px solid {Colors.BORDER};
}}
QPushButton {{
    font-size: 20px;
    padding: 4px 12px;
}}
"""


# --- Панель предпросмотра генератора ---
GENERATOR_PREVIEW_STYLE = f"""
QTextEdit {{
    font-family: 'Consolas', monospace;
    font-size: 12px;
    background-color: {Colors.BG_DARK};
    color: {Colors.TEXT_MAIN};
    border: 1px solid {Colors.BORDER};
}}
"""

# --- Схема потока данных (объяснение генератора) ---
DATA_FLOW_STYLE = f"""
QTextEdit {{
    font-family: 'Consolas', monospace;
    font-size: 12px;
    background-color: #1a1a2e;
    color: {Colors.BLUE};
    border: 1px solid #2a2a4a;
    padding: 8px;
}}
"""


# --- Отчёт в анализе ---
REPORT_STYLE = f"""
QTextEdit {{
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 15px;
    background-color: #242424;
    color: {Colors.TEXT_MAIN};
    padding: 10px;
    border: 1px solid {Colors.BORDER};
    line-height: 1.6;
}}
"""


# --- Сравнение попыток (анализ) ---
COMPARISON_TABLE_STYLE = f"""
QTableWidget {{
    font-size: 13px;
    gridline-color: {Colors.BORDER};
}}
QTableWidget::item {{
    padding: 6px;
}}
"""


# --- Пресеты (выпадающий список в генераторе) ---
PRESET_LIST_STYLE = f"""
QListWidget {{
    font-size: 14px;
    padding: 4px;
}}
QListWidget::item {{
    padding: 8px;
    border-bottom: 1px solid {Colors.BORDER};
}}
QListWidget::item:hover {{
    background-color: {Colors.BG_HOVER};
}}
QListWidget::item:selected {{
    background-color: #2a4a6b;
    color: {Colors.TEXT_BRIGHT};
}}
"""


# --- Логи (консольный стиль) ---
LOGS_CONSOLE_STYLE = f"""
QTextEdit {{
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    background-color: {Colors.BG_DARK};
    color: {Colors.TEXT_MAIN};
    border: 1px solid {Colors.BORDER};
}}
"""


# --- Рисовалка (Paint-подобный виджет) ---
CANVAS_TOOLBAR_STYLE = f"""
QFrame {{
    background-color: {Colors.BG_PANEL};
    border: 1px solid {Colors.BORDER};
    padding: 4px;
}}
QPushButton {{
    padding: 4px 8px;
    font-size: 14px;
    min-width: 32px;
    min-height: 32px;
}}
QPushButton:checked {{
    background-color: {Colors.BG_ACTIVE};
    border: 1px solid {Colors.BLUE};
}}
"""

CANVAS_AREA_STYLE = f"""
QFrame {{
    background-color: #ffffff;
    border: 2px solid {Colors.BORDER};
}}
"""


# --- Предупреждение о мусорных данных ---
MUD_WARNING_STYLE = f"""
QLabel {{
    color: {Colors.YELLOW};
    background-color: #2b2a1e;
    border: 1px solid {Colors.YELLOW};
    border-left: 4px solid {Colors.YELLOW};
    padding: 8px;
    font-size: 13px;
}}
"""

# --- Ошибка / критическое предупреждение ---
ERROR_STYLE = f"""
QLabel {{
    color: {Colors.RED};
    background-color: #2b1e1e;
    border: 1px solid {Colors.RED};
    border-left: 4px solid {Colors.RED};
    padding: 8px;
    font-size: 13px;
    font-weight: bold;
}}
"""

# --- Успех / подтверждение ---
SUCCESS_STYLE = f"""
QLabel {{
    color: {Colors.GREEN};
    background-color: #1e2b1e;
    border: 1px solid {Colors.GREEN};
    border-left: 4px solid {Colors.GREEN};
    padding: 8px;
    font-size: 13px;
}}
"""


# ============================================================
#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def get_status_color(status: str) -> str:
    """Возвращает цвет для текстового статуса.
    
    Аргументы:
        status: "success" | "warning" | "error" | "info" | "fun"
    """
    mapping = {
        "success": Colors.GREEN,
        "warning": Colors.YELLOW,
        "error":   Colors.RED,
        "info":    Colors.BLUE,
        "fun":     Colors.PURPLE,
    }
    return mapping.get(status, Colors.TEXT_MAIN)


def get_hint_style(mode: str) -> str:
    """Возвращает стиль подсказки по режиму.
    
    Аргументы:
        mode: "static" | "ai" | "warning"
    """
    if mode == "ai":
        return HINT_AI_STYLE
    elif mode == "warning":
        return HINT_WARNING_STYLE
    return HINT_STATIC_STYLE


def get_complexity_style(level: str) -> str:
    """Возвращает стиль для индикатора сложности.
    
    Аргументы:
        level: "easy" | "medium" | "hard" | "extreme"
    """
    return COMPLEXITY_STYLES.get(level, COMPLEXITY_STYLES["easy"])


def label_style(color: str, font_size: int = 13, bold: bool = False) -> str:
    """Генерирует быстрый инлайн-стиль для QLabel.
    
    Использование:
        my_label.setStyleSheet(label_style(Colors.GREEN, 16, bold=True))
    """
    weight = "bold" if bold else "normal"
    return f"color: {color}; font-size: {font_size}px; font-weight: {weight};"


def status_label_style(status: str) -> str:
    """Стиль для статус-лейбла (успех/ошибка/предупреждение).
    
    Аргументы:
        status: "success" | "warning" | "error" | "info"
    """
    styles = {
        "success": SUCCESS_STYLE,
        "warning": MUD_WARNING_STYLE,
        "error":   ERROR_STYLE,
    }
    if status in styles:
        return styles[status]
    return f"QLabel {{ color: {Colors.BLUE}; font-size: 13px; padding: 4px; }}"