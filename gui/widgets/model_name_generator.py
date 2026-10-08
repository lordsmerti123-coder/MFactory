"""
gui/widgets/model_name_generator.py
=====================================
Работает независимо, не создаёт ключей в shared_state.
"""

import random
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QLabel, QLineEdit
)
from PyQt5.QtCore import pyqtSignal


# Составляющие для генерации имён
ADJECTIVES = [
    "Нейро", "Квантовый", "Цифровой", "Умный", "Хитрый",
    "Быстрый", "Мудрый", "Космический", "Магический", "Электрический",
    "Тензорный", "Градиентный", "Эпохальный", "Лоссовый", "Свёрточный",
    "Рекуррентный", "Трансформерный", "Капсульный", "Диффузный", "Генеративный",
]

NOUNS = [
    "Вася", "Мозг", "Гений", "Умник", "Мыслитель",
    "Оракул", "Пророк", "Маг", "Волшебник", "Учёный",
    "Профессор", "Доцент", "Академик", "Мыслитель", "Философ",
    "Пиксель", "Байт", "Тензорчик", "Градиентик", "Нейрончик",
]

SUFFIXES = [
    "3000", "2000", "X", "Ultra", "Pro",
    "Max", "Mini", "Junior", "Senior", "Deluxe",
    "Turbo", "Mega", "Nano", "Alpha", "Omega",
    "-7", "-42", "-v2", "XL", "Supreme",
]

PHRASES = [
    "Мозгожуй-{n}",
    "Скайнет-младший",
    "Гендальф Серый",
    "Нейросетевичок",
    "Угадай-ка",
    "Думалка-{n}",
    "Решала-{n}",
    "Вычисляй-{n}",
    "Предсказатель-{n}",
    "Обучайка-{n}",
]


def generate_funny_name() -> str:
    """Генерирует одно весёлое имя для модели."""
    mode = random.randint(0, 2)
    if mode == 0:
        # Прилагательное + Существительное + Суффикс
        return f"{random.choice(ADJECTIVES)}{random.choice(NOUNS)}-{random.choice(SUFFIXES)}"
    elif mode == 1:
        # Фраза с числом
        phrase = random.choice(PHRASES)
        return phrase.replace("{n}", str(random.randint(1, 9999)))
    else:
        # Просто Прилагательное + Существительное
        return f"{random.choice(ADJECTIVES)} {random.choice(NOUNS)}"


def generate_batch_names(count: int = 5) -> list:
    """Генерирует несколько уникальных имён."""
    names = set()
    attempts = 0
    while len(names) < count and attempts < count * 10:
        names.add(generate_funny_name())
        attempts += 1
    return list(names)


class ModelNameGeneratorWidget(QWidget):
    """
    Виджет генерации и выбора имени модели.

    Сигналы:
        name_selected(str) — пользователь выбрал/подтвердил имя

    Использование:
        widget = ModelNameGeneratorWidget()
        widget.name_selected.connect(on_name_chosen)
    """

    name_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Метка
        label = QLabel("🏷 Имя модели:")
        label.setStyleSheet("font-weight: bold; color: #cc7832;")
        layout.addWidget(label)

        # Поле ввода
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Введи имя или нажми 🎲...")
        self.name_input.setToolTip(
            "Имя твоей нейросети. Можно ввести своё или сгенерировать весёлое. "
            "Это имя будет отображаться в карточке модели и в отчётах."
        )
        self.name_input.returnPressed.connect(self._confirm)
        layout.addWidget(self.name_input)

        # Кнопка генерации
        self.dice_btn = QPushButton("🎲")
        self.dice_btn.setFixedWidth(40)
        self.dice_btn.setToolTip("Сгенерировать случайное весёлое имя")
        self.dice_btn.clicked.connect(self._generate)
        layout.addWidget(self.dice_btn)

        # Кнопка подтверждения
        self.confirm_btn = QPushButton("✅")
        self.confirm_btn.setFixedWidth(40)
        self.confirm_btn.setToolTip("Подтвердить имя модели")
        self.confirm_btn.setStyleSheet("background-color: #385a3a;")
        self.confirm_btn.clicked.connect(self._confirm)
        layout.addWidget(self.confirm_btn)

        # Генерируем начальное имя
        self._generate()

    def _generate(self):
        """Генерирует новое случайное имя."""
        name = generate_funny_name()
        self.name_input.setText(name)

    def _confirm(self):
        """Подтверждает текущее имя."""
        name = self.name_input.text().strip()
        if not name:
            name = generate_funny_name()
            self.name_input.setText(name)
        self.name_selected.emit(name)

    def get_name(self) -> str:
        """Возвращает текущее имя."""
        return self.name_input.text().strip() or generate_funny_name()

    def set_name(self, name: str):
        """Устанавливает имя программно."""
        self.name_input.setText(name)