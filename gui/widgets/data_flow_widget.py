"""
gui/widgets/data_flow_widget.py
=================================
Показывает, как данные преобразуются от входа до модели.
Динамически обновляется при изменении настроек генератора.
Не создаёт ключей в shared_state.
"""

from typing import Optional, Dict, Any, List
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QFrame, QGroupBox
)
from PyQt5.QtCore import Qt


class DataFlowWidget(QWidget):
    """
    Виджет, показывающий схему потока данных:
    Вход → Токенизация → Модель → Выход

    Использование:
        flow = DataFlowWidget()
        flow.update_flow(
            data_type="text",
            sample_input="23+45",
            sample_output="68",
            vocab_size=15,
            model_type="transformer_seq2seq",
            embedding_dim=64
        )
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(4)

        # Заголовок
        title = QLabel("🔄 Поток данных (как данные идут в модель)")
        title.setStyleSheet("font-size: 13px; font-weight: bold; color: #cc7832;")
        layout.addWidget(title)

        # Контейнер для шагов
        self.steps_container = QVBoxLayout()
        self.steps_container.setSpacing(2)
        layout.addLayout(self.steps_container)

        # Начальное состояние
        self._show_placeholder()

    def _show_placeholder(self):
        self._clear_steps()
        lbl = QLabel("Выбери задачу в генераторе, чтобы увидеть как данные пойдут в модель.")
        lbl.setStyleSheet("color: #888; font-style: italic; padding: 10px;")
        lbl.setWordWrap(True)
        self.steps_container.addWidget(lbl)

    def _clear_steps(self):
        while self.steps_container.count():
            child = self.steps_container.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def update_flow(
        self,
        data_type: str = "text",
        sample_input: str = "",
        sample_output: str = "",
        vocab_size: int = 0,
        model_type: str = "",
        embedding_dim: int = 0,
        max_len: int = 0,
        image_shape: Optional[tuple] = None,
        num_classes: int = 0,
    ):
        """Обновляет визуализацию потока данных."""
        self._clear_steps()

        if data_type == "text":
            self._build_text_flow(sample_input, sample_output, vocab_size, model_type, embedding_dim, max_len)
        elif data_type == "image":
            self._build_image_flow(image_shape, num_classes, model_type)
        elif data_type == "audio":
            self._build_audio_flow(sample_input, model_type)
        elif data_type == "numeric":
            self._build_numeric_flow(sample_input, sample_output, model_type)
        else:
            self._show_placeholder()

    def _add_step(self, title: str, content: str, color: str = "#4a88c7"):
        """Добавляет один шаг в визуализацию."""
        frame = QFrame()
        frame.setStyleSheet(
            f"QFrame {{ background-color: #1e1e1e; border-left: 3px solid {color}; "
            f"padding: 6px; margin: 2px; }}"
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 12px;")
        layout.addWidget(title_lbl)

        content_lbl = QLabel(content)
        content_lbl.setStyleSheet("color: #a9b7c6; font-size: 11px;")
        content_lbl.setWordWrap(True)
        content_lbl.setTextFormat(Qt.RichText)
        layout.addWidget(content_lbl)

        self.steps_container.addWidget(frame)

    def _add_arrow(self):
        """Добавляет стрелку между шагами."""
        arrow = QLabel("        ⬇")
        arrow.setStyleSheet("color: #555; font-size: 14px; padding: 0px;")
        arrow.setAlignment(Qt.AlignCenter)
        self.steps_container.addWidget(arrow)

    def _build_text_flow(self, inp, out, vocab_size, model_type, emb_dim, max_len):
        # Шаг 1: Вход
        self._add_step(
            "📝 ВХОД (строка)",
            f'Пример: <code>"{inp}"</code> ({len(inp)} символов)',
            "#4a88c7"
        )
        self._add_arrow()

        # Шаг 2: Токенизация
        tokens_preview = ""
        if inp and vocab_size > 0:
            chars = list(inp[:10])
            tokens_preview = ", ".join([f"'{c}'→{i+4}" for i, c in enumerate(chars)])
            if len(inp) > 10:
                tokens_preview += ", ..."
        self._add_step(
            "🔤 ТОКЕНИЗАЦИЯ (символ → число)",
            f'Каждый символ получает номер из словаря ({vocab_size} токенов).<br>'
            f'Пример: {tokens_preview or "—"}<br>'
            f'Неизвестные символы → &lt;UNK&gt;',
            "#cc7832"
        )
        self._add_arrow()

        # Шаг 3: Эмбеддинг
        self._add_step(
            "🧮 ЭМБЕДДИНГ (число → вектор)",
            f'Каждый токен превращается в вектор размером {emb_dim}.<br>'
            f'Форма тензора: [batch, {max_len or "seq_len"}, {emb_dim}]',
            "#c678dd"
        )
        self._add_arrow()

        # Шаг 4: Модель
        self._add_step(
            f"🧠 МОДЕЛЬ ({model_type})",
            f'Нейросеть обрабатывает последовательность векторов и предсказывает следующий символ.',
            "#a6e3a1"
        )
        self._add_arrow()

        # Шаг 5: Выход
        self._add_step(
            "📤 ВЫХОД (предсказание)",
            f'Ожидаемый ответ: <code>"{out}"</code><br>'
            f'Модель предсказывает вероятность каждого символа из словаря.',
            "#385a3a"
        )

    def _build_image_flow(self, image_shape, num_classes, model_type):
        shape_str = f"{image_shape[0]}×{image_shape[1]}×{image_shape[2]}" if image_shape else "H×W×C"

        self._add_step(
            "🖼 ВХОД (изображение)",
            f'Размер: {shape_str} пикселей',
            "#4a88c7"
        )
        self._add_arrow()

        self._add_step(
            "📐 НОРМАЛИЗАЦИЯ",
            "Пиксели делятся на 255, чтобы значения были в диапазоне [0, 1].",
            "#cc7832"
        )
        self._add_arrow()

        self._add_step(
            f"🧠 МОДЕЛЬ ({model_type})",
            f"Свёрточные слои извлекают признаки (края, текстуры, формы).",
            "#a6e3a1"
        )
        self._add_arrow()

        self._add_step(
            "📤 ВЫХОД (класс)",
            f"Модель выдаёт вероятности для {num_classes} классов.<br>"
            f"Класс с максимальной вероятностью = ответ.",
            "#385a3a"
        )

    def _build_audio_flow(self, sample_input, model_type):
        self._add_step(
            "🔊 ВХОД (звук)",
            "Массив звуковых сэмплов (амплитуда колебаний).",
            "#4a88c7"
        )
        self._add_arrow()

        self._add_step(
            "📊 СПЕКТРОГРАММА",
            "Звук преобразуется в картинку частот (как нотоносец).",
            "#cc7832"
        )
        self._add_arrow()

        self._add_step(
            f"🧠 МОДЕЛЬ ({model_type})",
            "CNN/RNN анализирует спектрограмму как изображение.",
            "#a6e3a1"
        )
        self._add_arrow()

        self._add_step(
            "📤 ВЫХОД",
            "Предсказание: класс звука, текст, или очищенный сигнал.",
            "#385a3a"
        )

    def _build_numeric_flow(self, inp, out, model_type):
        self._add_step(
            "🔢 ВХОД (числа)",
            f'Пример: <code>{inp}</code>',
            "#4a88c7"
        )
        self._add_arrow()

        self._add_step(
            "📐 НОРМАЛИЗАЦИЯ",
            "Числа масштабируются (среднее=0, отклонение=1).",
            "#cc7832"
        )
        self._add_arrow()

        self._add_step(
            f"🧠 МОДЕЛЬ ({model_type})",
            "Полносвязные слои обрабатывают вектор чисел.",
            "#a6e3a1"
        )
        self._add_arrow()

        self._add_step(
            "📤 ВЫХОД (число)",
            f'Ожидаемый ответ: <code>{out}</code>',
            "#385a3a"
        )