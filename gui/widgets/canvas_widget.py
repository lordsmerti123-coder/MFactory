"""
gui/widgets/canvas_widget.py
=============================
Позволяет школьникам рисовать фигуры/цифры, размечать их и формировать датасет.
Не создаёт ключей в shared_state. Работает независимо.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QSpinBox, QColorDialog, QGroupBox, QFormLayout,
    QMessageBox, QSlider, QToolButton, QScrollArea, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QRect, QSize
from PyQt5.QtGui import (
    QPainter, QPen, QBrush, QColor, QPixmap, QImage,
    QPainterPath, QFont
)


class DrawableCanvas(QWidget):
    """Чистый холст для рисования мышью."""

    drawing_finished = pyqtSignal(np.ndarray)  # передаёт готовое изображение как массив

    def __init__(self, canvas_size: int = 128, parent=None):
        super().__init__(parent)
        self.canvas_size = canvas_size
        self.setMinimumSize(QSize(canvas_size + 40, canvas_size + 40))
        self.setMaximumSize(QSize(canvas_size + 40, canvas_size + 40))

        self.pixmap = QPixmap(canvas_size, canvas_size)
        self.pixmap.fill(Qt.white)

        self.drawing = False
        self.last_point = QPoint()
        self.pen_color = QColor(Qt.black)
        self.pen_width = 4
        self.tool = "pen"  # "pen" | "eraser" | "line" | "rect" | "circle" | "fill"
        self.start_point = QPoint()
        self.temp_pixmap = None

        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)

    def set_pen_color(self, color: QColor):
        self.pen_color = color

    def set_pen_width(self, width: int):
        self.pen_width = width

    def set_tool(self, tool: str):
        self.tool = tool

    def clear_canvas(self):
        self.pixmap.fill(Qt.white)
        self.update()

    def get_image_array(self) -> np.ndarray:
        """Возвращает изображение как numpy-массив (H, W) в градациях серого 0-255."""
        img = self.pixmap.toImage().convertToFormat(QImage.Format_Grayscale8)
        width = img.width()
        height = img.height()
        ptr = img.bits()
        ptr.setArraySize(height * width)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width))
        return arr.copy()

    def get_image_array_rgb(self) -> np.ndarray:
        """Возвращает изображение как numpy-массив (H, W, 3) RGB."""
        img = self.pixmap.toImage().convertToFormat(QImage.Format_RGB888)
        width = img.width()
        height = img.height()
        ptr = img.bits()
        ptr.setArraySize(height * width * 3)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 3))
        return arr.copy()

    def paintEvent(self, event):
        painter = QPainter(self)
        # Смещение для центрирования
        offset_x = (self.width() - self.canvas_size) // 2
        offset_y = (self.height() - self.canvas_size) // 2

        # Рамка холста
        painter.setPen(QPen(QColor("#555555"), 2))
        painter.drawRect(offset_x - 1, offset_y - 1, self.canvas_size + 2, self.canvas_size + 2)

        # Сам холст
        painter.drawPixmap(offset_x, offset_y, self.pixmap)

        # Временный превью для фигур
        if self.temp_pixmap is not None and self.drawing:
            painter.drawPixmap(offset_x, offset_y, self.temp_pixmap)

    def _canvas_pos(self, global_pos: QPoint) -> QPoint:
        offset_x = (self.width() - self.canvas_size) // 2
        offset_y = (self.height() - self.canvas_size) // 2
        return QPoint(global_pos.x() - offset_x, global_pos.y() - offset_y)

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        pos = self._canvas_pos(event.pos())
        if not self._in_canvas(pos):
            return

        self.drawing = True
        self.last_point = pos
        self.start_point = pos

        if self.tool == "fill":
            self._flood_fill(pos)
            self.drawing = False
            self.update()
            return

        if self.tool in ("line", "rect", "circle"):
            self.temp_pixmap = self.pixmap.copy()

    def mouseMoveEvent(self, event):
        if not self.drawing:
            return
        pos = self._canvas_pos(event.pos())
        if not self._in_canvas(pos):
            return

        if self.tool == "pen":
            self._draw_freehand(self.last_point, pos, self.pen_color)
        elif self.tool == "eraser":
            self._draw_freehand(self.last_point, pos, QColor(Qt.white))
        elif self.tool in ("line", "rect", "circle"):
            self._draw_shape_preview(pos)

        self.last_point = pos
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton or not self.drawing:
            return
        pos = self._canvas_pos(event.pos())
        if not self._in_canvas(pos):
            pos = self.last_point

        if self.tool == "line":
            self._draw_line(self.start_point, pos)
        elif self.tool == "rect":
            self._draw_rect(self.start_point, pos)
        elif self.tool == "circle":
            self._draw_circle(self.start_point, pos)

        self.drawing = False
        self.temp_pixmap = None
        self.update()

    def _in_canvas(self, pos: QPoint) -> bool:
        return 0 <= pos.x() < self.canvas_size and 0 <= pos.y() < self.canvas_size

    def _draw_freehand(self, from_pt: QPoint, to_pt: QPoint, color: QColor):
        painter = QPainter(self.pixmap)
        pen = QPen(color, self.pen_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(from_pt, to_pt)
        painter.end()

    def _draw_shape_preview(self, current_pt: QPoint):
        self.temp_pixmap = self.pixmap.copy()
        painter = QPainter(self.temp_pixmap)
        pen = QPen(self.pen_color, self.pen_width, Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        if self.tool == "line":
            painter.drawLine(self.start_point, current_pt)
        elif self.tool == "rect":
            rect = QRect(self.start_point, current_pt).normalized()
            painter.drawRect(rect)
        elif self.tool == "circle":
            rect = QRect(self.start_point, current_pt).normalized()
            painter.drawEllipse(rect)
        painter.end()

    def _draw_line(self, from_pt: QPoint, to_pt: QPoint):
        painter = QPainter(self.pixmap)
        pen = QPen(self.pen_color, self.pen_width, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(pen)
        painter.drawLine(from_pt, to_pt)
        painter.end()

    def _draw_rect(self, from_pt: QPoint, to_pt: QPoint):
        painter = QPainter(self.pixmap)
        pen = QPen(self.pen_color, self.pen_width, Qt.SolidLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        rect = QRect(from_pt, to_pt).normalized()
        painter.drawRect(rect)
        painter.end()

    def _draw_circle(self, from_pt: QPoint, to_pt: QPoint):
        painter = QPainter(self.pixmap)
        pen = QPen(self.pen_color, self.pen_width, Qt.SolidLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        rect = QRect(from_pt, to_pt).normalized()
        painter.drawEllipse(rect)
        painter.end()

    def _flood_fill(self, pos: QPoint):
        img = self.pixmap.toImage()
        target_color = img.pixelColor(pos)
        if target_color == self.pen_color:
            return
        self._flood_fill_recursive(img, pos.x(), pos.y(), target_color, self.pen_color)
        self.pixmap = QPixmap.fromImage(img)

    def _flood_fill_recursive(self, img, x, y, target, replacement):
        if x < 0 or x >= img.width() or y < 0 or y >= img.height():
            return
        if img.pixelColor(x, y) != target:
            return
        img.setPixelColor(x, y, replacement)
        self._flood_fill_recursive(img, x + 1, y, target, replacement)
        self._flood_fill_recursive(img, x - 1, y, target, replacement)
        self._flood_fill_recursive(img, x, y + 1, target, replacement)
        self._flood_fill_recursive(img, x, y - 1, target, replacement)


class CanvasWidget(QWidget):
    """
    Полноценный виджет-рисовалка с панелью инструментов.
    Сигналы:
        image_ready(np.ndarray, str) — изображение + метка класса
    """

    image_ready = pyqtSignal(np.ndarray, str)  # (массив, метка)

    def __init__(self, canvas_size: int = 128, parent=None):
        super().__init__(parent)
        self.canvas_size = canvas_size
        self.current_label = "без метки"
        self.drawn_images: List[Tuple[np.ndarray, str]] = []
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(8)

        # === Заголовок ===
        title = QLabel("🎨 Рисовалка: нарисуй и разметь")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #cc7832;")
        main_layout.addWidget(title)

        desc = QLabel(
            "Нарисуй фигуру или цифру, укажи что это, и нажми «Добавить в датасет». "
            "Можно нарисовать много примеров всем классом!"
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #a9b7c6; font-size: 12px; font-style: italic;")
        main_layout.addWidget(desc)

        # === Панель инструментов ===
        tools_group = QGroupBox("Инструменты")
        tools_layout = QHBoxLayout()

        # Кнопки инструментов
        self.tool_buttons = {}
        tool_defs = [
            ("pen", "✏️", "Карандаш (свободное рисование)"),
            ("eraser", "🧹", "Ластик (стирание)"),
            ("line", "📏", "Линия (прямая)"),
            ("rect", "⬜", "Прямоугольник"),
            ("circle", "⭕", "Круг / Эллипс"),
            ("fill", "🪣", "Заливка"),
        ]
        for tool_id, icon, tooltip in tool_defs:
            btn = QToolButton()
            btn.setText(icon)
            btn.setCheckable(True)
            btn.setToolTip(tooltip)
            btn.setFixedSize(36, 36)
            btn.setStyleSheet("font-size: 18px;")
            btn.clicked.connect(lambda checked, t=tool_id: self._set_tool(t))
            tools_layout.addWidget(btn)
            self.tool_buttons[tool_id] = btn

        self.tool_buttons["pen"].setChecked(True)

        tools_layout.addSpacing(20)

        # Толщина кисти
        tools_layout.addWidget(QLabel("Толщина:"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 30)
        self.width_spin.setValue(4)
        self.width_spin.setToolTip("Толщина линии рисования в пикселях")
        self.width_spin.valueChanged.connect(self._on_width_changed)
        tools_layout.addWidget(self.width_spin)

        # Цвет
        self.color_btn = QPushButton("🎨 Цвет")
        self.color_btn.setToolTip("Выбрать цвет рисования")
        self.color_btn.clicked.connect(self._pick_color)
        tools_layout.addWidget(self.color_btn)

        # Очистить
        self.clear_btn = QPushButton("🗑 Очистить")
        self.clear_btn.setToolTip("Очистить весь холст")
        self.clear_btn.clicked.connect(self._clear)
        tools_layout.addWidget(self.clear_btn)

        tools_layout.addStretch()
        tools_group.setLayout(tools_layout)
        main_layout.addWidget(tools_group)

        # === Холст + метка ===
        canvas_area_layout = QHBoxLayout()

        # Холст
        self.canvas = DrawableCanvas(self.canvas_size)
        canvas_area_layout.addWidget(self.canvas)

        # Правая панель: метка и статистика
        right_panel = QVBoxLayout()

        label_group = QGroupBox("Метка (что нарисовано?)")
        label_layout = QVBoxLayout()

        self.label_combo = QComboBox()
        self.label_combo.setEditable(True)
        self.label_combo.addItems([
            "круг", "квадрат", "треугольник", "прямоугольник",
            "звезда", "куб", "сфера", "пирамида",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "буква_А", "буква_Б", "другое"
        ])
        self.label_combo.setToolTip(
            "Укажи, что именно нарисовано. Это станет правильным ответом для нейросети. "
            "Можно ввести своё название."
        )
        label_layout.addWidget(self.label_combo)

        self.add_btn = QPushButton("➕ Добавить в датасет")
        self.add_btn.setToolTip("Сохранить рисунок и его метку в коллекцию датасета")
        self.add_btn.setStyleSheet(
            "background-color: #385a3a; font-weight: bold; padding: 8px;"
        )
        self.add_btn.clicked.connect(self._add_to_dataset)
        label_layout.addWidget(self.add_btn)

        self.count_label = QLabel("Рисунков в коллекции: 0")
        self.count_label.setStyleSheet("color: #4a88c7; font-size: 13px; margin-top: 8px;")
        label_layout.addWidget(self.count_label)

        self.export_btn = QPushButton("💾 Экспортировать как датасет")
        self.export_btn.setToolTip(
            "Сохранить все нарисованные примеры в формате .oai для обучения"
        )
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_dataset)
        label_layout.addWidget(self.export_btn)

        label_layout.addStretch()
        label_group.setLayout(label_layout)
        right_panel.addWidget(label_group)

        # Превью последних рисунков
        preview_group = QGroupBox("Последние рисунки")
        preview_layout = QVBoxLayout()
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_scroll.setFixedHeight(200)
        self.preview_container = QWidget()
        self.preview_grid = QVBoxLayout(self.preview_container)
        self.preview_scroll.setWidget(self.preview_container)
        preview_layout.addWidget(self.preview_scroll)
        preview_group.setLayout(preview_layout)
        right_panel.addWidget(preview_group)

        canvas_area_layout.addLayout(right_panel)
        main_layout.addLayout(canvas_area_layout)

        # === Размер холста ===
        size_group = QGroupBox("Размер холста")
        size_layout = QHBoxLayout()
        self.size_combo = QComboBox()
        self.size_combo.addItems(["28×28", "64×64", "128×128", "256×256"])
        self.size_combo.setCurrentText("128×128")
        self.size_combo.setToolTip(
            "Размер изображения. 28×28 — стандарт для MNIST. "
            "Больше размер — больше деталей, но медленнее обучение."
        )
        self.size_combo.currentIndexChanged.connect(self._on_size_changed)
        size_layout.addWidget(QLabel("Размер:"))
        size_layout.addWidget(self.size_combo)
        size_layout.addStretch()
        size_group.setLayout(size_layout)
        main_layout.addWidget(size_group)

    def _set_tool(self, tool: str):
        for tid, btn in self.tool_buttons.items():
            btn.setChecked(tid == tool)
        self.canvas.set_tool(tool)

    def _on_width_changed(self, val: int):
        self.canvas.set_pen_width(val)

    def _pick_color(self):
        color = QColorDialog.getColor(self.canvas.pen_color, self, "Выбор цвета")
        if color.isValid():
            self.canvas.set_pen_color(color)

    def _clear(self):
        self.canvas.clear_canvas()

    def _on_size_changed(self, index: int):
        sizes = [28, 64, 128, 256]
        self.canvas_size = sizes[index]
        self.canvas.canvas_size = self.canvas_size
        self.canvas.pixmap = QPixmap(self.canvas_size, self.canvas_size)
        self.canvas.pixmap.fill(Qt.white)
        self.canvas.setMinimumSize(QSize(self.canvas_size + 40, self.canvas_size + 40))
        self.canvas.setMaximumSize(QSize(self.canvas_size + 40, self.canvas_size + 40))
        self.canvas.update()

    def _add_to_dataset(self):
        label = self.label_combo.currentText().strip()
        if not label:
            QMessageBox.warning(self, "Внимание", "Укажи метку: что нарисовано?")
            return

        arr = self.canvas.get_image_array()
        self.drawn_images.append((arr.copy(), label))
        self.current_label = label

        self.count_label.setText(f"Рисунков в коллекции: {len(self.drawn_images)}")
        self.export_btn.setEnabled(len(self.drawn_images) > 0)

        # Мини-превью
        preview_label = QLabel(f"  #{len(self.drawn_images)}: {label}")
        preview_label.setStyleSheet("color: #a6e3a1; font-size: 11px; padding: 2px;")
        self.preview_grid.addWidget(preview_label)

        # Очистить холст для следующего рисунка
        self.canvas.clear_canvas()

    def _export_dataset(self):
        """Собирает все рисунки в формат, совместимый с DatasetContainer."""
        if not self.drawn_images:
            return

        from PyQt5.QtWidgets import QFileDialog
        from pathlib import Path
        import json

        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить датасет рисунков",
            "my_drawings.oai", "OracleAI Dataset (*.oai)"
        )
        if not path:
            return

        # Формируем данные в формате .oai
        inputs = []
        outputs = []
        for arr, label in self.drawn_images:
            inputs.append(arr.tolist())
            outputs.append(label)

        dataset_data = {
            "meta": {
                "format_version": "2.0",
                "type": "image",
                "task_type": "image_classification",
                "num_samples": len(inputs),
                "image_shape": [self.canvas_size, self.canvas_size, 1],
                "num_classes": len(set(outputs)),
                "description": f"Датасет из рисовалки ({len(inputs)} рисунков)",
                "generator_task": "custom_drawings",
            },
            "raw_inputs": inputs,
            "raw_outputs": outputs,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(dataset_data, f, ensure_ascii=False)

        QMessageBox.information(
            self, "Успех!",
            f"Датасет из {len(inputs)} рисунков сохранён в:\n{path}\n\n"
            "Теперь загрузи его во вкладке «Данные»."
        )

    def get_dataset_dict(self) -> Optional[Dict[str, Any]]:
        """Возвращает датасет как словарь (для программной интеграции)."""
        if not self.drawn_images:
            return None
        inputs = [arr.tolist() for arr, _ in self.drawn_images]
        outputs = [label for _, label in self.drawn_images]
        return {
            "meta": {
                "format_version": "2.0",
                "type": "image",
                "task_type": "image_classification",
                "num_samples": len(inputs),
                "image_shape": [self.canvas_size, self.canvas_size, 1],
                "num_classes": len(set(outputs)),
            },
            "raw_inputs": inputs,
            "raw_outputs": outputs,
        }