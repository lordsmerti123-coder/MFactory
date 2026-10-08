from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QFormLayout, QComboBox, QSpinBox, QPushButton,
                             QTextEdit, QFileDialog, QMessageBox, QLabel)
from PyQt5.QtCore import Qt
from core.generator import DatasetGenerator
from core.logger import get_logger
from pathlib import Path

logger = get_logger()

class GeneratorPanel(QWidget):
    def __init__(self, shared_state, parent=None):
        super().__init__(parent)
        self.shared_state = shared_state
        self.generator = DatasetGenerator()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        settings_group = QGroupBox("Настройки создания данных")
        settings_layout = QFormLayout()

        self.task_combo = QComboBox()
        # Расширенный список задач
        self.task_mapping = {
            "➕ Сложение (Арифметика)": "addition",
            "➖ Вычитание (Арифметика)": "subtraction",
            "✖️ Умножение (Арифметика)": "multiplication",
            "➗ Деление (Арифметика)": "division",
            "🧮 Смешанные операции (Приоритет)": "mixed_math",
            "📐 Линейные уравнения (Алгебра)": "linear_equation",
            "📈 Квадратные уравнения (Алгебра)": "quadratic",
            "🔐 Шифр Цезаря (Сдвиг)": "cipher_caesar",
            "🔏 Шифр Атбаш (Отражение)": "cipher_atbash"
        }
        self.task_combo.addItems(list(self.task_mapping.keys()))
        self.task_combo.setToolTip("Выберите тип задачи. От этого зависит, чему именно будет учиться нейросеть.")

        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(100, 1000000)
        self.samples_spin.setSingleStep(10000)
        self.samples_spin.setValue(20000)
        self.samples_spin.setToolTip("Количество примеров. Для простых сетей хватит 1000, для Трансформеров на сложных задачах нужно от 20 000 до 100 000.")

        # Диапазон чисел (для арифметических задач: сложение, вычитание и т.д.)
        self.range_min_spin = QSpinBox()
        self.range_min_spin.setRange(0, 100000)
        self.range_min_spin.setValue(0)
        self.range_min_spin.setToolTip("Нижняя граница чисел (например, 0 для «сложение 0–10»).")
        self.range_max_spin = QSpinBox()
        self.range_max_spin.setRange(1, 100000)
        self.range_max_spin.setValue(10)
        self.range_max_spin.setToolTip("Верхняя граница чисел (например, 10 для «сложение 0–10»).")

        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Диапазон чисел:"))
        range_row.addWidget(self.range_min_spin)
        range_row.addWidget(QLabel("—"))
        range_row.addWidget(self.range_max_spin)
        range_row.addStretch()

        settings_layout.addRow("Тип задачи:", self.task_combo)
        settings_layout.addRow("Количество примеров (строк):", self.samples_spin)
        settings_layout.addRow(range_row)
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        btn_layout = QHBoxLayout()
        self.generate_btn = QPushButton("✨ Сгенерировать датасет и сохранить")
        self.generate_btn.setToolTip("Создать файл с парами 'Вопрос-Ответ', на котором будет учиться ИИ.")
        self.generate_btn.clicked.connect(self.generate_dataset)
        btn_layout.addWidget(self.generate_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        info_group = QGroupBox("👨‍🏫 Разбор задачи (Что должен понять ИИ?)")
        info_layout = QVBoxLayout()
        self.explanation_text = QTextEdit()
        self.explanation_text.setReadOnly(True)
        self.explanation_text.setStyleSheet("font-family: 'Segoe UI', Arial, sans-serif; font-size: 14px; background-color: #242424;")
        self.explanation_text.setHtml(
            "<p style='color: #a9b7c6; text-align: center; margin-top: 20px;'>"
            "Выберите задачу сверху и нажмите <b>'Сгенерировать'</b>, <br>чтобы прочитать разбор алгоритма обучения.</p>"
        )
        info_layout.addWidget(self.explanation_text)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

    def _current_params(self) -> dict:
        """Параметры генерации из UI (диапазон чисел для арифметики)."""
        lo = self.range_min_spin.value()
        hi = self.range_max_spin.value()
        if lo > hi:
            lo, hi = hi, lo
        return {"num_range": (lo, hi)}

    def generate_dataset_silent(self, task_type: str, num_samples: int,
                                save_path, num_range=None) -> str:
        """
        Генерирует датасет без диалога сохранения (для ИИ-агента).
        Возвращает текст-разбор задачи или строку ошибки.
        """
        if task_type not in self.task_mapping.values():
            valid = ", ".join(sorted(set(self.task_mapping.values())))
            return f"неизвестная задача: {task_type}. Допустимые: {valid}"
        label = next(k for k, v in self.task_mapping.items() if v == task_type)
        self.task_combo.setCurrentText(label)
        self.samples_spin.setValue(max(100, min(int(num_samples), 1000000)))
        if num_range is not None:
            lo, hi = num_range
            self.range_min_spin.setValue(int(lo))
            self.range_max_spin.setValue(int(hi))
        params = self._current_params()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        explanation = self.generator.create_and_save_dataset(
            task_type, self.samples_spin.value(), save_path, params=params
        )
        self.explanation_text.setHtml(explanation)
        return (f"датасет создан: {self.samples_spin.value()} примеров, "
                f"диапазон {params['num_range'][0]}–{params['num_range'][1]}, "
                f"сохранён в {save_path}")

    def generate_dataset(self):
        task_name = self.task_combo.currentText()
        task_type = self.task_mapping[task_name]
        num_samples = self.samples_spin.value()

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить датасет", f"dataset_{task_type}.oai", "OracleAI Dataset (*.oai)"
        )

        if not save_path:
            return

        try:
            self.generate_btn.setEnabled(False)
            self.generate_btn.setText("⏳ Идет генерация тысяч примеров...")
            from PyQt5.QtWidgets import QApplication
            QApplication.processEvents()

            explanation = self.generator.create_and_save_dataset(
                task_type, num_samples, Path(save_path), params=self._current_params()
            )
            self.explanation_text.setHtml(explanation)
            
            QMessageBox.information(
                self, 
                "Успех!", 
                f"Датасет успешно сгенерирован!\n\nСохранен в:\n{save_path}\n\n"
                "Шаг 2: Перейдите на вкладку 'Данные' и загрузите этот файл."
            )
        except Exception as e:
            logger.error(f"Ошибка генерации: {e}")
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка при создании данных:\n{e}")
        finally:
            self.generate_btn.setEnabled(True)
            self.generate_btn.setText("✨ Сгенерировать датасет и сохранить")