"""
gui/widgets/vocab_editor.py
=============================
Позволяет просматривать, удалять и добавлять символы в словарь.
Не создаёт ключей в shared_state. Возвращает изменённый словарь через сигнал.
"""

from typing import Dict, Optional
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QLineEdit,
    QGroupBox, QMessageBox, QAbstractItemView, QSpinBox
)
from PyQt5.QtCore import Qt, pyqtSignal


class VocabEditorDialog(QDialog):
    """
    Модальное окно редактирования словаря.

    Сигналы:
        vocab_changed(dict) — словарь был изменён пользователем

    Использование:
        dlg = VocabEditorDialog(vocab, parent)
        dlg.vocab_changed.connect(on_vocab_updated)
        dlg.exec_()
    """

    vocab_changed = pyqtSignal(dict)

    # Специальные токены, которые нельзя удалять
    PROTECTED_TOKENS = {"<PAD>", "<UNK>", "<BOS>", "<EOS>"}

    def __init__(self, vocab: Dict[str, int], parent=None):
        super().__init__(parent)
        self.vocab = dict(vocab)  # копия
        self.original_vocab = dict(vocab)
        self.setWindowTitle("📖 Редактор словаря")
        self.setMinimumSize(500, 450)
        self._init_ui()
        self._populate_table()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Описание
        desc = QLabel(
            "Словарь — это «азбука» нейросети. Каждый символ имеет уникальный номер. "
            "Всё, чего нет в словаре, сеть видит как «?». "
            "Специальные токены (<PAD>, <UNK>, <BOS>, <EOS>) удалять нельзя."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #a9b7c6; font-size: 12px; font-style: italic; padding: 4px;")
        layout.addWidget(desc)

        # Статистика
        self.stats_label = QLabel()
        self.stats_label.setStyleSheet("color: #4a88c7; font-size: 13px; padding: 2px;")
        layout.addWidget(self.stats_label)

        # Таблица словаря
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Символ", "Индекс", "Частота"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setToolTip(
            "Список всех символов в словаре. "
            "Выдели строку и нажми «Удалить», чтобы убрать символ."
        )
        layout.addWidget(self.table)

        # Кнопки управления
        btn_layout = QHBoxLayout()

        self.delete_btn = QPushButton("🗑 Удалить выделенный")
        self.delete_btn.setToolTip("Удалить выделенный символ из словаря")
        self.delete_btn.clicked.connect(self._delete_selected)
        btn_layout.addWidget(self.delete_btn)

        btn_layout.addSpacing(20)

        # Добавление нового символа
        btn_layout.addWidget(QLabel("Добавить символ:"))
        self.new_char_input = QLineEdit()
        self.new_char_input.setMaximumWidth(60)
        self.new_char_input.setToolTip("Введи один символ для добавления в словарь")
        btn_layout.addWidget(self.new_char_input)

        self.add_btn = QPushButton("➕ Добавить")
        self.add_btn.setToolTip("Добавить новый символ в словарь")
        self.add_btn.clicked.connect(self._add_char)
        btn_layout.addWidget(self.add_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Предупреждение
        self.warning_label = QLabel("")
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet("color: #e06c75; font-size: 11px; padding: 4px;")
        layout.addWidget(self.warning_label)

        # Кнопки OK / Отмена
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()

        self.cancel_btn = QPushButton("Отмена")
        self.cancel_btn.setToolTip("Отменить все изменения и закрыть")
        self.cancel_btn.clicked.connect(self.reject)
        bottom_layout.addWidget(self.cancel_btn)

        self.apply_btn = QPushButton("✅ Применить изменения")
        self.apply_btn.setToolTip("Сохранить изменения словаря")
        self.apply_btn.setStyleSheet("background-color: #385a3a; font-weight: bold;")
        self.apply_btn.clicked.connect(self._apply)
        bottom_layout.addWidget(self.apply_btn)

        layout.addLayout(bottom_layout)

    def _populate_table(self):
        """Заполняет таблицу данными словаря."""
        sorted_vocab = sorted(self.vocab.items(), key=lambda x: x[1])
        self.table.setRowCount(len(sorted_vocab))

        for row, (char, idx) in enumerate(sorted_vocab):
            # Символ
            char_item = QTableWidgetItem(char if char not in self.PROTECTED_TOKENS else f"{char} 🔒")
            if char in self.PROTECTED_TOKENS:
                char_item.setForeground(Qt.gray)
            self.table.setItem(row, 0, char_item)

            # Индекс
            idx_item = QTableWidgetItem(str(idx))
            self.table.setItem(row, 1, idx_item)

            # Частота (заглушка, если нет данных)
            freq_item = QTableWidgetItem("—")
            self.table.setItem(row, 2, freq_item)

        self._update_stats()

    def _update_stats(self):
        total = len(self.vocab)
        protected = sum(1 for t in self.PROTECTED_TOKENS if t in self.vocab)
        self.stats_label.setText(
            f"Всего токенов: {total} | Специальных (защита): {protected} | "
            f"Пользовательских: {total - protected}"
        )

    def _delete_selected(self):
        """Удаляет выделенный символ из словаря."""
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            QMessageBox.warning(self, "Внимание", "Выдели строку для удаления.")
            return

        row = rows[0].row()
        char_text = self.table.item(row, 0).text().replace(" 🔒", "")

        if char_text in self.PROTECTED_TOKENS:
            QMessageBox.warning(
                self, "Нельзя удалить",
                f"Специальный токен «{char_text}» защищает работу модели.\n"
                "Его удаление сломает обучение."
            )
            return

        if char_text in self.vocab:
            del self.vocab[char_text]
            self.warning_label.setText(
                f"⚠️ Символ «{char_text}» удалён. Все его вхождения в данных станут <UNK>."
            )
            self._populate_table()

    def _add_char(self):
        """Добавляет новый символ в словарь."""
        char = self.new_char_input.text().strip()
        if not char:
            return
        if len(char) > 1:
            QMessageBox.warning(self, "Внимание", "Добавляй только ОДИН символ за раз.")
            return
        if char in self.vocab:
            QMessageBox.warning(self, "Внимание", f"Символ «{char}» уже есть в словаре.")
            return

        max_idx = max(self.vocab.values()) + 1
        self.vocab[char] = max_idx
        self.new_char_input.clear()
        self.warning_label.setText("")
        self._populate_table()

    def _apply(self):
        """Применяет изменения и закрывает окно."""
        if self.vocab != self.original_vocab:
            removed = set(self.original_vocab.keys()) - set(self.vocab.keys())
            added = set(self.vocab.keys()) - set(self.original_vocab.keys())
            msg_parts = []
            if removed:
                msg_parts.append(f"Удалено: {len(removed)} символов")
            if added:
                msg_parts.append(f"Добавлено: {len(added)} символов")

            confirm = QMessageBox.question(
                self, "Подтверждение",
                f"Изменения словаря:\n" + "\n".join(msg_parts) +
                "\n\nВсе удалённые символы станут <UNK> в данных.\nПродолжить?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if confirm != QMessageBox.Yes:
                return

        self.vocab_changed.emit(self.vocab)
        self.accept()

    def get_vocab(self) -> Dict[str, int]:
        """Возвращает текущий словарь."""
        return dict(self.vocab)