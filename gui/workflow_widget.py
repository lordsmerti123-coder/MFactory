"""
gui/workflow_widget.py
======================
Индикатор шагов рабочего процесса (WorkflowManager).

ОТВЕТСТВЕННОСТЬ:
  • Отображение текущего рекомендуемого шага.
  • Подсветка прогресса (выполнено / текущий / осталось).
  • Подсказка о следующем действии.

ЗАВИСИМОСТИ:
  • core.workflow_manager → WorkflowManager
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QProgressBar,
    QToolButton, QFrame,
)
from PyQt5.QtCore import Qt

from core.workflow_manager import WorkflowManager
from core.logger import get_logger

logger = get_logger(__name__)


class WorkflowWidget(QWidget):
    """Компактный индикатор шагов в верхней панели."""

    def __init__(self, workflow_manager: WorkflowManager, parent=None):
        super().__init__(parent)
        self.workflow_manager = workflow_manager
        self._expanded = True
        self.init_ui()
        self.refresh()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(4, 2, 4, 2)

        # === Первая строка: заголовок + прогресс ===
        header = QHBoxLayout()
        self.title_label = QLabel("🧭 Рабочий процесс")
        self.title_label.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: #a9b7c6;"
        )
        header.addWidget(self.title_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(120)
        self.progress_bar.setFixedHeight(14)
        self.progress_bar.setTextVisible(True)
        header.addWidget(self.progress_bar)

        self.toggle_btn = QToolButton()
        self.toggle_btn.setText("▾")
        self.toggle_btn.setToolTip("Свернуть / развернуть шаги")
        self.toggle_btn.clicked.connect(self._toggle)
        header.addWidget(self.toggle_btn)

        header.addStretch()
        layout.addLayout(header)

        # === Вторая строка: текущий шаг ===
        self.current_label = QLabel("")
        self.current_label.setWordWrap(True)
        self.current_label.setStyleSheet(
            "font-size: 12px; color: #e5c07b; padding: 2px;"
        )
        layout.addWidget(self.current_label)

        self.hint_label = QLabel("")
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet(
            "font-size: 11px; color: #888; padding: 2px;"
        )
        layout.addWidget(self.hint_label)

        # === Список шагов ===
        self.steps_label = QLabel("")
        self.steps_label.setWordWrap(True)
        self.steps_label.setStyleSheet(
            "font-size: 11px; color: #777; padding: 2px;"
        )
        layout.addWidget(self.steps_label)

    # ============================================================
    # ОБНОВЛЕНИЕ
    # ============================================================
    def refresh(self):
        try:
            progress = self.workflow_manager.get_progress()
            action = self.workflow_manager.get_next_action()
            steps = self.workflow_manager.get_all_steps()
        except Exception as e:
            logger.warning(f"Ошибка обновления WorkflowWidget: {e}")
            return

        self.progress_bar.setMaximum(max(progress["total"], 1))
        self.progress_bar.setValue(progress["completed"])
        self.progress_bar.setFormat(
            f"{progress['completed']}/{progress['total']} · {progress['percent']}%"
        )

        if action.get("action") == "all_done":
            self.current_label.setText("✅ Всё готово!")
            self.current_label.setStyleSheet(
                "font-size: 12px; color: #a6e3a1; padding: 2px;"
            )
            self.hint_label.setText(action.get("hint", ""))
        else:
            desc = action.get("description", "")
            self.current_label.setText(f"📋 {desc}")
            self.current_label.setStyleSheet(
                "font-size: 12px; color: #e5c07b; padding: 2px;"
            )
            self.hint_label.setText(action.get("hint", ""))

        # Список шагов (первые N)
        if self._expanded:
            lines = []
            for i, step in enumerate(steps):
                if step["completed"]:
                    mark = "✅"
                elif i == progress["completed"]:
                    mark = "👉"
                else:
                    mark = "⬜"
                optional = " (опц.)" if step["is_optional"] else ""
                lines.append(f"{mark} {i + 1}. {step['description']}{optional}")
            self.steps_label.setText("\n".join(lines[:8]))
        else:
            self.steps_label.setText("")

    def _toggle(self):
        self._expanded = not self._expanded
        self.toggle_btn.setText("▴" if self._expanded else "▾")
        self.refresh()
