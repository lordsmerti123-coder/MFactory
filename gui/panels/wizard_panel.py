"""
gui/panels/wizard_panel.py
==========================
Панель режима «Мастер» (ИИ-агент создаёт нейросеть наглядно, шаг за шагом).

ОТВЕТСТВЕННОСТЬ:
  • Кнопка запуска режима, выбор сценария (демо / интерактив).
  • Отображение плана шагов с галочками и текущей строки.
  • Комментарии Оракула по ходу выполнения.
  • Управление плеером: Старт / Пауза / Стоп, скорость.

ЗАВИСИМОСТИ:
  • core.oraculum.wizard → WizardPlanner, WizardPlayer
  • gui/widgets/highlight.py → подсветка вкладок
  • PyQt5
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSlider, QListWidget, QListWidgetItem, QInputDialog,
)
from PyQt5.QtCore import Qt, pyqtSignal

from core.oraculum.wizard import WizardPlanner, WizardPlayer
from gui.widgets.highlight import highlight_tab, clear_tab_highlight

STEP_NORMAL = "color: #a9b7c6;"
STEP_ACTIVE = (
    "background-color: #4a88c7; color: white; font-weight: bold;"
    "border-radius: 4px; padding: 2px 6px;"
)
STEP_DONE = "color: #a6e3a1;"


class WizardPanel(QWidget):
    """Панель режима «Мастер»."""

    # Передаётся в MainWindow: показать комментарий Оракула в чате
    panel_narration = pyqtSignal(str)
    wizard_started = pyqtSignal(str)   # scenario
    wizard_finished = pyqtSignal()
    wizard_aborted = pyqtSignal()

    def __init__(self, agent, execute_tool: Callable[[str, Dict[str, Any]], str],
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.agent = agent
        self.planner = WizardPlanner(agent)
        self.player = WizardPlayer(execute_tool, self)
        self._current_plan = None
        self.init_ui()
        self._connect_player()

    # ============================================================
    # UI
    # ============================================================
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        header = QHBoxLayout()
        title = QLabel("🚀 Мастер (Яндекс GPT)")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #c678dd;")
        header.addWidget(title)
        header.addStretch()
        layout.addLayout(header)

        # Сценарий
        row = QHBoxLayout()
        row.addWidget(QLabel("Сценарий:"))
        self.scenario_combo = QComboBox()
        self.scenario_combo.addItems([
            "Показать, как создаётся Трансформер (демо)",
            "Помочь создать мою модель (уточню)",
        ])
        self.scenario_combo.setToolTip(
            "Демо — Оракул сам создаст и обучит маленький Трансформер на сложении.\n"
            "Помощь — сначала уточнит имя проекта и модели."
        )
        row.addWidget(self.scenario_combo, stretch=1)
        layout.addLayout(row)

        # Кнопки управления
        btns = QHBoxLayout()
        self.btn_start = QPushButton("▶️ Запустить")
        self.btn_start.clicked.connect(self.start_wizard)
        btns.addWidget(self.btn_start)

        self.btn_pause = QPushButton("⏸ Пауза")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self.toggle_pause)
        btns.addWidget(self.btn_pause)

        self.btn_stop = QPushButton("⏹ Стоп")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_wizard)
        btns.addWidget(self.btn_stop)

        btns.addWidget(QLabel("Скорость:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(50, 200)
        self.speed_slider.setValue(100)
        self.speed_slider.setTickInterval(25)
        self.speed_slider.setToolTip("Скорость шагов: 0.5x – 2x")
        self.speed_slider.valueChanged.connect(self._on_speed_changed)
        btns.addWidget(self.speed_slider)
        btns.addStretch()
        layout.addLayout(btns)

        # План
        self.plan_list = QListWidget()
        self.plan_list.setStyleSheet(
            "font-size: 12px; background-color: #1e1e1e; color: #a9b7c6;"
            "border: 1px solid #555; border-radius: 4px;"
        )
        layout.addWidget(self.plan_list, stretch=1)

        # Комментарий
        self.narration_label = QLabel(
            "Нажмите «Запустить» — и Оракул покажет каждый шаг создания нейросети."
        )
        self.narration_label.setWordWrap(True)
        self.narration_label.setStyleSheet(
            "font-size: 12px; color: #d4d4d4; padding: 6px;"
            "background-color: #252526; border: 1px solid #555; border-radius: 4px;"
        )
        layout.addWidget(self.narration_label)

        # Журнал всех комментариев Оракула (ничего не проскакивает)
        self.comment_log = QListWidget()
        self.comment_log.setStyleSheet(
            "font-size: 11px; background-color: #1e1e1e; color: #a9b7c6;"
            "border: 1px solid #555; border-radius: 4px;"
        )
        self.comment_log.setMaximumHeight(120)
        self.comment_log.setToolTip("Журнал всех комментариев Оракула по ходу мастера.")
        layout.addWidget(self.comment_log)

    # ============================================================
    # УПРАВЛЕНИЕ
    # ============================================================
    def _connect_player(self):
        self.player.step_started.connect(self._on_step_started)
        self.player.narration.connect(self._on_narration)
        self.player.step_finished.connect(self._on_step_finished)
        self.player.finished.connect(self._on_finished)
        self.player.aborted.connect(self._on_aborted)
        self.player.error.connect(self._on_step_error)

    def start_wizard(self):
        scenario = self.scenario_combo.currentIndex()
        if scenario == 0:
            plan = self.planner.plan_demo_transformer()
        else:
            # Интерактив: уточняем у пользователя имя проекта и модели
            name, ok1 = QInputDialog.getText(self, "Мастер", "Имя проекта:")
            if not ok1 or not name.strip():
                self.narration_label.setText("Мастер отменён — не указано имя проекта.")
                return
            model_name, ok2 = QInputDialog.getText(
                self, "Мастер", "Имя модели:", text=f"{name.strip()}-модель"
            )
            if not ok2:
                self.narration_label.setText("Мастер отменён.")
                return
            plan = self.planner.plan_interactive({
                "name": name.strip(),
                "model_name": (model_name or "").strip() or f"{name.strip()}-модель",
            })

        self._current_plan = plan
        self.plan_list.clear()
        for i, step in enumerate(plan.steps):
            item = QListWidgetItem(f"  {i + 1}. {step.title}")
            item.setData(Qt.UserRole, step.step_id)
            item.setForeground(Qt.darkGray)
            self.plan_list.addItem(item)

        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_pause.setText("⏸ Пауза")
        self.btn_stop.setEnabled(True)
        self.wizard_started.emit(plan.scenario)
        self.player.start(plan)

    def toggle_pause(self):
        if self.player.is_running():
            self.player.pause()
            self.btn_pause.setText("▶️ Продолжить")
            self.narration_label.setText(self.narration_label.text() + "\n(пауза)")
        else:
            self.player.resume()
            self.btn_pause.setText("⏸ Пауза")

    def stop_wizard(self):
        self.player.stop()
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self._reset_styles()

    def _on_speed_changed(self, value: int):
        self.player.set_speed(value / 100.0)

    # ============================================================
    # СИГНАЛЫ ПЛЕЕРА
    # ============================================================
    def _on_step_started(self, step_id: str, info: dict):
        self._reset_styles()
        for i in range(self.plan_list.count()):
            item = self.plan_list.item(i)
            if item.data(Qt.UserRole) == step_id:
                item.setForeground(Qt.white)
                item.setBackground(Qt.darkBlue)
                self.plan_list.setCurrentRow(i)
                break
        # Подсветка вкладки, на которой происходит действие
        hl = info.get("highlight")
        tab_widget = self._tab_widget()
        if hl and tab_widget is not None:
            for i in range(tab_widget.count()):
                if hl in tab_widget.tabText(i):
                    highlight_tab(tab_widget, i)
                    break
        self.panel_narration.emit(f"🚀 {info.get('narration', '')}")

    def _on_narration(self, text: str):
        self.narration_label.setText(text)
        self.comment_log.addItem(text)
        self.comment_log.scrollToBottom()
        self.panel_narration.emit(text)

    def _on_step_finished(self, step_id: str, info: dict):
        for i in range(self.plan_list.count()):
            item = self.plan_list.item(i)
            if item.data(Qt.UserRole) == step_id:
                title = item.text()
                item.setText(f"  ✅{title[3:] if title.startswith('  ') else title}")
                item.setForeground(Qt.darkGreen)
                item.setBackground(Qt.transparent)
                break

    def _on_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.narration_label.setText("🎉 Мастер завершил работу. Спросите «что происходит».")
        self.panel_narration.emit("🎉 Мастер завершил работу.")
        self.wizard_finished.emit()

    def _on_aborted(self):
        self._reset_styles()

    def _on_step_error(self, step_id: str, error: str):
        self.narration_label.setText(f"⚠️ Ошибка на шаге «{step_id}»: {error}")
        self.panel_narration.emit(f"⚠️ Ошибка на шаге «{step_id}»: {error}")

    def _reset_styles(self):
        clear_tab_highlight(self._tab_widget())
        for i in range(self.plan_list.count()):
            item = self.plan_list.item(i)
            item.setForeground(Qt.gray)
            item.setBackground(Qt.transparent)

    def _tab_widget(self):
        return getattr(self, "_tab_widget_ref", None)

    def bind_tabs(self, tab_widget):
        """Привязывает QTabWidget для подсветки вкладок (вызывается из MainWindow)."""
        self._tab_widget_ref = tab_widget
