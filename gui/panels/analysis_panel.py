"""
gui/panels/analysis_panel.py
=============================
Панель полного анализа результатов обучения + песочница для тестирования модели.


Ответственность:
  • Полный структурированный отчёт об обучении (что делали, что хотели, что получили)
  • Рекомендации с кнопками действий («Попробовать автоматически», «Сохранить»)
  • Сравнение с предыдущими попытками (если есть история)
  • Песочница: ввод примера → предсказание → вердикт
  • Сигналы для повторного обучения и экспорта

Зависимости (импорт):
  • core.contracts — TrainingHistory, ModelConfig (если доступны; иначе словари)
  • core.explainer — Explainer (анализ графиков)
  • core.logger — get_logger
  • core.curator — Curator (комментарии к предсказаниям)

Сигналы (выходы):
  • retry_requested(dict) — запрос на повторное обучение с новым конфигом
  • export_requested() — запрос на переход к экспорту
  • save_requested() — запрос на сохранение модели

Входы (слоты):
  • run_analysis(dict | TrainingHistory) — вызывается по сигналу training_finished

shared_state ключи, которые читает:
  • "model" — nn.Module или None
  • "config" — dict (конфиг модели)
  • "dataset" — dict (DatasetContainer-совместимый)
  • "session" — SessionState или dict (опционально)
  • "project" — dict с "model_name", "scenario" (опционально)

shared_state ключи, которые пишет:
  • Ничего не пишет. Только читает.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QGroupBox, QLineEdit, QPushButton, QLabel,
    QMessageBox, QScrollArea, QFrame, QSizePolicy,
    QSplitter, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor

from core.explainer import Explainer
from core.logger import get_logger

logger = get_logger()

# ============================================================
#  ПОПЫТКА ИМПОРТА КОНТРАКТОВ (обратная совместимость)
# ============================================================
try:
    from core.contracts import (
        TrainingHistory, ModelConfig, config_to_dict, dataset_to_legacy_dict,
    )
    HAS_CONTRACTS = True
except ImportError:
    HAS_CONTRACTS = False
    TrainingHistory = None
    ModelConfig = None

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

# ============================================================
#  ПОПЫТКА ИМПОРТА КУРАТОРА (заглушка если нет)
# ============================================================
try:
    from core.curator import Curator
    HAS_CURATOR = True
except ImportError:
    HAS_CURATOR = False
    Curator = None


class AnalysisPanel(QWidget):
    """
    Панель анализа результатов обучения.

    Структура:
      ┌─────────────────────────────────────────────────────┐
      │ 📊 ПОЛНЫЙ ОТЧЁТ ОБ ОБУЧЕНИИ                        │
      │ (модель, время, результаты, что выучила)           │
      ├─────────────────────────────────────────────────────┤
      │ 💡 РЕКОМЕНДАЦИИ С КНОПКАМИ ДЕЙСТВИЙ               │
      │ [🚀 Попробовать автоматически] [💾 Сохранить]     │
      ├─────────────────────────────────────────────────────┤
      │ 📈 СРАВНЕНИЕ С ПРЕДЫДУЩИМИ ПОПЫТКАМИ              │
      ├─────────────────────────────────────────────────────┤
      │ 🧪 ПЕСОЧНИЦА: Проверь нейросеть сам               │
      │ [Ввод примера] [🚀 Спросить ИИ]                   │
      │ [Ответ ИИ]                                        │
      │ [Вердикт и пояснение]                             │
      └─────────────────────────────────────────────────────┘
    """

    # === СИГНАЛЫ ===
    retry_requested = pyqtSignal(dict)   # Новый конфиг для повторного обучения
    export_requested = pyqtSignal()       # Запрос перехода к экспорту
    save_requested = pyqtSignal()         # Запрос сохранения модели

    def __init__(self, shared_state: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.shared_state = shared_state

        # История попыток для сравнения
        self._previous_results: List[Dict[str, Any]] = []

        # Текущий отчёт (для кнопок действий)
        self._current_history: Optional[Dict] = None
        self._current_recommendations: List[Dict] = []

        # Куратор (если доступен)
        self._curator = None
        if HAS_CURATOR:
            try:
                session = shared_state.get("session")
                self._curator = Curator(session)
            except Exception:
                self._curator = None

        self.init_ui()

    # ============================================================
    #  ИНИЦИАЛИЗАЦИЯ ИНТЕРФЕЙСА
    # ============================================================
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Используем QSplitter для гибкого разделения пространства
        splitter = QSplitter(Qt.Vertical)

        # === ВЕРХНЯЯ ЧАСТЬ: Отчёт + Рекомендации ===
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)

        # --- Отчёт об обучении ---
        report_group = QGroupBox("📊 Полный отчёт об обучении")
        report_layout = QVBoxLayout()

        # Карточка модели (имя, архитектура, дата)
        self.model_card = QLabel("")
        self.model_card.setWordWrap(True)
        self.model_card.setTextFormat(Qt.RichText)
        self.model_card.setStyleSheet(
            "font-size: 13px; padding: 8px; "
            "background-color: #242424; border-left: 4px solid #4a88c7;"
        )
        self.model_card.setText(
            "<i>Пройдите процесс обучения, чтобы получить детальный отчёт.</i>"
        )
        report_layout.addWidget(self.model_card)

        # Основной текст отчёта (прокручиваемый)
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setStyleSheet(
            "font-family: 'Segoe UI', Arial, sans-serif; "
            "font-size: 14px; background-color: #1e1e1e; padding: 8px;"
        )
        self.report_text.setPlaceholderText("Здесь появится подробный анализ...")
        self.report_text.setMinimumHeight(120)
        report_layout.addWidget(self.report_text)

        report_group.setLayout(report_layout)
        top_layout.addWidget(report_group)

        # --- Рекомендации с кнопками ---
        recs_group = QGroupBox("💡 Вердикт Советника: Что делать дальше")
        recs_layout = QVBoxLayout()

        self.recommendations_text = QTextEdit()
        self.recommendations_text.setReadOnly(True)
        self.recommendations_text.setStyleSheet(
            "font-family: 'Segoe UI', Arial, sans-serif; "
            "font-size: 14px; background-color: #242424; padding: 8px;"
        )
        self.recommendations_text.setMinimumHeight(80)
        recs_layout.addWidget(self.recommendations_text)

        # Кнопки действий
        actions_layout = QHBoxLayout()

        self.btn_retry = QPushButton("🚀 Попробовать с рекомендациями")
        self.btn_retry.setToolTip(
            "Применить предложенные изменения и запустить обучение заново. "
            "Вы увидите, что именно будет изменено."
        )
        self.btn_retry.setStyleSheet(
            "QPushButton { background-color: #385a3a; border-color: #4a7a4c; "
            "font-weight: bold; padding: 8px 16px; }"
            "QPushButton:hover { background-color: #4a7a4c; }"
        )
        self.btn_retry.clicked.connect(self._on_retry_clicked)
        self.btn_retry.setEnabled(False)

        self.btn_save = QPushButton("💾 Сохранить модель")
        self.btn_save.setToolTip(
            "Сохранить текущую модель в файл (.pth или .oai). "
            "Если результат хороший — не потеряете!"
        )
        self.btn_save.setStyleSheet(
            "QPushButton { background-color: #36414f; border-color: #4c5052; "
            "padding: 8px 16px; }"
            "QPushButton:hover { background-color: #4c5052; }"
        )
        self.btn_save.clicked.connect(self._on_save_clicked)
        self.btn_save.setEnabled(False)

        self.btn_export = QPushButton("📤 Экспорт на флешку")
        self.btn_export.setToolTip(
            "Создать автономную папку с моделью, которую можно запустить "
            "на любом компьютере без установки программы."
        )
        self.btn_export.setStyleSheet(
            "QPushButton { background-color: #36414f; border-color: #4c5052; "
            "padding: 8px 16px; }"
            "QPushButton:hover { background-color: #4c5052; }"
        )
        self.btn_export.clicked.connect(self._on_export_clicked)
        self.btn_export.setEnabled(False)

        actions_layout.addWidget(self.btn_retry)
        actions_layout.addWidget(self.btn_save)
        actions_layout.addWidget(self.btn_export)
        actions_layout.addStretch()
        recs_layout.addLayout(actions_layout)

        recs_group.setLayout(recs_layout)
        top_layout.addWidget(recs_group)

        splitter.addWidget(top_widget)

        # === СРЕДНЯЯ ЧАСТЬ: Сравнение попыток ===
        comparison_widget = QWidget()
        comparison_layout = QVBoxLayout(comparison_widget)
        comparison_layout.setContentsMargins(0, 0, 0, 0)

        self.comparison_group = QGroupBox("📈 Сравнение с предыдущими попытками")
        comp_inner_layout = QVBoxLayout()

        self.comparison_table = QTableWidget()
        self.comparison_table.setColumnCount(5)
        self.comparison_table.setHorizontalHeaderLabels(
            ["Попытка", "Точность", "Val Loss", "Эпох", "Прогресс"]
        )
        self.comparison_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.comparison_table.setMaximumHeight(100)
        self.comparison_table.setVisible(False)
        comp_inner_layout.addWidget(self.comparison_table)

        self.comparison_label = QLabel(
            "<i>Пока нет предыдущих попыток для сравнения. "
            "Обучите модель несколько раз, чтобы увидеть прогресс.</i>"
        )
        self.comparison_label.setWordWrap(True)
        self.comparison_label.setStyleSheet("color: #888; font-size: 12px;")
        comp_inner_layout.addWidget(self.comparison_label)

        self.comparison_group.setLayout(comp_inner_layout)
        comparison_layout.addWidget(self.comparison_group)

        splitter.addWidget(comparison_widget)

        # === НИЖНЯЯ ЧАСТЬ: Песочница ===
        sandbox_widget = QWidget()
        sandbox_layout = QVBoxLayout(sandbox_widget)
        sandbox_layout.setContentsMargins(0, 0, 0, 0)

        sandbox_group = QGroupBox("🧪 Песочница: Проверь нейросеть сам")
        sb_layout = QVBoxLayout()

        # Подсказка
        sb_info = QLabel(
            "Введите свой пример, чтобы узнать, чему на самом деле научилась сеть. "
            "Программа покажет ответ и прокомментирует его."
        )
        sb_info.setWordWrap(True)
        sb_info.setStyleSheet("color: #a9b7c6; font-style: italic; font-size: 13px;")
        sb_layout.addWidget(sb_info)

        # Ввод + кнопка
        input_layout = QHBoxLayout()
        self.test_input = QLineEdit()
        self.test_input.setPlaceholderText(
            "Пример: 2+2  или  зашифрованное слово  или  1.5 2.0 3.0"
        )
        self.test_input.setStyleSheet("font-size: 16px; padding: 8px;")
        self.test_input.returnPressed.connect(self._run_prediction)
        input_layout.addWidget(self.test_input)

        self.btn_predict = QPushButton("🚀 Спросить ИИ")
        self.btn_predict.setStyleSheet(
            "font-size: 14px; font-weight: bold; padding: 8px 16px; "
            "background-color: #385a3a; border-color: #4a7a4c;"
        )
        self.btn_predict.clicked.connect(self._run_prediction)
        input_layout.addWidget(self.btn_predict)
        sb_layout.addLayout(input_layout)

        # Ответ
        output_layout = QHBoxLayout()
        output_label = QLabel("Ответ ИИ:")
        output_label.setStyleSheet("font-weight: bold; color: #cc7832; font-size: 16px;")
        self.test_output = QLineEdit()
        self.test_output.setReadOnly(True)
        self.test_output.setStyleSheet(
            "font-size: 16px; padding: 8px; background-color: #1e1e1e; "
            "color: #a6e3a1; font-weight: bold;"
        )
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.test_output)
        sb_layout.addLayout(output_layout)

        # Вердикт и пояснение
        self.verdict_label = QLabel("")
        self.verdict_label.setWordWrap(True)
        self.verdict_label.setTextFormat(Qt.RichText)
        self.verdict_label.setStyleSheet(
            "color: #a9b7c6; font-size: 13px; margin-top: 5px; "
            "padding: 6px; background-color: #242424; border-radius: 3px;"
        )
        self.verdict_label.setVisible(False)
        sb_layout.addWidget(self.verdict_label)

        sandbox_group.setLayout(sb_layout)
        sandbox_layout.addWidget(sandbox_group)

        splitter.addWidget(sandbox_widget)

        # Настройка пропорций сплиттера
        splitter.setSizes([400, 80, 200])

        layout.addWidget(splitter)

    # ============================================================
    #  ГЛАВНЫЙ СЛОТ: АНАЛИЗ ПОСЛЕ ОБУЧЕНИЯ
    # ============================================================
    def run_analysis(self, history):
        """
        Вызывается по сигналу training_finished из training_panel.

        Args:
            history: dict или TrainingHistory с ключами:
                - train_loss: List[float]
                - val_loss: List[float]
                - train_metric: List[float]
                - val_metric: List[float]
        """
        # Нормализуем вход (поддержка и dict, и dataclass)
        if HAS_CONTRACTS and isinstance(history, TrainingHistory):
            hist_dict = {
                "train_loss": history.train_loss,
                "val_loss": history.val_loss,
                "train_metric": history.train_metric,
                "val_metric": history.val_metric,
            }
        elif isinstance(history, dict):
            hist_dict = history
        else:
            hist_dict = {
                "train_loss": [],
                "val_loss": [],
                "train_metric": [],
                "val_metric": [],
            }

        self._current_history = hist_dict

        # 1. Формируем карточку модели
        self._build_model_card(hist_dict)

        # 2. Формируем основной отчёт
        self._build_report(hist_dict)

        # 3. Формируем рекомендации
        self._build_recommendations(hist_dict)

        # 4. Обновляем сравнение
        self._update_comparison(hist_dict)

        # 5. Активируем кнопки
        self.btn_save.setEnabled(True)
        self.btn_export.setEnabled(True)

        # Определяем, есть ли смысл предлагать повтор
        if self._current_recommendations:
            self.btn_retry.setEnabled(True)
        else:
            self.btn_retry.setEnabled(False)

        logger.info("Анализ обучения завершён. Отчёт сформирован.")

    # ============================================================
    #  КАРТОЧКА МОДЕЛИ
    # ============================================================
    def _build_model_card(self, hist_dict: Dict):
        """Формирует верхнюю карточку с информацией о модели."""
        config = config_to_dict(self.shared_state.get("config"))
        project = self.shared_state.get("project", {})

        model_name = project.get("model_name", "Безымянная модель")
        arch_type = config.get("type", "unknown").upper()
        data_type = config.get("data_type", "numeric")

        # Параметры модели
        model = self.shared_state.get("model")
        params_str = "—"
        if model is not None:
            try:
                params_count = sum(p.numel() for p in model.parameters())
                params_str = f"{params_count:,}"
            except Exception:
                pass

        # Время обучения
        epochs_done = len(hist_dict.get("train_loss", []))
        total_epochs = config.get("epochs", epochs_done)

        # Дата
        date_str = time.strftime("%d.%m.%Y, %H:%M")

        card_html = (
            f"<b>🧠 Модель: «{model_name}»</b><br>"
            f"📐 Архитектура: <b>{arch_type}</b> | "
            f"Параметров: {params_str}<br>"
            f"📊 Данные: {data_type} | "
            f"🔄 Эпох завершено: {epochs_done} из {total_epochs}<br>"
            f"📅 Дата: {date_str}"
        )
        self.model_card.setText(card_html)

    # ============================================================
    #  ОСНОВНОЙ ОТЧЁТ
    # ============================================================
    def _build_report(self, hist_dict: Dict):
        """Формирует подробный текстовый отчёт."""
        train_loss = hist_dict.get("train_loss", [])
        val_loss = hist_dict.get("val_loss", [])
        val_metric = hist_dict.get("val_metric", [])
        train_metric = hist_dict.get("train_metric", [])

        html_parts = []

        # --- Результаты ---
        html_parts.append("<h3>📈 РЕЗУЛЬТАТЫ:</h3>")

        if val_loss:
            final_val_loss = val_loss[-1]
            best_val_loss = min(val_loss)
            best_epoch = val_loss.index(best_val_loss) + 1
            html_parts.append(
                f"<p>• Финальная ошибка (Val Loss): <b>{final_val_loss:.4f}</b></p>"
            )
            html_parts.append(
                f"<p>• Лучшая ошибка: <b>{best_val_loss:.4f}</b> "
                f"(эпоха {best_epoch})</p>"
            )

        if val_metric:
            last_metric = val_metric[-1]
            if 0 <= last_metric <= 1:
                color = self._metric_color(last_metric)
                html_parts.append(
                    f"<p>• Точность предсказания: "
                    f"<b style='color:{color};'>{last_metric:.1%}</b></p>"
                )
            else:
                html_parts.append(
                    f"<p>• Финальная метрика: <b>{last_metric:.4f}</b></p>"
                )

        # --- Что сеть выучила ---
        html_parts.append("<hr>")
        html_parts.append("<h3>🎯 ЧТО СЕТЬ ВЫУЧИЛА:</h3>")

        config = config_to_dict(self.shared_state.get("config"))
        data_type = config.get("data_type", "numeric")

        if data_type == "text" and val_metric:
            last_metric = val_metric[-1]
            if last_metric >= 0.9:
                html_parts.append(
                    "<p style='color:#a6e3a1;'>✅ Модель отлично освоила задачу! "
                    "Точность выше 90%.</p>"
                )
            elif last_metric >= 0.7:
                html_parts.append(
                    "<p style='color:#e5c07b;'>⚠️ Модель понимает логику, "
                    "но иногда ошибается в деталях.</p>"
                )
            elif last_metric >= 0.4:
                html_parts.append(
                    "<p style='color:#e5c07b;'>⚠️ Модель улавливает общие "
                    "закономерности, но часто ошибается.</p>"
                )
            else:
                html_parts.append(
                    "<p style='color:#e06c75;'>❌ Модель пока не справляется "
                    "с задачей. Нужно больше данных или мощнее архитектура.</p>"
                )
        else:
            if val_loss and train_loss:
                ratio = val_loss[-1] / max(train_loss[-1], 1e-10)
                if ratio < 1.2:
                    html_parts.append(
                        "<p style='color:#a6e3a1;'>✅ Модель хорошо "
                        "обобщает (нет переобучения).</p>"
                    )
                elif ratio < 2.0:
                    html_parts.append(
                        "<p style='color:#e5c07b;'>⚠️ Есть признаки "
                        "небольшого переобучения.</p>"
                    )
                else:
                    html_parts.append(
                        "<p style='color:#e06c75;'>❌ Сильное переобучение: "
                        "сеть зубрит, а не понимает.</p>"
                    )

        # --- Динамика обучения ---
        if len(train_loss) >= 5:
            initial_loss = train_loss[0]
            final_loss = train_loss[-1]
            improvement = ((initial_loss - final_loss) / max(initial_loss, 1e-10)) * 100
            html_parts.append(
                f"<p>• Ошибка снизилась на <b>{improvement:.1f}%</b> "
                f"за время обучения.</p>"
            )

        self.report_text.setHtml("\n".join(html_parts))

    # ============================================================
    #  РЕКОМЕНДАЦИИ С КНОПКАМИ
    # ============================================================
    def _build_recommendations(self, hist_dict: Dict):
        """Формирует рекомендации и сохраняет для кнопки 'Попробовать'."""
        report = Explainer.analyze_training_results(hist_dict)

        # Новый Explainer возвращает FinalReport (dataclass), а не список строк.
        if hasattr(report, "recommendations"):
            recommendations = report.recommendations
        elif isinstance(report, (list, tuple)):
            recommendations = report
        else:
            recommendations = []

        # Сохраняем для кнопки действий
        self._current_recommendations = self._parse_recommendations_to_actions(
            hist_dict
        )

        # Формируем HTML
        html_parts = []
        for rec in recommendations:
            if isinstance(rec, str):
                text = rec
            else:
                title = getattr(rec, "title", "")
                desc = getattr(rec, "description", "")
                text = f"<b>{title}</b><br>{desc}" if title else desc
            html_parts.append(
                f"<div style='margin-bottom: 12px; padding: 10px; "
                f"border-left: 4px solid #4a88c7; "
                f"background-color: #1e1e1e;'>{text}</div>"
            )

        if not recommendations:
            html_parts.append(
                "<p style='color:#a6e3a1;'>🌟 Отличный результат! "
                "Дополнительных рекомендаций нет.</p>"
            )

        # Если есть предложения по изменению параметров
        if self._current_recommendations:
            html_parts.append("<hr>")
            html_parts.append(
                "<p><b>🔧 Предложенные изменения:</b></p><ul>"
            )
            for action in self._current_recommendations:
                html_parts.append(f"<li>{action['description']}</li>")
            html_parts.append("</ul>")
            html_parts.append(
                "<p style='color:#888; font-size:12px;'>"
                "Нажмите «Попробовать с рекомендациями», чтобы применить "
                "эти изменения и запустить обучение заново.</p>"
            )

        self.recommendations_text.setHtml("\n".join(html_parts))

    def _parse_recommendations_to_actions(
        self, hist_dict: Dict
    ) -> List[Dict[str, Any]]:
        """
        Анализирует историю и формирует конкретные изменения конфига.
        Возвращает список действий для кнопки «Попробовать».
        """
        actions = []
        config = config_to_dict(self.shared_state.get("config"))
        train_loss = hist_dict.get("train_loss", [])
        val_loss = hist_dict.get("val_loss", [])

        if not train_loss or not val_loss:
            return actions

        # Переобучение → увеличить weight_decay, уменьшить эпохи
        if len(val_loss) > 5 and val_loss[-1] > train_loss[-1] * 1.5:
            current_wd = config.get("weight_decay", 0.0)
            new_wd = max(current_wd, 0.01)
            if new_wd != current_wd:
                actions.append({
                    "param": "weight_decay",
                    "old": current_wd,
                    "new": new_wd,
                    "description": (
                        f"Увеличить штраф за зубрёжку (L2): "
                        f"{current_wd} → {new_wd}"
                    ),
                })

            current_epochs = config.get("epochs", 50)
            new_epochs = max(10, int(current_epochs * 0.6))
            if new_epochs != current_epochs:
                actions.append({
                    "param": "epochs",
                    "old": current_epochs,
                    "new": new_epochs,
                    "description": (
                        f"Уменьшить количество эпох: "
                        f"{current_epochs} → {new_epochs}"
                    ),
                })

        # Расходимость → уменьшить LR
        if len(val_loss) >= 5 and val_loss[-1] > val_loss[-5]:
            current_lr = config.get("learning_rate", 0.001)
            new_lr = current_lr / 10.0
            actions.append({
                "param": "learning_rate",
                "old": current_lr,
                "new": new_lr,
                "description": (
                    f"Уменьшить скорость обучения: "
                    f"{current_lr} → {new_lr}"
                ),
            })

        # Плато → увеличить модель (если трансформер)
        if len(train_loss) > 5 and (train_loss[-5] - train_loss[-1]) < 0.0001:
            arch_type = config.get("type", "")
            if arch_type == "transformer":
                current_embed = config.get("embedding_dim", 128)
                new_embed = min(current_embed * 2, 512)
                if new_embed != current_embed:
                    actions.append({
                        "param": "embedding_dim",
                        "old": current_embed,
                        "new": new_embed,
                        "description": (
                            f"Увеличить эмбеддинг: "
                            f"{current_embed} → {new_embed}"
                        ),
                    })

        return actions

    # ============================================================
    #  СРАВНЕНИЕ С ПРЕДЫДУЩИМИ ПОПЫТКАМИ
    # ============================================================
    def _update_comparison(self, hist_dict: Dict):
        """Добавляет текущий результат в историю и обновляет таблицу."""
        val_metric = hist_dict.get("val_metric", [])
        val_loss = hist_dict.get("val_loss", [])

        current_result = {
            "metric": val_metric[-1] if val_metric else 0.0,
            "val_loss": val_loss[-1] if val_loss else 0.0,
            "epochs": len(hist_dict.get("train_loss", [])),
            "timestamp": time.strftime("%H:%M:%S"),
        }
        self._previous_results.append(current_result)

        # Если только одна попытка — показываем сообщение
        if len(self._previous_results) < 2:
            self.comparison_table.setVisible(False)
            self.comparison_label.setVisible(True)
            self.comparison_label.setText(
                "<i>Первая попытка записана. Обучите модель ещё раз, "
                "чтобы увидеть сравнение.</i>"
            )
            return

        # Показываем таблицу
        self.comparison_table.setVisible(True)
        self.comparison_label.setVisible(False)

        n = len(self._previous_results)
        self.comparison_table.setRowCount(n)

        for i, res in enumerate(self._previous_results):
            self.comparison_table.setItem(
                i, 0, QTableWidgetItem(f"#{i + 1} ({res['timestamp']})")
            )

            metric_val = res["metric"]
            metric_item = QTableWidgetItem(f"{metric_val:.1%}")
            metric_item.setForeground(
                QColor(self._metric_color(metric_val))
            )
            self.comparison_table.setItem(i, 1, metric_item)

            self.comparison_table.setItem(
                i, 2, QTableWidgetItem(f"{res['val_loss']:.4f}")
            )
            self.comparison_table.setItem(
                i, 3, QTableWidgetItem(str(res["epochs"]))
            )

            # Прогресс относительно предыдущей попытки
            if i > 0:
                prev_metric = self._previous_results[i - 1]["metric"]
                diff = metric_val - prev_metric
                if diff > 0.01:
                    progress_str = f"📈 +{diff:.1%}"
                elif diff < -0.01:
                    progress_str = f"📉 {diff:.1%}"
                else:
                    progress_str = "➡️ Без изменений"
            else:
                progress_str = "— Первая попытка"

            self.comparison_table.setItem(i, 4, QTableWidgetItem(progress_str))

        # Поздравление при прогрессе
        if n >= 2:
            last = self._previous_results[-1]["metric"]
            prev = self._previous_results[-2]["metric"]
            if last > prev + 0.05:
                self.comparison_label.setVisible(True)
                self.comparison_label.setText(
                    f"<b style='color:#a6e3a1;'>🎉 Прогресс! "
                    f"Точность выросла с {prev:.1%} до {last:.1%}.</b>"
                )
                self.comparison_label.setStyleSheet("font-size: 13px;")

    # ============================================================
    #  ПЕСОЧНИЦА: ПРЕДСКАЗАНИЕ
    # ============================================================
    def _run_prediction(self):
        """Запускает предсказание на введённом примере."""
        model = self.shared_state.get("model")
        dataset = self.shared_state.get("dataset")

        if not model or not dataset:
            QMessageBox.warning(
                self, "Внимание",
                "Нейросеть ещё не готова!\n"
                "Создайте её во вкладке «Архитектура» и загрузите данные."
            )
            return

        input_text = self.test_input.text().strip()
        if not input_text:
            return

        ds_meta = dataset_to_legacy_dict(dataset).get("meta", {})
        data_type = ds_meta.get("type", "numeric")
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model.to(device)
        model.eval()

        try:
            with torch.no_grad():
                if data_type == "text":
                    result, details, is_correct = self._predict_text(
                        model, dataset, input_text, device
                    )
                else:
                    result, details, is_correct = self._predict_numeric(
                        model, ds_meta, input_text, device
                    )

            self.test_output.setText(result)
            self._show_verdict(input_text, result, details, is_correct)

        except ValueError as ve:
            QMessageBox.warning(self, "Ошибка ввода", str(ve))
        except Exception as e:
            logger.error(f"Сбой предсказания: {e}")
            self.test_output.setText("❌ Произошёл сбой нейросети")
            self.verdict_label.setText(
                f"<span style='color:#e06c75;'>Ошибка: {str(e)}</span>"
            )
            self.verdict_label.setVisible(True)

    def _predict_text(
        self, model, dataset: Dict, input_text: str, device
    ) -> Tuple[str, str, bool]:
        """Предсказание для текстовой модели."""
        dataset = dataset_to_legacy_dict(dataset)
        vocab = dataset.get(
            "vocab", {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        )
        inv_vocab = {v: k for k, v in vocab.items()}

        x_seq = [vocab.get(char, vocab.get("<UNK>", 1)) for char in input_text]
        unknowns = sum(1 for x in x_seq if x == vocab.get("<UNK>", 1))

        x_tensor = torch.tensor([x_seq], dtype=torch.long).to(device)

        if hasattr(model, "generate"):
            config = config_to_dict(self.shared_state.get("config"))
            max_len = config.get("max_len", 50)
            output_indices = model.generate(x_tensor, max_len=max_len)
            generated = output_indices[0].cpu().tolist()

            chars = []
            for idx in generated:
                if idx == vocab.get("<EOS>", 3):
                    break
                if idx in (vocab.get("<PAD>", 0), vocab.get("<BOS>", 2)):
                    continue
                chars.append(inv_vocab.get(idx, "?"))

            result = "".join(chars)

            details = (
                f"🧠 Сеть прочитала {len(x_seq)} символов "
                f"и сгенерировала {len(chars)} в ответ."
            )
            if unknowns > 0:
                details += (
                    f"\n⚠️ {unknowns} символов из вашего вопроса "
                    f"сеть видит впервые (заменены на UNK)!"
                )

            return result, details, True
        else:
            return (
                "Архитектура не поддерживает генерацию текста.",
                "",
                False
            )

    def _predict_numeric(
        self, model, ds_meta: Dict, input_text: str, device
    ) -> Tuple[str, str, bool]:
        """Предсказание для числовой модели."""
        try:
            x_vals = [
                float(x) for x in input_text.replace(",", " ").split()
            ]
        except ValueError:
            raise ValueError(
                "Я жду только числа, разделённые пробелом "
                "(например: '1.5 2.0')"
            )

        expected_dim = ds_meta.get("input_dim", 1)
        if len(x_vals) != expected_dim:
            raise ValueError(
                f"Эта сеть училась принимать на вход ровно "
                f"{expected_dim} чисел(а), а вы дали {len(x_vals)}."
            )

        x_tensor = torch.tensor(
            [x_vals], dtype=torch.float32
        ).to(device)
        output_tensor = model(x_tensor)
        result = str(np.round(output_tensor[0].cpu().numpy(), 4))

        details = f"🧠 Обработано {len(x_vals)} числовых признаков."
        return result, details, True

    def _show_verdict(
        self,
        input_text: str,
        result: str,
        details: str,
        is_correct: bool
    ):
        """Показывает вердикт с комментарием куратора."""
        verdict_html = ""

        # Комментарий куратора (если доступен)
        curator_comment = ""
        if self._curator and hasattr(self._curator, "comment_prediction"):
            try:
                curator_comment = self._curator.comment_prediction(
                    input_text, "", result
                )
            except Exception:
                pass

        if details:
            verdict_html += f"<p>{details}</p>"

        if curator_comment:
            verdict_html += (
                f"<p style='color:#e5c07b; font-style:italic;'>"
                f"💬 {curator_comment}</p>"
            )

        if not is_correct:
            verdict_html += (
                "<p style='color:#e06c75;'>❌ Модель не смогла "
                "обработать этот запрос.</p>"
            )

        self.verdict_label.setText(verdict_html)
        self.verdict_label.setVisible(bool(verdict_html))

    # ============================================================
    #  ОБРАБОТЧИКИ КНОПОК ДЕЙСТВИЙ
    # ============================================================
    def _on_retry_clicked(self):
        """
        Формирует новый конфиг на основе рекомендаций
        и отправляет сигнал для повторного обучения.
        """
        if not self._current_recommendations:
            QMessageBox.information(
                self, "Информация",
                "Нет конкретных рекомендаций для применения.\n"
                "Попробуйте изменить параметры вручную."
            )
            return

        # Копируем текущий конфиг
        config = config_to_dict(self.shared_state.get("config"))

        # Применяем изменения
        changes_summary = []
        for action in self._current_recommendations:
            param = action["param"]
            new_val = action["new"]
            config[param] = new_val
            changes_summary.append(
                f"• {action['description']}"
            )

        # Подтверждение
        msg = (
            "Будут применены следующие изменения:\n\n"
            + "\n".join(changes_summary)
            + "\n\nЗапустить обучение заново?"
        )
        reply = QMessageBox.question(
            self,
            "Повторное обучение",
            msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        if reply == QMessageBox.Yes:
            logger.info(
                f"Повторное обучение с изменениями: {changes_summary}"
            )
            self.retry_requested.emit(config)

    def _on_save_clicked(self):
        """Отправляет сигнал на сохранение модели."""
        self.save_requested.emit()

    def _on_export_clicked(self):
        """Отправляет сигнал на экспорт."""
        self.export_requested.emit()

    # ============================================================
    #  ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ============================================================
    @staticmethod
    def _metric_color(value: float) -> str:
        """Возвращает цвет для значения метрики."""
        if value >= 0.8:
            return "#a6e3a1"  # зелёный
        elif value >= 0.5:
            return "#e5c07b"  # жёлтый
        else:
            return "#e06c75"  # красный

    def reset(self):
        """Сбрасывает панель к начальному состоянию."""
        self.model_card.setText(
            "<i>Пройдите процесс обучения, чтобы получить детальный отчёт.</i>"
        )
        self.report_text.clear()
        self.recommendations_text.clear()
        self.test_output.clear()
        self.verdict_label.setVisible(False)
        self.btn_retry.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.btn_export.setEnabled(False)
        self._current_history = None
        self._current_recommendations = []