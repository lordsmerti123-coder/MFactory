"""
gui/panels/training_panel.py
=============================

Ответственность:
- Карточка модели (имя, архитектура, параметры, данные, статус)
- Управление циклом обучения (старт/пауза/стоп)
- Прогресс выполнения (эпохи + батчи)
- Безопасность (лимиты, подтверждение остановки, автосохранение)
- Сохранение/загрузка модели
- Логирование действий в сессию

Зависимости:
- core.contracts (ModelConfig, TrainingHistory, DataType, ArchitectureType)
- core.trainer (Trainer, TrainingCallback)
- core.dataset (OracleTorchDataset)
- core.model_factory (ModelFactory)
- core.curator (Curator) — для генерации имён и комментариев
- core.session (SessionManager) — через shared_state["session"]

Входы (через shared_state):
- shared_state["model"] — nn.Module или None
- shared_state["config"] — dict (конфиг модели и обучения)
- shared_state["dataset"] — DatasetContainer или None
- shared_state["session"] — SessionState
- shared_state["project"] — dict с "model_name", "scenario"

Выходы (сигналы):
- training_started(bool) — обучение началось, bool = is_text
- epoch_finished(int, dict) — эпоха завершена
- training_finished(dict) — обучение полностью завершено

Примечание по сборке:
- Файл самодостаточен. Импортирует только из контрактов и ядра.
- НЕ создаёт новых ключей в shared_state.
- Все виджеты имеют тултипы.
"""

import time
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QProgressBar, QMessageBox, QGroupBox, QFileDialog,
    QSpinBox, QCheckBox, QFormLayout, QFrame, QGridLayout
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QFont

from core.trainer import Trainer, TrainingCallback
from core.dataset import OracleTorchDataset
from core.model_factory import ModelFactory
from core.logger import get_logger

try:
    from core.contracts import (
        ModelConfig, TrainingHistory, config_to_dict, dataset_to_legacy_dict,
    )
except ImportError:
    ModelConfig = None
    TrainingHistory = None

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

logger = get_logger("TrainingPanel")


# ============================================================
# ПОТОК ОБУЧЕНИЯ
# ============================================================
class TrainerWorker(QThread, TrainingCallback):
    """
    Фоновый поток обучения. Реализует TrainingCallback
    для получения логов из Trainer.

    Сигналы:
    - epoch_done(int, dict) — завершена эпоха
    - batch_done(int, int, int, float, str) — завершён батч
    - finished(dict) — обучение завершено, передаёт history
    - error_occurred(str) — критическая ошибка
    - checkpoint_saved(int, str) — чекпоинт сохранён (эпоха, путь)
    """
    epoch_done = pyqtSignal(int, dict)
    batch_done = pyqtSignal(int, int, int, float, str)
    finished = pyqtSignal(object)  # TrainingHistory (dataclass) или dict
    error_occurred = pyqtSignal(str)
    checkpoint_saved = pyqtSignal(int, str)

    def __init__(self, trainer: Trainer, checkpoint_dir: Optional[Path] = None,
                 checkpoint_interval: int = 0):
        super().__init__()
        self.trainer = trainer
        self.trainer.add_callback(self)
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self._start_time = 0.0

    def run(self):
        """Запуск обучения в отдельном потоке."""
        self._start_time = time.time()
        try:
            history = self.trainer.train()
            elapsed = time.time() - self._start_time
            # TrainingHistory — dataclass, поэтому обращаемся к атрибуту,
            # а не к ключу (иначе TypeError).
            if hasattr(history, "total_time_seconds"):
                history.total_time_seconds = elapsed
            elif isinstance(history, dict):
                history["total_time_seconds"] = elapsed
            self.finished.emit(history)
        except Exception as e:
            logger.error(f"Критическая ошибка в потоке обучения: {e}")
            self.error_occurred.emit(str(e))

    # === Callbacks от Trainer ===
    def on_epoch_end(self, epoch: int, logs: dict):
        self.epoch_done.emit(epoch, logs)

        # Автосохранение чекпоинтов
        if (self.checkpoint_interval > 0 and
                self.checkpoint_dir is not None and
                epoch % self.checkpoint_interval == 0):
            try:
                ckpt_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
                ModelFactory.save_model(
                    self.trainer.model, ckpt_path, self.trainer.config
                )
                self.checkpoint_saved.emit(epoch, str(ckpt_path))
            except Exception as e:
                logger.warning(f"Не удалось сохранить чекпоинт: {e}")

    def on_batch_end(self, epoch: int, batch_idx: int,
                     total_batches: int, speed: float, eta_str: str):
        self.batch_done.emit(epoch, batch_idx, total_batches, speed, eta_str)


# ============================================================
# ПАНЕЛЬ ОБУЧЕНИЯ
# ============================================================
class TrainingPanel(QWidget):
    """
    Панель управления обучением нейросети.

    Отображает:
    - Карточку модели (имя, архитектура, параметры, данные)
    - Кнопки управления (старт/пауза/стоп)
    - Прогресс обучения (эпохи и батчи)
    - Настройки безопасности
    - Сохранение/загрузка весов
    """

    # === СИГНАЛЫ ===
    epoch_finished = pyqtSignal(int, dict)
    training_finished = pyqtSignal(dict)
    training_started = pyqtSignal(bool)  # True если текстовый режим

    def __init__(self, shared_state: dict, parent=None):
        super().__init__(parent)
        self.shared_state = shared_state
        self.worker: Optional[TrainerWorker] = None
        self.trainer: Optional[Trainer] = None
        self._training_start_time = 0.0
        self._is_training_active = False
        self.init_ui()

    # ============================================================
    # ИНИЦИАЛИЗАЦИЯ ИНТЕРФЕЙСА
    # ============================================================
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # === 1. КАРТОЧКА МОДЕЛИ ===
        self.model_card_group = QGroupBox("🧠 Карточка модели")
        self.model_card_group.setToolTip(
            "Полная информация о текущей модели в памяти.\n"
            "Если карточка пустая — создайте модель во вкладке «Архитектура»."
        )
        card_layout = QGridLayout()
        card_layout.setSpacing(4)

        # Имя модели
        self.name_label = QLabel("—")
        self.name_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.name_label.setStyleSheet("color: #4a88c7;")
        self.name_label.setToolTip("Имя вашей модели. Генерируется автоматически или задаётся вручную.")
        card_layout.addWidget(QLabel("Имя:"), 0, 0)
        card_layout.addWidget(self.name_label, 0, 1, 1, 3)

        # Архитектура
        self.arch_label = QLabel("—")
        self.arch_label.setToolTip("Тип нейросети (архитектура).")
        card_layout.addWidget(QLabel("Архитектура:"), 1, 0)
        card_layout.addWidget(self.arch_label, 1, 1, 1, 3)

        # Параметры
        self.params_label = QLabel("—")
        self.params_label.setToolTip(
            "Количество обучаемых параметров.\n"
            "Чем больше — тем мощнее модель, но дольше обучение и больше памяти."
        )
        card_layout.addWidget(QLabel("Параметры:"), 2, 0)
        card_layout.addWidget(self.params_label, 2, 1)

        # Память
        self.memory_label = QLabel("—")
        self.memory_label.setToolTip("Оценка занимаемой памяти модели (без учёта данных и градиентов).")
        card_layout.addWidget(QLabel("Память:"), 2, 2)
        card_layout.addWidget(self.memory_label, 2, 3)

        # Данные
        self.data_label = QLabel("—")
        self.data_label.setToolTip("Какие данные загружены для обучения (тип и размер).")
        card_layout.addWidget(QLabel("Данные:"), 3, 0)
        card_layout.addWidget(self.data_label, 3, 1, 1, 3)

        # Статус
        self.status_model_label = QLabel("—")
        self.status_model_label.setToolTip(
            "Откуда модель: создана заново или загружена из файла.\n"
            "Загруженную можно дообучить на новых данных."
        )
        card_layout.addWidget(QLabel("Статус:"), 4, 0)
        card_layout.addWidget(self.status_model_label, 4, 1, 1, 3)

        self.model_card_group.setLayout(card_layout)
        layout.addWidget(self.model_card_group)

        # === 2. УПРАВЛЕНИЕ ОБУЧЕНИЕМ ===
        control_group = QGroupBox("🚀 Управление обучением")
        control_group.setToolTip(
            "Кнопки запуска и остановки тренировочного процесса.\n"
            "Старт доступен только если модель создана и данные разбиты."
        )
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        # Старт
        self.btn_start = QPushButton("▶️  Старт обучения")
        self.btn_start.setObjectName("StartBtn")
        self.btn_start.setMinimumHeight(40)
        self.btn_start.setToolTip(
            "Запуск процесса обучения.\n"
            "Перед стартом убедитесь:\n"
            "• Модель создана (вкладка «Архитектура»)\n"
            "• Данные загружены и разбиты (вкладка «Данные»)\n"
            "• Гиперпараметры настроены (вкладка «Гиперпараметры»)"
        )
        self.btn_start.clicked.connect(self.start_training)
        btn_layout.addWidget(self.btn_start)

        # Пауза / Продолжить
        self.btn_pause = QPushButton("⏸  Пауза")
        self.btn_pause.setObjectName("PauseBtn")
        self.btn_pause.setMinimumHeight(40)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setToolTip(
            "Приостановить обучение.\n"
            "Модель запомнит текущее состояние.\n"
            "Нажмите снова для продолжения."
        )
        self.btn_pause.clicked.connect(self.toggle_pause)
        btn_layout.addWidget(self.btn_pause)

        # Стоп
        self.btn_stop = QPushButton("⏹  Стоп")
        self.btn_stop.setObjectName("StopBtn")
        self.btn_stop.setMinimumHeight(40)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setToolTip(
            "Остановить обучение после завершения текущей эпохи.\n"
            "Лучшие веса будут автоматически восстановлены.\n"
            "⚠️ Прогресс текущей эпохи будет потерян."
        )
        self.btn_stop.clicked.connect(self.stop_training)
        btn_layout.addWidget(self.btn_stop)

        control_group.setLayout(btn_layout)
        layout.addWidget(control_group)

        # === 3. ПРОГРЕСС ===
        progress_group = QGroupBox("📊 Прогресс выполнения")
        progress_group.setToolTip("Индикаторы прогресса обучения в реальном времени.")
        progress_layout = QVBoxLayout()

        # Эпохи
        self.epoch_progress = QProgressBar()
        self.epoch_progress.setFormat("Эпоха %v / %m")
        self.epoch_progress.setValue(0)
        self.epoch_progress.setMinimumHeight(22)
        self.epoch_progress.setToolTip(
            "Общий прогресс обучения по эпохам.\n"
            "Эпоха = один полный проход по всем тренировочным данным."
        )
        progress_layout.addWidget(self.epoch_progress)

        # Батчи
        self.batch_progress = QProgressBar()
        self.batch_progress.setFormat("Батч %v / %m")
        self.batch_progress.setValue(0)
        self.batch_progress.setMinimumHeight(18)
        self.batch_progress.setToolTip(
            "Прогресс внутри текущей эпохи.\n"
            "Батч = порция данных, обрабатываемая за один шаг."
        )
        progress_layout.addWidget(self.batch_progress)

        # Статусная строка
        self.status_label = QLabel("Статус: Жду команду на старт...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(
            "font-size: 14px; padding: 6px; "
            "background-color: #1e1e1e; border: 1px solid #333; border-radius: 3px;"
        )
        self.status_label.setToolTip("Текущее состояние процесса обучения.")
        progress_layout.addWidget(self.status_label)

        # Режим
        self.mode_label = QLabel("")
        self.mode_label.setAlignment(Qt.AlignCenter)
        self.mode_label.setStyleSheet("color: #777; font-size: 12px;")
        self.mode_label.setToolTip("Режим обучения: текстовый или числовой.")
        progress_layout.addWidget(self.mode_label)

        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # === 4. БЕЗОПАСНОСТЬ ===
        safety_group = QGroupBox("🛡️ Безопасность и автосохранение")
        safety_group.setToolTip(
            "Настройки защиты от аварий и потери прогресса.\n"
            "Рекомендуется оставить включёнными."
        )
        safety_layout = QFormLayout()

        # Лимит времени
        time_layout = QHBoxLayout()
        self.time_limit_check = QCheckBox("Ограничить время обучения")
        self.time_limit_check.setChecked(True)
        self.time_limit_check.setToolTip(
            "Если обучение длится дольше указанного времени — автоматическая остановка.\n"
            "Защита от забытых процессов."
        )
        self.time_limit_spin = QSpinBox()
        self.time_limit_spin.setRange(1, 48)
        self.time_limit_spin.setValue(4)
        self.time_limit_spin.setSuffix(" часов")
        self.time_limit_spin.setToolTip("Максимальное время обучения в часах.")
        time_layout.addWidget(self.time_limit_check)
        time_layout.addWidget(self.time_limit_spin)
        time_layout.addStretch()
        safety_layout.addRow("", time_layout)

        # Автосохранение чекпоинтов
        ckpt_layout = QHBoxLayout()
        self.checkpoint_check = QCheckBox("Автосохранение каждые")
        self.checkpoint_check.setChecked(True)
        self.checkpoint_check.setToolTip(
            "Периодически сохраняет веса модели на диск.\n"
            "Если обучение упадёт — можно восстановиться."
        )
        self.checkpoint_interval_spin = QSpinBox()
        self.checkpoint_interval_spin.setRange(1, 100)
        self.checkpoint_interval_spin.setValue(5)
        self.checkpoint_interval_spin.setSuffix(" эпох")
        self.checkpoint_interval_spin.setToolTip("Как часто сохранять контрольные точки.")
        ckpt_layout.addWidget(self.checkpoint_check)
        ckpt_layout.addWidget(self.checkpoint_interval_spin)
        ckpt_layout.addStretch()
        safety_layout.addRow("", ckpt_layout)

        # Подтверждение остановки
        self.confirm_stop_check = QCheckBox("Подтверждать остановку")
        self.confirm_stop_check.setChecked(True)
        self.confirm_stop_check.setToolTip(
            "Показывать диалог подтверждения при нажатии «Стоп».\n"
            "Защита от случайного клика."
        )
        safety_layout.addRow("", self.confirm_stop_check)

        safety_group.setLayout(safety_layout)
        layout.addWidget(safety_group)

        # === 5. СОХРАНЕНИЕ / ЗАГРУЗКА ===
        io_group = QGroupBox("💾 Сохранение и загрузка модели")
        io_group.setToolTip(
            "Сохраните обученную модель в файл (.pth), чтобы использовать позже.\n"
            "Загрузите ранее сохранённую модель для дообучения или тестирования."
        )
        io_layout = QHBoxLayout()

        self.btn_save = QPushButton("💾 Сохранить модель (.pth)")
        self.btn_save.setToolTip(
            "Сохранить веса модели + конфигурацию в файл.\n"
            "Файл можно загрузить позже для дообучения или экспорта."
        )
        self.btn_save.clicked.connect(self.save_model)
        io_layout.addWidget(self.btn_save)

        self.btn_load = QPushButton("📂 Загрузить модель (.pth)")
        self.btn_load.setToolTip(
            "Загрузить ранее сохранённую модель.\n"
            "После загрузки можно:\n"
            "• Дообучить на новых данных (нажать «Старт»)\n"
            "• Тестировать во вкладке «Анализ»\n"
            "• Экспортировать во вкладке «Экспорт»"
        )
        self.btn_load.clicked.connect(self.load_model)
        io_layout.addWidget(self.btn_load)

        io_group.setLayout(io_layout)
        layout.addWidget(io_group)

        layout.addStretch()

        # Таймер проверки лимита времени
        self._time_check_timer = QTimer(self)
        self._time_check_timer.setInterval(60000)  # каждую минуту
        self._time_check_timer.timeout.connect(self._check_time_limit)

    # ============================================================
    # ОБНОВЛЕНИЕ КАРТОЧКИ МОДЕЛИ
    # ============================================================
    def refresh_model_card(self):
        """Обновляет карточку модели из shared_state."""
        model = self.shared_state.get("model")
        config = config_to_dict(self.shared_state.get("config"))
        dataset = dataset_to_legacy_dict(self.shared_state.get("dataset"))
        project = self.shared_state.get("project", {})

        if not model:
            self.name_label.setText("⚠️ Модель не создана")
            self.name_label.setStyleSheet("color: #e06c75;")
            self.arch_label.setText("—")
            self.params_label.setText("—")
            self.memory_label.setText("—")
            self.data_label.setText("—")
            self.status_model_label.setText("Создайте модель во вкладке «Архитектура»")
            self.btn_start.setEnabled(False)
            return

        # Имя модели
        model_name = project.get("model_name", "") or config.get("model_name", "Безымянная")
        self.name_label.setText(model_name)
        self.name_label.setStyleSheet("color: #4a88c7;")

        # Архитектура
        arch_type = config.get("type", "unknown").upper()
        self.arch_label.setText(arch_type)

        # Параметры
        params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.params_label.setText(f"{params:,} (обучаемых: {trainable:,})")

        # Память
        mem_mb = ModelFactory.estimate_memory(model)
        if mem_mb > 500:
            self.memory_label.setText(f"~{mem_mb:.0f} МБ ⚠️")
            self.memory_label.setStyleSheet("color: #e06c75;")
        elif mem_mb > 100:
            self.memory_label.setText(f"~{mem_mb:.1f} МБ")
            self.memory_label.setStyleSheet("color: #e5c07b;")
        else:
            self.memory_label.setText(f"~{mem_mb:.2f} МБ")
            self.memory_label.setStyleSheet("color: #a6e3a1;")

        # Данные
        if dataset:
            meta = dataset.get("meta", {})
            data_type = meta.get("type", "unknown")
            num_samples = meta.get("num_samples", 0)
            self.data_label.setText(f"{data_type} ({num_samples:,} примеров)")
        else:
            self.data_label.setText("Датасет не загружен")
            self.data_label.setStyleSheet("color: #e06c75;")

        # Статус
        if config.get("loaded_from_file"):
            self.status_model_label.setText("📂 Загружена из файла (готова к дообучению или тестам)")
            self.status_model_label.setStyleSheet("color: #e5c07b;")
        else:
            self.status_model_label.setText("🛠 Создана в «Архитектуре» (новая, необученная)")
            self.status_model_label.setStyleSheet("color: #a6e3a1;")

        # Активируем кнопку старта если всё готово
        can_start = (
            model is not None and
            dataset is not None and
            "train_inputs" in dataset
        )
        self.btn_start.setEnabled(can_start and not self._is_training_active)

    def showEvent(self, event):
        """При показе вкладки обновляем карточку."""
        super().showEvent(event)
        self.refresh_model_card()

    # ============================================================
    # ЗАПУСК ОБУЧЕНИЯ
    # ============================================================
    def start_training(self, silent: bool = False):
        """
        Запуск процесса обучения с предварительной валидацией.

        silent=True — без модальных диалогов (для ИИ-агента Оракула).
        В silent-режиме возвращает строку ошибки или None при успехе.
        """
        # --- Валидация ---
        error = self._validate_before_start()
        if error:
            if silent:
                return error
            QMessageBox.warning(self, "⚠️ Проверка не пройдена", error)
            return

        # --- Проверка памяти ---
        model = self.shared_state["model"]
        mem_mb = ModelFactory.estimate_memory(model)
        if mem_mb > 2000 and not silent:
            reply = QMessageBox.question(
                self, "⚠️ Большая модель",
                f"Модель занимает ~{mem_mb:.0f} МБ.\n"
                "Обучение может потребовать много видеопамяти.\n"
                "Продолжить?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        # --- Подготовка данных ---
        ds = self.shared_state["dataset"]
        ds_dict = dataset_to_legacy_dict(ds)
        config = config_to_dict(self.shared_state["config"])
        meta = ds_dict.get("meta", {})
        data_type = meta.get("type", "numeric")
        config["data_type"] = data_type

        if data_type == "text":
            vocab = ds_dict.get("vocab", {})
            config["pad_idx"] = vocab.get("<PAD>", 0)
            config["bos_idx"] = vocab.get("<BOS>", 2)
            config["eos_idx"] = vocab.get("<EOS>", 3)
            if "teacher_forcing_ratio" not in config:
                config["teacher_forcing_ratio"] = 0.5

        # --- Создание загрузчиков данных ---
        try:
            train_ds = OracleTorchDataset(ds, split="train")
            val_ds = OracleTorchDataset(ds, split="val")
            batch_size = config.get("batch_size", 64)

            train_loader = DataLoader(
                train_ds, batch_size=batch_size,
                shuffle=True, collate_fn=train_ds.collate_fn
            )
            val_loader = DataLoader(
                val_ds, batch_size=batch_size,
                shuffle=False, collate_fn=val_ds.collate_fn
            )
        except Exception as e:
            if silent:
                return f"Ошибка данных: {e}"
            QMessageBox.critical(self, "Ошибка данных", f"Не удалось создать загрузчики:\n{e}")
            return

        # --- Автосохранение чекпоинтов ---
        checkpoint_dir = None
        checkpoint_interval = 0
        if self.checkpoint_check.isChecked():
            checkpoint_interval = self.checkpoint_interval_spin.value()
            from config import MODELS_DIR
            checkpoint_dir = MODELS_DIR / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # --- Создание тренера и потока ---
        try:
            model_config = ModelConfig.from_dict(config) if ModelConfig is not None else config
            self.trainer = Trainer(model, train_loader, val_loader, model_config)
            self.worker = TrainerWorker(
                self.trainer,
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval=checkpoint_interval
            )

            # Подписка на сигналы
            self.worker.epoch_done.connect(self._on_epoch_done)
            self.worker.batch_done.connect(self._on_batch_done)
            self.worker.finished.connect(self._on_training_finished)
            self.worker.error_occurred.connect(self._on_training_error)
            self.worker.checkpoint_saved.connect(self._on_checkpoint_saved)

        except Exception as e:
            if silent:
                return f"Критическая ошибка: {e}"
            QMessageBox.critical(self, "Критическая ошибка", f"Не удалось инициализировать обучение:\n{e}")
            return

        # --- UI: подготовка к обучению ---
        epochs = config.get("epochs", 50)
        self.epoch_progress.setMaximum(epochs)
        self.epoch_progress.setValue(0)
        self.batch_progress.setValue(0)
        self.btn_start.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self._is_training_active = True
        self._training_start_time = time.time()

        # Режим
        is_text = data_type == "text"
        if is_text:
            vocab_size = len(ds_dict.get("vocab", {}))
            self.mode_label.setText(
                f"📝 Текстовый режим | Словарь: {vocab_size} токенов | "
                f"Батч: {batch_size}"
            )
        else:
            self.mode_label.setText(
                f"🔢 Числовой режим | Размер батча: {batch_size}"
            )

        self.status_label.setText("Статус: 🚀 Обучение запущено...")
        self.status_label.setStyleSheet(
            "font-size: 14px; padding: 6px; color: #a6e3a1; "
            "background-color: #1e1e1e; border: 1px solid #385a3a; border-radius: 3px;"
        )

        # Логирование
        self._log_action("training_start", f"Модель: {config.get('model_name', '?')}")

        # Запуск таймера лимита
        if self.time_limit_check.isChecked():
            self._time_check_timer.start()

        # Эмиссия сигнала
        self.training_started.emit(is_text)

        # Запуск потока
        self.worker.start()

    # ============================================================
    # ВАЛИДАЦИЯ ПЕРЕД СТАРТОМ
    # ============================================================
    def _validate_before_start(self) -> str:
        """Проверяет готовность к обучению. Возвращает текст ошибки или ''."""
        model = self.shared_state.get("model")
        dataset = dataset_to_legacy_dict(self.shared_state.get("dataset"))

        if not model:
            return (
                "Модель не создана!\n"
                "Перейдите во вкладку «Архитектура» и нажмите «Создать модель»."
            )

        if not dataset:
            return (
                "Данные не загружены!\n"
                "Перейдите во вкладку «Данные» и загрузите датасет."
            )

        if "train_inputs" not in dataset:
            return (
                "Датасет НЕ разбит!\n"
                "Во вкладке «Данные» нажмите «Применить разбиение»."
            )

        if len(dataset.get("train_inputs", [])) == 0:
            return "Тренировочная выборка пуста. Проверьте параметры разбиения."

        # Проверка совместимости
        config = config_to_dict(self.shared_state.get("config"))
        data_type = dataset.get("meta", {}).get("type", "numeric")
        model_type = config.get("type", "mlp")

        if data_type == "text" and model_type == "mlp":
            return "MLP не поддерживает текст. Выберите Transformer или RNN/LSTM."

        return ""

    # ============================================================
    # ОБРАБОТКА СИГНАЛОВ ОТ ПОТОКА
    # ============================================================
    def _on_epoch_done(self, epoch: int, logs: dict):
        """Эпоха завершена — обновляем UI и эмитим сигнал."""
        self.epoch_progress.setValue(epoch)

        loss = logs.get("train_loss", 0)
        val_loss = logs.get("val_loss", 0)
        metric = logs.get("val_metric", 0)
        cfg = config_to_dict(self.shared_state.get("config"))
        total_epochs = cfg.get("epochs", 50)
        data_type = cfg.get("data_type", "numeric")

        if data_type == "text":
            status = (
                f"Эпоха {epoch}/{total_epochs} | "
                f"Loss: {loss:.4f} | Точность: {metric:.1%}"
            )
        else:
            status = (
                f"Эпоха {epoch}/{total_epochs} | "
                f"Train: {loss:.4f} | Val: {val_loss:.4f}"
            )
        self.status_label.setText(status)

        # Эмиссия для мониторинга
        self.epoch_finished.emit(epoch, logs)

    def _on_batch_done(self, epoch: int, batch_idx: int,
                       total_batches: int, speed: float, eta_str: str):
        """Батч завершён — обновляем прогресс-бар батчей."""
        if self.batch_progress.maximum() != total_batches:
            self.batch_progress.setMaximum(total_batches)
        self.batch_progress.setValue(batch_idx)

    def _on_training_finished(self, history):
        """Обучение полностью завершено."""
        # history может быть TrainingHistory (dataclass) от Trainer.
        raw_history = history
        if not isinstance(history, dict):
            history = history.to_dict() if hasattr(history, "to_dict") else {}

        # Сохраняем историю в shared_state — её читают анализ, диагностика и Оракул.
        if raw_history is not None:
            self.shared_state["history"] = raw_history

        self._is_training_active = False
        self._time_check_timer.stop()

        self.btn_start.setEnabled(True)
        self.btn_load.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_pause.setText("⏸  Пауза")

        elapsed = history.get("total_time_seconds", 0)
        mins, secs = divmod(int(elapsed), 60)
        hours, mins = divmod(mins, 60)
        time_str = f"{hours:02d}:{mins:02d}:{secs:02d}"

        self.status_label.setText(
            f"✅ Обучение завершено за {time_str}! Лучшие веса восстановлены."
        )
        self.status_label.setStyleSheet(
            "font-size: 14px; padding: 6px; color: #a6e3a1; "
            "background-color: #1e1e1e; border: 1px solid #385a3a; border-radius: 3px;"
        )
        self.mode_label.setText(
            "🏆 Лучшие веса (наименьший Val Loss) автоматически загружены в память."
        )

        # Логирование
        self._log_action("training_finish", f"Время: {time_str}")

        # Уведомление
        QMessageBox.information(
            self, "🎉 Обучение завершено!",
            f"Обучение завершено за {time_str}!\n\n"
            "Программа автоматически откатила модель к лучшей эпохе "
            "(где ошибка на новых данных была минимальной).\n\n"
            "Перейдите во вкладку «Анализ» для проверки,\n"
            "или во вкладку «Экспорт» для сохранения."
        )

        # Эмиссия
        self.training_finished.emit(history)

    def _on_training_error(self, error_msg: str):
        """Критическая ошибка в потоке обучения."""
        self._is_training_active = False
        self._time_check_timer.stop()

        self.btn_start.setEnabled(True)
        self.btn_load.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)

        self.status_label.setText("❌ Авария! Обучение остановлено.")
        self.status_label.setStyleSheet(
            "font-size: 14px; padding: 6px; color: #e06c75; "
            "background-color: #1e1e1e; border: 1px solid #8f4646; border-radius: 3px;"
        )

        logger.error(f"Авария обучения: {error_msg}")
        self._log_action("training_error", error_msg[:200])

        QMessageBox.critical(
            self, "🚨 Авария обучения",
            f"Обучение остановлено с ошибкой:\n{error_msg}\n\n"
            "Возможные причины:\n"
            "• Слишком высокая скорость обучения (LR)\n"
            "• Недостаточно видеопамяти (уменьшите Batch Size)\n"
            "• Ошибка в данных (проверьте датасет)\n"
            "• NaN в градиентах (увеличьте Clip)"
        )

    def _on_checkpoint_saved(self, epoch: int, path: str):
        """Чекпоинт сохранён."""
        logger.info(f"💾 Чекпоинт сохранён: эпоха {epoch} → {path}")

    # ============================================================
    # УПРАВЛЕНИЕ ПАУЗОЙ И ОСТАНОВКОЙ
    # ============================================================
    def toggle_pause(self):
        """Переключение пауза/продолжить."""
        if not self.trainer:
            return

        if self.trainer.paused:
            self.trainer.resume()
            self.btn_pause.setText("⏸  Пауза")
            self.status_label.setText("Статус: 🚀 Обучение продолжается...")
            self.status_label.setStyleSheet(
                "font-size: 14px; padding: 6px; color: #a6e3a1; "
                "background-color: #1e1e1e; border: 1px solid #385a3a; border-radius: 3px;"
            )
            self._log_action("training_resume")
        else:
            self.trainer.pause()
            self.btn_pause.setText("▶️  Продолжить")
            self.status_label.setText("Статус: ⏸ Обучение на паузе")
            self.status_label.setStyleSheet(
                "font-size: 14px; padding: 6px; color: #e5c07b; "
                "background-color: #1e1e1e; border: 1px solid #856a3e; border-radius: 3px;"
            )
            self._log_action("training_pause")

    def stop_training(self, silent: bool = False):
        """Остановка обучения с подтверждением."""
        if not self.trainer:
            return

        # Подтверждение (silent — для ИИ-агента, без диалога)
        if self.confirm_stop_check.isChecked() and not silent:
            reply = QMessageBox.question(
                self, "⚠️ Остановка обучения",
                "Вы уверены, что хотите остановить обучение?\n"
                "Прогресс текущей эпохи будет потерян.\n"
                "Лучшие веса будут автоматически восстановлены.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        self.trainer.stop()
        self.status_label.setText("Статус: ⏹ Остановка после текущей эпохи...")
        self.status_label.setStyleSheet(
            "font-size: 14px; padding: 6px; color: #e5c07b; "
            "background-color: #1e1e1e; border: 1px solid #856a3e; border-radius: 3px;"
        )
        self.btn_stop.setEnabled(False)
        self._log_action("training_stop")

    # ============================================================
    # ПРОВЕРКА ЛИМИТА ВРЕМЕНИ
    # ============================================================
    def _check_time_limit(self):
        """Вызывается по таймеру. Проверяет не превышен ли лимит."""
        if not self._is_training_active:
            return
        if not self.time_limit_check.isChecked():
            return

        elapsed_hours = (time.time() - self._training_start_time) / 3600.0
        limit_hours = self.time_limit_spin.value()

        if elapsed_hours >= limit_hours:
            logger.warning(
                f"⏰ Лимит времени ({limit_hours}ч) превышен. Автостоп."
            )
            self.trainer.stop()
            self.status_label.setText(
                f"⏰ Лимит времени ({limit_hours}ч) достигнут. Автоостановка."
            )
            QMessageBox.warning(
                self, "⏰ Лимит времени",
                f"Обучение длилось более {limit_hours} часов.\n"
                "Автоматическая остановка для защиты ПК.\n"
                "Лучшие веса будут восстановлены."
            )
            self._time_check_timer.stop()

    # ============================================================
    # СОХРАНЕНИЕ / ЗАГРУЗКА МОДЕЛИ
    # ============================================================
    def save_model(self):
        """Сохранение модели в файл .pth."""
        model = self.shared_state.get("model")
        if not model:
            QMessageBox.warning(self, "Внимание", "Нет модели для сохранения.")
            return

        # Имя файла по умолчанию
        project = self.shared_state.get("project", {})
        model_name = project.get("model_name", "model")
        default_name = f"{model_name.replace(' ', '_')}.pth"

        path, _ = QFileDialog.getSaveFileName(
            self, "💾 Сохранить модель",
            default_name,
            "PyTorch Model (*.pth);;Все файлы (*)"
        )
        if not path:
            return

        try:
            ModelFactory.save_model(model, Path(path), self.shared_state["config"])
            self._log_action("model_saved", path)
            QMessageBox.information(
                self, "✅ Сохранено",
                f"Модель сохранена в:\n{path}\n\n"
                "Вы можете загрузить её позже для дообучения или экспорта."
            )
        except Exception as e:
            logger.error(f"Ошибка сохранения модели: {e}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить:\n{e}")

    def load_model(self):
        """Загрузка модели из файла .pth."""
        if self._is_training_active:
            QMessageBox.warning(
                self, "Внимание",
                "Нельзя загружать модель во время обучения.\n"
                "Сначала остановите текущее обучение."
            )
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "📂 Загрузить модель",
            "",
            "PyTorch Model (*.pth);;Все файлы (*)"
        )
        if not path:
            return

        try:
            model, config, _extra = ModelFactory.load_model(Path(path))
            self.shared_state["model"] = model
            config.loaded_from_file = True
            self.shared_state["config"] = config

            self.refresh_model_card()
            self._log_action("model_loaded", path)

            QMessageBox.information(
                self, "✅ Загружено",
                "Модель успешно загружена!\n\n"
                "Теперь вы можете:\n"
                "• Дообучить на новых данных (нажмите «Старт»)\n"
                "• Тестировать во вкладке «Анализ»\n"
                "• Экспортировать во вкладке «Экспорт»"
            )
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            QMessageBox.critical(
                self, "❌ Ошибка загрузки",
                f"Файл повреждён или несовместим:\n{e}"
            )

    # ============================================================
    # ЛОГИРОВАНИЕ ДЕЙСТВИЙ
    # ============================================================
    def _log_action(self, action: str, details: str = ""):
        """Записывает действие в сессию пользователя."""
        session_manager = self.shared_state.get("session_manager")
        if session_manager:
            session_manager.log_action("training_panel", action, details)

    # ============================================================
    # ВНЕШНЕЕ ОБНОВЛЕНИЕ (вызывается из main_window)
    # ============================================================
    def refresh(self):
        """Обновление при переключении вкладок или изменении данных."""
        self.refresh_model_card()