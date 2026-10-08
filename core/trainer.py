"""
core/trainer.py
===============
Универсальный тренер для всех типов данных и архитектур зоопарка.
Контракт: contracts.py, FORMAT_VERSION="2.0".

Ответственность:
  • Цикл обучения для всех DataType / TaskType / ArchitectureType.
  • Поддержка колбэков для GUI (epoch_end, batch_end, sample_predictions).
  • Пауза / стоп / возобновление.
  • Автосохранение лучших весов.
  • Ранняя остановка (Early Stopping).
  • Генерация живых примеров для мониторинга.
  • Безопасность: не падает на мусорных данных, логирует и продолжает.

Зависимости:
  • core.contracts  (ModelConfig, TrainingHistory, DataType, TaskType, ArchitectureType)
  • core.optimizer_factory (create_optimizer, create_scheduler)
  • core.metrics (accuracy, mae, rmse, r2_score, f1_score)
  • core.logger (get_logger)

НЕ создаёт новых ключей в shared_state.
НЕ содержит GUI-кода. Только логика обучения.
"""

import time
import copy
import math
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from core.contracts import (
    ModelConfig,
    TrainingHistory,
    DataType,
    TaskType,
    ArchitectureType,
    FORMAT_VERSION,
)
from core.optimizer_factory import create_optimizer, create_scheduler
from core.metrics import accuracy, mae, rmse
from core.logger import get_logger

logger = get_logger(__name__)


# ============================================================
#  КОЛБЭК-ИНТЕРФЕЙС
# ============================================================

class TrainingCallback:
    """
    Базовый класс колбэка. GUI-панели наследуют и переопределяют.
    Все методы вызываются из потока обучения.
    """

    def on_training_begin(self, config: ModelConfig):
        """Вызывается один раз перед первой эпохой."""
        pass

    def on_epoch_end(self, epoch: int, logs: Dict[str, float]):
        """
        Вызывается после каждой эпохи.
        logs содержит: train_loss, val_loss, train_metric, val_metric, lr
        """
        pass

    def on_batch_end(
        self,
        epoch: int,
        batch_idx: int,
        total_batches: int,
        speed: float,
        eta_str: str,
    ):
        """Вызывается периодически внутри эпохи для прогресс-бара."""
        pass

    def on_sample_predictions(self, predictions: List[Dict[str, Any]]):
        """
        Вызывается каждые N эпох с живыми примерами.
        Каждый элемент: {"input": ..., "expected": ..., "predicted": ..., "correct": bool}
        """
        pass

    def on_early_stop(self, epoch: int, best_epoch: int):
        """Вызывается при срабатывании ранней остановки."""
        pass

    def on_training_end(self, history: TrainingHistory):
        """Вызывается один раз после завершения всего обучения."""
        pass


# ============================================================
#  МОДУЛЬ РАННЕЙ ОСТАНОВКИ
# ============================================================

class EarlyStopping:
    """Отслеживает отсутствие улучшения и сигнализирует об остановке."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: Optional[float] = None
        self.best_epoch = 0
        self.should_stop = False

    def step(self, val_loss: float, epoch: int) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        return False


# ============================================================
#  ОСНОВНОЙ ТРЕНЕР
# ============================================================

class Trainer:
    """
    Универсальный тренер. Работает с любой архитектурой из зоопарка.

    Использование:
        trainer = Trainer(model, train_loader, val_loader, config)
        trainer.add_callback(my_gui_callback)
        history = trainer.train()
    """

    # Как часто генерировать живые примеры (каждые N эпох)
    SAMPLE_PREDICTION_INTERVAL = 5

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: ModelConfig,
        test_loader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config

        # Устройство
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        # Оптимизатор и планировщик
        self.optimizer = create_optimizer(model, config)
        self.scheduler = create_scheduler(self.optimizer, config)

        # Определяем режим по типу данных и задаче
        self.data_type = config.data_type
        self.task_type = config.task_type
        self.architecture = config.architecture

        # Спецтокены для текста (должны быть заданы ДО построения критерия,
        # иначе _build_criterion для seq2seq упадёт с AttributeError: pad_idx)
        self.pad_idx = 0
        self.bos_idx = 2
        self.eos_idx = 3
        if self.data_type == DataType.TEXT:
            # Извлекаем из arch_params если есть
            ap = config.arch_params
            self.pad_idx = ap.get("pad_idx", 0)
            self.bos_idx = ap.get("bos_idx", 2)
            self.eos_idx = ap.get("eos_idx", 3)

        # Флаги для определения режима обучения
        self.is_seq2seq = self._is_seq2seq_task()
        self.is_autoencoder = self._is_autoencoder_task()
        self.is_gan = self._is_gan_task()
        self.is_vae = self._is_vae_task()
        self.is_diffusion = self._is_diffusion_task()
        self.is_classification = self._is_classification_task()
        self.is_regression = self._is_regression_task()

        # Функция потерь
        self.criterion = self._build_criterion()

        # Параметры обучения из конфига
        self.epochs = config.epochs
        self.clip_grad = config.gradient_clipping
        self.teacher_forcing_ratio = config.teacher_forcing_ratio

        # Ранняя остановка
        self.early_stopping: Optional[EarlyStopping] = None
        if config.early_stopping_patience > 0:
            self.early_stopping = EarlyStopping(
                patience=config.early_stopping_patience,
                min_delta=1e-4,
            )
            logger.info(
                f"Ранняя остановка включена: терпение={config.early_stopping_patience}"
            )

        # Колбэки
        self.callbacks: List[TrainingCallback] = []

        # Управление состоянием
        self.stop_training = False
        self.paused = False

        # Лучшие веса
        self.best_val_loss = float("inf")
        self.best_model_state: Optional[Dict] = None
        self.best_epoch = 0

        # Для генерации живых примеров
        self._sample_pool: List[Dict[str, Any]] = []
        self._prepare_sample_pool()

        logger.info(
            f"Trainer инициализирован: арх={self.architecture.value}, "
            f"данные={self.data_type.value}, задача={self.task_type.value}, "
            f"устройство={self.device}"
        )

    # ============================================================
    #  ОПРЕДЕЛЕНИЕ РЕЖИМА ОБУЧЕНИЯ
    # ============================================================

    def _is_seq2seq_task(self) -> bool:
        """Seq2Seq: текст→текст, перевод, генерация текста."""
        return (
            self.data_type == DataType.TEXT
            and self.task_type in (
                TaskType.SEQ2SEQ,
                TaskType.TRANSLATION,
            )
        )

    def _is_autoencoder_task(self) -> bool:
        """Автоэнкодер: вход = выход (реконструкция)."""
        return self.architecture in (
            ArchitectureType.AUTOENCODER,
            ArchitectureType.CONV_AE,
        )

    def _is_gan_task(self) -> bool:
        """GAN / DCGAN."""
        return self.architecture in (
            ArchitectureType.GAN,
            ArchitectureType.DCGAN,
        )

    def _is_vae_task(self) -> bool:
        """VAE / CVAE."""
        return self.architecture in (
            ArchitectureType.VAE,
            ArchitectureType.CVAE,
        )

    def _is_diffusion_task(self) -> bool:
        """Диффузионные модели."""
        return self.architecture in (
            ArchitectureType.DIFFUSION,
            ArchitectureType.LATENT_DIFFUSION,
        )

    def _is_classification_task(self) -> bool:
        """Классификация (изображения, текст, звук, графы)."""
        return self.task_type in (
            TaskType.CLASSIFICATION,
            TaskType.IMAGE_CLASSIFICATION,
            TaskType.AUDIO_CLASSIFICATION,
        )

    def _is_regression_task(self) -> bool:
        """Регрессия (числовые, временные ряды)."""
        return self.task_type in (
            TaskType.REGRESSION,
        )

    # ============================================================
    #  ПОСТРОЕНИЕ ФУНКЦИИ ПОТЕРЬ
    # ============================================================

    def _build_criterion(self) -> nn.Module:
        """Выбирает функцию потерь на основе типа задачи."""

        if self.is_seq2seq:
            logger.info(
                f"Seq2Seq режим: CrossEntropyLoss(ignore_index={self.pad_idx})"
            )
            return nn.CrossEntropyLoss(ignore_index=self.pad_idx)

        if self.is_classification:
            logger.info("Классификация: CrossEntropyLoss")
            return nn.CrossEntropyLoss()

        if self.is_gan:
            logger.info("GAN: BCELoss (бинарная кросс-энтропия)")
            return nn.BCELoss()

        if self.is_vae:
            # VAE использует составной лосс, обрабатывается в _forward
            logger.info("VAE: MSE + KL (составной)")
            return nn.MSELoss(reduction="sum")

        if self.is_diffusion:
            logger.info("Diffusion: MSELoss (шум)")
            return nn.MSELoss()

        # Автоэнкодер, регрессия, прочее
        loss_name = self.config.loss_function.lower() if self.config.loss_function else "mse"
        if loss_name == "mae":
            logger.info("Регрессия: L1Loss (MAE)")
            return nn.L1Loss()
        elif loss_name == "cross_entropy":
            logger.info("CrossEntropyLoss")
            return nn.CrossEntropyLoss()
        else:
            logger.info("Регрессия/Реконструкция: MSELoss")
            return nn.MSELoss()

    # ============================================================
    #  ПОДГОТОВКА ПУЛА ПРИМЕРОВ ДЛЯ ЖИВЫХ ПРЕДСКАЗАНИЙ
    # ============================================================

    def _prepare_sample_pool(self):
        """Берёт несколько примеров из val_loader для живого мониторинга."""
        self._sample_pool = []
        try:
            for batch in self.val_loader:
                x = batch.get("x")
                y = batch.get("y")
                if x is None:
                    continue
                # Берём до 10 примеров из первого батча
                n = min(x.size(0), 10)
                for i in range(n):
                    self._sample_pool.append({
                        "x": x[i].clone(),
                        "y": y[i].clone() if y is not None else None,
                    })
                break  # только первый батч
        except Exception as e:
            logger.warning(f"Не удалось подготовить пул примеров: {e}")

    # ============================================================
    #  ПРЯМОЙ ПРОХОД (УНИВЕРСАЛЬНЫЙ)
    # ============================================================

    def _forward_pass(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        is_training: bool = True,
    ) -> Tuple[torch.Tensor, Any]:
        """
        Универсальный прямой проход. Возвращает (loss, outputs).
        Адаптируется под тип задачи.
        """

        # --- GAN: отдельная логика ---
        if self.is_gan:
            return self._forward_gan(inputs, targets, is_training)

        # --- VAE: составной лосс ---
        if self.is_vae:
            return self._forward_vae(inputs, targets, is_training)

        # --- Диффузия ---
        if self.is_diffusion:
            return self._forward_diffusion(inputs, targets, is_training)

        # --- Seq2Seq (текст → текст) ---
        if self.is_seq2seq:
            tf_ratio = self.teacher_forcing_ratio if is_training else 0.0
            outputs = self.model(inputs, targets, teacher_forcing_ratio=tf_ratio)
            target_shifted = targets[:, 1:].contiguous()
            loss = self.criterion(
                outputs.reshape(-1, outputs.size(-1)),
                target_shifted.reshape(-1),
            )
            return loss, outputs

        # --- Автоэнкодер: вход = цель ---
        if self.is_autoencoder:
            outputs = self.model(inputs)
            # Цель — сам вход (реконструкция)
            loss = self.criterion(outputs, inputs)
            return loss, outputs

        # --- Классификация / Регрессия / Прочее ---
        outputs = self.model(inputs)

        # Убедимся что размерности совпадают
        if outputs.shape != targets.shape:
            # Для классификации: если targets — скалярные метки
            if targets.dim() == 1 or (targets.dim() == 2 and targets.size(1) == 1):
                targets = targets.long().squeeze()
            else:
                targets = targets.float()

        loss = self.criterion(outputs, targets)
        return loss, outputs

    # ============================================================
    #  СПЕЦИАЛЬНЫЕ ПРЯМЫЕ ПРОХОДЫ
    # ============================================================

    def _forward_gan(
        self,
        real_data: torch.Tensor,
        targets: torch.Tensor,
        is_training: bool,
    ) -> Tuple[torch.Tensor, Any]:
        """
        GAN: упрощённый проход.
        В полной реализации здесь будет чередование G и D.
        Сейчас — заглушка, которая не падает.
        """
        try:
            # Пытаемся использовать модель как есть
            outputs = self.model(real_data)
            loss = self.criterion(
                outputs,
                torch.ones_like(outputs) * 0.9,  # label smoothing
            )
            return loss, outputs
        except Exception as e:
            logger.warning(f"GAN forward fallback: {e}")
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            return loss, real_data

    def _forward_vae(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        is_training: bool,
    ) -> Tuple[torch.Tensor, Any]:
        """
        VAE: reconstruction + KL divergence.
        Ожидается что модель возвращает (recon, mu, logvar).
        """
        try:
            recon, mu, logvar = self.model(inputs)
            recon_loss = self.criterion(recon, inputs)
            kl_loss = -0.5 * torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp()
            )
            total_loss = recon_loss + kl_loss
            return total_loss, recon
        except Exception as e:
            # Если модель не возвращает му/логвар — просто MSE
            logger.warning(f"VAE forward fallback: {e}")
            outputs = self.model(inputs)
            loss = self.criterion(outputs, inputs)
            return loss, outputs

    def _forward_diffusion(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        is_training: bool,
    ) -> Tuple[torch.Tensor, Any]:
        """
        Диффузия: упрощённый проход.
        В полной реализации — предсказание шума на случайном шаге t.
        """
        try:
            # Генерируем случайный уровень шума
            batch_size = inputs.size(0)
            t = torch.randint(0, 100, (batch_size,), device=self.device)
            noise = torch.randn_like(inputs)
            noisy_inputs = inputs + noise * 0.1  # упрощение

            outputs = self.model(noisy_inputs, t)
            loss = self.criterion(outputs, noise)
            return loss, outputs
        except Exception as e:
            logger.warning(f"Diffusion forward fallback: {e}")
            outputs = self.model(inputs)
            loss = self.criterion(outputs, inputs)
            return loss, outputs

    # ============================================================
    #  ВЫЧИСЛЕНИЕ МЕТРИК
    # ============================================================

    def _compute_metric(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """Выбирает метрику на основе типа задачи."""

        if self.is_seq2seq:
            return self._compute_text_accuracy(outputs, targets)

        if self.is_classification:
            try:
                return accuracy(outputs, targets)
            except Exception:
                return 0.0

        if self.is_autoencoder:
            # Для автоэнкодера метрика = 1 - нормализованный MSE
            try:
                mse_val = nn.functional.mse_loss(outputs, targets).item()
                return max(0.0, 1.0 - mse_val)
            except Exception:
                return 0.0

        if self.is_gan or self.is_vae or self.is_diffusion:
            # Генеративные модели: метрика = отрицательный лосс (чем меньше, тем лучше)
            return 0.0  # заглушка

        # Регрессия
        try:
            return mae(outputs, targets)
        except Exception:
            return 0.0

    def _compute_text_accuracy(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """Точность предсказания токенов для текста."""
        try:
            preds = outputs.argmax(dim=-1)
            target_shifted = targets[:, 1:]
            mask = target_shifted != self.pad_idx
            if mask.sum() == 0:
                return 0.0
            correct = ((preds == target_shifted) & mask).sum().item()
            return correct / mask.sum().item()
        except Exception:
            return 0.0

    # ============================================================
    #  ГЕНЕРАЦИЯ ЖИВЫХ ПРИМЕРОВ
    # ============================================================

    def _generate_sample_predictions(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Берёт n случайных примеров из пула и предсказывает.
        Возвращает список словарей для мониторинга.
        """
        if not self._sample_pool:
            return []

        self.model.eval()
        predictions = []
        samples = random.sample(self._sample_pool, min(n, len(self._sample_pool)))

        with torch.no_grad():
            for sample in samples:
                x = sample["x"].unsqueeze(0).to(self.device)
                y_true = sample["y"]

                try:
                    if self.is_seq2seq and hasattr(self.model, "generate"):
                        # Генерация текста
                        output_indices = self.model.generate(x, max_len=50)
                        predicted = output_indices[0].cpu().tolist()
                        expected = y_true.cpu().tolist() if y_true is not None else []
                        predictions.append({
                            "input": x[0].cpu().tolist(),
                            "expected": expected,
                            "predicted": predicted,
                            "correct": predicted[:len(expected)] == expected,
                        })
                    elif self.is_autoencoder:
                        # Реконструкция
                        output = self.model(x)
                        predicted = output[0].cpu().tolist()
                        expected = x[0].cpu().tolist()
                        mse_val = sum(
                            (a - b) ** 2 for a, b in zip(predicted, expected)
                        ) / max(len(expected), 1)
                        predictions.append({
                            "input": expected,
                            "expected": expected,
                            "predicted": predicted,
                            "correct": mse_val < 0.01,
                            "mse": mse_val,
                        })
                    else:
                        # Классификация / регрессия
                        output = self.model(x)
                        if self.is_classification:
                            pred_class = output.argmax(dim=-1).item()
                            true_class = y_true.item() if y_true is not None else -1
                            predictions.append({
                                "input": x[0].cpu().tolist(),
                                "expected": true_class,
                                "predicted": pred_class,
                                "correct": pred_class == true_class,
                            })
                        else:
                            pred_val = output[0].cpu().tolist()
                            true_val = y_true.cpu().tolist() if y_true is not None else []
                            predictions.append({
                                "input": x[0].cpu().tolist(),
                                "expected": true_val,
                                "predicted": pred_val,
                                "correct": False,  # для регрессии не бинарно
                            })
                except Exception as e:
                    logger.warning(f"Ошибка генерации примера: {e}")
                    predictions.append({
                        "input": x[0].cpu().tolist(),
                        "expected": [],
                        "predicted": [],
                        "correct": False,
                        "error": str(e),
                    })

        self.model.train()
        return predictions

    # ============================================================
    #  ОСНОВНОЙ ЦИКЛ ОБУЧЕНИЯ
    # ============================================================

    def train(self) -> TrainingHistory:
        """
        Запускает полный цикл обучения.
        Возвращает TrainingHistory.
        """
        history = TrainingHistory()
        start_time = time.time()

        # Уведомляем колбэки о начале
        for cb in self.callbacks:
            cb.on_training_begin(self.config)

        # Оценка сложности
        params_count = sum(p.numel() for p in self.model.parameters())
        batch_size = self.config.batch_size
        logger.info(
            f"Параметров: {params_count:,} | "
            f"Батч: {batch_size} | Эпох: {self.epochs}"
        )
        logger.info(f"🚀 Начало обучения на устройстве: {self.device}")

        # Инициализация лучших весов
        self.best_val_loss = float("inf")
        self.best_model_state = copy.deepcopy(self.model.state_dict())
        self.best_epoch = 0

        avg_step_time = 0.0
        last_ui_update = 0.0

        for epoch in range(1, self.epochs + 1):
            # --- Проверка остановки ---
            if self.stop_training:
                logger.info("⏹ Обучение остановлено пользователем.")
                break

            # --- Пауза ---
            while self.paused:
                time.sleep(0.1)
                if self.stop_training:
                    break
            if self.stop_training:
                break

            # === ТРЕНИРОВКА ===
            self.model.train()
            total_train_loss = 0.0
            total_train_metric = 0.0
            total_batches = max(len(self.train_loader), 1)

            for batch_idx, batch in enumerate(self.train_loader, 1):
                if self.stop_training:
                    break

                batch_start = time.time()

                inputs = batch["x"].to(self.device)
                targets = batch["y"].to(self.device)

                self.optimizer.zero_grad()

                try:
                    loss, outputs = self._forward_pass(
                        inputs, targets, is_training=True
                    )

                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(
                            f"Эпоха {epoch}, батч {batch_idx}: "
                            f"loss = NaN/Inf. Пропускаем батч."
                        )
                        continue

                    loss.backward()

                    # Клиппинг градиентов
                    if self.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.clip_grad
                        )

                    self.optimizer.step()
                    total_train_loss += loss.item()

                    # Метрика
                    metric = self._compute_metric(outputs, targets)
                    total_train_metric += metric

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.error(
                            f"⚠️ CUDA OOM на эпохе {epoch}, батче {batch_idx}. "
                            f"Уменьшите размер батча или модель."
                        )
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        self.stop_training = True
                        break
                    else:
                        logger.error(f"RuntimeError: {e}")
                        continue

                # --- ТАЙМИНГ И ОБНОВЛЕНИЕ ПРОГРЕССА ---
                batch_end = time.time()
                step_time = batch_end - batch_start

                if avg_step_time == 0.0:
                    avg_step_time = step_time
                else:
                    avg_step_time = 0.9 * avg_step_time + 0.1 * step_time

                # Троттлинг: не чаще 5 раз в секунду
                if batch_end - last_ui_update > 0.2 or batch_idx == total_batches:
                    speed = 1.0 / avg_step_time if avg_step_time > 0 else 0.0
                    remaining_batches = (
                        (self.epochs - epoch) * total_batches
                        + (total_batches - batch_idx)
                    )
                    eta_sec = remaining_batches * avg_step_time
                    m, s = divmod(int(eta_sec), 60)
                    h, m = divmod(m, 60)
                    eta_str = f"{h:02d}:{m:02d}:{s:02d}"

                    for cb in self.callbacks:
                        cb.on_batch_end(
                            epoch, batch_idx, total_batches, speed, eta_str
                        )
                    last_ui_update = time.time()

            if self.stop_training:
                break

            avg_train_loss = total_train_loss / total_batches
            avg_train_metric = total_train_metric / total_batches

            # === ВАЛИДАЦИЯ ===
            self.model.eval()
            total_val_loss = 0.0
            total_val_metric = 0.0
            n_val = max(len(self.val_loader), 1)

            with torch.no_grad():
                for batch in self.val_loader:
                    inputs = batch["x"].to(self.device)
                    targets = batch["y"].to(self.device)

                    try:
                        loss, outputs = self._forward_pass(
                            inputs, targets, is_training=False
                        )
                        total_val_loss += loss.item()
                        metric = self._compute_metric(outputs, targets)
                        total_val_metric += metric
                    except Exception as e:
                        logger.warning(f"Ошибка валидации: {e}")
                        continue

            avg_val_loss = total_val_loss / n_val
            avg_val_metric = total_val_metric / n_val

            # === АВТОСОХРАНЕНИЕ ЛУЧШЕЙ ЭПОХИ ===
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch

            # === ПЛАНИРОВЩИК ===
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self.scheduler.step(avg_val_loss)
                else:
                    self.scheduler.step()

            # Текущий LR
            current_lr = self.optimizer.param_groups[0]["lr"]

            # === ЗАПИСЬ В ИСТОРИЮ ===
            history.train_loss.append(avg_train_loss)
            history.val_loss.append(avg_val_loss)
            history.train_metric.append(avg_train_metric)
            history.val_metric.append(avg_val_metric)
            history.epochs_completed = epoch

            logs = {
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_metric": avg_train_metric,
                "val_metric": avg_val_metric,
                "lr": current_lr,
            }

            for cb in self.callbacks:
                cb.on_epoch_end(epoch, logs)

            # === ЖИВЫЕ ПРИМЕРЫ ===
            if epoch % self.SAMPLE_PREDICTION_INTERVAL == 0 or epoch == 1:
                sample_preds = self._generate_sample_predictions(n=5)
                if sample_preds:
                    history.sample_predictions = sample_preds
                    for cb in self.callbacks:
                        cb.on_sample_predictions(sample_preds)

            # === РАННЯЯ ОСТАНОВКА ===
            if self.early_stopping is not None:
                if self.early_stopping.step(avg_val_loss, epoch):
                    logger.info(
                        f"⏹ Ранняя остановка на эпохе {epoch}. "
                        f"Лучшая эпоха: {self.early_stopping.best_epoch}"
                    )
                    history.epochs_completed = epoch
                    history.best_epoch = self.early_stopping.best_epoch
                    for cb in self.callbacks:
                        cb.on_early_stop(epoch, self.early_stopping.best_epoch)
                    break

            # === ЛОГИРОВАНИЕ ===
            if epoch % 5 == 0 or epoch == 1:
                metric_name = (
                    "Token Acc" if self.is_seq2seq
                    else "Class Acc" if self.is_classification
                    else "Metric"
                )
                logger.info(
                    f"Эпоха {epoch}/{self.epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f} | "
                    f"{metric_name}: {avg_val_metric:.4f} | "
                    f"LR: {current_lr:.6f}"
                )

        # === ВОССТАНОВЛЕНИЕ ЛУЧШИХ ВЕСОВ ===
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(
                f"✅ Восстановлены лучшие веса модели "
                f"(эпоха {self.best_epoch}, Val Loss: {self.best_val_loss:.4f}). "
                f"Модель готова к тестированию!"
            )

        # Финализация истории
        history.best_epoch = self.best_epoch
        history.best_val_loss = self.best_val_loss
        history.total_time_seconds = time.time() - start_time

        # Тестовая метрика (если есть тестовый лоадер)
        if self.test_loader is not None:
            test_metric = self._evaluate_on_loader(self.test_loader)
            logger.info(f"📊 Тестовая метрика: {test_metric:.4f}")

        # Уведомляем о завершении
        for cb in self.callbacks:
            cb.on_training_end(history)

        return history

    # ============================================================
    #  ОЦЕНКА НА ЛОАДЕРЕ (для теста)
    # ============================================================

    def _evaluate_on_loader(self, loader: DataLoader) -> float:
        """Запускает оценку на любом лоадере. Возвращает среднюю метрику."""
        self.model.eval()
        total_metric = 0.0
        n = 0
        with torch.no_grad():
            for batch in loader:
                inputs = batch["x"].to(self.device)
                targets = batch["y"].to(self.device)
                try:
                    _, outputs = self._forward_pass(
                        inputs, targets, is_training=False
                    )
                    metric = self._compute_metric(outputs, targets)
                    total_metric += metric
                    n += 1
                except Exception:
                    continue
        self.model.train()
        return total_metric / max(n, 1)

    # ============================================================
    #  УПРАВЛЕНИЕ СОСТОЯНИЕМ
    # ============================================================

    def pause(self):
        """Ставит обучение на паузу."""
        self.paused = True
        logger.info("⏸ Обучение приостановлено.")

    def resume(self):
        """Возобновляет обучение."""
        self.paused = False
        logger.info("▶️ Обучение возобновлено.")

    def stop(self):
        """Запрашивает остановку после текущей эпохи/батча."""
        self.stop_training = True
        logger.info("⏹ Запрошена остановка обучения.")

    def add_callback(self, callback: TrainingCallback):
        """Регистрирует колбэк."""
        self.callbacks.append(callback)

    def remove_callback(self, callback: TrainingCallback):
        """Удаляет колбэк."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)