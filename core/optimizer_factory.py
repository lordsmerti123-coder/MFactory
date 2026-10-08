"""
core/optimizer_factory.py
=========================
Фабрика оптимизаторов и планировщиков скорости обучения.
Отвечает ТОЛЬКО за создание оптимизаторов, планировщиков,
валидацию параметров, кривые LR и контекстные подсказки.

Зависимости:
  - core.contracts  — типы и версии
  - core.logger     — логирование
  - config          — константы

НЕ зависит от:
  - model_factory.py, trainer.py, dataset.py, любых GUI-панелей

Обратная совместимость:
  - принимает старый формат Dict[str, Any]
  - принимает новый формат ModelConfig из contracts
  - сохраняет функции верхнего уровня create_optimizer / create_scheduler

Контракт с trainer.py:
  trainer вызывает:
    opt   = OptimizerFactory.create(model, config)
    sched = OptimizerFactory.create_scheduler(opt, config)
  Для GAN:
    g_opt, d_opt = OptimizerFactory.create_gan_optimizers(gen, disc, config)

Контракт с hyperparams_panel.py:
  панель вызывает:
    OptimizerFactory.get_lr_curve(config, epochs) -> List[float]
    OptimizerFactory.get_contextual_hint(config)   -> str
    OptimizerFactory.validate(config)              -> List[str]
"""

from __future__ import annotations

import math
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR,
    MultiStepLR,
    ExponentialLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    LambdaLR,
)
from typing import Any, Dict, List, Optional, Tuple, Union

from core.logger import get_logger

# ---------------------------------------------------------------------------
# Защищённый импорт контрактов.
# Если контракты ещё не готовы (параллельная разработка), модуль
# продолжает работать со старым dict-форматом.
# ---------------------------------------------------------------------------
try:
    from core.contracts import ModelConfig, FORMAT_VERSION
    _HAS_CONTRACTS = True
except ImportError:
    ModelConfig = None
    FORMAT_VERSION = "2.0"
    _HAS_CONTRACTS = False

logger = get_logger(__name__)


# ============================================================
#  РЕЕСТР ОПТИМИЗАТОРОВ
# ============================================================
# Каждый элемент: (класс, минимальная версия PyTorch, описание для подсказки)
OPTIMIZER_REGISTRY: Dict[str, Dict[str, Any]] = {
    "adam": {
        "class": optim.Adam,
        "desc": (
            "Adam — самый популярный «умный» алгоритм. Сам понимает, где "
            "нужно сделать большой шаг, а где притормозить. Идеален для новичков."
        ),
        "best_for": ["mlp", "cnn", "transformer", "rnn", "lstm", "gru", "autoencoder"],
    },
    "adamw": {
        "class": optim.AdamW,
        "desc": (
            "AdamW — Adam с улучшенным штрафом за «зубрёжку» (weight decay). "
            "Стандарт для Трансформеров и больших моделей."
        ),
        "best_for": ["transformer", "vit", "resnet", "diffusion", "latent_diffusion"],
    },
    "sgd": {
        "class": optim.SGD,
        "desc": (
            "SGD — классика. Медленный, требует точной настройки скорости обучения, "
            "но иногда даёт лучшее качество. Для терпеливых экспериментаторов."
        ),
        "best_for": ["resnet", "cnn"],
    },
    "rmsprop": {
        "class": optim.RMSprop,
        "desc": (
            "RMSprop — хорошо справляется с Рекуррентными сетями (RNN/LSTM/GRU). "
            "Адаптивно масштабирует градиенты."
        ),
        "best_for": ["rnn", "lstm", "gru", "gan", "dcgan"],
    },
    "adagrad": {
        "class": optim.Adagrad,
        "desc": (
            "Adagrad — накапливает историю градиентов. Хорош для разреженных данных, "
            "но скорость обучения быстро падает до нуля."
        ),
        "best_for": ["rbfn", "som"],
    },
    "adadelta": {
        "class": optim.Adadelta,
        "desc": (
            "Adadelta — развитие Adagrad, не даёт скорости обучения упасть до нуля. "
            "Работает без явного указания LR."
        ),
        "best_for": ["rnn", "lstm"],
    },
    "adamax": {
        "class": optim.Adamax,
        "desc": (
            "Adamax — вариант Adam на основе бесконечной нормы. "
            "Стабильнее на моделях с эмбеддингами."
        ),
        "best_for": ["transformer", "gnn", "gcn", "gat"],
    },
}

# Оптимизаторы, доступные только в новых версиях PyTorch
_OPTIMIZERS_OPTIONAL: Dict[str, str] = {
    "nadam": "NAdam",
    "radam": "RAdam",
}
for _name, _cls_name in _OPTIMIZERS_OPTIONAL.items():
    if hasattr(optim, _cls_name):
        OPTIMIZER_REGISTRY[_name] = {
            "class": getattr(optim, _cls_name),
            "desc": (
                f"{_cls_name} — улучшенная версия Adam с коррекцией разогрева. "
                "Полезен для больших моделей и нестабильного обучения."
            ),
            "best_for": ["transformer", "vit", "clip_like"],
        }


# ============================================================
#  РЕЕСТР ПЛАНИРОВЩИКОВ
# ============================================================
SCHEDULER_REGISTRY: Dict[str, Dict[str, Any]] = {
    "none": {
        "desc": "Планировщик выключен. Скорость обучения постоянна на протяжении всего обучения.",
    },
    "step": {
        "desc": (
            "Ступенчатый: каждые N эпох скорость обучения умножается на γ (например, падает в 10 раз). "
            "Простой и предсказуемый."
        ),
        "params": {"step_size": 10, "gamma": 0.1},
    },
    "multistep": {
        "desc": (
            "Ступени на заданных эпохах: например, на 30-й и 60-й эпохе LR падает. "
            "Удобно, когда вы знаете, на какой эпохе обучение «застревает»."
        ),
        "params": {"milestones": [30, 60], "gamma": 0.1},
    },
    "exponential": {
        "desc": (
            "Экспоненциальный спад: каждую эпоху LR умножается на γ (< 1). "
            "Плавно замедляет обучение к концу."
        ),
        "params": {"gamma": 0.95},
    },
    "cosine": {
        "desc": (
            "Косинусный: LR плавно снижается по кривой косинуса от начального до минимального. "
            "Считается одним из лучших для Трансформеров и CNN."
        ),
        "params": {"eta_min": 0.0},
    },
    "cosine_restarts": {
        "desc": (
            "Косинусный с рестартами: LR плавно падает, потом резко возвращается вверх. "
            "Помогает «выпрыгнуть» из локальных минимумов."
        ),
        "params": {"T_0": 10, "T_mult": 2, "eta_min": 0.0},
    },
    "plateau": {
        "desc": (
            "По плато: если ошибка на валидации не падает несколько эпох подряд — "
            "автоматически уменьшаем LR. Не требует ручной настройки."
        ),
        "params": {"patience": 5, "factor": 0.5, "min_lr": 1e-7},
    },
    "linear_warmup_cosine": {
        "desc": (
            "Линейный разогрев + косинусный спад: первые N эпох LR растёт от малого до целевого, "
            "затем плавно снижается по косинусу. Стандарт для Трансформеров."
        ),
        "params": {"warmup_epochs": 5, "warmup_start_lr": 1e-6},
    },
    "linear_warmup_linear_decay": {
        "desc": (
            "Линейный разогрев + линейный спад: LR растёт, потом линейно падает до нуля. "
            "Простая и предсказуемая кривая."
        ),
        "params": {"warmup_epochs": 5, "warmup_start_lr": 1e-6},
    },
    "linear_warmup": {
        "desc": (
            "Линейный разогрев + линейный спад: LR плавно растёт в начале, затем "
            "линейно снижается. Простая и предсказуемая кривая."
        ),
        "params": {"warmup_epochs": 5, "warmup_start_lr": 1e-6},
    },
    "polynomial": {
        "desc": (
            "Полиномиальный спад: LR падает по полиномиальной кривой. "
            "Используется в задачах сегментации и диффузии."
        ),
        "params": {"power": 1.0, "total_iters": None},
    },
}


# ============================================================
#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def _normalize_config(config: Any) -> Dict[str, Any]:
    """
    Принимает ModelConfig или dict, возвращает dict.
    Обеспечивает обратную совместимость со старым форматом.
    """
    if _HAS_CONTRACTS and isinstance(config, ModelConfig):
        return config.to_dict()
    if isinstance(config, dict):
        return dict(config)
    raise TypeError(
        f"Ожидался ModelConfig или dict, получен {type(config).__name__}"
    )


def _get_scheduler_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Извлекает параметры планировщика из конфига.
    Ищет в нескольких местах для обратной совместимости:
      1. config["scheduler_params"]  (новый формат)
      2. config["arch_params"]["scheduler_params"]  (ModelConfig)
      3. пустой словарь
    """
    if "scheduler_params" in config:
        return dict(config["scheduler_params"])
    arch = config.get("arch_params", {})
    if isinstance(arch, dict) and "scheduler_params" in arch:
        return dict(arch["scheduler_params"])
    return {}


def _safe_get(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Безопасное извлечение значения из конфига."""
    val = config.get(key)
    return val if val is not None else default


# ============================================================
#  КЛАСС ФАБРИКИ
# ============================================================

class OptimizerFactory:
    """
    Единая точка создания оптимизаторов и планировщиков.
    Все модули проекта используют ТОЛЬКО этот класс.
    """

    # ----------------------------------------------------------
    #  СОЗДАНИЕ ОПТИМИЗАТОРА
    # ----------------------------------------------------------
    @staticmethod
    def create(
        model: torch.nn.Module,
        config: Any,
    ) -> optim.Optimizer:
        """
        Создаёт оптимизатор для модели.

        Аргументы:
            model  — nn.Module (обучаемая модель)
            config — ModelConfig или dict с ключами:
                - optimizer:       str   (имя оптимизатора)
                - learning_rate:   float
                - weight_decay:    float (L2-штраф)
                - momentum:        float (для SGD, по умолчанию 0.9)
                - beta1 / beta2:   float (для Adam-подобных)

        Возвращает:
            torch.optim.Optimizer

        Исключения:
            ValueError — если имя оптимизатора неизвестно или параметры некорректны.
        """
        cfg = _normalize_config(config)
        errors = OptimizerFactory.validate(cfg)
        if errors:
            raise ValueError(
                "Оптимизатор не может быть создан:\n" + "\n".join(f"• {e}" for e in errors)
            )

        opt_name = _safe_get(cfg, "optimizer", "adam").lower().strip()
        lr = float(_safe_get(cfg, "learning_rate", 0.001))
        wd = float(_safe_get(cfg, "weight_decay", 0.0))
        momentum = float(_safe_get(cfg, "momentum", 0.9))
        beta1 = float(_safe_get(cfg, "beta1", 0.9))
        beta2 = float(_safe_get(cfg, "beta2", 0.999))
        eps = float(_safe_get(cfg, "eps", 1e-8))

        if opt_name not in OPTIMIZER_REGISTRY:
            available = ", ".join(sorted(OPTIMIZER_REGISTRY.keys()))
            raise ValueError(
                f"Неизвестный оптимизатор «{opt_name}».\n"
                f"Доступные: {available}"
            )

        entry = OPTIMIZER_REGISTRY[opt_name]
        cls = entry["class"]

        # --- Собираем kwargs в зависимости от типа оптимизатора ---
        kwargs: Dict[str, Any] = {"lr": lr}

        if opt_name == "sgd":
            kwargs["momentum"] = momentum
            kwargs["weight_decay"] = wd
            kwargs["nesterov"] = bool(_safe_get(cfg, "nesterov", False))
        elif opt_name in ("adam", "adamw", "adamax", "nadam", "radam"):
            kwargs["betas"] = (beta1, beta2)
            kwargs["eps"] = eps
            kwargs["weight_decay"] = wd
        elif opt_name == "rmsprop":
            kwargs["alpha"] = float(_safe_get(cfg, "rmsprop_alpha", 0.99))
            kwargs["eps"] = eps
            kwargs["weight_decay"] = wd
            kwargs["momentum"] = momentum
        elif opt_name == "adagrad":
            kwargs["weight_decay"] = wd
            kwargs["eps"] = eps
        elif opt_name == "adadelta":
            kwargs["weight_decay"] = wd
            kwargs["rho"] = float(_safe_get(cfg, "rho", 0.95))
            kwargs["eps"] = eps

        optimizer = cls(model.parameters(), **kwargs)
        logger.info(
            f"Оптимизатор создан: {opt_name.upper()} | LR={lr} | WD={wd}"
        )
        return optimizer

    # ----------------------------------------------------------
    #  СОЗДАНИЕ ОПТИМИЗАТОРОВ ДЛЯ GAN
    # ----------------------------------------------------------
    @staticmethod
    def create_gan_optimizers(
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        config: Any,
    ) -> Tuple[optim.Optimizer, optim.Optimizer]:
        """
        Создаёт пару оптимизаторов для GAN / DCGAN / CycleGAN / Pix2Pix.

        По умолчанию для GAN рекомендуется Adam с beta1=0.5, beta2=0.999.
        Если в конфиге нет явных параметров, применяются GAN-пресеты.

        Возвращает:
            (optimizer_generator, optimizer_discriminator)
        """
        cfg = _normalize_config(config)

        # GAN-пресеты по умолчанию
        gan_defaults = {
            "optimizer": _safe_get(cfg, "optimizer", "adam"),
            "learning_rate": float(_safe_get(cfg, "learning_rate", 0.0002)),
            "beta1": float(_safe_get(cfg, "beta1", 0.5)),
            "beta2": float(_safe_get(cfg, "beta2", 0.999)),
            "weight_decay": float(_safe_get(cfg, "weight_decay", 0.0)),
        }

        # Допускаем разный LR для генератора и дискриминатора
        g_cfg = dict(gan_defaults)
        d_cfg = dict(gan_defaults)
        if "generator_lr" in cfg:
            g_cfg["learning_rate"] = float(cfg["generator_lr"])
        if "discriminator_lr" in cfg:
            d_cfg["learning_rate"] = float(cfg["discriminator_lr"])

        g_opt = OptimizerFactory.create(generator, g_cfg)
        d_opt = OptimizerFactory.create(discriminator, d_cfg)
        logger.info("GAN: созданы оптимизаторы для генератора и дискриминатора.")
        return g_opt, d_opt

    # ----------------------------------------------------------
    #  СОЗДАНИЕ ПЛАНИРОВЩИКА
    # ----------------------------------------------------------
    @staticmethod
    def create_scheduler(
        optimizer: optim.Optimizer,
        config: Any,
    ) -> Optional[Any]:
        """
        Создаёт планировщик скорости обучения.

        Аргументы:
            optimizer — созданный оптимизатор
            config    — ModelConfig или dict с ключом "scheduler"

        Возвращает:
            Планировщик (объект _LRScheduler / ReduceLROnPlateau) или None.

        Примечание для trainer.py:
            ReduceLROnPlateau требует вызова .step(metric) с метрикой.
            Остальные планировщики — .step() без аргументов.
            Проверяйте тип: isinstance(sched, ReduceLROnPlateau).
        """
        cfg = _normalize_config(config)
        sched_name = _safe_get(cfg, "scheduler", "none").lower().strip()
        total_epochs = int(_safe_get(cfg, "epochs", 50))
        params = _get_scheduler_params(cfg)

        if sched_name == "none":
            return None

        # --- StepLR ---
        if sched_name == "step":
            step_size = int(params.get("step_size", 10))
            gamma = float(params.get("gamma", 0.1))
            logger.info(f"Планировщик: StepLR(step_size={step_size}, gamma={gamma})")
            return StepLR(optimizer, step_size=step_size, gamma=gamma)

        # --- MultiStepLR ---
        if sched_name == "multistep":
            milestones = params.get("milestones", [30, 60])
            if isinstance(milestones, str):
                milestones = [int(x.strip()) for x in milestones.split(",")]
            milestones = sorted(int(m) for m in milestones)
            gamma = float(params.get("gamma", 0.1))
            logger.info(f"Планировщик: MultiStepLR(milestones={milestones}, gamma={gamma})")
            return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

        # --- ExponentialLR ---
        if sched_name == "exponential":
            gamma = float(params.get("gamma", 0.95))
            logger.info(f"Планировщик: ExponentialLR(gamma={gamma})")
            return ExponentialLR(optimizer, gamma=gamma)

        # --- CosineAnnealingLR ---
        if sched_name == "cosine":
            t_max = int(params.get("T_max", total_epochs))
            eta_min = float(params.get("eta_min", 0.0))
            logger.info(f"Планировщик: CosineAnnealingLR(T_max={t_max}, eta_min={eta_min})")
            return CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

        # --- CosineAnnealingWarmRestarts ---
        if sched_name == "cosine_restarts":
            t_0 = int(params.get("T_0", 10))
            t_mult = int(params.get("T_mult", 2))
            eta_min = float(params.get("eta_min", 0.0))
            logger.info(
                f"Планировщик: CosineAnnealingWarmRestarts(T_0={t_0}, T_mult={t_mult})"
            )
            return CosineAnnealingWarmRestarts(
                optimizer, T_0=t_0, T_mult=t_mult, eta_min=eta_min
            )

        # --- ReduceLROnPlateau ---
        if sched_name == "plateau":
            patience = int(params.get("patience", 5))
            factor = float(params.get("factor", 0.5))
            min_lr = float(params.get("min_lr", 1e-7))
            logger.info(
                f"Планировщик: ReduceLROnPlateau(patience={patience}, factor={factor})"
            )
            return ReduceLROnPlateau(
                optimizer, mode="min", patience=patience,
                factor=factor, min_lr=min_lr,
            )

        # --- Linear Warmup + Cosine Decay ---
        if sched_name == "linear_warmup_cosine":
            warmup_epochs = int(params.get("warmup_epochs", 5))
            warmup_start = float(params.get("warmup_start_lr", 1e-6))
            base_lr = optimizer.defaults.get("lr", 0.001)
            eta_min = float(params.get("eta_min", 0.0))

            # Защита от недопустимых значений: разогрев обязан быть меньше
            # общего числа эпох, иначе планировщик «не работает».
            if warmup_epochs < 1:
                warmup_epochs = 1
            if warmup_epochs >= total_epochs:
                warmup_epochs = max(1, total_epochs - 1)
                logger.warning(
                    f"linear_warmup_cosine: warmup_epochs >= total_epochs, "
                    f"уменьшаю разогрев до {warmup_epochs} (всего эпох: {total_epochs})."
                )

            # Функция-замыкание для LambdaLR
            def lr_lambda(epoch: int) -> float:
                if epoch < warmup_epochs:
                    # Линейный рост от warmup_start до base_lr
                    if base_lr <= 0:
                        return 1.0
                    alpha = epoch / max(warmup_epochs, 1)
                    return (warmup_start + alpha * (base_lr - warmup_start)) / base_lr
                else:
                    # Косинусный спад
                    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
                    progress = min(progress, 1.0)
                    cosine_val = 0.5 * (1.0 + math.cos(math.pi * progress))
                    min_ratio = eta_min / base_lr if base_lr > 0 else 0.0
                    return min_ratio + (1.0 - min_ratio) * cosine_val

            logger.info(
                f"Планировщик: LinearWarmup+Cosine(warmup={warmup_epochs}, total={total_epochs})"
            )
            return LambdaLR(optimizer, lr_lambda=lr_lambda)

        # --- Linear Warmup + Linear Decay ---
        if sched_name in ("linear_warmup_linear_decay", "linear_warmup"):
            # Имя «linear_warmup» приходит из GUI (hyperparams_panel),
            # «linear_warmup_linear_decay» — полное имя реестра. Оба значат
            # «разогрев + линейный спад».
            warmup_epochs = int(params.get("warmup_epochs") or max(total_epochs // 10, 1))
            warmup_start = float(params.get("warmup_start_lr", 1e-6))
            base_lr = optimizer.defaults.get("lr", 0.001)

            # Защита от недопустимых значений.
            if warmup_epochs < 1:
                warmup_epochs = 1
            if warmup_epochs >= total_epochs:
                warmup_epochs = max(1, total_epochs - 1)
                logger.warning(
                    f"linear_warmup_linear_decay: warmup_epochs >= total_epochs, "
                    f"уменьшаю разогрев до {warmup_epochs} (всего эпох: {total_epochs})."
                )

            def lr_lambda_linear(epoch: int) -> float:
                if epoch < warmup_epochs:
                    if base_lr <= 0:
                        return 1.0
                    alpha = epoch / max(warmup_epochs, 1)
                    return (warmup_start + alpha * (base_lr - warmup_start)) / base_lr
                else:
                    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
                    progress = min(progress, 1.0)
                    return max(1.0 - progress, 0.0)

            logger.info(
                f"Планировщик: LinearWarmup+LinearDecay(warmup={warmup_epochs})"
            )
            return LambdaLR(optimizer, lr_lambda=lr_lambda_linear)

        # --- Polynomial ---
        if sched_name == "polynomial":
            power = float(params.get("power", 1.0))
            total_iters = params.get("total_iters", total_epochs)
            if total_iters is None:
                total_iters = total_epochs
            total_iters = int(total_iters)

            def lr_lambda_poly(epoch: int) -> float:
                if epoch >= total_iters:
                    return 0.0
                factor = (1.0 - epoch / total_iters) ** power
                return max(factor, 0.0)

            logger.info(f"Планировщик: Polynomial(power={power}, iters={total_iters})")
            return LambdaLR(optimizer, lr_lambda=lr_lambda_poly)

        # --- Неизвестный планировщик ---
        available = ", ".join(sorted(SCHEDULER_REGISTRY.keys()))
        logger.warning(
            f"Неизвестный планировщик «{sched_name}». Доступные: {available}. "
            "Планировщик не создан."
        )
        return None

    # ----------------------------------------------------------
    #  КРИВАЯ LR ДЛЯ ВИЗУАЛИЗАЦИИ
    # ----------------------------------------------------------
    @staticmethod
    def get_lr_curve(
        config: Any,
        total_epochs: Optional[int] = None,
        num_points: int = 200,
    ) -> List[float]:
        """
        Возвращает список значений скорости обучения по эпохам
        для визуализации в GUI.

        Для ReduceLROnPlateau возвращает прямую линию (зависит от метрики).

        Аргументы:
            config       — конфиг с параметрами планировщика
            total_epochs — количество эпох (если не задано, берётся из конфига)
            num_points   — количество точек для плавности графика

        Возвращает:
            List[float] — значения LR по эпохам.
        """
        cfg = _normalize_config(config)
        if total_epochs is None:
            total_epochs = int(_safe_get(cfg, "epochs", 50))
        total_epochs = max(total_epochs, 1)

        sched_name = _safe_get(cfg, "scheduler", "none").lower().strip()
        base_lr = float(_safe_get(cfg, "learning_rate", 0.001))
        params = _get_scheduler_params(cfg)

        if sched_name == "none":
            return [base_lr] * total_epochs

        if sched_name == "plateau":
            # Невозможно предсказать без метрики
            return [base_lr] * total_epochs

        # Создаём фиктивный параметр и оптимизатор для расчёта кривой
        dummy = torch.tensor([1.0], requires_grad=True)
        dummy_opt = optim.SGD([dummy], lr=base_lr)

        # Временно подменяем конфиг на полный
        full_cfg = dict(cfg)
        full_cfg["epochs"] = total_epochs
        full_cfg["learning_rate"] = base_lr

        try:
            scheduler = OptimizerFactory.create_scheduler(dummy_opt, full_cfg)
        except Exception:
            return [base_lr] * total_epochs

        if scheduler is None:
            return [base_lr] * total_epochs

        curve: List[float] = []
        for epoch in range(total_epochs):
            current_lr = dummy_opt.param_groups[0]["lr"]
            curve.append(current_lr)

            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(current_lr)  # фиктивная метрика
            else:
                scheduler.step()

        return curve

    # ----------------------------------------------------------
    #  ОПИСАНИЯ И ПОДСКАЗКИ
    # ----------------------------------------------------------
    @staticmethod
    def get_optimizer_description(optimizer_name: str) -> str:
        """Возвращает человекочитаемое описание оптимизатора."""
        name = optimizer_name.lower().strip()
        if name in OPTIMIZER_REGISTRY:
            return OPTIMIZER_REGISTRY[name]["desc"]
        return f"Оптимизатор «{name}» неизвестен."

    @staticmethod
    def get_scheduler_description(scheduler_name: str) -> str:
        """Возвращает человекочитаемое описание планировщика."""
        name = scheduler_name.lower().strip()
        if name in SCHEDULER_REGISTRY:
            return SCHEDULER_REGISTRY[name]["desc"]
        return f"Планировщик «{name}» неизвестен."

    @staticmethod
    def get_contextual_hint(config: Any) -> str:
        """
        Динамическая подсказка, зависящая от:
          - выбранного оптимизатора
          - выбранного планировщика
          - текущего значения LR
          - типа модели (если есть в конфиге)

        Используется в hyperparams_panel.py и hint_engine.py.
        """
        cfg = _normalize_config(config)
        opt_name = _safe_get(cfg, "optimizer", "adam").lower().strip()
        sched_name = _safe_get(cfg, "scheduler", "none").lower().strip()
        lr = float(_safe_get(cfg, "learning_rate", 0.001))
        wd = float(_safe_get(cfg, "weight_decay", 0.0))
        model_type = _safe_get(cfg, "type", "").lower()
        data_type = _safe_get(cfg, "data_type", "numeric")

        parts: List[str] = []

        # --- Оценка LR ---
        if lr > 0.1:
            parts.append(
                "⚠️ Скорость обучения ОЧЕНЬ высокая. Сеть может «разойтись» — "
                "ошибка улетит в бесконечность. Уменьшите до 0.001–0.01."
            )
        elif lr > 0.01 and model_type in ("transformer", "transformer_seq2seq", "vit"):
            parts.append(
                "⚠️ Для Трансформеров LR выше 0.01 обычно слишком высок. "
                "Попробуйте 1e-4 — 1e-3."
            )
        elif lr < 1e-6:
            parts.append(
                "🐢 Скорость обучения экстремально мала. Обучение может занять "
                "вечность. Попробуйте 0.001."
            )
        else:
            parts.append(f"✅ Скорость обучения {lr} выглядит разумной.")

        # --- Оценка weight decay ---
        if wd > 0.1:
            parts.append(
                "⚠️ Штраф за зубрёжку (L2) очень большой. Сеть может «забыть» всё. "
                "Обычно достаточно 0.0–0.01."
            )
        elif wd > 0 and opt_name == "adam":
            parts.append(
                "💡 Совет: для Adam лучше использовать AdamW — он корректнее "
                "применяет weight decay."
            )

        # --- Совместимость оптимизатора и архитектуры ---
        if opt_name in OPTIMIZER_REGISTRY:
            best_for = OPTIMIZER_REGISTRY[opt_name].get("best_for", [])
            if model_type and best_for and model_type not in best_for:
                parts.append(
                    f"💡 {opt_name.upper()} не является стандартным выбором для "
                    f"«{model_type}». Проверьте, подходит ли он для вашей задачи."
                )

        # --- Планировщик ---
        if sched_name != "none":
            parts.append(
                f"📈 Планировщик: {OptimizerFactory.get_scheduler_description(sched_name)}"
            )
            if sched_name == "plateau":
                parts.append(
                    "💡 ReduceLROnPlateau не требует ручной настройки кривой — "
                    "он сам уменьшит LR, когда обучение застрянет."
                )
            if sched_name in ("linear_warmup_cosine", "linear_warmup_linear_decay", "linear_warmup"):
                try:
                    sp = _get_scheduler_params(cfg)
                    we = int(sp.get("warmup_epochs", 5))
                    ep = int(_safe_get(cfg, "epochs", 50))
                    if we >= ep:
                        parts.append(
                            f"⚠️ Разогрев ({we} эпох) не меньше общего числа эпох ({ep}). "
                            "Уменьшите warmup или увеличьте эпохи — иначе разогрев не успеет сработать."
                        )
                except Exception:
                    pass
        else:
            parts.append(
                "📈 Планировщик выключен. Если обучение «застрянет», "
                "попробуйте включить «Косинусный» или «По плато»."
            )

        return "\n".join(parts)

    # ----------------------------------------------------------
    #  ВАЛИДАЦИЯ КОНФИГА
    # ----------------------------------------------------------
    @staticmethod
    def validate(config: Any) -> List[str]:
        """
        Проверяет корректность параметров оптимизатора и планировщика.
        Возвращает список ошибок (пустой список = всё ок).

        НЕ бросает исключение — возвращает список строк.
        """
        cfg = _normalize_config(config)
        errors: List[str] = []

        # --- Learning Rate ---
        lr = cfg.get("learning_rate")
        if lr is not None:
            try:
                lr = float(lr)
                if lr <= 0:
                    errors.append("Скорость обучения (LR) должна быть > 0.")
                elif lr > 10:
                    errors.append(
                        f"Скорость обучения {lr} подозрительно велика. "
                        "Обычно используют 0.0001–0.1."
                    )
            except (TypeError, ValueError):
                errors.append(f"Скорость обучения должна быть числом, получено: {lr!r}")

        # --- Weight Decay ---
        wd = cfg.get("weight_decay")
        if wd is not None:
            try:
                wd = float(wd)
                if wd < 0:
                    errors.append("Штраф за зубрёжку (weight_decay) не может быть отрицательным.")
                elif wd > 1:
                    errors.append(
                        f"Weight decay = {wd} экстремально велик. "
                        "Обычно используют 0.0–0.1."
                    )
            except (TypeError, ValueError):
                errors.append(f"Weight decay должен быть числом, получено: {wd!r}")

        # --- Optimizer name ---
        opt_name = _safe_get(cfg, "optimizer", "adam")
        if opt_name and str(opt_name).lower().strip() not in OPTIMIZER_REGISTRY:
            available = ", ".join(sorted(OPTIMIZER_REGISTRY.keys()))
            errors.append(
                f"Неизвестный оптимизатор «{opt_name}». Доступные: {available}"
            )

        # --- Scheduler name ---
        sched_name = _safe_get(cfg, "scheduler", "none")
        if sched_name and str(sched_name).lower().strip() not in SCHEDULER_REGISTRY:
            available = ", ".join(sorted(SCHEDULER_REGISTRY.keys()))
            errors.append(
                f"Неизвестный планировщик «{sched_name}». Доступные: {available}"
            )

        # --- Epochs ---
        epochs = cfg.get("epochs")
        if epochs is not None:
            try:
                epochs = int(epochs)
                if epochs <= 0:
                    errors.append("Количество эпох должно быть > 0.")
                elif epochs > 100000:
                    errors.append(
                        f"Количество эпох ({epochs}) подозрительно велико. "
                        "Обучение может занять очень много времени."
                    )
            except (TypeError, ValueError):
                errors.append(f"Количество эпох должно быть целым числом, получено: {epochs!r}")

        # --- Batch size ---
        bs = cfg.get("batch_size")
        if bs is not None:
            try:
                bs = int(bs)
                if bs <= 0:
                    errors.append("Размер батча должен быть > 0.")
            except (TypeError, ValueError):
                errors.append(f"Размер батча должен быть целым числом, получено: {bs!r}")

        return errors

    # ----------------------------------------------------------
    #  ПРОВЕРКА СОВМЕСТИМОСТИ
    # ----------------------------------------------------------
    @staticmethod
    def check_compatibility(config: Any) -> List[str]:
        """
        Проверяет совместимость оптимизатора с типом модели и данных.
        Возвращает список ПРЕДУПРЕЖДЕНИЙ (не блокирующих).
        """
        cfg = _normalize_config(config)
        warnings: List[str] = []

        opt_name = _safe_get(cfg, "optimizer", "adam").lower().strip()
        model_type = _safe_get(cfg, "type", "").lower()
        data_type = _safe_get(cfg, "data_type", "numeric")
        lr = float(_safe_get(cfg, "learning_rate", 0.001))

        # Трансформеры + Adam без AdamW
        if model_type in ("transformer", "transformer_seq2seq", "vit") and opt_name == "adam":
            warnings.append(
                "💡 Для Трансформеров рекомендуется AdamW вместо Adam. "
                "Он корректнее применяет регуляризацию."
            )

        # GAN + не-Adam
        if model_type in ("gan", "dcgan") and opt_name not in ("adam", "rmsprop"):
            warnings.append(
                "⚠️ Для GAN обычно используют Adam (β1=0.5, β2=0.999) или RMSprop. "
                f"Текущий выбор: {opt_name}."
            )

        # RNN + AdamW (не ошибка, но не стандарт)
        if model_type in ("rnn", "lstm", "gru") and opt_name == "adamw":
            warnings.append(
                "💡 Для рекуррентных сетей чаще используют Adam или RMSprop, "
                "но AdamW тоже может сработать."
            )

        # Очень высокий LR для трансформеров
        if model_type in ("transformer", "transformer_seq2seq", "vit") and lr > 0.01:
            warnings.append(
                f"⚠️ LR={lr} слишком высок для Трансформера. "
                "Рекомендуемый диапазон: 1e-5 — 5e-3."
            )

        return warnings

    # ----------------------------------------------------------
    #  СПРАВКА ПО ДОСТУПНЫМ ВАРИАНТАМ
    # ----------------------------------------------------------
    @staticmethod
    def list_optimizers() -> List[str]:
        """Возвращает список доступных оптимизаторов (для GUI)."""
        return sorted(OPTIMIZER_REGISTRY.keys())

    @staticmethod
    def list_schedulers() -> List[str]:
        """Возвращает список доступных планировщиков (для GUI)."""
        return sorted(SCHEDULER_REGISTRY.keys())

    @staticmethod
    def get_scheduler_default_params(scheduler_name: str) -> Dict[str, Any]:
        """Возвращает параметры по умолчанию для планировщика (для GUI)."""
        name = scheduler_name.lower().strip()
        if name in SCHEDULER_REGISTRY:
            return dict(SCHEDULER_REGISTRY[name].get("params", {}))
        return {}


# ============================================================
#  ОБЁРТКИ ОБРАТНОЙ СОВМЕСТИМОСТИ
# ============================================================
# Старый интерфейс (функции верхнего уровня) сохраняется,
# чтобы существующий код не сломался при замене файла.

def create_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """
    Устаревший интерфейс. Используйте OptimizerFactory.create().
    Сохранён для обратной совместимости с trainer.py версии 1.x.
    """
    return OptimizerFactory.create(model, config)


def create_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any]):
    """
    Устаревший интерфейс. Используйте OptimizerFactory.create_scheduler().
    Сохранён для обратной совместимости с trainer.py версии 1.x.
    """
    return OptimizerFactory.create_scheduler(optimizer, config)


# ============================================================
#  САМОТЕСТ (запуск: python -m core.optimizer_factory)
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ТЕСТ: core/optimizer_factory.py")
    print("=" * 60)

    # 1. Тест создания оптимизатора
    dummy_model = torch.nn.Linear(10, 2)
    test_config = {
        "optimizer": "adamw",
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "epochs": 50,
        "scheduler": "cosine",
        "scheduler_params": {"eta_min": 1e-6},
    }

    opt = OptimizerFactory.create(dummy_model, test_config)
    print(f"✅ Оптимизатор: {opt.__class__.__name__}")

    sched = OptimizerFactory.create_scheduler(opt, test_config)
    print(f"✅ Планировщик: {sched.__class__.__name__}")

    # 2. Тест кривой LR
    curve = OptimizerFactory.get_lr_curve(test_config, total_epochs=50)
    print(f"✅ Кривая LR: {len(curve)} точек, "
          f"первая={curve[0]:.6f}, последняя={curve[-1]:.6f}")

    # 3. Тест валидации
    bad_config = {"learning_rate": -0.5, "optimizer": "nonexistent"}
    errs = OptimizerFactory.validate(bad_config)
    print(f"✅ Валидация плохого конфига: {len(errs)} ошибок")
    for e in errs:
        print(f"   • {e}")

    # 4. Тест подсказки
    hint = OptimizerFactory.get_contextual_hint(test_config)
    print(f"✅ Контекстная подсказка:\n{hint}")

    # 5. Тест GAN
    gen = torch.nn.Linear(10, 10)
    disc = torch.nn.Linear(10, 1)
    g_opt, d_opt = OptimizerFactory.create_gan_optimizers(gen, disc, {
        "optimizer": "adam",
        "learning_rate": 0.0002,
        "beta1": 0.5,
        "beta2": 0.999,
    })
    print(f"✅ GAN оптимизаторы: G={g_opt.__class__.__name__}, D={d_opt.__class__.__name__}")

    # 6. Список доступных
    print(f"✅ Оптимизаторы: {OptimizerFactory.list_optimizers()}")
    print(f"✅ Планировщики: {OptimizerFactory.list_schedulers()}")

    print("\n" + "=" * 60)
    print("ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
    print("=" * 60)