"""
core/preset_registry.py
=======================
Модуль: Реестр пресетов датасетов и моделей.

НАЗНАЧЕНИЕ:
    Хранит готовые, проверенные комбинации «задача генератора → параметры →
    рекомендуемая архитектура → рекомендуемые гиперпараметры».
    Пресеты позволяют ученику начать работу в один клик, а продвинутому
    пользователю — быстро оттолкнуться от разумной базы.

ЗАВИСИМОСТИ (по ТЗ):
    - Импортирует типы из `core/contracts.py` (если файл уже создан).
    - Если контракты ещё не готовы — использует локальные совместимые константы,
      чтобы модуль собирался и работал независимо (слепая сборка).

ПРАВИЛА ИСПОЛЬЗОВАНИЯ:
    1. Другие модули НЕ должны копировать словарь пресетов.
       Использовать ТОЛЬКО через `PresetRegistry`.
    2. Каждый пресет обязан содержать поле `format_version`.
    3. Пресеты с `available == False` — заглушки под будущие задачи.
       Они видны в интерфейсе, но помечены как «скоро».

ВХОДЫ:
    Нет внешних входов. Реестр статический.

ВЫХОДЫ:
    - `PresetRegistry.get_all()` -> List[Dict]
    - `PresetRegistry.get_by_id(id)` -> Dict | None
    - `PresetRegistry.get_available()` -> List[Dict]
    - `PresetRegistry.get_by_category(cat)` -> List[Dict]
    - `PresetRegistry.get_by_difficulty(level)` -> List[Dict]
    - `PresetRegistry.validate()` -> List[str] (список ошибок)
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional

# ============================================================
# СОВМЕСТИМОСТЬ С КОНТРАКТАМИ (слепая сборка)
# ============================================================
# По ТЗ все типы должны браться из core/contracts.py.
# Если контракты ещё не созданы,
# определяем совместимые строковые константы локально,
# чтобы модуль был работоспособен. При появлении контрактов
# импорт подхватится автоматически без изменения логики.
try:
    from core.contracts import (
        DataType,
        TaskType,
        ArchitectureType,
        FORMAT_VERSION,
    )
    _CONTRACTS_AVAILABLE = True
except ImportError:  # pragma: no cover — контракты ещё не созданы
    _CONTRACTS_AVAILABLE = False

    FORMAT_VERSION = "2.0"

    class DataType(str, Enum):
        TEXT = "text"
        NUMERIC = "numeric"
        IMAGE = "image"
        AUDIO = "audio"
        GRAPH = "graph"
        TIME_SERIES = "time_series"
        MULTIMODAL = "multimodal"

    class TaskType(str, Enum):
        CLASSIFICATION = "classification"
        REGRESSION = "regression"
        SEQ2SEQ = "seq2seq"
        IMAGE_CLASSIFICATION = "image_classification"
        IMAGE_GENERATION = "image_generation"
        AUDIO_CLASSIFICATION = "audio_classification"
        AUDIO_GENERATION = "audio_generation"
        SPEECH_TO_TEXT = "speech_to_text"
        TEXT_TO_SPEECH = "text_to_speech"
        TRANSLATION = "translation"
        ANOMALY_DETECTION = "anomaly_detection"
        CLUSTERING = "clustering"
        REINFORCEMENT = "reinforcement"

    class ArchitectureType(str, Enum):
        MLP = "mlp"
        CNN = "cnn"
        RNN = "rnn"
        LSTM = "lstm"
        GRU = "gru"
        TRANSFORMER_SEQ2SEQ = "transformer_seq2seq"
        TRANSFORMER_ENCODER = "transformer_encoder"
        TRANSFORMER_DECODER = "transformer_decoder"
        AUTOENCODER = "autoencoder"
        VAE = "vae"
        CVAE = "cvae"
        GAN = "gan"
        DCGAN = "dcgan"
        DIFFUSION = "diffusion"
        LATENT_DIFFUSION = "latent_diffusion"
        VIT = "vit"
        CAPSNET = "capsnet"
        RESNET = "resnet"
        GNN = "gnn"
        GCN = "gcn"
        GAT = "gat"
        RBFN = "rbfn"
        SOM = "som"
        LIQUID_NN = "liquid_nn"
        SPIKING_NN = "spiking_nn"
        PINN = "pinn"
        NEURAL_ODE = "neural_ode"
        CONV_AE = "conv_ae"
        CNN_RNN = "cnn_rnn"
        CLIP_LIKE = "clip_like"
        GRAPH_TRANSFORMER = "graph_transformer"
        CUSTOM = "custom"


# Локальный логгер (зависит от core/logger при наличии)
try:
    from core.logger import get_logger
    _logger = get_logger(__name__)
except ImportError:  # pragma: no cover
    _logger = logging.getLogger(__name__)


# ============================================================
# ТИПЫ И КОНСТАНТЫ МОДУЛЯ
# ============================================================
class PresetCategory(str, Enum):
    """Категории пресетов для группировки в интерфейсе."""
    MATH = "math"
    CRYPTO = "crypto"
    IMAGES = "images"
    AUDIO = "audio"
    TIME_SERIES = "time_series"
    GRAPHS = "graphs"
    TABULAR = "tabular"
    GENERATIVE = "generative"
    MULTIMODAL = "multimodal"


class PresetDifficulty(str, Enum):
    """Уровень сложности пресета (для подсветки и фильтрации)."""
    BEGINNER = "beginner"          # первый запуск, всё просто
    INTERMEDIATE = "intermediate"  # требует понимания
    ADVANCED = "advanced"          # для продвинутых / требует GPU


# Минимальная версия формата пресета
PRESET_FORMAT_VERSION = "1.0"


# ============================================================
# СТРУКТУРА ПРЕСЕТА
# ============================================================
@dataclass
class Preset:
    """
    Один пресет — готовый рецепт эксперимента.

    Поля обязательны для заполнения. При изменении структуры
    необходимо обновить `PRESET_FORMAT_VERSION` и добавить
    миграцию в `PresetRegistry._migrate_preset`.
    """
    # Идентификация
    id: str                              # уникальный машинный идентификатор
    name: str                            # человекочитаемое имя с эмодзи
    description: str                     # подробное описание для ученика
    category: PresetCategory             # категория для группировки
    difficulty: PresetDifficulty         # сложность

    # Данные
    task: str                            # ID задачи генератора (из generator.TASKS)
    data_type: DataType                  # тип данных (совместимо с contracts)
    task_type: TaskType                  # тип задачи (совместимо с contracts)
    params: Dict[str, Any]               # параметры для генератора
    recommended_num_samples: int         # рекомендуемое число примеров

    # Модель
    recommended_architecture: ArchitectureType  # рекомендуемая архитектура
    recommended_config: Dict[str, Any]   # конфиг для ModelFactory / панели

    # Служебные
    available: bool = True               # реализовано ли сейчас (иначе «скоро»)
    tags: List[str] = field(default_factory=list)
    notes: str = ""                      # заметки, ограничения
    format_version: str = PRESET_FORMAT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в JSON-совместимый словарь."""
        d = asdict(self)
        # Преобразуем enum в строки для JSON
        d["category"] = self.category.value
        d["difficulty"] = self.difficulty.value
        d["data_type"] = self.data_type.value
        d["task_type"] = self.task_type.value
        d["recommended_architecture"] = self.recommended_architecture.value
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Preset":
        """Десериализация с обратным преобразованием строк в enum."""
        d = dict(d)  # копия, чтобы не портить исходник
        d["category"] = PresetCategory(d.get("category", "math"))
        d["difficulty"] = PresetDifficulty(d.get("difficulty", "beginner"))
        d["data_type"] = DataType(d.get("data_type", "text"))
        d["task_type"] = TaskType(d.get("task_type", "seq2seq"))
        d["recommended_architecture"] = ArchitectureType(
            d.get("recommended_architecture", "mlp")
        )
        # Допустимы лишние поля — сохраняем их в notes через extra
        known_keys = {f for f in Preset.__dataclass_fields__}  # type: ignore[attr-defined]
        extra = {k: v for k, v in d.items() if k not in known_keys}
        for k in extra:
            d.pop(k, None)
        if extra:
            existing_notes = d.get("notes", "")
            d["notes"] = (existing_notes + f" | extra_fields={extra}").strip()
        return Preset(**d)


# ============================================================
# РЕЕСТР ПРЕСЕТОВ
# ============================================================
# Внимание: пресеты с `available=True` обязаны ссылаться на задачи,
# которые УЖЕ реализованы в текущем генераторе (9 текстовых задач).
# Пресеты с `available=False` — заглушки под будущие задачи
# (изображения, звук, графы, временные ряды, генеративные модели).
# ============================================================

_PRESETS: List[Preset] = [

    # =========================================================
    # МАТЕМАТИКА (доступно сейчас)
    # =========================================================
    Preset(
        id="math_addition_simple",
        name="🧮 Простое сложение (однозначные)",
        description=(
            "Сложение однозначных чисел. Идеальный первый эксперимент: "
            "словарь крошечный, примеры короткие, Трансформер учится быстро."
        ),
        category=PresetCategory.MATH,
        difficulty=PresetDifficulty.BEGINNER,
        task="addition",
        data_type=DataType.TEXT,
        task_type=TaskType.SEQ2SEQ,
        params={"num_range": (1, 9)},
        recommended_num_samples=5000,
        recommended_architecture=ArchitectureType.TRANSFORMER_SEQ2SEQ,
        recommended_config={
            "type": "transformer",
            "embedding_dim": 64,
            "num_heads": 4,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "dim_feedforward": 256,
            "activation": "relu",
            "max_len": 20,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "epochs": 20,
            "batch_size": 64,
        },
        available=True,
        tags=["арифметика", "сложение", "первый старт"],
        notes="Если хочется проще — уменьшите количество примеров до 2000.",
    ),

    Preset(
        id="math_addition_two_digit",
        name="🔢 Двузначное сложение (до 99)",
        description=(
            "Сложение двузначных чисел. Нейросети придётся освоить перенос разрядов. "
            "Хороший следующий шаг после однозначного сложения."
        ),
        category=PresetCategory.MATH,
        difficulty=PresetDifficulty.INTERMEDIATE,
        task="addition",
        data_type=DataType.TEXT,
        task_type=TaskType.SEQ2SEQ,
        params={"num_range": (10, 99)},
        recommended_num_samples=20000,
        recommended_architecture=ArchitectureType.TRANSFORMER_SEQ2SEQ,
        recommended_config={
            "type": "transformer",
            "embedding_dim": 128,
            "num_heads": 8,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "dim_feedforward": 512,
            "activation": "relu",
            "max_len": 30,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "epochs": 30,
            "batch_size": 64,
        },
        available=True,
        tags=["арифметика", "сложение", "двузначные"],
    ),

    Preset(
        id="math_subtraction_with_negative",
        name="➖ Вычитание с отрицательными результатами",
        description=(
            "Вычитание, где ответ может быть отрицательным. Сеть должна понять знак. "
            "Хорошо демонстрирует, как формат выхода влияет на словарь."
        ),
        category=PresetCategory.MATH,
        difficulty=PresetDifficulty.INTERMEDIATE,
        task="subtraction",
        data_type=DataType.TEXT,
        task_type=TaskType.SEQ2SEQ,
        params={"num_range": (1, 99), "allow_negative": True},
        recommended_num_samples=20000,
        recommended_architecture=ArchitectureType.TRANSFORMER_SEQ2SEQ,
        recommended_config={
            "type": "transformer",
            "embedding_dim": 128,
            "num_heads": 8,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "dim_feedforward": 512,
            "max_len": 30,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "epochs": 30,
        },
        available=True,
        tags=["арифметика", "вычитание", "отрицательные"],
    ),

    Preset(
        id="math_multiplication_small",
        name="✖️ Таблица умножения (до 12)",
        description=(
            "Умножение однозначных чисел до 12. Классика: модель фактически "
            "заучивает таблицу умножения вместе с вами."
        ),
        category=PresetCategory.MATH,
        difficulty=PresetDifficulty.BEGINNER,
        task="multiplication",
        data_type=DataType.TEXT,
        task_type=TaskType.SEQ2SEQ,
        params={"num_range": (1, 12)},
        recommended_num_samples=5000,
        recommended_architecture=ArchitectureType.TRANSFORMER_SEQ2SEQ,
        recommended_config={
            "type": "transformer",
            "embedding_dim": 64,
            "num_heads": 4,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "dim_feedforward": 256,
            "max_len": 20,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "epochs": 20,
        },
        available=True,
        tags=["арифметика", "умножение", "таблица умножения"],
    ),

    Preset(
        id="math_division_exact",
        name="➗ Деление нацело",
        description=(
            "Деление без остатка. Сеть учится обратной операции и «переворачивает» "
            "логику умножения. Хороший тест на обобщение."
        ),
        category=PresetCategory.MATH,
        difficulty=PresetDifficulty.INTERMEDIATE,
        task="division",
        data_type=DataType.TEXT,
        task_type=TaskType.SEQ2SEQ,
        params={"num_range": (1, 99)},
        recommended_num_samples=15000,
        recommended_architecture=ArchitectureType.TRANSFORMER_SEQ2SEQ,
        recommended_config={
            "type": "transformer",
            "embedding_dim": 128,
            "num_heads": 8,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "dim_feedforward": 512,
            "max_len": 30,
        },
        available=True,
        tags=["арифметика", "деление"],
    ),

    Preset(
        id="math_mixed_operations",
        name="🧮 Смешанные операции (приоритет действий)",
        description=(
            "Выражения вида a*b+c. Сеть должна освоить приоритет операций. "
            "Сложная задача, требует внимательности и достаточного количества данных."
        ),
        category=PresetCategory.MATH,
        difficulty=PresetDifficulty.ADVANCED,
        task="mixed_math",
        data_type=DataType.TEXT,
        task_type=TaskType.SEQ2SEQ,
        params={"num_range": (1, 20)},
        recommended_num_samples=30000,
        recommended_architecture=ArchitectureType.TRANSFORMER_SEQ2SEQ,
        recommended_config={
            "type": "transformer",
            "embedding_dim": 128,
            "num_heads": 8,
            "num_encoder_layers": 3,
            "num_decoder_layers": 2,
            "dim_feedforward": 512,
            "max_len": 40,
            "learning_rate": 0.0005,
            "epochs": 40,
        },
        available=True,
        tags=["арифметика", "приоритет операций", "продвинутый"],
    ),

    Preset(
        id="math_linear_equation",
        name="📐 Линейные уравнения (ax + b = c)",
        description=(
            "Решение линейных уравнений. Нейросеть учится «переносить» переменную "
            "и инвертировать операции. Настоящий тест логического вывода."
        ),
        category=PresetCategory.MATH,
        difficulty=PresetDifficulty.INTERMEDIATE,
        task="linear_equation",
        data_type=DataType.TEXT,
        task_type=TaskType.SEQ2SEQ,
        params={"coeff_range": (1, 15), "root_range": (-50, 50)},
        recommended_num_samples=20000,
        recommended_architecture=ArchitectureType.TRANSFORMER_SEQ2SEQ,
        recommended_config={
            "type": "transformer",
            "embedding_dim": 128,
            "num_heads": 8,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "dim_feedforward": 512,
            "max_len": 40,
        },
        available=True,
        tags=["алгебра", "уравнения"],
    ),

    Preset(
        id="math_quadratic_roots",
        name="📈 Квадратные уравнения (корни)",
        description=(
            "Нахождение корней квадратного уравнения. Сеть неявно ищет аналог "
            "формулы дискриминанта. Требует глубокой сети и больших данных."
        ),
        category=PresetCategory.MATH,
        difficulty=PresetDifficulty.ADVANCED,
        task="quadratic",
        data_type=DataType.TEXT,
        task_type=TaskType.SEQ2SEQ,
        params={"root_range": (-15, 15)},
        recommended_num_samples=50000,
        recommended_architecture=ArchitectureType.TRANSFORMER_SEQ2SEQ,
        recommended_config={
            "type": "transformer",
            "embedding_dim": 256,
            "num_heads": 8,
            "num_encoder_layers": 4,
            "num_decoder_layers": 2,
            "dim_feedforward": 1024,
            "max_len": 60,
            "learning_rate": 0.0003,
            "epochs": 50,
        },
        available=True,
        tags=["алгебра", "квадратные уравнения", "продвинутый"],
    ),

    # =========================================================
    # КРИПТОГРАФИЯ (доступно сейчас)
    # =========================================================
    Preset(
        id="crypto_caesar_shift",
        name="🔐 Шифр Цезаря (сдвиг на 3)",
        description=(
            "Расшифровка шифра Цезаря со сдвигом на 3. Отличная задача для "
            "понимания биективных отображений и механизмов внимания."
        ),
        category=PresetCategory.CRYPTO,
        difficulty=PresetDifficulty.BEGINNER,
        task="cipher_caesar",
        data_type=DataType.TEXT,
        task_type=TaskType.SEQ2SEQ,
        params={"shift": 3, "word_length": (5, 10)},
        recommended_num_samples=10000,
        recommended_architecture=ArchitectureType.TRANSFORMER_SEQ2SEQ,
        recommended_config={
            "type": "transformer",
            "embedding_dim": 64,
            "num_heads": 4,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "dim_feedforward": 256,
            "max_len": 30,
        },
        available=True,
        tags=["шифр", "цезарь", "криптография"],
    ),

    Preset(
        id="crypto_atbash_mirror",
        name="🔏 Шифр Атбаш (отражение алфавита)",
        description=(
            "Расшифровка Атбаша, где алфавит перевёрнут. Модель должна выучить "
            "полную таблицу подстановок. Идеально для Seq2Seq."
        ),
        category=PresetCategory.CRYPTO,
        difficulty=PresetDifficulty.INTERMEDIATE,
        task="cipher_atbash",
        data_type=DataType.TEXT,
        task_type=TaskType.SEQ2SEQ,
        params={"word_length": (6, 12)},
        recommended_num_samples=10000,
        recommended_architecture=ArchitectureType.TRANSFORMER_SEQ2SEQ,
        recommended_config={
            "type": "transformer",
            "embedding_dim": 64,
            "num_heads": 4,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "dim_feedforward": 256,
            "max_len": 30,
        },
        available=True,
        tags=["шифр", "атбаш", "криптография"],
    ),

    # =========================================================
    # ИЗОБРАЖЕНИЯ (заглушки: генератор ещё не реализован)
    # =========================================================
    Preset(
        id="image_shapes_beginner",
        name="🔷 Фигуры для новичков (круг/квадрат/треугольник)",
        description=(
            "Классификация простых 2D фигур. Свёрточная сеть учится выделять "
            "края и формы. Отличный старт для компьютерного зрения."
        ),
        category=PresetCategory.IMAGES,
        difficulty=PresetDifficulty.BEGINNER,
        task="shapes_2d",
        data_type=DataType.IMAGE,
        task_type=TaskType.IMAGE_CLASSIFICATION,
        params={"size": (28, 28), "shapes": ["circle", "square", "triangle"], "noise": 0.0},
        recommended_num_samples=3000,
        recommended_architecture=ArchitectureType.CNN,
        recommended_config={
            "type": "cnn",
            "conv_layers": 2,
            "kernel_size": 3,
            "filters": [16, 32],
            "dropout": 0.2,
            "learning_rate": 0.001,
            "epochs": 15,
            "batch_size": 32,
        },
        available=False,  # заглушка
        tags=["изображения", "классификация", "фигуры"],
        notes="Задача будет доступна после реализации генератора изображений.",
    ),

    Preset(
        id="image_shapes_3d",
        name="🧊 3D проекции (куб/сфера/пирамида)",
        description=(
            "Классификация 2D проекций 3D фигур. Модель учится узнавать объёмные "
            "объекты по плоскому изображению. Сложнее, чем обычные фигуры."
        ),
        category=PresetCategory.IMAGES,
        difficulty=PresetDifficulty.INTERMEDIATE,
        task="shapes_3d",
        data_type=DataType.IMAGE,
        task_type=TaskType.IMAGE_CLASSIFICATION,
        params={"size": (64, 64), "shapes": ["cube", "sphere", "pyramid", "cylinder"]},
        recommended_num_samples=5000,
        recommended_architecture=ArchitectureType.CNN,
        recommended_config={
            "type": "cnn",
            "conv_layers": 3,
            "kernel_size": 3,
            "filters": [16, 32, 64],
            "dropout": 0.3,
        },
        available=False,
        tags=["изображения", "3D", "классификация"],
    ),

    Preset(
        id="image_denoise_autoencoder",
        name="🔇 Очистка шума (Автоэнкодер)",
        description=(
            "Автоэнкодер учится убирать шум с изображений. Вход — зашумлённая "
            "картинка, выход — чистая. Наглядная демонстрация сжатия признаков."
        ),
        category=PresetCategory.IMAGES,
        difficulty=PresetDifficulty.INTERMEDIATE,
        task="noise_denoise",
        data_type=DataType.IMAGE,
        task_type=TaskType.IMAGE_GENERATION,
        params={"noise_level": 0.3, "size": (28, 28)},
        recommended_num_samples=5000,
        recommended_architecture=ArchitectureType.CONV_AE,
        recommended_config={
            "type": "conv_ae",
            "latent_dim": 32,
            "learning_rate": 0.001,
            "epochs": 30,
        },
        available=False,
        tags=["изображения", "автоэнкодер", "шумоподавление"],
    ),

    Preset(
        id="image_gan_patterns",
        name="🎨 Генерация паттернов (простой GAN)",
        description=(
            "Генеративно-состязательная сеть учится рисовать простые паттерны. "
            "Генератор подделывает, дискриминатор разоблачает. Эксперимент для смелых."
        ),
        category=PresetCategory.GENERATIVE,
        difficulty=PresetDifficulty.ADVANCED,
        task="pattern_generation",
        data_type=DataType.IMAGE,
        task_type=TaskType.IMAGE_GENERATION,
        params={"size": (32, 32)},
        recommended_num_samples=2000,
        recommended_architecture=ArchitectureType.DCGAN,
        recommended_config={
            "type": "dcgan",
            "latent_dim": 64,
            "learning_rate": 0.0002,
            "epochs": 50,
        },
        available=False,
        tags=["изображения", "генерация", "GAN"],
        notes="Требует стабильного GPU и аккуратного подбора скорости обучения.",
    ),

    # =========================================================
    # ЗВУК (заглушки: генератор ещё не реализован)
    # =========================================================
    Preset(
        id="audio_tone_classifier",
        name="🎵 Угадай ноту (частоты)",
        description=(
            "Классификация звуковых тонов по частоте. Модель учится различать "
            "ноты по спектру. Хороший старт в аудио-задачах."
        ),
        category=PresetCategory.AUDIO,
        difficulty=PresetDifficulty.BEGINNER,
        task="tone_classification",
        data_type=DataType.AUDIO,
        task_type=TaskType.AUDIO_CLASSIFICATION,
        params={"frequencies": [262, 294, 330, 349, 392], "duration": 0.5, "sample_rate": 16000},
        recommended_num_samples=2000,
        recommended_architecture=ArchitectureType.CNN,
        recommended_config={
            "type": "cnn",
            "input_as_spectrogram": True,
            "conv_layers": 2,
            "filters": [16, 32],
            "dropout": 0.2,
        },
        available=False,
        tags=["звук", "частоты", "классификация"],
        notes="Звук подаётся как спектрограмма. Требуется генератор аудио.",
    ),

    Preset(
        id="audio_frequency_regression",
        name="📊 Предсказание частоты звука",
        description=(
            "Регрессия: по фрагменту звука предсказать его частоту. "
            "Модель учится извлекать числовой признак из сигнала."
        ),
        category=PresetCategory.AUDIO,
        difficulty=PresetDifficulty.INTERMEDIATE,
        task="frequency_regression",
        data_type=DataType.AUDIO,
        task_type=TaskType.REGRESSION,
        params={"freq_range": (100, 1000), "duration": 0.3},
        recommended_num_samples=3000,
        recommended_architecture=ArchitectureType.MLP,
        recommended_config={
            "type": "mlp",
            "hidden_layers": [64, 64],
            "activation": "relu",
            "dropout": 0.1,
        },
        available=False,
        tags=["звук", "регрессия"],
    ),

    # =========================================================
    # ВРЕМЕННЫЕ РЯДЫ (заглушки)
    # =========================================================
    Preset(
        id="series_sine_prediction",
        name="📈 Предскажи синус (LSTM)",
        description=(
            "LSTM учится предсказывать следующее значение синусоиды. "
            "Классика для понимания рекуррентных сетей и памяти."
        ),
        category=PresetCategory.TIME_SERIES,
        difficulty=PresetDifficulty.INTERMEDIATE,
        task="sine_wave",
        data_type=DataType.TIME_SERIES,
        task_type=TaskType.REGRESSION,
        params={"sequence_length": 50, "noise": 0.05},
        recommended_num_samples=1000,
        recommended_architecture=ArchitectureType.LSTM,
        recommended_config={
            "type": "lstm",
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "epochs": 40,
        },
        available=False,
        tags=["временные ряды", "синус", "LSTM"],
    ),

    Preset(
        id="series_anomaly_detection",
        name="🚨 Поиск аномалий в ряду",
        description=(
            "Автоэнкодер учится отличать нормальный сигнал от аномального. "
            "Основа промышленного мониторинга оборудования."
        ),
        category=PresetCategory.TIME_SERIES,
        difficulty=PresetDifficulty.ADVANCED,
        task="anomaly_detection",
        data_type=DataType.TIME_SERIES,
        task_type=TaskType.ANOMALY_DETECTION,
        params={"anomaly_ratio": 0.05, "sequence_length": 100},
        recommended_num_samples=2000,
        recommended_architecture=ArchitectureType.AUTOENCODER,
        recommended_config={
            "type": "autoencoder",
            "latent_dim": 16,
        },
        available=False,
        tags=["временные ряды", "аномалии", "автоэнкодер"],
    ),

    # =========================================================
    # ТАБЛИЦЫ / ЧИСЛА (частично доступно)
    # =========================================================
    Preset(
        id="tabular_xor_mlp",
        name="🔀 Классика: XOR на перцептроне",
        description=(
            "Легендарная задача, которую не мог решить однослойный перцептрон. "
            "Два скрытых слоя — и вот сеть учится логике «исключающего ИЛИ»."
        ),
        category=PresetCategory.TABULAR,
        difficulty=PresetDifficulty.BEGINNER,
        task="xor_problem",
        data_type=DataType.NUMERIC,
        task_type=TaskType.CLASSIFICATION,
        params={"dim": 2},
        recommended_num_samples=1000,
        recommended_architecture=ArchitectureType.MLP,
        recommended_config={
            "type": "mlp",
            "hidden_layers": [8, 8],
            "activation": "tanh",
            "dropout": 0.0,
            "learning_rate": 0.05,
            "epochs": 100,
            "batch_size": 16,
        },
        available=False,  # задача будет добавлена в генератор
        tags=["таблицы", "классификация", "XOR", "MLP"],
    ),

    Preset(
        id="tabular_function_approx",
        name="📉 Аппроксимация функции",
        description=(
            "Нейросеть учится приближать неизвестную функцию по точкам. "
            "Наглядный способ понять, как сеть «запоминает» форму кривой."
        ),
        category=PresetCategory.TABULAR,
        difficulty=PresetDifficulty.INTERMEDIATE,
        task="function_approximation",
        data_type=DataType.NUMERIC,
        task_type=TaskType.REGRESSION,
        params={"function": "sin", "x_range": (-5, 5), "noise": 0.1},
        recommended_num_samples=2000,
        recommended_architecture=ArchitectureType.MLP,
        recommended_config={
            "type": "mlp",
            "hidden_layers": [64, 64, 64],
            "activation": "tanh",
            "learning_rate": 0.01,
            "epochs": 100,
        },
        available=False,
        tags=["таблицы", "регрессия", "аппроксимация"],
    ),

    Preset(
        id="tabular_som_clustering",
        name="🗺️ Карта Кохонена (кластеризация)",
        description=(
            "Самоорганизующаяся карта Кохонена учится раскладывать данные "
            "по кластерам без учителя. Красивая визуализация топологии."
        ),
        category=PresetCategory.TABULAR,
        difficulty=PresetDifficulty.INTERMEDIATE,
        task="clustering_blobs",
        data_type=DataType.NUMERIC,
        task_type=TaskType.CLUSTERING,
        params={"n_clusters": 4, "dim": 2},
        recommended_num_samples=1000,
        recommended_architecture=ArchitectureType.SOM,
        recommended_config={
            "type": "som",
            "grid_size": (10, 10),
            "learning_rate": 0.5,
        },
        available=False,
        tags=["кластеризация", "без учителя", "SOM"],
    ),

    # =========================================================
    # ГРАФЫ (заглушки)
    # =========================================================
    Preset(
        id="graph_connectivity_basic",
        name="🕸️ Связность графа",
        description=(
            "Графовая нейросеть учится определять, связен ли граф. "
            "Знакомство с миром узлов, рёбер и сообщений между ними."
        ),
        category=PresetCategory.GRAPHS,
        difficulty=PresetDifficulty.ADVANCED,
        task="graph_connectivity",
        data_type=DataType.GRAPH,
        task_type=TaskType.CLASSIFICATION,
        params={"nodes_range": (5, 15)},
        recommended_num_samples=500,
        recommended_architecture=ArchitectureType.GCN,
        recommended_config={
            "type": "gcn",
            "hidden_dim": 32,
            "num_layers": 3,
        },
        available=False,
        tags=["графы", "связность", "продвинутый"],
    ),

    # =========================================================
    # МУЛЬТИМОДАЛЬНОСТЬ (заглушки)
    # =========================================================
    Preset(
        id="multimodal_translation_stub",
        name="🌍 Пары переводов (заготовка под ИИ)",
        description=(
            "Заготовка под будущую генерацию пар переводов через локальную модель. "
            "Сейчас пары будут создаваться из заранее подготовленного словаря."
        ),
        category=PresetCategory.MULTIMODAL,
        difficulty=PresetDifficulty.ADVANCED,
        task="translation_pairs",
        data_type=DataType.TEXT,
        task_type=TaskType.TRANSLATION,
        params={"language_pair": ("ru", "en"), "num_pairs": 1000},
        recommended_num_samples=1000,
        recommended_architecture=ArchitectureType.TRANSFORMER_SEQ2SEQ,
        recommended_config={
            "type": "transformer",
            "embedding_dim": 128,
            "num_heads": 8,
        },
        available=False,
        tags=["перевод", "мультимодальность", "будущее"],
        notes="Генерация пар может быть дополнена локальной моделью (см. curator).",
    ),
]


# ============================================================
# ПУБЛИЧНЫЙ РЕЕСТР
# ============================================================
class PresetRegistry:
    """
    Единая точка доступа к пресетам.

    Инстанс не обязателен: все методы статические, оперируют
    глобальным списком `_PRESETS`. Класс нужен для группировки
    и возможности подмены в тестах.
    """

    # ----------------------------------------------------------
    # Доступ
    # ----------------------------------------------------------
    @staticmethod
    def get_all(include_unavailable: bool = True) -> List[Dict[str, Any]]:
        """
        Возвращает список всех пресетов в виде словарей.

        :param include_unavailable: если False, исключает заглушки.
        :return: список словарей (копий) с данными пресетов.
        """
        result = []
        for p in _PRESETS:
            if not include_unavailable and not p.available:
                continue
            result.append(p.to_dict())
        return result

    @staticmethod
    def get_by_id(preset_id: str) -> Optional[Dict[str, Any]]:
        """Возвращает пресет по уникальному идентификатору или None."""
        for p in _PRESETS:
            if p.id == preset_id:
                return p.to_dict()
        return None

    @staticmethod
    def get_available() -> List[Dict[str, Any]]:
        """Только пресеты, которые можно запустить прямо сейчас."""
        return PresetRegistry.get_all(include_unavailable=False)

    @staticmethod
    def get_by_category(category: PresetCategory, include_unavailable: bool = True) -> List[Dict[str, Any]]:
        """Фильтр по категории."""
        return [
            p.to_dict() for p in _PRESETS
            if p.category == category and (include_unavailable or p.available)
        ]

    @staticmethod
    def get_by_difficulty(difficulty: PresetDifficulty, include_unavailable: bool = True) -> List[Dict[str, Any]]:
        """Фильтр по сложности."""
        return [
            p.to_dict() for p in _PRESETS
            if p.difficulty == difficulty and (include_unavailable or p.available)
        ]

    @staticmethod
    def get_categories() -> List[PresetCategory]:
        """Список всех категорий для построения меню."""
        return list(PresetCategory)

    @staticmethod
    def get_difficulties() -> List[PresetDifficulty]:
        """Список всех уровней сложности."""
        return list(PresetDifficulty)

    # ----------------------------------------------------------
    # Адаптация под существующие панели
    # ----------------------------------------------------------
    @staticmethod
    def to_generator_params(preset_id: str) -> Optional[Dict[str, Any]]:
        """
        Возвращает параметры для генератора:
        {'task': ..., 'num_samples': ..., 'params': {...}}.
        Используется в generator_panel для быстрого старта.
        """
        p = PresetRegistry.get_by_id(preset_id)
        if not p:
            return None
        return {
            "task": p["task"],
            "num_samples": p.get("recommended_num_samples", 1000),
            "params": copy.deepcopy(p.get("params", {})),
        }

    @staticmethod
    def to_model_config(preset_id: str) -> Optional[Dict[str, Any]]:
        """
        Возвращает рекомендуемый конфиг модели для передачи в
        ModelFactory / architecture_panel.
        """
        p = PresetRegistry.get_by_id(preset_id)
        if not p:
            return None
        return copy.deepcopy(p.get("recommended_config", {}))

    # ----------------------------------------------------------
    # Валидация и целостность
    # ----------------------------------------------------------
    @staticmethod
    def validate() -> List[str]:
        """
        Проверяет целостность реестра. Возвращает список ошибок.
        Пустой список — всё в порядке. Вызывать при запуске и в тестах.
        """
        errors: List[str] = []

        ids = [p.id for p in _PRESETS]
        # Дубликаты идентификаторов
        seen = set()
        for pid in ids:
            if pid in seen:
                errors.append(f"Дубликат идентификатора пресета: '{pid}'")
            seen.add(pid)

        # Минимальное количество (по ТЗ — не менее 15)
        if len(_PRESETS) < 15:
            errors.append(f"Пресетов слишком мало: {len(_PRESETS)} (нужно ≥15)")

        # Проверка каждого пресета
        for p in _PRESETS:
            if not p.id:
                errors.append("Найден пресет без идентификатора")
            if not p.name:
                errors.append(f"Пресет '{p.id}' без имени")
            if not p.task:
                errors.append(f"Пресет '{p.id}' без задачи генератора")
            if not p.description:
                errors.append(f"Пресет '{p.id}' без описания")
            if p.recommended_num_samples <= 0:
                errors.append(f"Пресет '{p.id}': recommended_num_samples <= 0")

            # Для доступных пресетов задача обязана быть реализована
            if p.available:
                known_tasks = {
                    "addition", "subtraction", "multiplication", "division",
                    "mixed_math", "linear_equation", "quadratic",
                    "cipher_caesar", "cipher_atbash",
                }
                if p.task not in known_tasks:
                    errors.append(
                        f"Пресет '{p.id}' помечен как доступный, но задача "
                        f"'{p.task}' не реализована в текущем генераторе"
                    )

            # Совместимость типа данных и архитектуры (упрощённая проверка)
            dt = p.data_type
            arch = p.recommended_architecture
            if dt == DataType.TEXT and arch in (
                ArchitectureType.CNN, ArchitectureType.MLP, ArchitectureType.DCGAN
            ):
                errors.append(
                    f"Пресет '{p.id}': текстовые данные несовместимы с архитектурой {arch.value}"
                )
            if dt == DataType.IMAGE and arch in (
                ArchitectureType.TRANSFORMER_SEQ2SEQ, ArchitectureType.MLP
            ):
                errors.append(
                    f"Пресет '{p.id}': изображения несовместимы с архитектурой {arch.value}"
                )

        return errors

    # ----------------------------------------------------------
    # Миграция (будущее)
    # ----------------------------------------------------------
    @staticmethod
    def _migrate_preset(raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Заглушка миграции пресетов старого формата.
        При изменении `PRESET_FORMAT_VERSION` добавить сюда логику.
        """
        if raw.get("format_version") != PRESET_FORMAT_VERSION:
            raw = dict(raw)
            raw["format_version"] = PRESET_FORMAT_VERSION
        return raw


# ============================================================
# САМОПРОВЕРКА ПРИ ИМПОРТЕ
# ============================================================
# Если реестр повреждён — лучше узнать об этом сразу, но не ронять
# приложение. Логируем предупреждение и выдаём пустой список.
_validation_errors = PresetRegistry.validate()
if _validation_errors:
    _logger.warning(
        "PresetRegistry: обнаружены проблемы целостности (%d):\n%s",
        len(_validation_errors),
        "\n".join(f" - {e}" for e in _validation_errors),
    )
else:
    _logger.info(
        "PresetRegistry: загружено %d пресетов (доступно сейчас: %d)",
        len(_PRESETS),
        sum(1 for p in _PRESETS if p.available),
    )


# ============================================================
# ПУБЛИЧНЫЙ ЭКСПОРТ
# ============================================================
__all__ = [
    "Preset",
    "PresetRegistry",
    "PresetCategory",
    "PresetDifficulty",
    "PRESET_FORMAT_VERSION",
]