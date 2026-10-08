"""
core/contracts.py
=================
ЕДИНСТВЕННЫЙ источник типов и контрактов данных проекта.
Версия формата: 2.0

ПРАВИЛА МОДУЛЯ:
1. Никакой модуль не создаёт свои dataclass для тех же сущностей — только импорт отсюда.
2. Все сохраняемые файлы содержат поле "format_version".
3. from_dict принимает данные версии 1.x (старый проект) и мигрирует их.
4. Неизвестные поля при десериализации сохраняются в поле "extra", не отбрасываются.
5. to_dict возвращает JSON-совместимый словарь (без tuple, set, dataclass).
6. Зависимости: ТОЛЬКО стандартная библиотека. Никакого torch, numpy, PyQt.
7. Любой модуль может импортировать этот файл без риска циклических зависимостей.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ============================================================
# ВЕРСИЯ ФОРМАТА
# ============================================================
FORMAT_VERSION = "2.0"
MIN_SUPPORTED_VERSION = "0.4"  # старая версия проекта, которую мы умеем читать


# ============================================================
# ПЕРЕЧИСЛЕНИЯ
# ============================================================

class DataType(str, Enum):
    """Тип данных, с которыми работает модель."""
    TEXT = "text"
    NUMERIC = "numeric"
    IMAGE = "image"
    AUDIO = "audio"
    GRAPH = "graph"
    TIME_SERIES = "time_series"
    MULTIMODAL = "multimodal"

    @staticmethod
    def from_str(value: str) -> "DataType":
        """Безопасное преобразование строки в DataType."""
        value = value.strip().lower()
        try:
            return DataType(value)
        except ValueError:
            return DataType.NUMERIC  # fallback для старых форматов


class TaskType(str, Enum):
    """Тип задачи обучения."""
    # Общие
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    SEQ2SEQ = "seq2seq"
    # Изображения
    IMAGE_CLASSIFICATION = "image_classification"
    IMAGE_GENERATION = "image_generation"
    IMAGE_DENOISING = "image_denoising"
    IMAGE_CAPTIONING = "image_captioning"
    # Звук
    AUDIO_CLASSIFICATION = "audio_classification"
    AUDIO_GENERATION = "audio_generation"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    # Текст
    TRANSLATION = "translation"
    TEXT_GENERATION = "text_generation"
    # Графы
    GRAPH_CLASSIFICATION = "graph_classification"
    NODE_CLASSIFICATION = "node_classification"
    LINK_PREDICTION = "link_prediction"
    # Временные ряды
    FORECASTING = "forecasting"
    ANOMALY_DETECTION = "anomaly_detection"
    # Другое
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    REINFORCEMENT = "reinforcement"

    @staticmethod
    def from_str(value: str) -> "TaskType":
        value = value.strip().lower()
        try:
            return TaskType(value)
        except ValueError:
            return TaskType.REGRESSION


class ArchitectureType(str, Enum):
    """Тип архитектуры нейросети."""
    # Базовые
    MLP = "mlp"
    CNN = "cnn"
    RNN = "rnn"
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER_SEQ2SEQ = "transformer_seq2seq"
    TRANSFORMER_ENCODER = "transformer_encoder"
    TRANSFORMER_DECODER = "transformer_decoder"
    # Генеративные
    AUTOENCODER = "autoencoder"
    VAE = "vae"
    CVAE = "cvae"
    GAN = "gan"
    DCGAN = "dcgan"
    DIFFUSION = "diffusion"
    LATENT_DIFFUSION = "latent_diffusion"
    # Продвинутые
    VIT = "vit"
    CAPSNET = "capsnet"
    RESNET = "resnet"
    GNN = "gnn"
    GCN = "gcn"
    GAT = "gat"
    GRAPH_TRANSFORMER = "graph_transformer"
    RBFN = "rbfn"
    SOM = "som"
    LIQUID_NN = "liquid_nn"
    SPIKING_NN = "spiking_nn"
    PINN = "pinn"
    NEURAL_ODE = "neural_ode"
    # Комбинированные
    CONV_AE = "conv_ae"
    CNN_RNN = "cnn_rnn"
    CLIP_LIKE = "clip_like"
    # Служебные
    MULTIMODAL = "multimodal"
    CUSTOM = "custom"

    @staticmethod
    def from_str(value: str) -> "ArchitectureType":
        """Безопасное преобразование. Старый формат: 'transformer' -> TRANSFORMER_SEQ2SEQ."""
        value = value.strip().lower()
        # Миграция со старых имён
        migration_map = {
            "transformer": "transformer_seq2seq",
        }
        value = migration_map.get(value, value)
        try:
            return ArchitectureType(value)
        except ValueError:
            return ArchitectureType.CUSTOM


class HintMode(str, Enum):
    """Режим подсказок."""
    STATIC = "static"      # Только правило-ориентированные
    AI = "ai"              # Только ИИ
    HYBRID = "hybrid"      # ИИ если доступен, иначе статические (по умолчанию)
    NONE = "none"          # Без подсказок

    @staticmethod
    def from_str(value: str) -> "HintMode":
        try:
            return HintMode(value.strip().lower())
        except ValueError:
            return HintMode.HYBRID


class ScenarioType(str, Enum):
    """Сценарий входа пользователя."""
    NEW = "new"            # Создаю с нуля
    FINETUNE = "finetune"  # Дообучаю готовую
    PLAY = "play"          # Играю с моделью / песочница

    @staticmethod
    def from_str(value: str) -> "ScenarioType":
        try:
            return ScenarioType(value.strip().lower())
        except ValueError:
            return ScenarioType.NEW


class UserLevel(str, Enum):
    """Уровень пользователя (для адаптации подсказок)."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class ExportFormat(str, Enum):
    """Формат экспорта."""
    OAI_PROJECT = "oai_project"       # Полный проект .oai
    PTH_MODEL = "pth_model"           # Только веса .pth
    FLASH_CHAT = "flash_chat"         # Папка для флешки
    HTML_SANDBOX = "html_sandbox"     # HTML с ONNX
    ONNX = "onnx"                     # ONNX формат
    REPORT = "report"                 # Отчёт .md/.html


# ============================================================
# КОНТРАКТ: МЕТАДАННЫЕ ДАТАСЕТА
# ============================================================

@dataclass
class DatasetMeta:
    """Метаданные датасета. Описывают ЧТО внутри, не сами данные."""
    format_version: str = FORMAT_VERSION
    data_type: DataType = DataType.TEXT
    task_type: TaskType = TaskType.SEQ2SEQ
    num_samples: int = 0
    # Размерности
    input_dim: Optional[Union[int, Tuple[int, ...]]] = None
    output_dim: Optional[Union[int, Tuple[int, ...]]] = None
    # Для текста
    vocab_size: int = 0
    max_sequence_length: int = 0
    # Для изображений
    image_shape: Optional[Tuple[int, int, int]] = None  # (H, W, C)
    # Для аудио
    audio_sample_rate: Optional[int] = None
    audio_channels: int = 1
    audio_duration_seconds: Optional[float] = None
    # Для графов
    num_nodes: Optional[int] = None
    num_edges: Optional[int] = None
    # Классификация
    num_classes: int = 0
    class_names: List[str] = field(default_factory=list)
    # Описание
    description: str = ""
    generator_task: str = ""  # какая задача генератора создала
    dataset_id: str = ""
    # Расширение для неизвестных полей
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "format_version": self.format_version,
            "data_type": self.data_type.value,
            "task_type": self.task_type.value,
            "num_samples": self.num_samples,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "vocab_size": self.vocab_size,
            "max_sequence_length": self.max_sequence_length,
            "image_shape": list(self.image_shape) if self.image_shape else None,
            "audio_sample_rate": self.audio_sample_rate,
            "audio_channels": self.audio_channels,
            "audio_duration_seconds": self.audio_duration_seconds,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "description": self.description,
            "generator_task": self.generator_task,
            "dataset_id": self.dataset_id,
        }
        d.update(self.extra)
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DatasetMeta":
        """Десериализация с миграцией со старых форматов."""
        known_keys = {
            "format_version", "data_type", "task_type", "num_samples",
            "input_dim", "output_dim", "vocab_size", "max_sequence_length",
            "image_shape", "audio_sample_rate", "audio_channels",
            "audio_duration_seconds", "num_nodes", "num_edges",
            "num_classes", "class_names", "description", "generator_task",
            "dataset_id", "type",  # "type" — старое имя для "data_type"
        }
        extra = {k: v for k, v in d.items() if k not in known_keys}

        # Миграция: старое поле "type" -> "data_type"
        raw_data_type = d.get("data_type", d.get("type", "numeric"))

        image_shape = d.get("image_shape")
        if image_shape is not None:
            image_shape = tuple(image_shape)

        input_dim = d.get("input_dim")
        if isinstance(input_dim, list):
            input_dim = tuple(input_dim)
        output_dim = d.get("output_dim")
        if isinstance(output_dim, list):
            output_dim = tuple(output_dim)

        return DatasetMeta(
            format_version=d.get("format_version", FORMAT_VERSION),
            data_type=DataType.from_str(str(raw_data_type)),
            task_type=TaskType.from_str(d.get("task_type", "regression")),
            num_samples=d.get("num_samples", 0),
            input_dim=input_dim,
            output_dim=output_dim,
            vocab_size=d.get("vocab_size", 0),
            max_sequence_length=d.get("max_sequence_length", 0),
            image_shape=image_shape,
            audio_sample_rate=d.get("audio_sample_rate"),
            audio_channels=d.get("audio_channels", 1),
            audio_duration_seconds=d.get("audio_duration_seconds"),
            num_nodes=d.get("num_nodes"),
            num_edges=d.get("num_edges"),
            num_classes=d.get("num_classes", 0),
            class_names=d.get("class_names", []),
            description=d.get("description", ""),
            generator_task=d.get("generator_task", d.get("task_type", "")),
            dataset_id=d.get("dataset_id", ""),
            extra=extra,
        )


# ============================================================
# КОНТРАКТ: КОНТЕЙНЕР ДАТАСЕТА
# ============================================================

@dataclass
class DatasetContainer:
    """
    Единый контейнер датасета.
    ВСЕ модули проекта работают ТОЛЬКО с этим типом.
    """
    meta: DatasetMeta = field(default_factory=DatasetMeta)
    # Исходные данные (до разбиения)
    raw_inputs: List[Any] = field(default_factory=list)
    raw_outputs: List[Any] = field(default_factory=list)
    # Разбиение
    train_inputs: List[Any] = field(default_factory=list)
    train_outputs: List[Any] = field(default_factory=list)
    val_inputs: List[Any] = field(default_factory=list)
    val_outputs: List[Any] = field(default_factory=list)
    test_inputs: List[Any] = field(default_factory=list)
    test_outputs: List[Any] = field(default_factory=list)
    # Для текста
    vocab: Dict[str, int] = field(default_factory=dict)
    # Для числовых данных
    normalization: Optional[Dict[str, Any]] = None  # {"mean": [...], "std": [...]}
    # Для изображений
    augmentation_config: Optional[Dict[str, Any]] = None
    # Расширение
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_split(self) -> bool:
        """Проверка: данные разбиты на выборки?"""
        return len(self.train_inputs) > 0

    @property
    def data_type(self) -> DataType:
        return self.meta.data_type

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) if self.vocab else self.meta.vocab_size

    def get_split_size(self, split: str) -> int:
        """Размер конкретной выборки: 'train', 'val', 'test'."""
        mapping = {
            "train": self.train_inputs,
            "val": self.val_inputs,
            "test": self.test_inputs,
        }
        return len(mapping.get(split, []))

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в JSON-совместимый словарь."""
        d = {
            "format_version": FORMAT_VERSION,
            "meta": self.meta.to_dict(),
            "raw_inputs": self.raw_inputs,
            "raw_outputs": self.raw_outputs,
            "train_inputs": self.train_inputs,
            "train_outputs": self.train_outputs,
            "val_inputs": self.val_inputs,
            "val_outputs": self.val_outputs,
            "test_inputs": self.test_inputs,
            "test_outputs": self.test_outputs,
            "vocab": self.vocab,
            "normalization": self.normalization,
            "augmentation_config": self.augmentation_config,
        }
        d.update(self.extra)
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DatasetContainer":
        """
        Десериализация с миграцией.
        Старый формат (v0.4): {"meta": {...}, "raw_inputs": [...], "raw_outputs": [...]}
        """
        known_keys = {
            "format_version", "meta", "raw_inputs", "raw_outputs",
            "train_inputs", "train_outputs", "val_inputs", "val_outputs",
            "test_inputs", "test_outputs", "vocab", "normalization",
            "augmentation_config",
        }
        extra = {k: v for k, v in d.items() if k not in known_keys}

        # Метаданные
        meta_dict = d.get("meta", {})
        meta = DatasetMeta.from_dict(meta_dict)

        # Если в старом формате не было vocab_size в мета, но есть словарь
        vocab = d.get("vocab", {})
        if vocab and meta.vocab_size == 0:
            meta.vocab_size = len(vocab)

        return DatasetContainer(
            meta=meta,
            raw_inputs=d.get("raw_inputs", []),
            raw_outputs=d.get("raw_outputs", []),
            train_inputs=d.get("train_inputs", []),
            train_outputs=d.get("train_outputs", []),
            val_inputs=d.get("val_inputs", []),
            val_outputs=d.get("val_outputs", []),
            test_inputs=d.get("test_inputs", []),
            test_outputs=d.get("test_outputs", []),
            vocab=vocab,
            normalization=d.get("normalization"),
            augmentation_config=d.get("augmentation_config"),
            extra=extra,
        )

    def save(self, path: Path) -> None:
        """Сохранить датасет в .oai файл."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: Path) -> "DatasetContainer":
        """Загрузить датасет из .oai файла."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return DatasetContainer.from_dict(data)


# ============================================================
# КОНТРАКТ: КОНФИГ МОДЕЛИ
# ============================================================

@dataclass
class ModelConfig:
    """Полная конфигурация модели. Используется для создания и воссоздания."""
    format_version: str = FORMAT_VERSION
    # Идентификация
    model_name: str = "Модель"
    model_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    architecture: ArchitectureType = ArchitectureType.MLP
    data_type: DataType = DataType.NUMERIC
    task_type: TaskType = TaskType.REGRESSION
    # Размерности
    input_dim: Optional[Union[int, Tuple[int, ...]]] = None
    output_dim: Optional[Union[int, Tuple[int, ...]]] = None
    vocab_size: int = 0
    num_classes: int = 0
    image_shape: Optional[Tuple[int, int, int]] = None
    # Архитектурные параметры (уникальные для каждого типа)
    arch_params: Dict[str, Any] = field(default_factory=dict)
    # Параметры обучения
    optimizer: str = "adam"
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    gradient_clipping: float = 1.0
    epochs: int = 50
    batch_size: int = 64
    loss_function: str = "mse"
    teacher_forcing_ratio: float = 0.5
    scheduler: str = "none"
    early_stopping_patience: int = 0  # 0 = выключено
    # Токенизация (для текста)
    pad_idx: int = 0
    bos_idx: int = 2
    eos_idx: int = 3
    unk_idx: int = 1
    max_len: int = 100
    # Метаданные
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    description: str = ""
    loaded_from_file: bool = False
    # Расширение
    extra: Dict[str, Any] = field(default_factory=dict)

    def get_arch_param(self, key: str, default: Any = None) -> Any:
        """Безопасное получение архитектурного параметра."""
        return self.arch_params.get(key, default)

    def set_arch_param(self, key: str, value: Any) -> None:
        """Установка архитектурного параметра."""
        self.arch_params[key] = value

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "format_version": self.format_version,
            "model_name": self.model_name,
            "model_id": self.model_id,
            "type": self.architecture.value,  # "type" для обратной совместимости
            "architecture": self.architecture.value,
            "data_type": self.data_type.value,
            "task_type": self.task_type.value,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "vocab_size": self.vocab_size,
            "num_classes": self.num_classes,
            "image_shape": list(self.image_shape) if self.image_shape else None,
            "arch_params": self.arch_params,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gradient_clipping": self.gradient_clipping,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "loss_function": self.loss_function,
            "teacher_forcing_ratio": self.teacher_forcing_ratio,
            "scheduler": self.scheduler,
            "early_stopping_patience": self.early_stopping_patience,
            "pad_idx": self.pad_idx,
            "bos_idx": self.bos_idx,
            "eos_idx": self.eos_idx,
            "unk_idx": self.unk_idx,
            "max_len": self.max_len,
            "created_at": self.created_at,
            "description": self.description,
            "loaded_from_file": self.loaded_from_file,
        }
        d.update(self.extra)
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ModelConfig":
        """
        Десериализация с миграцией.
        Старый формат: {"type": "transformer", "data_type": "text", "embedding_dim": 128, ...}
        """
        known_keys = {
            "format_version", "model_name", "model_id", "type", "architecture",
            "data_type", "task_type", "input_dim", "output_dim", "vocab_size",
            "num_classes", "image_shape", "arch_params", "optimizer",
            "learning_rate", "weight_decay", "gradient_clipping", "epochs",
            "batch_size", "loss_function", "teacher_forcing_ratio", "scheduler",
            "early_stopping_patience", "pad_idx", "bos_idx", "eos_idx",
            "unk_idx", "max_len", "created_at", "description", "loaded_from_file",
            # Старые архитектурные параметры (из старого конфига)
            "embedding_dim", "num_heads", "num_layers", "num_encoder_layers",
            "num_decoder_layers", "dim_feedforward", "activation", "dropout",
            "hidden_layers", "hidden_size", "rnn_type", "batch_norm",
        }
        extra = {k: v for k, v in d.items() if k not in known_keys}

        # Определяем архитектуру
        arch_str = d.get("architecture", d.get("type", "mlp"))
        architecture = ArchitectureType.from_str(str(arch_str))

        # Миграция старых архитектурных параметров в arch_params
        arch_params = d.get("arch_params", {})
        legacy_arch_keys = [
            "embedding_dim", "num_heads", "num_layers", "num_encoder_layers",
            "num_decoder_layers", "dim_feedforward", "activation", "dropout",
            "hidden_layers", "hidden_size", "rnn_type", "batch_norm",
        ]
        for key in legacy_arch_keys:
            if key in d and key not in arch_params:
                arch_params[key] = d[key]

        input_dim = d.get("input_dim")
        if isinstance(input_dim, list):
            input_dim = tuple(input_dim)
        output_dim = d.get("output_dim")
        if isinstance(output_dim, list):
            output_dim = tuple(output_dim)
        image_shape = d.get("image_shape")
        if image_shape is not None:
            image_shape = tuple(image_shape)

        return ModelConfig(
            format_version=d.get("format_version", FORMAT_VERSION),
            model_name=d.get("model_name", "Модель"),
            model_id=d.get("model_id", str(uuid.uuid4())[:8]),
            architecture=architecture,
            data_type=DataType.from_str(d.get("data_type", "numeric")),
            task_type=TaskType.from_str(d.get("task_type", "regression")),
            input_dim=input_dim,
            output_dim=output_dim,
            vocab_size=d.get("vocab_size", 0),
            num_classes=d.get("num_classes", 0),
            image_shape=image_shape,
            arch_params=arch_params,
            optimizer=d.get("optimizer", "adam"),
            learning_rate=d.get("learning_rate", 0.001),
            weight_decay=d.get("weight_decay", 0.0),
            gradient_clipping=d.get("gradient_clipping", 1.0),
            epochs=d.get("epochs", 50),
            batch_size=d.get("batch_size", 64),
            loss_function=d.get("loss_function", "mse"),
            teacher_forcing_ratio=d.get("teacher_forcing_ratio", 0.5),
            scheduler=d.get("scheduler", "none"),
            early_stopping_patience=d.get("early_stopping_patience", 0),
            pad_idx=d.get("pad_idx", 0),
            bos_idx=d.get("bos_idx", 2),
            eos_idx=d.get("eos_idx", 3),
            unk_idx=d.get("unk_idx", 1),
            max_len=d.get("max_len", 100),
            created_at=d.get("created_at", ""),
            description=d.get("description", ""),
            loaded_from_file=d.get("loaded_from_file", False),
            extra=extra,
        )


# ============================================================
# КОНТРАКТ: ЧЕКПОИНТ МОДЕЛИ (сохранённые веса)
# ============================================================

@dataclass
class ModelCheckpoint:
    """Формат сохранённой модели (.pth / .oai)."""
    format_version: str = FORMAT_VERSION
    model_state_dict: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)  # ModelConfig.to_dict()
    history: Dict[str, List[float]] = field(default_factory=dict)
    vocab: Dict[str, int] = field(default_factory=dict)
    normalization: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_config(self) -> ModelConfig:
        """Извлечь ModelConfig из чекпоинта."""
        return ModelConfig.from_dict(self.config)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_version": self.format_version,
            "model_state_dict": self.model_state_dict,
            "config": self.config,
            "history": self.history,
            "vocab": self.vocab,
            "normalization": self.normalization,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ModelCheckpoint":
        return ModelCheckpoint(
            format_version=d.get("format_version", FORMAT_VERSION),
            model_state_dict=d.get("model_state_dict", {}),
            config=d.get("config", {}),
            history=d.get("history", {}),
            vocab=d.get("vocab", {}),
            normalization=d.get("normalization"),
            metadata=d.get("metadata", {}),
        )


# ============================================================
# КОНТРАКТ: ИСТОРИЯ ОБУЧЕНИЯ
# ============================================================

@dataclass
class TrainingHistory:
    """История обучения модели. Заполняется тренером."""
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_metric: List[float] = field(default_factory=list)
    val_metric: List[float] = field(default_factory=list)
    epochs_completed: int = 0
    epochs_total: int = 0
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    total_time_seconds: float = 0.0
    device: str = ""
    # Живые примеры предсказаний (для мониторинга)
    sample_predictions: List[Dict[str, Any]] = field(default_factory=list)
    # События (аварии, предупреждения)
    events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_metric": self.train_metric,
            "val_metric": self.val_metric,
            "epochs_completed": self.epochs_completed,
            "epochs_total": self.epochs_total,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "total_time_seconds": self.total_time_seconds,
            "device": self.device,
            "sample_predictions": self.sample_predictions,
            "events": self.events,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TrainingHistory":
        return TrainingHistory(
            train_loss=d.get("train_loss", []),
            val_loss=d.get("val_loss", []),
            train_metric=d.get("train_metric", []),
            val_metric=d.get("val_metric", []),
            epochs_completed=d.get("epochs_completed", 0),
            epochs_total=d.get("epochs_total", 0),
            best_epoch=d.get("best_epoch", 0),
            best_val_loss=d.get("best_val_loss", float("inf")),
            total_time_seconds=d.get("total_time_seconds", 0.0),
            device=d.get("device", ""),
            sample_predictions=d.get("sample_predictions", []),
            events=d.get("events", []),
        )


# ============================================================
# КОНТРАКТ: СЕАНС ПОЛЬЗОВАТЕЛЯ
# ============================================================

@dataclass
class SessionState:
    """Состояние текущего и исторического сеанса пользователя."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    total_sessions: int = 0
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    current_tab: str = ""
    scenario: ScenarioType = ScenarioType.NEW
    completed_steps: List[str] = field(default_factory=list)
    user_actions_log: List[Dict[str, Any]] = field(default_factory=list)
    mistakes_count: int = 0
    successful_models: int = 0
    failed_trainings: int = 0
    last_model_name: str = ""
    hint_mode: HintMode = HintMode.HYBRID
    user_level: UserLevel = UserLevel.BEGINNER
    # Какие элементы вызывали затруднения (для адаптации подсказок)
    struggle_widgets: Dict[str, int] = field(default_factory=dict)

    def log_action(self, panel: str, action: str, details: str = "") -> None:
        """Записать действие пользователя."""
        self.user_actions_log.append({
            "timestamp": datetime.now().isoformat(),
            "panel": panel,
            "action": action,
            "details": details,
        })
        # Ограничиваем лог, чтобы не разрастался бесконечно
        if len(self.user_actions_log) > 500:
            self.user_actions_log = self.user_actions_log[-200:]

    def record_mistake(self, widget_id: str) -> None:
        """Записать ошибку пользователя."""
        self.mistakes_count += 1
        self.struggle_widgets[widget_id] = self.struggle_widgets.get(widget_id, 0) + 1

    def record_success(self, model_name: str) -> None:
        """Записать успешное обучение."""
        self.successful_models += 1
        self.last_model_name = model_name

    def get_user_level(self) -> UserLevel:
        """Определить уровень пользователя."""
        if self.total_sessions <= 2 and self.successful_models == 0:
            return UserLevel.BEGINNER
        elif self.successful_models >= 5 or self.total_sessions >= 10:
            return UserLevel.ADVANCED
        return UserLevel.INTERMEDIATE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "total_sessions": self.total_sessions,
            "started_at": self.started_at,
            "current_tab": self.current_tab,
            "scenario": self.scenario.value,
            "completed_steps": self.completed_steps,
            "user_actions_log": self.user_actions_log[-100:],  # последние 100
            "mistakes_count": self.mistakes_count,
            "successful_models": self.successful_models,
            "failed_trainings": self.failed_trainings,
            "last_model_name": self.last_model_name,
            "hint_mode": self.hint_mode.value,
            "user_level": self.get_user_level().value,
            "struggle_widgets": self.struggle_widgets,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SessionState":
        return SessionState(
            session_id=d.get("session_id", str(uuid.uuid4())[:12]),
            total_sessions=d.get("total_sessions", 0),
            started_at=d.get("started_at", ""),
            current_tab=d.get("current_tab", ""),
            scenario=ScenarioType.from_str(d.get("scenario", "new")),
            completed_steps=d.get("completed_steps", []),
            user_actions_log=d.get("user_actions_log", []),
            mistakes_count=d.get("mistakes_count", 0),
            successful_models=d.get("successful_models", 0),
            failed_trainings=d.get("failed_trainings", 0),
            last_model_name=d.get("last_model_name", ""),
            hint_mode=HintMode.from_str(d.get("hint_mode", "hybrid")),
            user_level=UserLevel(d.get("user_level", "beginner")),
            struggle_widgets=d.get("struggle_widgets", {}),
        )


# ============================================================
# КОНТРАКТ: КОНТЕКСТ ПОДСКАЗКИ
# ============================================================

@dataclass
class HintContext:
    """Контекст для запроса подсказки."""
    widget_id: str = ""
    panel_name: str = ""
    model_config: Optional[ModelConfig] = None
    dataset_meta: Optional[DatasetMeta] = None
    session: Optional[SessionState] = None
    current_value: Any = None
    # Дополнительный контекст
    extra: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# КОНТРАКТ: ШАГ ПАЙПЛАЙНА (ОРКЕСТРАТОР)
# ============================================================

@dataclass
class PipelineStep:
    """Один шаг в пайплайне оркестратора."""
    step_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    model_config: ModelConfig = field(default_factory=ModelConfig)
    # model_state_dict хранится отдельно в файле пайплайна
    input_transform: Optional[str] = None   # имя функции предобработки
    output_transform: Optional[str] = None  # имя функции постобработки
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "model_config": self.model_config.to_dict(),
            "input_transform": self.input_transform,
            "output_transform": self.output_transform,
            "description": self.description,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PipelineStep":
        return PipelineStep(
            step_id=d.get("step_id", ""),
            model_config=ModelConfig.from_dict(d.get("model_config", {})),
            input_transform=d.get("input_transform"),
            output_transform=d.get("output_transform"),
            description=d.get("description", ""),
        )


@dataclass
class PipelineConfig:
    """Конфигурация пайплайна (оркестратора)."""
    format_version: str = FORMAT_VERSION
    pipeline_name: str = ""
    steps: List[PipelineStep] = field(default_factory=list)
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_version": self.format_version,
            "pipeline_name": self.pipeline_name,
            "steps": [s.to_dict() for s in self.steps],
            "description": self.description,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PipelineConfig":
        return PipelineConfig(
            format_version=d.get("format_version", FORMAT_VERSION),
            pipeline_name=d.get("pipeline_name", ""),
            steps=[PipelineStep.from_dict(s) for s in d.get("steps", [])],
            description=d.get("description", ""),
            created_at=d.get("created_at", ""),
        )


# ============================================================
# КОНТРАКТ: ПРЕСЕТ
# ============================================================

@dataclass
class PresetInfo:
    """Пресет датасета или конфигурации."""
    preset_id: str = ""
    name: str = ""
    description: str = ""
    category: str = ""  # "math", "images", "audio", "graphs", "text"
    task: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    recommended_architecture: str = ""
    recommended_config: Dict[str, Any] = field(default_factory=dict)
    difficulty: str = "easy"  # "easy" | "medium" | "hard" | "extreme"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "preset_id": self.preset_id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "task": self.task,
            "params": self.params,
            "recommended_architecture": self.recommended_architecture,
            "recommended_config": self.recommended_config,
            "difficulty": self.difficulty,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PresetInfo":
        return PresetInfo(
            preset_id=d.get("preset_id", ""),
            name=d.get("name", ""),
            description=d.get("description", ""),
            category=d.get("category", ""),
            task=d.get("task", ""),
            params=d.get("params", {}),
            recommended_architecture=d.get("recommended_architecture", ""),
            recommended_config=d.get("recommended_config", {}),
            difficulty=d.get("difficulty", "easy"),
        )


# ============================================================
# КОНТРАКТ: ПРОЕКТ (для сохранения/загрузки)
# ============================================================

@dataclass
class ProjectState:
    """Полное состояние проекта для сохранения."""
    format_version: str = FORMAT_VERSION
    project_name: str = ""
    scenario: ScenarioType = ScenarioType.NEW
    model_name: str = ""
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    dataset: Optional[DatasetContainer] = None
    config: Optional[ModelConfig] = None
    history: Optional[TrainingHistory] = None
    session: Optional[SessionState] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_version": self.format_version,
            "project_name": self.project_name,
            "scenario": self.scenario.value,
            "model_name": self.model_name,
            "description": self.description,
            "created_at": self.created_at,
            "dataset": self.dataset.to_dict() if self.dataset else None,
            "config": self.config.to_dict() if self.config else None,
            "history": self.history.to_dict() if self.history else None,
            "session": self.session.to_dict() if self.session else None,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ProjectState":
        dataset = None
        if d.get("dataset"):
            dataset = DatasetContainer.from_dict(d["dataset"])
        config = None
        if d.get("config"):
            config = ModelConfig.from_dict(d["config"])
        history = None
        if d.get("history"):
            history = TrainingHistory.from_dict(d["history"])
        session = None
        if d.get("session"):
            session = SessionState.from_dict(d["session"])

        return ProjectState(
            format_version=d.get("format_version", FORMAT_VERSION),
            project_name=d.get("project_name", ""),
            scenario=ScenarioType.from_str(d.get("scenario", "new")),
            model_name=d.get("model_name", ""),
            description=d.get("description", ""),
            created_at=d.get("created_at", ""),
            dataset=dataset,
            config=config,
            history=history,
            session=session,
        )


# ============================================================
# УТИЛИТЫ ВАЛИДАЦИИ
# ============================================================

def validate_dataset_container(container: DatasetContainer) -> List[str]:
    """
    Проверяет датасет на корректность.
    Возвращает список предупреждений (пустой = всё ок).
    НЕ бросает исключений.
    """
    warnings: List[str] = []

    if container.meta.num_samples == 0 and not container.raw_inputs:
        warnings.append("Датасет пуст: нет ни одного примера.")

    if container.raw_inputs and container.raw_outputs:
        if len(container.raw_inputs) != len(container.raw_outputs):
            warnings.append(
                f"Рассинхрон: {len(container.raw_inputs)} входов, "
                f"но {len(container.raw_outputs)} выходов."
            )

    if container.is_split:
        total_split = (
            len(container.train_inputs)
            + len(container.val_inputs)
            + len(container.test_inputs)
        )
        if total_split == 0:
            warnings.append("Разбиение применено, но все выборки пусты.")
        if len(container.train_inputs) == 0:
            warnings.append("Train-выборка пуста. Обучение невозможно.")

    if container.meta.data_type == DataType.TEXT:
        if not container.vocab and container.is_split:
            warnings.append("Текстовые данные, но словарь пуст.")

    if container.meta.data_type == DataType.IMAGE:
        if container.meta.image_shape is None:
            warnings.append("Изображения без указания формы (H, W, C).")

    if container.meta.data_type == DataType.AUDIO:
        if container.meta.audio_sample_rate is None:
            warnings.append("Аудио без указания частоты дискретизации.")

    return warnings


def validate_model_config(config: ModelConfig) -> List[str]:
    """
    Проверяет конфиг модели на корректность.
    Возвращает список ошибок (пустой = всё ок).
    """
    errors: List[str] = []

    if config.learning_rate <= 0:
        errors.append("Скорость обучения должна быть > 0.")

    if config.epochs < 1:
        errors.append("Количество эпох должно быть >= 1.")

    if config.batch_size < 1:
        errors.append("Размер батча должен быть >= 1.")

    if config.architecture == ArchitectureType.TRANSFORMER_SEQ2SEQ:
        embed = config.get_arch_param("embedding_dim", 128)
        heads = config.get_arch_param("num_heads", 8)
        if heads > 0 and embed % heads != 0:
            errors.append(
                f"Эмбеддинг ({embed}) должен делиться на головы ({heads})."
            )

    if config.data_type == DataType.TEXT and config.vocab_size == 0:
        errors.append("Текстовая модель без словаря (vocab_size=0).")

    return errors


def get_compatible_architectures(data_type: DataType) -> List[ArchitectureType]:
    """Возвращает список архитектур, совместимых с данным типом данных."""
    compatibility: Dict[DataType, List[ArchitectureType]] = {
        DataType.TEXT: [
            ArchitectureType.RNN, ArchitectureType.LSTM, ArchitectureType.GRU,
            ArchitectureType.TRANSFORMER_SEQ2SEQ, ArchitectureType.TRANSFORMER_ENCODER,
            ArchitectureType.TRANSFORMER_DECODER, ArchitectureType.MLP,
        ],
        DataType.NUMERIC: [
            ArchitectureType.MLP, ArchitectureType.AUTOENCODER, ArchitectureType.VAE,
            ArchitectureType.RBFN, ArchitectureType.SOM, ArchitectureType.RNN,
            ArchitectureType.LSTM, ArchitectureType.GRU, ArchitectureType.PINN,
        ],
        DataType.IMAGE: [
            ArchitectureType.CNN, ArchitectureType.AUTOENCODER, ArchitectureType.VAE,
            ArchitectureType.CVAE, ArchitectureType.GAN, ArchitectureType.DCGAN,
            ArchitectureType.DIFFUSION, ArchitectureType.LATENT_DIFFUSION,
            ArchitectureType.VIT, ArchitectureType.CAPSNET, ArchitectureType.RESNET,
            ArchitectureType.CONV_AE,
        ],
        DataType.AUDIO: [
            ArchitectureType.CNN, ArchitectureType.RNN, ArchitectureType.LSTM,
            ArchitectureType.GRU, ArchitectureType.TRANSFORMER_ENCODER,
        ],
        DataType.GRAPH: [
            ArchitectureType.GNN, ArchitectureType.GCN, ArchitectureType.GAT,
            ArchitectureType.GRAPH_TRANSFORMER,
        ],
        DataType.TIME_SERIES: [
            ArchitectureType.RNN, ArchitectureType.LSTM, ArchitectureType.GRU,
            ArchitectureType.TRANSFORMER_ENCODER, ArchitectureType.NEURAL_ODE,
            ArchitectureType.LIQUID_NN, ArchitectureType.MLP,
        ],
        DataType.MULTIMODAL: [
            ArchitectureType.CNN_RNN, ArchitectureType.CLIP_LIKE,
            ArchitectureType.MULTIMODAL, ArchitectureType.CUSTOM,
        ],
    }
    return compatibility.get(data_type, [ArchitectureType.CUSTOM])


# ============================================================
# КОНСТАНТЫ СПЕЦИАЛЬНЫХ ТОКЕНОВ
# ============================================================

SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<BOS>": 2,
    "<EOS>": 3,
}

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"

PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3


# ============================================================
# УТИЛИТЫ СОВМЕСТИМОСТИ С GUI (старый v0.4 dict-контракт)
# ============================================================
#
# Часть GUI-панелей была написана под старый формат, где shared_state
# хранил обычные dict'ы (config как dict, dataset как dict с meta["type"]).
# Ядро v2.0 хранит ModelConfig и DatasetContainer (dataclass'ы).
# Эти функции дают панелям лёгкий dict-вид БЕЗ изменения формата .oai.

def config_to_dict(config: Any) -> Dict[str, Any]:
    """
    Нормализует конфиг модели (ModelConfig или dict) в обычный dict.

    Дополнительно поднимает ключи arch_params на верхний уровень
    (embedding_dim, num_heads, ...), чтобы панели старого контракта
    могли читать config['embedding_dim'] и т.п.
    Не выполняет копирования данных модели/датасета.
    """
    if config is None:
        return {}
    if isinstance(config, dict):
        return config
    if hasattr(config, "to_dict"):
        d = config.to_dict()
        arch = d.get("arch_params")
        if isinstance(arch, dict):
            for k, v in arch.items():
                d.setdefault(k, v)
        return d
    return {}


def dataset_to_legacy_dict(dataset: Any) -> Dict[str, Any]:
    """
    Нормализует датасет (DatasetContainer или dict) в dict старого формата.

    Возвращает лёгкий словарь (списки НЕ копируются — только ссылки) с ключами
    meta (включая legacy-ключ "type"), vocab, train_inputs, ... как в v0.4.
    """
    if dataset is None:
        return {}
    if isinstance(dataset, dict):
        return dataset
    if hasattr(dataset, "to_dict"):
        d = dataset.to_dict()
        meta = d.get("meta")
        if isinstance(meta, dict) and "type" not in meta and "data_type" in meta:
            meta["type"] = meta["data_type"]
        return d
    return {}