"""
core/orchestrator.py
====================
Оркестратор пайплайнов: цепочки из нескольких моделей.

Позволяет соединять модели в последовательность:
  [Аудио] → [Speech-to-Text] → [Текст] → [Переводчик] → [Текст] → [TTS] → [Аудио]

Зависимости (импорт):
  - core.contracts  (ModelConfig, DataType, TaskType, ArchitectureType, FORMAT_VERSION)
  - core.model_factory (ModelFactory — для воссоздания моделей при загрузке)
  - core.logger (get_logger)

Входы:
  - Список PipelineStep (модель + конфиг + трансформы)
  - Данные любого типа (определяются первым шагом)

Выходы:
  - Результат последнего шага пайплайна
  - Файл .oai-pipeline при сохранении

Правила:
  - Все типы данных берутся ИЗ contracts.py
  - format_version обязателен в каждом сохраняемом файле
  - Мусор на входе → предупреждение, не падение
  - Каждый шаг изолирован: ошибка в одном не убивает весь пайплайн
"""

import json
import copy
import traceback
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np

from core.contracts import (
    FORMAT_VERSION,
    ModelConfig,
    DataType,
    TaskType,
    ArchitectureType,
)
from core.logger import get_logger

logger = get_logger("Orchestrator")


# ============================================================
# КОНТРАКТ ШАГА ПАЙПЛАЙНА
# ============================================================

@dataclass
class PipelineStep:
    """
    Один шаг пайплайна: модель + конфиг + опциональные трансформы.

    Поля:
      step_id          — уникальный идентификатор шага (для логирования)
      model            — nn.Module (или None для заглушек)
      config           — ModelConfig из contracts.py
      input_transform  — функция предобработки ВХОДА перед моделью
      output_transform — функция постобработки ВЫХОДА после модели
      description      — человекочитаемое описание шага

    Трансформы принимают и возвращают данные. Сигнатура:
      input_transform(data: Any) -> Any
      output_transform(data: Any) -> Any
    """
    step_id: str = ""
    model: Optional[nn.Module] = None
    config: Optional[ModelConfig] = None
    input_transform: Optional[Callable] = None
    output_transform: Optional[Callable] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация метаинформации шага (без модели и функций)."""
        return {
            "step_id": self.step_id,
            "config": self.config.to_dict() if self.config else None,
            "description": self.description,
            "has_input_transform": self.input_transform is not None,
            "has_output_transform": self.output_transform is not None,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PipelineStep":
        """Десериализация метаинформации. Модель воссоздаётся отдельно."""
        config = None
        if d.get("config"):
            config = ModelConfig.from_dict(d["config"])
        return PipelineStep(
            step_id=d.get("step_id", ""),
            config=config,
            description=d.get("description", ""),
        )


# ============================================================
# РЕЗУЛЬТАТ ВЫПОЛНЕНИЯ ПАЙПЛАЙНА
# ============================================================

@dataclass
class PipelineResult:
    """
    Результат выполнения пайплайна.

    Поля:
      output       — финальный результат
      success      — успешность всего пайплайна
      step_results — промежуточные результаты каждого шага
      errors       — список ошибок (если были)
      warnings     — список предупреждений
    """
    output: Any = None
    success: bool = True
    step_results: List[Any] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ============================================================
# РЕЕСТР ВСТРОЕННЫХ ТРАНСФОРМ
# ============================================================

class TransformRegistry:
    """
    Реестр встроенных трансформ для связки шагов.
    Свою трансформу можно зарегистрировать через @register.
    """

    _transforms: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str):
        """Декоратор для регистрации трансформы."""
        def decorator(func: Callable):
            cls._transforms[name] = func
            return func
        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Callable]:
        return cls._transforms.get(name)

    @classmethod
    def list_available(cls) -> List[str]:
        return list(cls._transforms.keys())


# --- Встроенные трансформы ---

@TransformRegistry.register("text_to_token_ids")
def text_to_token_ids(data: Any) -> Any:
    """
    Строка → список индексов токенов (символьный уровень).
    Ожидает на входе строку или список строк.
    Возвращает тензор [1, seq_len] или [batch, seq_len].
    """
    if isinstance(data, str):
        data = [data]
    if not isinstance(data, list):
        raise ValueError(f"text_to_token_ids ожидает строку или список строк, получил {type(data)}")
    # Простая символьная токенизация (без словаря — по ord)
    # В реальном пайплайне словарь берётся из конфига модели
    sequences = []
    for text in data:
        seq = [ord(c) % 5000 for c in str(text)]  # грубая заглушка
        sequences.append(seq)
    # Выравнивание по максимальной длине
    max_len = max(len(s) for s in sequences) if sequences else 0
    padded = [s + [0] * (max_len - len(s)) for s in sequences]
    return torch.tensor(padded, dtype=torch.long)


@TransformRegistry.register("token_ids_to_text")
def token_ids_to_text(data: Any) -> Any:
    """
    Тензор индексов → строка. Обратная к text_to_token_ids.
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().tolist()
    if isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], list):
            data = data[0]  # берём первый элемент батча
    chars = []
    for idx in data:
        if idx == 0:  # pad
            continue
        try:
            chars.append(chr(idx))
        except (ValueError, OverflowError):
            chars.append("?")
    return "".join(chars)


@TransformRegistry.register("numpy_to_tensor")
def numpy_to_tensor(data: Any) -> Any:
    """numpy массив или список чисел → float32 тензор."""
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).float().unsqueeze(0)
    if isinstance(data, (list, tuple)):
        return torch.tensor([data], dtype=torch.float32)
    if isinstance(data, (int, float)):
        return torch.tensor([[data]], dtype=torch.float32)
    raise ValueError(f"numpy_to_tensor не поддерживает тип {type(data)}")


@TransformRegistry.register("tensor_to_numpy")
def tensor_to_numpy(data: Any) -> Any:
    """Тензор → numpy массив."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data


@TransformRegistry.register("normalize_image")
def normalize_image(data: Any) -> Any:
    """Нормализация изображения в [0, 1] и конвертация в тензор."""
    if isinstance(data, np.ndarray):
        arr = data.astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]  # (1, H, W)
        elif arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
            arr = np.transpose(arr, (2, 0, 1))  # (C, H, W)
        return torch.from_numpy(arr).unsqueeze(0)
    raise ValueError(f"normalize_image ожидает numpy массив, получил {type(data)}")


@TransformRegistry.register("audio_to_spectrogram")
def audio_to_spectrogram(data: Any) -> Any:
    """
    Аудио массив → спектрограмма (заглушка).
    В будущем: torch.stft или torchaudio.
    """
    if isinstance(data, np.ndarray):
        # Заглушка: просто нормализуем и подаём как 1D сигнал
        arr = data.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 32768.0  # нормализация int16
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    raise ValueError(f"audio_to_spectrogram ожидает numpy массив, получил {type(data)}")


@TransformRegistry.register("identity")
def identity(data: Any) -> Any:
    """Трансформа-заглушка: передаёт данные без изменений."""
    return data


# ============================================================
# ОРКЕСТРАТОР
# ============================================================

class Orchestrator:
    """
    Оркестратор пайплайнов из нескольких моделей.

    Использование:
        steps = [
            PipelineStep(step_id="stt", model=model1, config=cfg1, ...),
            PipelineStep(step_id="translate", model=model2, config=cfg2, ...),
            PipelineStep(step_id="tts", model=model3, config=cfg3, ...),
        ]
        orch = Orchestrator(steps)
        result = orch.run("Привет, мир!")

    Правила:
      - Каждый шаг изолирован: ошибка логируется, пайплайн останавливается
      - Мусор на входе → предупреждение в PipelineResult, не исключение
      - Модель может быть None (заглушка) → шаг пропускается с предупреждением
    """

    def __init__(self, steps: Optional[List[PipelineStep]] = None):
        self.steps: List[PipelineStep] = steps or []
        self.pipeline_name: str = "Безымянный пайплайн"
        self.description: str = ""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            f"Orchestrator создан: {len(self.steps)} шагов, устройство={self.device}"
        )

    # --------------------------------------------------------
    # ДОБАВЛЕНИЕ / УДАЛЕНИЕ ШАГОВ
    # --------------------------------------------------------

    def add_step(
        self,
        step_id: str,
        model: Optional[nn.Module],
        config: ModelConfig,
        input_transform: Optional[Callable] = None,
        output_transform: Optional[Callable] = None,
        description: str = "",
    ) -> None:
        """Добавляет шаг в конец пайплайна."""
        step = PipelineStep(
            step_id=step_id,
            model=model,
            config=config,
            input_transform=input_transform,
            output_transform=output_transform,
            description=description,
        )
        self.steps.append(step)
        logger.info(f"Шаг '{step_id}' добавлен ({config.architecture.value})")

    def remove_step(self, step_id: str) -> bool:
        """Удаляет шаг по step_id. Возвращает True если удалён."""
        for i, s in enumerate(self.steps):
            if s.step_id == step_id:
                self.steps.pop(i)
                logger.info(f"Шаг '{step_id}' удалён")
                return True
        return False

    def get_step(self, step_id: str) -> Optional[PipelineStep]:
        """Возвращает шаг по step_id или None."""
        for s in self.steps:
            if s.step_id == step_id:
                return s
        return None

    # --------------------------------------------------------
    # ВАЛИДАЦИЯ ПАЙПЛАЙНА
    # --------------------------------------------------------

    def validate(self) -> List[str]:
        """
        Проверяет пайплайн на логические ошибки.
        Возвращает список предупреждений (пустой = всё ок).
        Не блокирует запуск — только предупреждает.
        """
        warnings = []

        if not self.steps:
            warnings.append("Пайплайн пуст: нет ни одного шага.")
            return warnings

        for i, step in enumerate(self.steps):
            if step.model is None:
                warnings.append(
                    f"Шаг {i} ('{step.step_id}'): модель = None. "
                    f"Шаг будет пропущен при выполнении."
                )
            if step.config is None:
                warnings.append(
                    f"Шаг {i} ('{step.step_id}'): конфиг отсутствует. "
                    f"Невозможно определить тип данных."
                )

        # Проверка совместимости типов данных между соседними шагами
        for i in range(len(self.steps) - 1):
            curr = self.steps[i]
            nxt = self.steps[i + 1]
            if curr.config and nxt.config:
                curr_out_type = self._infer_output_type(curr.config)
                nxt_in_type = nxt.config.data_type
                if curr_out_type != nxt_in_type and nxt.input_transform is None:
                    warnings.append(
                        f"Шаг {i}→{i+1}: выход '{curr_out_type.value}' "
                        f"не совпадает со входом '{nxt_in_type.value}'. "
                        f"Добавьте input_transform для конвертации."
                    )

        return warnings

    @staticmethod
    def _infer_output_type(config: ModelConfig) -> DataType:
        """Определяет тип данных на выходе модели по её задаче."""
        task = config.task_type
        if task in (TaskType.SEQ2SEQ, TaskType.TRANSLATION, TaskType.SPEECH_TO_TEXT):
            return DataType.TEXT
        if task in (TaskType.TEXT_TO_SPEECH, TaskType.AUDIO_GENERATION):
            return DataType.AUDIO
        if task in (TaskType.IMAGE_GENERATION,):
            return DataType.IMAGE
        if task in (TaskType.CLASSIFICATION, TaskType.REGRESSION):
            return config.data_type
        return config.data_type

    # --------------------------------------------------------
    # ВЫПОЛНЕНИЕ ПАЙПЛАЙНА
    # --------------------------------------------------------

    def run(self, input_data: Any) -> PipelineResult:
        """
        Выполняет пайплайн от начала до конца.

        Аргументы:
          input_data — данные любого типа (определяются первым шагом)

        Возвращает:
          PipelineResult с output, step_results, errors, warnings

        НЕ бросает исключений — все ошибки фиксируются в result.errors.
        """
        result = PipelineResult()
        data = input_data

        if not self.steps:
            result.success = False
            result.errors.append("Пайплайн пуст. Добавьте хотя бы один шаг.")
            return result

        # Предварительная валидация
        pre_warnings = self.validate()
        result.warnings.extend(pre_warnings)

        logger.info(f"🚀 Запуск пайплайна '{self.pipeline_name}' ({len(self.steps)} шагов)")

        for i, step in enumerate(self.steps):
            step_label = f"[{i}:{step.step_id}]"

            # Пропуск шагов без модели (заглушки)
            if step.model is None:
                msg = f"{step_label} Модель отсутствует (заглушка). Шаг пропущен."
                logger.warning(msg)
                result.warnings.append(msg)
                result.step_results.append(None)
                continue

            try:
                # 1. Входная трансформа
                if step.input_transform is not None:
                    data = step.input_transform(data)

                # 2. Инференс модели
                data = self._infer(step.model, data, step.config)

                # 3. Выходная трансформа
                if step.output_transform is not None:
                    data = step.output_transform(data)

                result.step_results.append(data)
                logger.info(f"{step_label} ✅ Успешно")

            except Exception as e:
                error_msg = f"{step_label} ❌ Ошибка: {e}"
                logger.error(error_msg)
                logger.debug(traceback.format_exc())
                result.errors.append(error_msg)
                result.success = False
                result.output = data  # возвращаем то, что было до ошибки
                return result

        result.output = data
        result.success = len(result.errors) == 0
        logger.info(f"🏁 Пайплайн завершён. Успех: {result.success}")
        return result

    def _infer(
        self,
        model: nn.Module,
        data: Any,
        config: Optional[ModelConfig],
    ) -> Any:
        """
        Универсальный инференс одной модели.
        Определяет стратегию по типу задачи из конфига.

        Стратегии:
          - seq2seq / translation → model.generate()
          - classification → model(x) → argmax
          - regression → model(x)
          - image_generation → model(x) (заглушка)
          - прочее → model(x)
        """
        model.eval()
        device = self.device

        # Приводим данные к тензору, если ещё не тензор
        tensor_data = self._ensure_tensor(data, config)
        tensor_data = tensor_data.to(device)
        model.to(device)

        task_type = config.task_type if config else TaskType.REGRESSION

        with torch.no_grad():
            # --- Генеративные задачи (есть метод generate) ---
            if task_type in (TaskType.SEQ2SEQ, TaskType.TRANSLATION):
                if hasattr(model, "generate"):
                    max_len = 100
                    if config and config.arch_params:
                        max_len = config.arch_params.get("max_len", 100)
                    output = model.generate(tensor_data, max_len=max_len)
                    return output
                else:
                    # Fallback: прямой forward
                    logger.warning(
                        "Модель не имеет метода generate(). "
                        "Используется прямой forward pass."
                    )
                    return model(tensor_data)

            # --- Классификация ---
            elif task_type in (
                TaskType.CLASSIFICATION,
                TaskType.IMAGE_CLASSIFICATION,
                TaskType.AUDIO_CLASSIFICATION,
            ):
                output = model(tensor_data)
                # Если выход — логиты, берём argmax
                if output.dim() > 1 and output.size(-1) > 1:
                    return output.argmax(dim=-1)
                return output

            # --- Регрессия и прочее ---
            else:
                output = model(tensor_data)
                return output

    @staticmethod
    def _ensure_tensor(data: Any, config: Optional[ModelConfig]) -> torch.Tensor:
        """
        Приводит данные к тензору. Безопасно: мусор → предупреждение + нулевой тензор.
        """
        if isinstance(data, torch.Tensor):
            return data

        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).float().unsqueeze(0)

        if isinstance(data, (list, tuple)):
            try:
                return torch.tensor([data], dtype=torch.float32)
            except (ValueError, TypeError):
                pass

        if isinstance(data, (int, float)):
            return torch.tensor([[data]], dtype=torch.float32)

        if isinstance(data, str):
            # Строка → символьные индексы (грубая заглушка)
            seq = [ord(c) % 5000 for c in data]
            return torch.tensor([seq], dtype=torch.long)

        # Мусор: предупреждение и нулевой тензор
        logger.warning(
            f"Невозможно преобразовать данные типа {type(data)} в тензор. "
            f"Подставлен нулевой тензор [1, 1]."
        )
        return torch.zeros(1, 1, dtype=torch.float32)

    # --------------------------------------------------------
    # СОХРАНЕНИЕ / ЗАГРУЗКА ПАЙПЛАЙНА
    # --------------------------------------------------------

    def save_pipeline(self, path: Union[str, Path]) -> None:
        """
        Сохраняет пайплайн в файл .oai-pipeline (JSON + веса моделей).

        Структура файла:
        {
            "format_version": "2.0",
            "pipeline_name": "...",
            "description": "...",
            "steps": [ {meta шага}, ... ],
            "model_weights": { "step_id": state_dict, ... }
        }
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        steps_meta = [step.to_dict() for step in self.steps]

        # Собираем веса моделей
        model_weights = {}
        for step in self.steps:
            if step.model is not None:
                # Сериализуем state_dict в CPU-тензоры
                state = {
                    k: v.cpu() for k, v in step.model.state_dict().items()
                }
                model_weights[step.step_id] = state

        payload = {
            "format_version": FORMAT_VERSION,
            "pipeline_name": self.pipeline_name,
            "description": self.description,
            "steps": steps_meta,
            "model_weights": model_weights,
        }

        torch.save(payload, path)
        logger.info(f"Пайплайн сохранён: {path} ({len(self.steps)} шагов)")

    @staticmethod
    def load_pipeline(path: Union[str, Path]) -> "Orchestrator":
        """
        Загружает пайплайн из файла .oai-pipeline.
        Воссоздаёт модели через ModelFactory.

        Аргументы:
          path — путь к файлу

        Возвращает:
          Orchestrator с загруженными моделями

        Бросает:
          FileNotFoundError — файл не найден
          ValueError — несовместимая версия формата
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Файл пайплайна не найден: {path}")

        payload = torch.load(path, map_location="cpu")

        # Проверка версии формата
        fmt_ver = payload.get("format_version", "1.0")
        if fmt_ver != FORMAT_VERSION:
            logger.warning(
                f"Версия формата пайплайна {fmt_ver} != {FORMAT_VERSION}. "
                f"Попытка миграции."
            )
            # В будущем: миграция между версиями
            # Пока просто логируем и продолжаем

        orch = Orchestrator()
        orch.pipeline_name = payload.get("pipeline_name", "Загруженный пайплайн")
        orch.description = payload.get("description", "")

        steps_meta = payload.get("steps", [])
        model_weights = payload.get("model_weights", {})

        for meta in steps_meta:
            step = PipelineStep.from_dict(meta)

            # Воссоздаём модель из конфига и весов
            if step.config and step.step_id in model_weights:
                try:
                    from core.model_factory import ModelFactory
                    model = ModelFactory.create_model(step.config)
                    model.load_state_dict(model_weights[step.step_id])
                    step.model = model
                    logger.info(f"Шаг '{step.step_id}': модель восстановлена")
                except Exception as e:
                    logger.error(
                        f"Шаг '{step.step_id}': не удалось восстановить модель: {e}"
                    )
                    step.model = None

            orch.steps.append(step)

        logger.info(
            f"Пайплайн загружен: {orch.pipeline_name} ({len(orch.steps)} шагов)"
        )
        return orch

    # --------------------------------------------------------
    # ИНФОРМАЦИЯ О ПАЙПЛАЙНЕ
    # --------------------------------------------------------

    def summary(self) -> str:
        """Возвращает человекочитаемое описание пайплайна."""
        lines = [
            f"🔗 Пайплайн: {self.pipeline_name}",
            f"   Описание: {self.description or '—'}",
            f"   Шагов: {len(self.steps)}",
            f"   Устройство: {self.device}",
            "",
        ]
        for i, step in enumerate(self.steps):
            status = "✅" if step.model is not None else "⬜ (заглушка)"
            arch = step.config.architecture.value if step.config else "?"
            task = step.config.task_type.value if step.config else "?"
            lines.append(
                f"   {i}. [{step.step_id}] {status} "
                f"arch={arch}, task={task}"
            )
            if step.description:
                lines.append(f"      {step.description}")
        return "\n".join(lines)

    def get_param_count(self) -> int:
        """Суммарное число параметров всех моделей пайплайна."""
        total = 0
        for step in self.steps:
            if step.model is not None:
                total += sum(p.numel() for p in step.model.parameters())
        return total

    # --------------------------------------------------------
    # ЗАГОТОВКИ ДЛЯ БУДУЩИХ ПАЙПЛАЙНОВ
    # --------------------------------------------------------

    @staticmethod
    def create_speech_translate_tts_stub() -> "Orchestrator":
        """
        Заглушка: Речь → Текст → Перевод → Голос.
        Все модели = None. Для будущей реализации.
        """
        orch = Orchestrator()
        orch.pipeline_name = "Речь → Перевод → Голос"
        orch.description = (
            "Пайплайн перевода речи: аудио → текст → перевод → синтез речи. "
            "Требует предзагруженных моделей (заглушки)."
        )

        orch.add_step(
            step_id="speech_to_text",
            model=None,
            config=ModelConfig(
                model_name="STT-заглушка",
                architecture=ArchitectureType.TRANSFORMER_ENCODER,
                data_type=DataType.AUDIO,
                task_type=TaskType.SPEECH_TO_TEXT,
                description="Распознавание речи (заглушка)",
            ),
            input_transform=TransformRegistry.get("audio_to_spectrogram"),
            description="Аудио → Текст",
        )

        orch.add_step(
            step_id="translator",
            model=None,
            config=ModelConfig(
                model_name="Переводчик-заглушка",
                architecture=ArchitectureType.TRANSFORMER_SEQ2SEQ,
                data_type=DataType.TEXT,
                task_type=TaskType.TRANSLATION,
                description="Перевод текста (заглушка)",
            ),
            description="Текст → Переведённый текст",
        )

        orch.add_step(
            step_id="text_to_speech",
            model=None,
            config=ModelConfig(
                model_name="TTS-заглушка",
                architecture=ArchitectureType.TRANSFORMER_DECODER,
                data_type=DataType.TEXT,
                task_type=TaskType.TEXT_TO_SPEECH,
                description="Синтез речи (заглушка)",
            ),
            output_transform=TransformRegistry.get("tensor_to_numpy"),
            description="Текст → Аудио",
        )

        logger.info("Создан пайплайн-заглушка: Речь → Перевод → Голос")
        return orch

    @staticmethod
    def create_image_caption_stub() -> "Orchestrator":
        """
        Заглушка: Изображение → Описание (CNN + RNN / Transformer).
        """
        orch = Orchestrator()
        orch.pipeline_name = "Изображение → Описание"
        orch.description = "Пайплайн генерации описания изображения."

        orch.add_step(
            step_id="image_encoder",
            model=None,
            config=ModelConfig(
                model_name="CNN-энкодер-заглушка",
                architecture=ArchitectureType.CNN,
                data_type=DataType.IMAGE,
                task_type=TaskType.IMAGE_CLASSIFICATION,
                description="Извлечение признаков изображения",
            ),
            input_transform=TransformRegistry.get("normalize_image"),
            description="Изображение → Вектор признаков",
        )

        orch.add_step(
            step_id="text_decoder",
            model=None,
            config=ModelConfig(
                model_name="Текстовый декодер-заглушка",
                architecture=ArchitectureType.TRANSFORMER_DECODER,
                data_type=DataType.TEXT,
                task_type=TaskType.SEQ2SEQ,
                description="Генерация описания по вектору",
            ),
            output_transform=TransformRegistry.get("token_ids_to_text"),
            description="Вектор признаков → Текст описания",
        )

        logger.info("Создан пайплайн-заглушка: Изображение → Описание")
        return orch

    @staticmethod
    def create_denoise_stub() -> "Orchestrator":
        """
        Заглушка: Зашумлённое изображение → Чистое (ConvAE).
        """
        orch = Orchestrator()
        orch.pipeline_name = "Очистка изображения от шума"
        orch.description = "Автоэнкодер убирает шум с изображения."

        orch.add_step(
            step_id="denoise_ae",
            model=None,
            config=ModelConfig(
                model_name="ConvAE-заглушка",
                architecture=ArchitectureType.CONV_AE,
                data_type=DataType.IMAGE,
                task_type=TaskType.ANOMALY_DETECTION,
                description="Шумоподавление",
            ),
            input_transform=TransformRegistry.get("normalize_image"),
            output_transform=TransformRegistry.get("tensor_to_numpy"),
            description="Зашумлённое → Чистое изображение",
        )

        logger.info("Создан пайплайн-заглушка: Очистка от шума")
        return orch


# ============================================================
# ПРЕСЕТЫ ПАЙПЛАЙНОВ (для preset_registry)
# ============================================================

PIPELINE_PRESETS = [
    {
        "id": "speech_translate_tts",
        "name": "🎤 Речь → Перевод → Голос",
        "description": "Полный пайплайн перевода речи. Требует предзагруженных моделей.",
        "factory": Orchestrator.create_speech_translate_tts_stub,
        "available": False,  # заглушка
    },
    {
        "id": "image_caption",
        "name": "🖼 Изображение → Описание",
        "description": "CNN извлекает признаки, декодер генерирует текст.",
        "factory": Orchestrator.create_image_caption_stub,
        "available": False,
    },
    {
        "id": "denoise",
        "name": "🔇 Очистка от шума",
        "description": "Автоэнкодер убирает шум с изображения.",
        "factory": Orchestrator.create_denoise_stub,
        "available": False,
    },
]