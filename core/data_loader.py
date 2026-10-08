"""
core/data_loader.py
===================
Загрузка, валидация, очистка и разбиение датасетов.

Модуль работает с объектами DatasetContainer из core/contracts.py.

Поддерживаемые форматы файлов:
  .oai   — внутренний JSON-формат проекта (основной)
  .json  — произвольный JSON (список пар или словарь)
  .csv   — таблица (последние 2 колонки = вход/выход)
  .pt    — PyTorch-тензор / state_dict
  .wav   — аудиофайл (одиночный)
  папка  — набор изображений (.png/.jpg/.bmp) или аудио (.wav/.mp3)

Поддерживаемые типы данных (DataType):
  text, numeric, image, audio, graph, time_series, multimodal

Принципы:
  1. Мусор на входе — не ошибка, а урок. Программа НЕ падает.
  2. Все результаты возвращаются как DatasetContainer.
  3. format_version обязателен в каждом сохраняемом файле.
  4. Обратная совместимость: файлы без format_version мигрируются.
"""

import json
import csv
import warnings
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from core.logger import get_logger
from core.text_tokenizer import TextTokenizer

# ============================================================
# ИМПОРТ КОНТРАКТОВ — ЕДИНСТВЕННЫЙ ИСТОЧНИК ТИПОВ
# ============================================================
from core.contracts import (
    DataType,
    TaskType,
    DatasetMeta,
    DatasetContainer,
    FORMAT_VERSION,
)

logger = get_logger(__name__)


# ============================================================
#  КОНСТАНТЫ
# ============================================================

# Максимальная длина строки при анализе текстовых данных.
# Всё, что длиннее, считается аномалией (например, base64-картинка).
MAX_TEXT_LENGTH_THRESHOLD = 1500

# Максимальный размер файла для загрузки без предупреждения (МБ)
MAX_FILE_SIZE_MB_WARNING = 500

# Допустимые расширения изображений
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

# Допустимые расширения аудио
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}


# ============================================================
#  DatasetWizard — анализ, очистка, отчёт о качестве
# ============================================================

class DatasetWizard:
    """
    Мастер импорта и валидации произвольных пользовательских датасетов.

    Определяет тип данных, отсекает аномалии, формирует отчёт.
    НЕ падает на мусоре — возвращает предупреждения.
    """

    @staticmethod
    def analyze_and_clean(
        raw_inputs: List[Any],
        raw_outputs: List[Any],
        max_len_threshold: int = MAX_TEXT_LENGTH_THRESHOLD,
    ) -> Tuple[List, List, Dict[str, Any]]:
        """
        Проводит Type Inference (определение типа) и отсечение аномалий.

        Возвращает:
            (cleaned_inputs, cleaned_outputs, report_dict)

        report_dict содержит:
            - type: str               — определённый тип данных
            - num_samples: int        — количество после очистки
            - input_dim: int          — размерность входа
            - output_dim: int         — размерность выхода
            - dropped_empty: int      — удалено пустых строк
            - dropped_length: int     — удалено слишком длинных
            - dropped_nan_numeric: int— удалено NaN/Inf (для числовых)
            - warnings: List[str]     — текстовые предупреждения
        """
        cleaned_in: List[Any] = []
        cleaned_out: List[Any] = []
        dropped_empty = 0
        dropped_length = 0
        dropped_nan_numeric = 0
        warnings_list: List[str] = []

        # --- Определяем базовый тип по первому валидному элементу ---
        is_numeric = True

        for x, y in zip(raw_inputs, raw_outputs):
            # 1. Отсев пустых / None / NaN
            if x is None or y is None:
                dropped_empty += 1
                continue
            str_x = str(x).strip()
            str_y = str(y).strip()
            if str_x == "" or str_y == "":
                dropped_empty += 1
                continue
            if str_x.lower() == "nan" or str_y.lower() == "nan":
                dropped_empty += 1
                continue

            # 2. Детектор аномальной длины (защита от base64 и мусора)
            if len(str_x) > max_len_threshold:
                dropped_length += 1
                continue

            # 3. Проверка: числовой ли тип?
            if is_numeric:
                try:
                    if isinstance(x, (list, tuple)):
                        [float(i) for i in x]
                    elif isinstance(x, np.ndarray):
                        if np.isnan(x).any() or np.isinf(x).any():
                            dropped_nan_numeric += 1
                            continue
                    else:
                        float(x)
                except (ValueError, TypeError):
                    is_numeric = False

            cleaned_in.append(x)
            cleaned_out.append(y)

        # --- Формируем отчёт ---
        if dropped_empty > 0:
            warnings_list.append(
                f"Удалено {dropped_empty} пустых строк (None / NaN / '')."
            )
        if dropped_length > 0:
            warnings_list.append(
                f"Удалено {dropped_length} строк длиннее {max_len_threshold} символов. "
                f"Возможно, это закодированные изображения или мусор."
            )
        if dropped_nan_numeric > 0:
            warnings_list.append(
                f"Удалено {dropped_nan_numeric} числовых записей с NaN/Inf."
            )

        # Определяем размерность
        input_dim = 1
        output_dim = 1
        if cleaned_in:
            first = cleaned_in[0]
            if isinstance(first, (list, tuple)):
                input_dim = len(first)
            elif isinstance(first, np.ndarray):
                input_dim = first.shape[0] if first.ndim > 0 else 1
        if cleaned_out:
            first_out = cleaned_out[0]
            if isinstance(first_out, (list, tuple)):
                output_dim = len(first_out)
            elif isinstance(first_out, np.ndarray):
                output_dim = first_out.shape[0] if first_out.ndim > 0 else 1

        data_type = "numeric" if is_numeric else "text"

        meta_report = {
            "type": data_type,
            "num_samples": len(cleaned_in),
            "input_dim": input_dim,
            "output_dim": output_dim,
            "dropped_empty": dropped_empty,
            "dropped_length": dropped_length,
            "dropped_nan_numeric": dropped_nan_numeric,
            "warnings": warnings_list,
        }

        for w in warnings_list:
            logger.warning(f"DatasetWizard: {w}")

        return cleaned_in, cleaned_out, meta_report

    @staticmethod
    def validate_dataset(dataset: DatasetContainer) -> List[str]:
        """
        Финальная валидация DatasetContainer перед использованием.
        Возвращает список предупреждений (пустой = всё ок).
        НЕ бросает исключений.
        """
        issues: List[str] = []
        meta = dataset.meta

        # Проверка наличия данных
        total = meta.num_samples
        if total == 0:
            issues.append("Датасет пуст (0 примеров).")
            return issues

        # Проверка разбиения
        has_split = bool(dataset.train_inputs)
        if not has_split:
            issues.append(
                "Датасет не разбит на Train/Val/Test. "
                "Нажмите 'Применить разбиение' во вкладке 'Данные'."
            )

        # Проверка словаря для текста
        if meta.data_type == DataType.TEXT:
            if has_split and not dataset.vocab:
                issues.append(
                    "Текстовый датасет разбит, но словарь пуст. "
                    "Словарь должен быть построен при разбиении."
                )

        # Проверка изображений
        if meta.data_type == DataType.IMAGE:
            if meta.image_shape is None:
                issues.append(
                    "Для изображений не указана форма (image_shape). "
                    "Ожидается (H, W, C)."
                )

        # Проверка аудио
        if meta.data_type == DataType.AUDIO:
            if meta.audio_sample_rate is None:
                issues.append(
                    "Для аудио не указана частота дискретизации (audio_sample_rate)."
                )

        # Проверка согласованности длин
        for split_name in ("train", "val", "test"):
            inputs = getattr(dataset, f"{split_name}_inputs", [])
            outputs = getattr(dataset, f"{split_name}_outputs", [])
            if inputs and outputs and len(inputs) != len(outputs):
                issues.append(
                    f"Рассинхрон в '{split_name}': "
                    f"{len(inputs)} входов, {len(outputs)} выходов."
                )

        for issue in issues:
            logger.warning(f"Валидация: {issue}")

        return issues

    @staticmethod
    def detect_type_from_file(path: Path) -> Optional[str]:
        """
        Определяет предполагаемый тип данных по расширению / содержимому.
        Возвращает строку типа или None.
        """
        suffix = path.suffix.lower()

        if suffix in IMAGE_EXTENSIONS:
            return "image"
        if suffix in AUDIO_EXTENSIONS:
            return "audio"
        if suffix in (".oai", ".json", ".csv", ".pt"):
            return None  # определяется по содержимому
        if path.is_dir():
            # Проверяем содержимое папки
            files = list(path.iterdir())
            if not files:
                return None
            exts = {f.suffix.lower() for f in files if f.is_file()}
            if exts & IMAGE_EXTENSIONS:
                return "image"
            if exts & AUDIO_EXTENSIONS:
                return "audio"
        return None


# ============================================================
#  DatasetLoader — загрузка, миграция, разбиение
# ============================================================

class DatasetLoader:
    """
    Загрузчик датасетов. Единая точка входа для получения данных.

    Все методы статические. Результат — DatasetContainer.
    """

    # --------------------------------------------------------
    #  ЗАГРУЗКА
    # --------------------------------------------------------

    @staticmethod
    def load_dataset(path: Path) -> DatasetContainer:
        """
        Загружает датасет из файла или папки.

        Поддерживаемые форматы:
          .oai, .json, .csv, .pt  — файлы данных
          папка с изображениями    — классификация изображений
          папка с аудио            — классификация аудио

        Возвращает DatasetContainer.
        Бросает ValueError только при критической невозможности чтения.
        """
        path = Path(path)

        # Проверка размера файла
        if path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > MAX_FILE_SIZE_MB_WARNING:
                logger.warning(
                    f"Файл {path.name} имеет размер {size_mb:.1f} МБ. "
                    f"Загрузка может занять время."
                )

        # Если это папка — загружаем как набор файлов
        if path.is_dir():
            return DatasetLoader._load_from_directory(path)

        suffix = path.suffix.lower()

        try:
            if suffix == ".oai":
                return DatasetLoader._load_oai(path)
            elif suffix == ".json":
                return DatasetLoader._load_json(path)
            elif suffix == ".csv":
                return DatasetLoader._load_csv(path)
            elif suffix == ".pt":
                return DatasetLoader._load_pt(path)
            elif suffix in IMAGE_EXTENSIONS:
                return DatasetLoader._load_single_image(path)
            elif suffix in AUDIO_EXTENSIONS:
                return DatasetLoader._load_single_audio(path)
            else:
                raise ValueError(f"Неподдерживаемый формат: {suffix}")
        except Exception as e:
            logger.error(f"Ошибка загрузки датасета из '{path}': {e}")
            raise

    # --------------------------------------------------------
    #  ВНУТРЕННИЕ ЗАГРУЗЧИКИ ПО ФОРМАТАМ
    # --------------------------------------------------------

    @staticmethod
    def _load_oai(path: Path) -> DatasetContainer:
        """Загрузка внутреннего формата .oai (JSON)."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return DatasetLoader._migrate_raw_dict(data)

    @staticmethod
    def _load_json(path: Path) -> DatasetContainer:
        """
        Загрузка произвольного JSON.
        Ожидаем список словарей [{"x":..., "y":...}] или {"inputs":[], "outputs":[]}.
        """
        with open(path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # Вариант 1: список пар
        if isinstance(raw_data, list):
            inputs = [item.get("x", item.get("input", "")) for item in raw_data]
            outputs = [item.get("y", item.get("output", "")) for item in raw_data]
            clean_in, clean_out, report = DatasetWizard.analyze_and_clean(inputs, outputs)
            return DatasetLoader._build_container_from_raw(
                clean_in, clean_out, report, source_path=path
            )

        # Вариант 2: словарь с ключами
        if isinstance(raw_data, dict):
            # Проверяем, может это уже формат с meta
            if "meta" in raw_data or "raw_inputs" in raw_data:
                return DatasetLoader._migrate_raw_dict(raw_data)
            # Пробуем как {"inputs": [...], "outputs": [...]}
            inputs = raw_data.get("inputs", raw_data.get("raw_inputs", []))
            outputs = raw_data.get("outputs", raw_data.get("raw_outputs", []))
            if inputs and outputs:
                clean_in, clean_out, report = DatasetWizard.analyze_and_clean(
                    inputs, outputs
                )
                return DatasetLoader._build_container_from_raw(
                    clean_in, clean_out, report, source_path=path
                )

        raise ValueError("Не удалось распознать структуру JSON-файла.")

    @staticmethod
    def _load_csv(path: Path) -> DatasetContainer:
        """
        Загрузка CSV. По умолчанию последние 2 колонки = вход и выход.
        Первая строка считается заголовком.
        """
        inputs: List[str] = []
        outputs: List[str] = []

        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)  # пропускаем заголовок
            for row in reader:
                if len(row) >= 2:
                    inputs.append(row[-2])
                    outputs.append(row[-1])

        if not inputs:
            raise ValueError("CSV-файл пуст или не содержит данных.")

        clean_in, clean_out, report = DatasetWizard.analyze_and_clean(inputs, outputs)
        return DatasetLoader._build_container_from_raw(
            clean_in, clean_out, report, source_path=path
        )

    @staticmethod
    def _load_pt(path: Path) -> DatasetContainer:
        """Загрузка PyTorch-файла (.pt). Ожидаем словарь или тензоры."""
        data = torch.load(path, map_location="cpu", weights_only=False)

        # Если это уже словарь с данными
        if isinstance(data, dict):
            if "raw_inputs" in data or "meta" in data:
                return DatasetLoader._migrate_raw_dict(data)
            # Может быть словарь тензоров
            if "x" in data and "y" in data:
                inputs = data["x"].numpy().tolist()
                outputs = data["y"].numpy().tolist()
                report = {"type": "numeric", "num_samples": len(inputs),
                          "input_dim": data["x"].shape[-1] if data["x"].dim() > 1 else 1,
                          "output_dim": 1, "warnings": []}
                return DatasetLoader._build_container_from_raw(
                    inputs, outputs, report, source_path=path
                )

        raise ValueError("Не удалось распознать структуру .pt файла.")

    @staticmethod
    def _load_from_directory(path: Path) -> DatasetContainer:
        """
        Загрузка папки с файлами.
        Изображения: каждая подпапка = класс.
        Аудио: аналогично.
        """
        files = sorted(path.iterdir())

        # Определяем тип по расширениям
        image_files = [f for f in files if f.suffix.lower() in IMAGE_EXTENSIONS]
        audio_files = [f for f in files if f.suffix.lower() in AUDIO_EXTENSIONS]

        # Проверяем подпапки (структура: папка/класс/файлы)
        subdirs = [d for d in files if d.is_dir()]

        if subdirs:
            # Структура: папка/класс1/..., папка/класс2/...
            return DatasetLoader._load_directory_by_class(path, subdirs)
        elif image_files:
            return DatasetLoader._load_flat_images(image_files, path)
        elif audio_files:
            return DatasetLoader._load_flat_audio(audio_files, path)
        else:
            raise ValueError(
                f"Папка '{path}' не содержит поддерживаемых файлов "
                f"(изображения: {IMAGE_EXTENSIONS}, аудио: {AUDIO_EXTENSIONS})."
            )

    @staticmethod
    def _load_directory_by_class(
        root: Path, subdirs: List[Path]
    ) -> DatasetContainer:
        """Загрузка изображений/аудио из структуры папка/класс/файлы."""
        inputs: List[str] = []   # пути к файлам
        outputs: List[str] = []  # имена классов
        data_type = None

        for class_dir in subdirs:
            class_name = class_dir.name
            for f in sorted(class_dir.iterdir()):
                if f.suffix.lower() in IMAGE_EXTENSIONS:
                    inputs.append(str(f))
                    outputs.append(class_name)
                    data_type = "image"
                elif f.suffix.lower() in AUDIO_EXTENSIONS:
                    inputs.append(str(f))
                    outputs.append(class_name)
                    data_type = "audio"

        if not inputs:
            raise ValueError(f"В подпапках '{root}' не найдено поддерживаемых файлов.")

        num_classes = len(subdirs)

        meta = DatasetMeta(
            format_version=FORMAT_VERSION,
            data_type=DataType.IMAGE if data_type == "image" else DataType.AUDIO,
            task_type=TaskType.IMAGE_CLASSIFICATION if data_type == "image"
                      else TaskType.AUDIO_CLASSIFICATION,
            num_samples=len(inputs),
            num_classes=num_classes,
            description=f"Загружено из папки '{root.name}' ({num_classes} классов)",
        )

        container = DatasetContainer(
            meta=meta,
            raw_inputs=inputs,
            raw_outputs=outputs,
        )

        logger.info(
            f"Загружено {len(inputs)} файлов из {num_classes} классов "
            f"из папки '{root}'."
        )
        return container

    @staticmethod
    def _load_flat_images(
        image_files: List[Path], root: Path
    ) -> DatasetContainer:
        """Загрузка плоской папки с изображениями (без классов)."""
        inputs = [str(f) for f in image_files]
        outputs = ["unknown"] * len(inputs)  # класс неизвестен

        meta = DatasetMeta(
            format_version=FORMAT_VERSION,
            data_type=DataType.IMAGE,
            task_type=TaskType.IMAGE_CLASSIFICATION,
            num_samples=len(inputs),
            num_classes=0,
            description=f"Изображения из папки '{root.name}' (без разметки)",
        )

        return DatasetContainer(meta=meta, raw_inputs=inputs, raw_outputs=outputs)

    @staticmethod
    def _load_flat_audio(
        audio_files: List[Path], root: Path
    ) -> DatasetContainer:
        """Загрузка плоской папки с аудио (без классов)."""
        inputs = [str(f) for f in audio_files]
        outputs = ["unknown"] * len(inputs)

        meta = DatasetMeta(
            format_version=FORMAT_VERSION,
            data_type=DataType.AUDIO,
            task_type=TaskType.AUDIO_CLASSIFICATION,
            num_samples=len(inputs),
            num_classes=0,
            description=f"Аудио из папки '{root.name}' (без разметки)",
        )

        return DatasetContainer(meta=meta, raw_inputs=inputs, raw_outputs=outputs)

    @staticmethod
    def _load_single_image(path: Path) -> DatasetContainer:
        """Загрузка одного изображения (для тестов / песочницы)."""
        meta = DatasetMeta(
            format_version=FORMAT_VERSION,
            data_type=DataType.IMAGE,
            task_type=TaskType.IMAGE_CLASSIFICATION,
            num_samples=1,
            description=f"Одиночное изображение: {path.name}",
        )
        return DatasetContainer(
            meta=meta,
            raw_inputs=[str(path)],
            raw_outputs=["unknown"],
        )

    @staticmethod
    def _load_single_audio(path: Path) -> DatasetContainer:
        """Загрузка одного аудиофайла."""
        meta = DatasetMeta(
            format_version=FORMAT_VERSION,
            data_type=DataType.AUDIO,
            task_type=TaskType.AUDIO_CLASSIFICATION,
            num_samples=1,
            description=f"Одиночное аудио: {path.name}",
        )
        return DatasetContainer(
            meta=meta,
            raw_inputs=[str(path)],
            raw_outputs=["unknown"],
        )

    # --------------------------------------------------------
    #  МИГРАЦИЯ И КОНСТРУИРОВАНИЕ КОНТЕЙНЕРА
    # --------------------------------------------------------

    @staticmethod
    def _migrate_raw_dict(data: Dict[str, Any]) -> DatasetContainer:
        """
        Миграция сырого словаря (из файла) в DatasetContainer.
        Поддерживает старый формат (без format_version) и новый.
        """
        # Извлекаем meta
        raw_meta = data.get("meta", {})
        format_ver = raw_meta.get("format_version", data.get("format_version", "1.x"))

        # Определяем тип данных
        type_str = raw_meta.get("type", raw_meta.get("data_type", "text"))
        try:
            data_type = DataType(type_str)
        except ValueError:
            data_type = DataType.TEXT if type_str in ("text", "string") else DataType.NUMERIC

        task_str = raw_meta.get("task_type", "seq2seq" if data_type == DataType.TEXT else "regression")
        try:
            task_type = TaskType(task_str)
        except ValueError:
            task_type = TaskType.SEQ2SEQ if data_type == DataType.TEXT else TaskType.REGRESSION

        meta = DatasetMeta(
            format_version=FORMAT_VERSION,
            data_type=data_type,
            task_type=task_type,
            num_samples=raw_meta.get("num_samples", 0),
            input_dim=raw_meta.get("input_dim"),
            output_dim=raw_meta.get("output_dim"),
            vocab_size=raw_meta.get("vocab_size", 0),
            image_shape=tuple(raw_meta["image_shape"]) if raw_meta.get("image_shape") else None,
            audio_sample_rate=raw_meta.get("audio_sample_rate"),
            audio_channels=raw_meta.get("audio_channels", 1),
            num_classes=raw_meta.get("num_classes", 0),
            max_sequence_length=raw_meta.get("max_sequence_length", 0),
            description=raw_meta.get("description", raw_meta.get("explanation", "")),
            generator_task=raw_meta.get("task_type", raw_meta.get("generator_task", "")),
        )

        container = DatasetContainer(
            meta=meta,
            raw_inputs=data.get("raw_inputs", []),
            raw_outputs=data.get("raw_outputs", []),
            train_inputs=data.get("train_inputs", []),
            train_outputs=data.get("train_outputs", []),
            val_inputs=data.get("val_inputs", []),
            val_outputs=data.get("val_outputs", []),
            test_inputs=data.get("test_inputs", []),
            test_outputs=data.get("test_outputs", []),
            vocab=data.get("vocab", {}),
            normalization=data.get("normalization"),
            augmentation_config=data.get("augmentation_config"),
            extra=data.get("extra", {}),
        )

        # Если данных в meta меньше, чем реально — пересчитываем
        if meta.num_samples == 0 and container.raw_inputs:
            meta.num_samples = len(container.raw_inputs)

        if format_ver != FORMAT_VERSION:
            logger.info(
                f"Миграция данных: формат '{format_ver}' → '{FORMAT_VERSION}'."
            )

        return container

    @staticmethod
    def _build_container_from_raw(
        inputs: List[Any],
        outputs: List[Any],
        report: Dict[str, Any],
        source_path: Optional[Path] = None,
    ) -> DatasetContainer:
        """Создаёт DatasetContainer из очищенных данных и отчёта."""
        type_str = report.get("type", "text")
        try:
            data_type = DataType(type_str)
        except ValueError:
            data_type = DataType.TEXT

        task_type = (
            TaskType.SEQ2SEQ if data_type == DataType.TEXT else TaskType.REGRESSION
        )

        meta = DatasetMeta(
            format_version=FORMAT_VERSION,
            data_type=data_type,
            task_type=task_type,
            num_samples=len(inputs),
            input_dim=report.get("input_dim", 1),
            output_dim=report.get("output_dim", 1),
            description=f"Загружено из {source_path.name}" if source_path else "",
        )

        container = DatasetContainer(
            meta=meta,
            raw_inputs=inputs,
            raw_outputs=outputs,
        )

        # Сохраняем предупреждения в extra
        warnings_list = report.get("warnings", [])
        if warnings_list:
            container.extra["load_warnings"] = warnings_list

        return container

    # --------------------------------------------------------
    #  АНАЛИЗ
    # --------------------------------------------------------

    @staticmethod
    def analyze_dataset(dataset: DatasetContainer) -> Dict[str, Any]:
        """
        Возвращает сводку по датасету в виде словаря.
        Используется для отображения метаданных в GUI.
        """
        meta = dataset.meta
        info: Dict[str, Any] = {
            "type": meta.data_type,
            "task_type": meta.task_type,
            "num_samples": meta.num_samples,
            "input_dim": meta.input_dim,
            "output_dim": meta.output_dim,
            "vocab_size": meta.vocab_size,
            "num_classes": meta.num_classes,
        }

        # Добавляем информацию о разбиении
        has_split = bool(dataset.train_inputs)
        info["has_split"] = has_split
        if has_split:
            info["train_samples"] = len(dataset.train_inputs)
            info["val_samples"] = len(dataset.val_inputs)
            info["test_samples"] = len(dataset.test_inputs)

        # Предупреждения
        info["warnings"] = dataset.extra.get("load_warnings", [])

        return info

    # --------------------------------------------------------
    #  РАЗБИЕНИЕ И СЛОВАРЬ
    # --------------------------------------------------------

    @staticmethod
    def split_dataset(
        dataset: DatasetContainer,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        max_vocab_size: Optional[int] = None,
    ) -> DatasetContainer:
        """
        Разбивает датасет на Train / Val / Test.
        Для текстовых данных строит словарь (только на Train).

        Аргументы:
            dataset:        DatasetContainer с заполненными raw_inputs/raw_outputs
            train_ratio:    доля для обучения (0.0–1.0)
            val_ratio:      доля для валидации
            test_ratio:     доля для теста
            max_vocab_size: ограничение словаря (только для текста)

        Возвращает:
            Тот же DatasetContainer с заполненными split-полями и словарём.

        Бросает ValueError только если данных нет вообще.
        """
        inputs = dataset.raw_inputs
        outputs = dataset.raw_outputs

        if not inputs or not outputs:
            raise ValueError("Нет данных для разбиения (raw_inputs / raw_outputs пусты).")

        if len(inputs) != len(outputs):
            raise ValueError(
                f"Рассинхрон: {len(inputs)} входов и {len(outputs)} выходов. "
                "Каждому входу должен соответствовать ровно один выход."
            )

        total = len(inputs)
        if total < 3:
            raise ValueError(
                f"Слишком мало данных для разбиения: {total} примеров. "
                "Нужно минимум 3 (по одному на каждый сплит)."
            )

        # Проверяем суммы
        ratio_sum = train_ratio + val_ratio + test_ratio
        if abs(ratio_sum - 1.0) > 0.001:
            raise ValueError(
                f"Сумма долей разбиения должна быть 1.0, а получила {ratio_sum:.4f}."
            )

        # Перемешиваем индексы
        indices = np.random.permutation(total)

        train_end = max(int(total * train_ratio), 1)
        val_end = max(train_end + int(total * val_ratio), train_end + 1)

        def get_split(start: int, end: int) -> Tuple[List, List]:
            split_idx = indices[start:end]
            return (
                [inputs[i] for i in split_idx],
                [outputs[i] for i in split_idx],
            )

        dataset.train_inputs, dataset.train_outputs = get_split(0, train_end)
        dataset.val_inputs, dataset.val_outputs = get_split(train_end, val_end)
        dataset.test_inputs, dataset.test_outputs = get_split(val_end, total)

        # Обновляем мета
        dataset.meta.num_samples = total

        # --- ПОСТРОЕНИЕ СЛОВАРЯ (только для текста) ---
        data_type = dataset.meta.data_type
        if data_type == DataType.TEXT:
            tokenizer = TextTokenizer()
            # Словарь строим ТОЛЬКО на обучающей выборке!
            train_texts = dataset.train_inputs + dataset.train_outputs
            tokenizer.build_vocab(train_texts, max_vocab_size=max_vocab_size)
            dataset.vocab = tokenizer.vocab
            dataset.meta.vocab_size = len(tokenizer.vocab)
            logger.info(
                f"Словарь построен на {len(train_texts)} текстовых примерах: "
                f"{len(tokenizer.vocab)} токенов."
            )
        elif data_type == DataType.NUMERIC:
            # Для числовых данных вычисляем нормализацию
            DatasetLoader._compute_normalization(dataset)

        # Логируем результат
        logger.info(
            f"Разбиение применено: Train={len(dataset.train_inputs)}, "
            f"Val={len(dataset.val_inputs)}, Test={len(dataset.test_inputs)}."
        )

        return dataset

    @staticmethod
    def _compute_normalization(dataset: DatasetContainer):
        """Вычисляет mean/std для числовых данных и сохраняет в контейнер."""
        try:
            train_array = np.array(dataset.train_inputs, dtype=np.float32)
            mean = train_array.mean(axis=0).tolist()
            std = train_array.std(axis=0).tolist()
            dataset.normalization = {"mean": mean, "std": std}
            logger.info("Нормализация вычислена для числовых данных.")
        except Exception as e:
            logger.warning(f"Не удалось вычислить нормализацию: {e}")

    # --------------------------------------------------------
    #  РЕДАКТИРОВАНИЕ СЛОВАРЯ
    # --------------------------------------------------------

    @staticmethod
    def get_vocab_table(dataset: DatasetContainer) -> List[Dict[str, Any]]:
        """
        Возвращает таблицу словаря для отображения в GUI.
        Каждая строка: {символ, индекс, частота}.
        """
        vocab = dataset.vocab
        if not vocab:
            return []

        # Считаем частоты в train
        from collections import Counter
        char_counts: Counter = Counter()
        for text in dataset.train_inputs + dataset.train_outputs:
            char_counts.update(str(text))

        table = []
        for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
            table.append({
                "token": token,
                "index": idx,
                "frequency": char_counts.get(token, 0),
            })
        return table

    @staticmethod
    def rebuild_vocab(
        dataset: DatasetContainer,
        max_vocab_size: Optional[int] = None,
    ) -> DatasetContainer:
        """
        Перестраивает словарь заново (после ручного редактирования).
        Вызывается из GUI-редактора словаря.
        """
        if not dataset.train_inputs:
            logger.warning("Нельзя перестроить словарь: train_inputs пусты.")
            return dataset

        tokenizer = TextTokenizer()
        train_texts = dataset.train_inputs + dataset.train_outputs
        tokenizer.build_vocab(train_texts, max_vocab_size=max_vocab_size)
        dataset.vocab = tokenizer.vocab
        dataset.meta.vocab_size = len(tokenizer.vocab)

        logger.info(f"Словарь перестроен: {len(tokenizer.vocab)} токенов.")
        return dataset

    # --------------------------------------------------------
    #  СОХРАНЕНИЕ
    # --------------------------------------------------------

    @staticmethod
    def save_dataset(dataset: DatasetContainer, path: Path):
        """
        Сохраняет DatasetContainer в файл .oai (JSON).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Используем встроенный метод сериализации
        dataset.save(path)

        logger.info(f"Датасет сохранён: {path} ({dataset.meta.num_samples} примеров)")