"""
core/dataset.py
===============
Универсальный PyTorch Dataset для всех типов данных проекта.

Поддерживаемые типы:
  • text         — символьная токенизация (последовательности)
  • numeric      — числовые векторы / таблицы
  • image        — изображения (H, W, C)
  • audio        — звуковые волны / спектрограммы
  • graph        — графы (узлы, рёбра, признаки)
  • time_series  — временные ряды (последовательности чисел)
  • multimodal   — пары разнородных данных

Версия формата: 2.0

ПРАВИЛА:
  - Работает ТОЛЬКО с DatasetContainer из core.contracts
  - НЕ использует словари для внутреннего представления
  - Все данные извлекаются через атрибуты объекта
  - Безопасно обрабатывает None и отсутствующие данные
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import json
import warnings

from core.contracts import (
    DataType,
    TaskType,
    DatasetMeta,
    DatasetContainer,
    FORMAT_VERSION,
)
from core.logger import get_logger

logger = get_logger(__name__)


# ============================================================
#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ПРИВЕДЕНИЯ ТИПОВ
# ============================================================

def _ensure_container(data) -> DatasetContainer:
    """
    Приводит входные данные к DatasetContainer.
    Принимает: DatasetContainer, dict (старый или новый формат).
    """
    if isinstance(data, DatasetContainer):
        return data
    if isinstance(data, dict):
        # Если уже содержит format_version и структурирован как новый
        if "format_version" in data and "meta" in data:
            return DatasetContainer.from_dict(data)
        # Старый формат — миграция через from_dict
        return DatasetContainer.from_dict(data)
    raise TypeError(
        f"Ожидался DatasetContainer или dict, получен: {type(data).__name__}"
    )


def _safe_tensor_from_array(val) -> torch.Tensor:
    """Безопасное преобразование в float-тензор."""
    if val is None:
        return torch.zeros(1, dtype=torch.float32)
    if isinstance(val, (int, float)):
        return torch.tensor([val], dtype=torch.float32)
    if isinstance(val, str):
        try:
            parts = val.replace(",", " ").split()
            return torch.tensor([float(p) for p in parts], dtype=torch.float32)
        except ValueError:
            logger.warning(f"Не удалось распарсить '{val}' как числа. Нули.")
            return torch.zeros(1, dtype=torch.float32)
    if isinstance(val, torch.Tensor):
        return val.float()
    try:
        arr = np.array(val, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(arr).float()
    except (ValueError, TypeError):
        return torch.zeros(1, dtype=torch.float32)


def _safe_label(val) -> torch.Tensor:
    """Преобразует метку в тензор."""
    if isinstance(val, (int, np.integer)):
        return torch.tensor(int(val), dtype=torch.long)
    if isinstance(val, float):
        return torch.tensor(val, dtype=torch.float32)
    if isinstance(val, str):
        try:
            return torch.tensor(int(val), dtype=torch.long)
        except ValueError:
            try:
                return torch.tensor(float(val), dtype=torch.float32)
            except ValueError:
                return torch.tensor(0, dtype=torch.long)
    if isinstance(val, (list, tuple)):
        try:
            return torch.tensor(val, dtype=torch.float32)
        except (ValueError, TypeError):
            return torch.tensor(0, dtype=torch.long)
    if isinstance(val, dict):
        return torch.tensor(0, dtype=torch.long)
    if isinstance(val, torch.Tensor):
        return val
    return torch.tensor(0, dtype=torch.long)


# ============================================================
#  ОСНОВНОЙ КЛАСС
# ============================================================

class OracleTorchDataset(Dataset):
    """
    Универсальный PyTorch Dataset для всех типов данных.

    Использование:
        container = DatasetContainer.load("dataset.oai")
        ds = OracleTorchDataset(container, split="train")
        loader = DataLoader(ds, batch_size=64, collate_fn=ds.collate_fn)

    Поддерживает:
        - text: символьная токенизация, PAD/UNK/BOS/EOS
        - numeric: float-тензоры, опциональная нормализация
        - image: (C, H, W) тензоры, нормализация [0, 1]
        - audio: (C, samples) тензоры
        - graph: dict с node_features, edge_index, labels
        - time_series: (seq_len, features) тензоры
        - multimodal: dict с несколькими модальностями
    """

    def __init__(self, data, split: str = "train"):
        """
        Args:
            data: DatasetContainer (или dict для обратной совместимости)
            split: "train" | "val" | "test"
        """
        self.split = split
        self.container = _ensure_container(data)
        self.meta: DatasetMeta = self.container.meta
        self.data_type: DataType = self.meta.data_type

        # Извлекаем данные для нужного сплита из контейнера
        self.inputs: List[Any] = getattr(self.container, f"{split}_inputs", [])
        self.outputs: List[Any] = getattr(self.container, f"{split}_outputs", [])

        # Валидация
        self._validate_data()

        # Специфичные атрибуты по типу данных
        self.vocab: Dict[str, int] = {}
        self.pad_idx: int = 0
        self.unk_idx: int = 1
        self.bos_idx: int = 2
        self.eos_idx: int = 3

        if self.data_type == DataType.TEXT:
            self.vocab = self.container.vocab or {
                "<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3
            }
            self.pad_idx = self.vocab.get("<PAD>", 0)
            self.unk_idx = self.vocab.get("<UNK>", 1)
            self.bos_idx = self.vocab.get("<BOS>", 2)
            self.eos_idx = self.vocab.get("<EOS>", 3)

        # Для числовых данных: нормализация
        self._norm_mean: Optional[np.ndarray] = None
        self._norm_std: Optional[np.ndarray] = None
        if self.data_type in (DataType.NUMERIC, DataType.TIME_SERIES):
            norm = self.container.normalization
            if norm and "mean" in norm and "std" in norm:
                self._norm_mean = np.array(norm["mean"], dtype=np.float32)
                self._norm_std = np.array(norm["std"], dtype=np.float32)
                self._norm_std = np.where(
                    np.abs(self._norm_std) < 1e-8, 1.0, self._norm_std
                )

        # Для изображений: ожидаемая форма
        self._image_shape: Optional[Tuple[int, int, int]] = self.meta.image_shape

        # Для аудио
        self._audio_sr: Optional[int] = self.meta.audio_sample_rate
        self._audio_channels: int = self.meta.audio_channels

        logger.info(
            f"OracleTorchDataset создан: split={split}, "
            f"type={self.data_type.value}, samples={len(self.inputs)}"
        )

    # --------------------------------------------------------
    #  ВАЛИДАЦИЯ
    # --------------------------------------------------------
    def _validate_data(self):
        """
        Проверяет данные на целостность.
        Не блокирует обучение, но логирует проблемы.
        """
        if not self.inputs and not self.outputs:
            raise ValueError(
                f"Выборка '{self.split}' пуста!\n"
                f"Вы забыли нажать 'Применить разбиение' во вкладке 'Данные'."
            )

        if len(self.inputs) != len(self.outputs):
            raise ValueError(
                f"Рассинхрон данных в '{self.split}'!\n"
                f"Входов: {len(self.inputs)}, Выходов: {len(self.outputs)}.\n"
                f"Каждому вопросу должен соответствовать ровно один ответ."
            )

        # Проверяем первые 100 элементов
        issues = 0
        sample_size = min(len(self.inputs), 100)

        for i in range(sample_size):
            try:
                self._check_single_item(self.inputs[i], self.outputs[i])
            except (ValueError, TypeError) as e:
                issues += 1
                if issues <= 3:
                    logger.warning(
                        f"Проблема в элементе [{i}] сплита '{self.split}': {e}"
                    )

        if issues > 0:
            logger.warning(
                f"⚠️ Обнаружено {issues} проблемных элементов из "
                f"{sample_size} проверенных. Они будут пропущены при загрузке."
            )

        # Проверка на NaN/Inf для числовых данных
        if self.data_type in (DataType.NUMERIC, DataType.TIME_SERIES):
            self._check_numeric_health(sample_size)

    def _check_single_item(self, x, y):
        """Проверяет один элемент данных на валидность."""
        if x is None or y is None:
            raise ValueError("None в данных")
        if isinstance(x, str) and x.strip() == "":
            raise ValueError("Пустая строка на входе")
        if isinstance(y, str) and y.strip() == "":
            raise ValueError("Пустая строка на выходе")

    def _check_numeric_health(self, sample_size: int):
        """Проверяет числовые данные на NaN/Inf."""
        nan_count = 0
        inf_count = 0
        for i in range(min(sample_size, len(self.inputs))):
            try:
                arr = np.array(self.inputs[i], dtype=np.float32)
                if np.isnan(arr).any():
                    nan_count += 1
                if np.isinf(arr).any():
                    inf_count += 1
            except (ValueError, TypeError):
                pass

        if nan_count > 0 or inf_count > 0:
            logger.warning(
                f"⚠️ В числовых данных: {nan_count} NaN, {inf_count} Inf. "
                f"Эти элементы будут заменены нулями при загрузке."
            )

    # --------------------------------------------------------
    #  ОСНОВНЫЕ МЕТОДЫ DATASET
    # --------------------------------------------------------
    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Возвращает словарь {"x": ..., "y": ...} для одного элемента.
        Формат зависит от типа данных.
        """
        x_raw = self.inputs[idx]
        y_raw = self.outputs[idx]

        try:
            if self.data_type == DataType.TEXT:
                return self._get_text_item(x_raw, y_raw)
            elif self.data_type == DataType.NUMERIC:
                return self._get_numeric_item(x_raw, y_raw)
            elif self.data_type == DataType.IMAGE:
                return self._get_image_item(x_raw, y_raw)
            elif self.data_type == DataType.AUDIO:
                return self._get_audio_item(x_raw, y_raw)
            elif self.data_type == DataType.GRAPH:
                return self._get_graph_item(x_raw, y_raw)
            elif self.data_type == DataType.TIME_SERIES:
                return self._get_time_series_item(x_raw, y_raw)
            elif self.data_type == DataType.MULTIMODAL:
                return self._get_multimodal_item(x_raw, y_raw)
            else:
                return self._get_numeric_item(x_raw, y_raw)
        except Exception as e:
            logger.warning(f"Элемент [{idx}] повреждён ({e}). Возвращаю заглушку.")
            return self._get_fallback_item()

    def _get_fallback_item(self) -> Dict[str, Any]:
        """Заглушка для повреждённых элементов."""
        if self.data_type == DataType.TEXT:
            return {
                "x": torch.tensor([self.pad_idx], dtype=torch.long),
                "y": torch.tensor([self.bos_idx, self.eos_idx], dtype=torch.long),
            }
        elif self.data_type == DataType.IMAGE:
            shape = self._image_shape or (28, 28, 1)
            h, w, c = shape
            return {
                "x": torch.zeros(c, h, w, dtype=torch.float32),
                "y": torch.tensor(0, dtype=torch.long),
            }
        elif self.data_type == DataType.AUDIO:
            return {
                "x": torch.zeros(1, 16000, dtype=torch.float32),
                "y": torch.tensor(0, dtype=torch.long),
            }
        elif self.data_type == DataType.GRAPH:
            return {
                "x": {"node_features": torch.zeros(1, 1),
                       "edge_index": torch.zeros(2, 0, dtype=torch.long)},
                "y": torch.tensor(0, dtype=torch.long),
            }
        else:
            dim = self.meta.input_dim if self.meta.input_dim else 1
            if isinstance(dim, (tuple, list)):
                dim = int(np.prod(dim))
            return {
                "x": torch.zeros(dim, dtype=torch.float32),
                "y": torch.zeros(1, dtype=torch.float32),
            }

    # --------------------------------------------------------
    #  TEXT
    # --------------------------------------------------------
    def _get_text_item(self, x_raw, y_raw) -> Dict[str, torch.Tensor]:
        """
        Текст → последовательности индексов.
        Вход: [символы без BOS]
        Выход: [BOS, символы, EOS]
        """
        x_seq = [self.vocab.get(ch, self.unk_idx) for ch in str(x_raw)]
        y_seq = (
            [self.bos_idx]
            + [self.vocab.get(ch, self.unk_idx) for ch in str(y_raw)]
            + [self.eos_idx]
        )
        return {
            "x": torch.tensor(x_seq, dtype=torch.long),
            "y": torch.tensor(y_seq, dtype=torch.long),
        }

    # --------------------------------------------------------
    #  NUMERIC
    # --------------------------------------------------------
    def _get_numeric_item(self, x_raw, y_raw) -> Dict[str, torch.Tensor]:
        """
        Числовые данные → float-тензоры.
        Применяет нормализацию, если задана.
        """
        x = self._to_float_array(x_raw)
        y = self._to_float_array(y_raw)

        # Нормализация
        if self._norm_mean is not None and self._norm_std is not None:
            if x.shape[-1] == len(self._norm_mean):
                x = (x - self._norm_mean) / self._norm_std

        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
        }

    def _to_float_array(self, val) -> np.ndarray:
        """Безопасное преобразование в float-массив."""
        if val is None:
            return np.zeros(1, dtype=np.float32)
        if isinstance(val, (int, float)):
            return np.array([val], dtype=np.float32)
        if isinstance(val, str):
            try:
                parts = val.replace(",", " ").split()
                return np.array([float(p) for p in parts], dtype=np.float32)
            except ValueError:
                logger.warning(f"Не удалось распарсить '{val}' как числа. Нули.")
                return np.zeros(1, dtype=np.float32)
        try:
            arr = np.array(val, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            return arr
        except (ValueError, TypeError):
            return np.zeros(1, dtype=np.float32)

    # --------------------------------------------------------
    #  IMAGE
    # --------------------------------------------------------
    def _get_image_item(self, x_raw, y_raw) -> Dict[str, Any]:
        """
        Изображение → тензор (C, H, W), нормализованный [0, 1].
        Метка → long (для классификации) или тензор (для генерации).
        """
        img = self._to_image_tensor(x_raw)

        # Метка: для классификации — число, для генерации — само изображение
        if self.meta.task_type == TaskType.IMAGE_GENERATION:
            y_tensor = img.clone()
        elif self.meta.task_type == TaskType.ANOMALY_DETECTION:
            y_tensor = self._to_image_tensor(y_raw)
        else:
            y_tensor = _safe_label(y_raw)

        return {"x": img, "y": y_tensor}

    def _to_image_tensor(self, val) -> torch.Tensor:
        """
        Преобразует изображение в тензор (C, H, W).
        Принимает: ndarray (H,W,C), список списков, путь к файлу.
        """
        if isinstance(val, str):
            path = Path(val)
            if path.exists() and path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                try:
                    return self._load_image_file(path)
                except Exception as e:
                    logger.warning(f"Не удалось загрузить изображение '{val}': {e}")
                    return self._default_image_tensor()
            try:
                parts = val.replace(",", " ").split()
                arr = np.array([float(p) for p in parts], dtype=np.float32)
                return self._reshape_to_image(arr)
            except ValueError:
                return self._default_image_tensor()

        if isinstance(val, np.ndarray):
            arr = val.astype(np.float32)
        elif isinstance(val, (list, tuple)):
            try:
                arr = np.array(val, dtype=np.float32)
            except (ValueError, TypeError):
                return self._default_image_tensor()
        elif isinstance(val, torch.Tensor):
            arr = val.float().numpy()
        else:
            return self._default_image_tensor()

        if arr.max() > 1.0:
            arr = arr / 255.0

        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        elif arr.ndim == 3:
            if arr.shape[0] in (1, 3, 4):
                pass
            elif arr.shape[2] in (1, 3, 4):
                arr = np.transpose(arr, (2, 0, 1))
        else:
            return self._reshape_to_image(arr.flatten())

        return torch.tensor(arr, dtype=torch.float32)

    def _reshape_to_image(self, flat: np.ndarray) -> torch.Tensor:
        """Пытается ресейпить плоский массив в изображение."""
        expected = self._image_shape
        if expected:
            h, w, c = expected
            if flat.size == h * w * c:
                arr = flat.reshape(h, w, c)
                arr = np.transpose(arr, (2, 0, 1))
                if arr.max() > 1.0:
                    arr = arr / 255.0
                return torch.tensor(arr, dtype=torch.float32)
        return torch.tensor(flat[:1].reshape(1, 1, 1), dtype=torch.float32)

    def _default_image_tensor(self) -> torch.Tensor:
        """Заглушка для повреждённых изображений."""
        shape = self._image_shape or (28, 28, 1)
        h, w, c = shape
        return torch.zeros(c, h, w, dtype=torch.float32)

    def _load_image_file(self, path: Path) -> torch.Tensor:
        """Загружает изображение из файла (без PIL — заглушка)."""
        try:
            import struct
            logger.warning(
                f"Загрузка изображений из файлов требует Pillow. "
                f"Файл '{path.name}' заменён заглушкой."
            )
            return self._default_image_tensor()
        except Exception:
            return self._default_image_tensor()

    # --------------------------------------------------------
    #  AUDIO
    # --------------------------------------------------------
    def _get_audio_item(self, x_raw, y_raw) -> Dict[str, torch.Tensor]:
        """
        Аудио → тензор (channels, samples).
        Метка → класс или регрессия.
        """
        waveform = self._to_audio_tensor(x_raw)
        y_tensor = _safe_label(y_raw)
        return {"x": waveform, "y": y_tensor}

    def _to_audio_tensor(self, val) -> torch.Tensor:
        """Преобразует аудио в тензор (channels, samples)."""
        if isinstance(val, str):
            path = Path(val)
            if path.exists() and path.suffix.lower() in (".wav", ".mp3", ".flac"):
                logger.warning(
                    f"Загрузка аудио из файлов требует torchaudio. "
                    f"Файл '{path.name}' заменён тишиной."
                )
            return torch.zeros(self._audio_channels or 1, 16000)

        if isinstance(val, np.ndarray):
            arr = val.astype(np.float32)
        elif isinstance(val, (list, tuple)):
            try:
                arr = np.array(val, dtype=np.float32)
            except (ValueError, TypeError):
                return torch.zeros(self._audio_channels or 1, 16000)
        elif isinstance(val, torch.Tensor):
            arr = val.float().numpy()
        elif isinstance(val, dict):
            samples = val.get("samples", val.get("waveform", []))
            try:
                arr = np.array(samples, dtype=np.float32)
            except (ValueError, TypeError):
                return torch.zeros(self._audio_channels or 1, 16000)
        else:
            return torch.zeros(self._audio_channels or 1, 16000)

        if np.abs(arr).max() > 1.0:
            arr = arr / np.abs(arr).max()

        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        elif arr.ndim == 2:
            if arr.shape[0] > arr.shape[1]:
                arr = arr.T

        return torch.tensor(arr, dtype=torch.float32)

    # --------------------------------------------------------
    #  GRAPH
    # --------------------------------------------------------
    def _get_graph_item(self, x_raw, y_raw) -> Dict[str, Any]:
        """
        Граф → словарь с node_features, edge_index.
        Формат: {"node_features": (N, F), "edge_index": (2, E)}
        """
        graph_data = self._to_graph_dict(x_raw)
        y_tensor = _safe_label(y_raw)
        return {"x": graph_data, "y": y_tensor}

    def _to_graph_dict(self, val) -> Dict[str, torch.Tensor]:
        """Преобразует граф в стандартный формат."""
        if isinstance(val, dict):
            nf = val.get("node_features", val.get("nodes", [[0.0]]))
            ei = val.get("edge_index", val.get("edges", []))

            node_features = self._safe_to_tensor(nf, torch.float32)
            if node_features.ndim == 1:
                node_features = node_features.unsqueeze(0)

            if len(ei) > 0:
                edge_index = self._safe_to_tensor(ei, torch.long)
                if edge_index.ndim == 1:
                    edge_index = edge_index.unsqueeze(0)
            else:
                edge_index = torch.zeros(2, 0, dtype=torch.long)

            return {"node_features": node_features, "edge_index": edge_index}

        if isinstance(val, (list, tuple)) and len(val) == 2:
            nf, ei = val
            node_features = self._safe_to_tensor(nf, torch.float32)
            if node_features.ndim == 1:
                node_features = node_features.unsqueeze(0)
            edge_index = self._safe_to_tensor(ei, torch.long)
            if edge_index.ndim == 1:
                edge_index = edge_index.unsqueeze(0)
            return {"node_features": node_features, "edge_index": edge_index}

        return {
            "node_features": torch.zeros(1, 1, dtype=torch.float32),
            "edge_index": torch.zeros(2, 0, dtype=torch.long),
        }

    def _safe_to_tensor(self, val, dtype) -> torch.Tensor:
        """Безопасное преобразование в тензор."""
        try:
            if isinstance(val, torch.Tensor):
                return val.to(dtype)
            arr = np.array(val)
            return torch.tensor(arr, dtype=dtype)
        except (ValueError, TypeError):
            return torch.zeros(1, dtype=dtype)

    # --------------------------------------------------------
    #  TIME SERIES
    # --------------------------------------------------------
    def _get_time_series_item(self, x_raw, y_raw) -> Dict[str, torch.Tensor]:
        """
        Временной ряд → (seq_len, features).
        Выход: следующее значение или окно.
        """
        x_seq = self._to_sequence_array(x_raw)
        y_val = self._to_float_array(y_raw)

        if self._norm_mean is not None and self._norm_std is not None:
            if x_seq.ndim == 2 and x_seq.shape[1] == len(self._norm_mean):
                x_seq = (x_seq - self._norm_mean) / self._norm_std
            elif x_seq.ndim == 1 and len(self._norm_mean) == 1:
                x_seq = (x_seq - self._norm_mean[0]) / self._norm_std[0]

        if x_seq.ndim == 1:
            x_seq = x_seq[:, np.newaxis]

        return {
            "x": torch.tensor(x_seq, dtype=torch.float32),
            "y": torch.tensor(y_val, dtype=torch.float32),
        }

    def _to_sequence_array(self, val) -> np.ndarray:
        """Преобразует временной ряд в numpy-массив."""
        if isinstance(val, str):
            try:
                parts = val.replace(",", " ").split()
                return np.array([float(p) for p in parts], dtype=np.float32)
            except ValueError:
                return np.zeros(10, dtype=np.float32)
        if isinstance(val, (list, tuple)):
            try:
                arr = np.array(val, dtype=np.float32)
                return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            except (ValueError, TypeError):
                return np.zeros(10, dtype=np.float32)
        if isinstance(val, np.ndarray):
            return np.nan_to_num(val.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if isinstance(val, (int, float)):
            return np.array([val], dtype=np.float32)
        return np.zeros(10, dtype=np.float32)

    # --------------------------------------------------------
    #  MULTIMODAL
    # --------------------------------------------------------
    def _get_multimodal_item(self, x_raw, y_raw) -> Dict[str, Any]:
        """
        Мультимодальные данные: пары разных типов.
        x_raw может быть dict: {"text": ..., "image": ...}
        """
        result = {}

        if isinstance(x_raw, dict):
            for key, val in x_raw.items():
                if key == "text":
                    seq = [self.vocab.get(ch, self.unk_idx) for ch in str(val)]
                    result[key] = torch.tensor(seq, dtype=torch.long)
                elif key == "image":
                    result[key] = self._to_image_tensor(val)
                elif key == "audio":
                    result[key] = self._to_audio_tensor(val)
                else:
                    result[key] = _safe_tensor_from_array(val)
        else:
            result["x"] = _safe_tensor_from_array(x_raw)

        y_tensor = _safe_label(y_raw)
        return {"x": result, "y": y_tensor}

    # --------------------------------------------------------
    #  COLLATE FUNCTIONS
    # --------------------------------------------------------
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Объединяет элементы батча.
        Маршрутизирует по типу данных.
        """
        if not batch:
            return {"x": torch.tensor([]), "y": torch.tensor([])}

        if self.data_type == DataType.TEXT:
            return self._collate_text(batch)
        elif self.data_type == DataType.IMAGE:
            return self._collate_image(batch)
        elif self.data_type == DataType.AUDIO:
            return self._collate_audio(batch)
        elif self.data_type == DataType.GRAPH:
            return self._collate_graph(batch)
        elif self.data_type == DataType.TIME_SERIES:
            return self._collate_time_series(batch)
        elif self.data_type == DataType.MULTIMODAL:
            return self._collate_multimodal(batch)
        else:
            return self._collate_numeric(batch)

    def _collate_text(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Текст: pad_sequence до самой длинной последовательности."""
        x_tensors = [item["x"] for item in batch]
        y_tensors = [item["y"] for item in batch]

        x_padded = pad_sequence(
            x_tensors, batch_first=True, padding_value=self.pad_idx
        )
        y_padded = pad_sequence(
            y_tensors, batch_first=True, padding_value=self.pad_idx
        )
        return {"x": x_padded, "y": y_padded}

    def _collate_numeric(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Числовые: stack (все одинаковой размерности)."""
        try:
            x_batch = torch.stack([item["x"] for item in batch])
            y_batch = torch.stack([item["y"] for item in batch])
        except RuntimeError:
            max_x = max(item["x"].shape[0] for item in batch)
            max_y = max(item["y"].shape[0] for item in batch)
            x_list, y_list = [], []
            for item in batch:
                x = item["x"]
                y = item["y"]
                if x.shape[0] < max_x:
                    x = torch.cat([x, torch.zeros(max_x - x.shape[0])])
                if y.shape[0] < max_y:
                    y = torch.cat([y, torch.zeros(max_y - y.shape[0])])
                x_list.append(x)
                y_list.append(y)
            x_batch = torch.stack(x_list)
            y_batch = torch.stack(y_list)
        return {"x": x_batch, "y": y_batch}

    def _collate_image(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Изображения: stack (все одного размера)."""
        try:
            x_batch = torch.stack([item["x"] for item in batch])
        except RuntimeError:
            x_batch = self._pad_images([item["x"] for item in batch])

        y_items = [item["y"] for item in batch]
        if all(isinstance(y, torch.Tensor) for y in y_items):
            try:
                y_batch = torch.stack(y_items)
            except RuntimeError:
                y_batch = torch.stack([y.flatten() for y in y_items])
        else:
            y_batch = torch.tensor(y_items)

        return {"x": x_batch, "y": y_batch}

    def _pad_images(self, images: List[torch.Tensor]) -> torch.Tensor:
        """Паддит изображения до максимального размера в батче."""
        if not images:
            return torch.zeros(1, 1, 28, 28)

        max_c = max(img.shape[0] for img in images)
        max_h = max(img.shape[1] for img in images)
        max_w = max(img.shape[2] for img in images)

        padded = []
        for img in images:
            c, h, w = img.shape
            pad = torch.zeros(max_c, max_h, max_w, dtype=img.dtype)
            pad[:c, :h, :w] = img
            padded.append(pad)
        return torch.stack(padded)

    def _collate_audio(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Аудио: паддим до самой длинной записи."""
        x_tensors = [item["x"] for item in batch]
        y_items = [item["y"] for item in batch]

        max_samples = max(x.shape[-1] for x in x_tensors)
        max_channels = max(x.shape[0] for x in x_tensors)

        x_padded = []
        for x in x_tensors:
            c, s = x.shape
            pad = torch.zeros(max_channels, max_samples, dtype=x.dtype)
            pad[:c, :s] = x
            x_padded.append(pad)
        x_batch = torch.stack(x_padded)

        try:
            y_batch = torch.stack(y_items)
        except (RuntimeError, TypeError):
            y_batch = torch.tensor([y.item() if isinstance(y, torch.Tensor) else y
                                    for y in y_items])

        return {"x": x_batch, "y": y_batch}

    def _collate_graph(self, batch: List[Dict]) -> Dict[str, Any]:
        """
        Графы: батчинг с объединением узлов и рёбер.
        Возвращает единый граф с батч-индексом.
        """
        all_nodes = []
        all_edges = []
        all_labels = []
        batch_indices = []
        node_offset = 0

        for i, item in enumerate(batch):
            graph = item["x"]
            nf = graph["node_features"]
            ei = graph["edge_index"]

            num_nodes = nf.shape[0]
            all_nodes.append(nf)

            if ei.numel() > 0:
                all_edges.append(ei + node_offset)

            batch_indices.extend([i] * num_nodes)
            all_labels.append(item["y"])
            node_offset += num_nodes

        if all_nodes:
            node_features = torch.cat(all_nodes, dim=0)
        else:
            node_features = torch.zeros(1, 1)

        if all_edges:
            edge_index = torch.cat(all_edges, dim=1)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)

        try:
            labels = torch.stack(all_labels)
        except (RuntimeError, TypeError):
            labels = torch.tensor(all_labels)

        return {
            "x": {
                "node_features": node_features,
                "edge_index": edge_index,
                "batch": torch.tensor(batch_indices, dtype=torch.long),
            },
            "y": labels,
        }

    def _collate_time_series(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Временные ряды: паддим по длине последовательности."""
        x_tensors = [item["x"] for item in batch]
        y_tensors = [item["y"] for item in batch]

        max_len = max(x.shape[0] for x in x_tensors)
        features = x_tensors[0].shape[1] if x_tensors[0].ndim > 1 else 1

        x_padded = []
        for x in x_tensors:
            if x.ndim == 1:
                x = x.unsqueeze(1)
            seq_len, feat = x.shape
            pad = torch.zeros(max_len, feat, dtype=x.dtype)
            pad[:seq_len, :] = x
            x_padded.append(pad)
        x_batch = torch.stack(x_padded)

        try:
            y_batch = torch.stack(y_tensors)
        except RuntimeError:
            max_y = max(y.shape[0] for y in y_tensors)
            y_list = []
            for y in y_tensors:
                if y.shape[0] < max_y:
                    y = torch.cat([y, torch.zeros(max_y - y.shape[0])])
                y_list.append(y)
            y_batch = torch.stack(y_list)

        return {"x": x_batch, "y": y_batch}

    def _collate_multimodal(self, batch: List[Dict]) -> Dict[str, Any]:
        """Мультимодальные: группируем по ключам."""
        if not batch:
            return {"x": {}, "y": torch.tensor([])}

        first_x = batch[0]["x"]
        result_x = {}

        if isinstance(first_x, dict):
            for key in first_x.keys():
                items = [item["x"][key] for item in batch]
                if all(isinstance(t, torch.Tensor) for t in items):
                    try:
                        result_x[key] = torch.stack(items)
                    except RuntimeError:
                        if items[0].dtype == torch.long:
                            result_x[key] = pad_sequence(
                                items, batch_first=True, padding_value=0
                            )
                        else:
                            result_x[key] = items
                else:
                    result_x[key] = items
        else:
            try:
                result_x = torch.stack([item["x"] for item in batch])
            except RuntimeError:
                result_x = [item["x"] for item in batch]

        y_items = [item["y"] for item in batch]
        try:
            y_batch = torch.stack(y_items)
        except (RuntimeError, TypeError):
            y_batch = y_items

        return {"x": result_x, "y": y_batch}

    # --------------------------------------------------------
    #  УТИЛИТЫ
    # --------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        """Размер словаря (для текстовых данных)."""
        return len(self.vocab)

    @property
    def num_samples(self) -> int:
        """Количество элементов в данном сплите."""
        return len(self)

    def get_info(self) -> Dict[str, Any]:
        """Информация о датасете для отладки и GUI."""
        return {
            "split": self.split,
            "data_type": self.data_type.value,
            "task_type": self.meta.task_type.value,
            "num_samples": len(self),
            "vocab_size": self.vocab_size if self.data_type == DataType.TEXT else None,
            "image_shape": self._image_shape,
            "audio_sample_rate": self._audio_sr,
            "has_normalization": self._norm_mean is not None,
        }


# ============================================================
#  ФАБРИКА ДЛЯ СОЗДАНИЯ ЗАГРУЗЧИКОВ
# ============================================================

def create_data_loaders(
    data,
    batch_size: int = 64,
    shuffle_train: bool = True,
) -> Tuple:
    """
    Удобная функция: создаёт train/val/test DataLoader из одного источника.

    Args:
        data: DatasetContainer (или dict для обратной совместимости)
        batch_size: размер батча
        shuffle_train: перемешивать ли обучающую выборку

    Returns:
        (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader

    # Приводим к контейнеру
    container = _ensure_container(data)

    train_ds = OracleTorchDataset(container, split="train")
    val_ds = OracleTorchDataset(container, split="val")
    test_ds = OracleTorchDataset(container, split="test")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=shuffle_train, collate_fn=train_ds.collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, collate_fn=val_ds.collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, collate_fn=test_ds.collate_fn,
    )
    return train_loader, val_loader, test_loader