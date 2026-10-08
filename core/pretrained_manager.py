"""
core/pretrained_manager.py
==========================
Менеджер предобученных моделей OracleAI Studio v2.0.

ОТВЕТСТВЕННОСТЬ:
  • Загрузка манифеста доступных моделей (data/shared/pretrained_models/manifest.json).
  • Определение установленных моделей (data/models/installed_models.json).
  • Проверка целостности (sha256).
  • Загрузка моделей через ModelFactory (проверяет совместимость).
  • Удаление моделей.

БЕЗ ИНТЕРНЕТА:
  Программа не падает и не требует сеть. Если манифеста нет — создаётся
  встроенный список. Скачивание работает через requests, при его отсутствии
  или ошибке сети возвращается False без исключений.

ЗАВИСИМОСТИ:
  • core/pretrained_model_info.py → PretrainedModelInfo
  • core/model_factory.py        → ModelFactory
  • config.py                    → PRETRAINED_MODELS_DIR, INSTALLED_MODELS_FILE
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.pretrained_model_info import PretrainedModelInfo
from core.logger import get_logger
from config import PRETRAINED_MODELS_DIR, INSTALLED_MODELS_FILE

logger = get_logger(__name__)

MANIFEST_FILE_NAME = "manifest.json"

# ============================================================
# ВСТРОЕННЫЙ МАНИФЕСТ (если файла нет)
# ============================================================
DEFAULT_MANIFEST = {
    "format_version": "2.0",
    "models": [
        {
            "id": "tiny_transformer_math",
            "name": "Маленький Трансформер для математики",
            "description": "Умеет складывать и вычитать числа до 100.",
            "data_type": "text",
            "architecture": "transformer_seq2seq",
            "params_count": 150000,
            "size_mb": 0.6,
            "source_url": "",
            "sha256_hash": "",
            "category": "text",
        },
        {
            "id": "tiny_rnn_math",
            "name": "RNN-калькулятор (мини)",
            "description": "Складывает однозначные числа. Обучается за минуты.",
            "data_type": "text",
            "architecture": "rnn",
            "params_count": 80000,
            "size_mb": 0.3,
            "source_url": "",
            "sha256_hash": "",
            "category": "text",
        },
        {
            "id": "image_classifier_cnn",
            "name": "Классификатор изображений (мини)",
            "description": "Распознаёт 10 классов: цифры и простые фигуры.",
            "data_type": "image",
            "architecture": "cnn",
            "params_count": 200000,
            "size_mb": 0.8,
            "source_url": "",
            "sha256_hash": "",
            "category": "image",
        },
        {
            "id": "melody_generator",
            "name": "Генератор мелодий (мини)",
            "description": "Генерирует простые мелодии по последовательности нот.",
            "data_type": "audio",
            "architecture": "lstm",
            "params_count": 120000,
            "size_mb": 0.5,
            "source_url": "",
            "sha256_hash": "",
            "category": "music",
        },
        {
            "id": "translator_ru_en_tiny",
            "name": "Переводчик RU↔EN (мини)",
            "description": "Переводит простые слова и короткие фразы.",
            "data_type": "text",
            "architecture": "transformer_seq2seq",
            "params_count": 400000,
            "size_mb": 1.6,
            "source_url": "",
            "sha256_hash": "",
            "category": "translation",
        },
    ],
}


class PretrainedManager:
    """Управление предобученными моделями."""

    def __init__(self, shared_dir: Path = PRETRAINED_MODELS_DIR,
                 installed_file: Path = INSTALLED_MODELS_FILE):
        self.shared_dir = Path(shared_dir)
        self.installed_file = Path(installed_file)
        self.manifest: Dict[str, Any] = {}
        self.models: Dict[str, PretrainedModelInfo] = {}

        self.shared_dir.mkdir(parents=True, exist_ok=True)
        self.installed_file.parent.mkdir(parents=True, exist_ok=True)

        self.load_manifest()
        self.refresh_installed()

    # ============================================================
    # МАНИФЕСТ
    # ============================================================
    def load_manifest(self) -> Dict[str, Any]:
        """Загружает манифест. Если файла нет — использует встроенный."""
        manifest_path = self.shared_dir / MANIFEST_FILE_NAME
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    self.manifest = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Манифест повреждён ({e}). Используется встроенный.")
                self.manifest = DEFAULT_MANIFEST
        else:
            self.manifest = DEFAULT_MANIFEST
            self.save_manifest()  # создаём файл для редактирования

        raw_models = self.manifest.get("models", [])
        for item in raw_models:
            info = PretrainedModelInfo.from_dict(item)
            self.models[info.model_id] = info
        logger.info(f"Манифест загружен: {len(self.models)} моделей")
        return self.manifest

    def save_manifest(self) -> bool:
        path = self.shared_dir / MANIFEST_FILE_NAME
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.manifest, f, ensure_ascii=False, indent=2)
            return True
        except OSError as e:
            logger.error(f"Не удалось сохранить манифест: {e}")
            return False

    # ============================================================
    # УСТАНОВЛЕННЫЕ МОДЕЛИ
    # ============================================================
    def _load_installed(self) -> Dict[str, Dict[str, Any]]:
        if self.installed_file.exists():
            try:
                with open(self.installed_file, "r", encoding="utf-8") as f:
                    return json.load(f).get("models", {})
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_installed(self, installed: Dict[str, Dict[str, Any]]):
        data = {"format_version": "2.0", "models": installed}
        tmp = self.installed_file.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        if self.installed_file.exists():
            self.installed_file.unlink()
        tmp.rename(self.installed_file)

    def refresh_installed(self):
        """Обновляет статус installed у моделей по файлам на диске."""
        installed = self._load_installed()
        for model_id, info in self.models.items():
            path = self._model_path(model_id)
            exists = path.exists() and path.stat().st_size > 0
            entry = installed.get(model_id, {})
            info.installed = bool(exists or entry.get("installed", False))
            info.local_path = str(path) if exists else entry.get("local_path", "")

    # ============================================================
    # СПИСКИ
    # ============================================================
    def get_available_models(self) -> List[PretrainedModelInfo]:
        return [m for m in self.models.values() if not m.installed]

    def get_installed_models(self) -> List[PretrainedModelInfo]:
        return [m for m in self.models.values() if m.installed]

    def get_all_models(self) -> List[PretrainedModelInfo]:
        return list(self.models.values())

    def get_model(self, model_id: str) -> Optional[PretrainedModelInfo]:
        return self.models.get(model_id)

    def check_model_exists(self, model_id: str) -> bool:
        info = self.models.get(model_id)
        if info is None:
            return False
        return self._model_path(model_id).exists()

    def get_model_size(self, model_id: str) -> int:
        """Размер файла модели в МБ (0 если нет)."""
        path = self._model_path(model_id)
        if path.exists():
            return int(path.stat().st_size / (1024 * 1024))
        info = self.models.get(model_id)
        return int(info.size_mb) if info else 0

    def _model_path(self, model_id: str) -> Path:
        return self.shared_dir / f"{model_id}.pth"

    # ============================================================
    # СКАЧИВАНИЕ
    # ============================================================
    def download_model(self, model_id: str,
                       progress_callback: Optional[Callable[[int, int], None]] = None,
                       source_url: str = "") -> bool:
        """
        Скачивает модель. Если URL пуст или сети нет — создаёт заглушку
        (встроенную мини-модель из ModelFactory), чтобы пайплайн работал
        офлайн. Возвращает True при успехе.
        """
        info = self.models.get(model_id)
        if info is None:
            logger.error(f"Модель '{model_id}' не найдена в манифесте")
            return False
        if self.check_model_exists(model_id):
            logger.info(f"Модель '{model_id}' уже установлена")
            info.installed = True
            return True

        target = self._model_path(model_id)
        url = source_url or info.source_url

        # Попытка скачивания (если есть URL и requests)
        if url:
            ok = self._download_via_http(url, target, info.size_mb, progress_callback)
            if ok:
                info.installed = True
                self._record_installed(info)
                return True

        # Офлайн-заглушка: создаём реальную мини-модель через ModelFactory
        ok = self._create_stub_model(info, target)
        if ok:
            info.installed = True
            self._record_installed(info)
        return ok

    def _download_via_http(self, url: str, target: Path, size_mb: float,
                           progress_callback) -> bool:
        try:
            import requests
        except ImportError:
            logger.warning("requests не установлен. Скачивание невозможно.")
            return False
        try:
            response = requests.get(url, stream=True, timeout=30)
            if response.status_code != 200:
                logger.warning(f"HTTP {response.status_code} при скачивании {url}")
                return False
            total = int(response.headers.get("content-length", int(size_mb * 1024 * 1024)))
            downloaded = 0
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as f:
                for chunk in response.iter_content(chunk_size=65536):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        try:
                            progress_callback(downloaded, total)
                        except Exception:
                            pass
            return True
        except Exception as e:
            logger.warning(f"Ошибка скачивания {url}: {e}")
            return False

    def _create_stub_model(self, info: PretrainedModelInfo, target: Path) -> bool:
        """Создаёт обучаемую мини-модель под описание из манифеста."""
        try:
            from core.model_factory import ModelFactory
            from core.contracts import ModelConfig, TaskType

            task_type = TaskType.SEQ2SEQ
            if info.data_type.value in ("image", "audio"):
                task_type = TaskType.CLASSIFICATION
            elif info.architecture.value in ("mlp", "rbfn", "som", "pinn"):
                task_type = TaskType.REGRESSION

            config = ModelConfig(
                model_name=info.name,
                model_id=info.model_id,
                architecture=info.architecture,
                data_type=info.data_type,
                task_type=task_type,
                vocab_size=64,
                num_classes=10,
                input_dim=8,
                output_dim=1,
                image_shape=(16, 16, 1) if info.data_type.value == "image" else None,
                arch_params={
                    "embedding_dim": 32,
                    "num_heads": 4,
                    "num_encoder_layers": 1,
                    "num_decoder_layers": 1,
                    "dim_feedforward": 128,
                    "hidden_layers": [32, 32],
                },
            )
            model = ModelFactory.create_model(config)
            # Сохраняем как обычный чекпоинт
            ModelFactory.save_model(
                model, target, config,
                metadata={"source": "offline_stub", "pretrained_id": info.model_id},
            )
            logger.info(f"Создана офлайн-заглушка модели '{info.model_id}'")
            return True
        except Exception as e:
            logger.error(f"Не удалось создать заглушку '{info.model_id}': {e}")
            return False

    def _record_installed(self, info: PretrainedModelInfo):
        installed = self._load_installed()
        installed[info.model_id] = {
            "installed": True,
            "installed_at": __import__("datetime").datetime.now().isoformat(),
            "local_path": str(self._model_path(info.model_id)),
            "size_mb": self.get_model_size(info.model_id),
        }
        self._save_installed(installed)

    # ============================================================
    # ЗАГРУЗКА / ПРОВЕРКА / УДАЛЕНИЕ
    # ============================================================
    def load_model(self, model_id: str):
        """
        Загружает модель через ModelFactory.
        Возвращает (model, config, extra) или бросает FileNotFoundError.
        """
        info = self.models.get(model_id)
        if info is None:
            raise FileNotFoundError(f"Модель '{model_id}' не в манифесте")
        path = self._model_path(model_id)
        if not path.exists():
            raise FileNotFoundError(
                f"Модель '{model_id}' не установлена. Скачайте её."
            )
        from core.model_factory import ModelFactory
        model, config, extra = ModelFactory.load_model(path)
        return model, config, extra

    def verify_integrity(self, model_id: str) -> bool:
        """Проверяет sha256 модель, если хеш указан в манифесте."""
        info = self.models.get(model_id)
        if info is None:
            return False
        if not info.sha256_hash:
            return True  # хеша нет — нечего проверять
        path = self._model_path(model_id)
        if not path.exists():
            return False
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest() == info.sha256_hash

    def remove_model(self, model_id: str) -> bool:
        """Удаляет файл модели и запись об установке."""
        info = self.models.get(model_id)
        if info is None:
            return False
        path = self._model_path(model_id)
        removed = False
        if path.exists():
            try:
                path.unlink()
                removed = True
            except OSError as e:
                logger.error(f"Не удалось удалить {path}: {e}")
        installed = self._load_installed()
        installed.pop(model_id, None)
        self._save_installed(installed)
        info.installed = False
        info.local_path = ""
        return removed

    # ============================================================
    # СЛУЖЕБНОЕ
    # ============================================================
    def total_installed_size_mb(self) -> float:
        total = 0.0
        for info in self.get_installed_models():
            total += self.get_model_size(info.model_id)
        return total

    def status_summary(self) -> dict:
        return {
            "available": len(self.get_available_models()),
            "installed": len(self.get_installed_models()),
            "total_size_mb": self.total_installed_size_mb(),
        }
