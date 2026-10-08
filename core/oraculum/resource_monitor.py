"""
core/oraculum/resource_monitor.py
=================================
Контроль ресурсов для Оракула.

ОТВЕТСТВЕННОСТЬ:
  • Проверка доступности обработки (RAM/CPU).
  • Контроль загрузки локальных моделей.
  • Ограничение частоты вызовов внешних API.
  • Отчёт об использовании ресурсов.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import (
    ORACULUM_MAX_RAM_MB,
    ORACULUM_MAX_MODEL_SIZE_MB,
    ORACULUM_MAX_CPU_PERCENT,
    ORACULUM_API_CALLS_PER_MINUTE,
    ORACULUM_MAX_LLM_MODEL_SIZE_MB,
)
from core.logger import get_logger

logger = get_logger(__name__)


def _get_psutil():
    """Безопасный импорт psutil (может отсутствовать)."""
    try:
        import psutil
        return psutil
    except ImportError:
        return None


class ResourceMonitor:
    """Следит за ресурсами и защищает систему от перегрузки."""

    def __init__(self, limits: Optional[Dict[str, Any]] = None):
        self.limits = {
            "max_ram_mb": ORACULUM_MAX_RAM_MB,
            "max_model_ram_mb": ORACULUM_MAX_MODEL_SIZE_MB,
            "max_cpu_percent": ORACULUM_MAX_CPU_PERCENT,
            "api_calls_per_minute": ORACULUM_API_CALLS_PER_MINUTE,
            "max_llm_model_ram_mb": ORACULUM_MAX_LLM_MODEL_SIZE_MB,
        }
        if limits:
            self.limits.update({k: v for k, v in limits.items() if v is not None})
        self._api_calls: List[float] = []
        self._model_loaded = False
        self._model_size_mb = 0.0
        self._psutil = _get_psutil()

    # ============================================================
    # ПРОВЕРКИ
    # ============================================================
    def can_process(self) -> bool:
        """Может ли агент обработать запрос прямо сейчас."""
        if self._psutil is not None:
            try:
                mem = self._psutil.virtual_memory()
                if mem.used / mem.total > 0.9:
                    return False
                cpu = self._psutil.cpu_percent(interval=None)
                if cpu > self.limits["max_cpu_percent"]:
                    return False
            except Exception:
                pass  # при сбое psutil считаем, что ресурсы в порядке
        return True

    def can_load_model(self, model_path: Path) -> bool:
        """Можно ли загрузить локальную модель в память."""
        model_path = Path(model_path)
        if not model_path.exists():
            return False
        size_mb = model_path.stat().st_size / (1024 * 1024)
        if size_mb > self.limits["max_model_ram_mb"]:
            logger.warning(
                f"Модель {size_mb:.1f}МБ больше лимита "
                f"{self.limits['max_model_ram_mb']}МБ"
            )
            return False
        if self._psutil is not None:
            try:
                free_ram = self._psutil.virtual_memory().available / (1024 * 1024)
                if free_ram < size_mb * 2:
                    logger.warning("Недостаточно свободной RAM для модели")
                    return False
            except Exception:
                pass
        self._model_size_mb = size_mb
        return True

    def can_load_llm(self, model_path: Path) -> bool:
        """
        Проверка для больших GGUF/LLM-моделей (больше 512 МБ).
        Использует отдельный лимит max_llm_model_ram_mb и проверяет свободную RAM.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            return False
        size_mb = model_path.stat().st_size / (1024 * 1024)
        if size_mb > self.limits.get("max_llm_model_ram_mb", ORACULUM_MAX_LLM_MODEL_SIZE_MB):
            logger.warning(
                f"LLM-модель {size_mb:.1f}МБ больше лимита "
                f"{self.limits.get('max_llm_model_ram_mb', ORACULUM_MAX_LLM_MODEL_SIZE_MB)}МБ"
            )
            return False
        if self._psutil is not None:
            try:
                free_ram = self._psutil.virtual_memory().available / (1024 * 1024)
                if free_ram < size_mb * 1.5:
                    logger.warning("Недостаточно свободной RAM для LLM-модели")
                    return False
            except Exception:
                pass
        self._model_size_mb = size_mb
        return True

    def can_use_api(self) -> bool:
        """Не превышен ли лимит запросов к API за минуту."""
        now = time.time()
        self._api_calls = [t for t in self._api_calls if now - t < 60]
        if len(self._api_calls) >= self.limits["api_calls_per_minute"]:
            return False
        self._api_calls.append(now)
        return True

    # ============================================================
    # УПРАВЛЕНИЕ МОДЕЛЬЮ
    # ============================================================
    def mark_model_loaded(self, size_mb: float = 0.0):
        self._model_loaded = True
        self._model_size_mb = size_mb or self._model_size_mb

    def mark_model_unloaded(self):
        self._model_loaded = False
        self._model_size_mb = 0.0

    def is_model_loaded(self) -> bool:
        return self._model_loaded

    def check_and_unload_if_needed(self, agent=None) -> bool:
        """
        Проверяет память и при превышении 85% — просит агента выгрузить модель.
        Возвращает True, если модель была выгружена.
        """
        if not self._model_loaded:
            return False
        if self._psutil is not None:
            try:
                mem = self._psutil.virtual_memory()
                if mem.used / mem.total > 0.85:
                    if agent is not None and hasattr(agent, "unload_local_model"):
                        agent.unload_local_model()
                        return True
            except Exception:
                pass
        return False

    # ============================================================
    # ОТЧЁТ
    # ============================================================
    def get_usage_report(self) -> Dict[str, Any]:
        mem = ram_total = cpu = 0.0
        if self._psutil is not None:
            try:
                m = self._psutil.virtual_memory()
                mem = m.used / (1024 * 1024)
                ram_total = m.total / (1024 * 1024)
                cpu = self._psutil.cpu_percent(interval=None)
            except Exception:
                pass
        now = time.time()
        return {
            "ram_used_mb": round(mem, 1),
            "ram_total_mb": round(ram_total, 1),
            "cpu_percent": round(cpu, 1),
            "model_loaded": self._model_loaded,
            "model_size_mb": round(self._model_size_mb, 1),
            "api_calls_last_minute": len([t for t in self._api_calls if now - t < 60]),
        }
