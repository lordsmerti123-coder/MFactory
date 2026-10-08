"""
main.py
=======
Точка входа в OracleAI Studio v2.0.

Отвечает ТОЛЬКО за:
1. Настройку QApplication (High DPI, шрифты, имя приложения)
2. Логирование версий окружения
3. Создание главного окна
4. Аварийное завершение с понятным сообщением

НЕ содержит бизнес-логики. Вся работа — в gui.main_window.MainWindow.
"""

import os
import sys
import traceback

# ============================================================
# HighDPI: настройка ДО импорта PyQt5 и создания QApplication.
#
# ПРОБЛЕМА: на 2K-экранах Qt 5.15 с AA_EnableHighDpiScaling может
# неверно определить масштаб (например dpr=2.0 при реальном 1.5),
# из-за чего интерфейс перемасштабируется и «плывёт».
#
# РЕШЕНИЕ: детерминированный QT_SCALE_FACTOR от физической высоты
# экрана (нормализация к 1080p). Все шрифты, виджеты и px-стили
# масштабируются равномерно, а логические координаты сохраняются —
# разметка не ломается.
# ============================================================

from config import (
    __version__,
    APP_NAME,
    ORG_NAME,
    UI_REFERENCE_HEIGHT,
    UI_SCALE_MIN,
    UI_SCALE_MAX,
)


def _compute_ui_scale() -> float:
    """
    Вычисляет масштаб интерфейса по физической высоте экрана.
    На 1080p даёт 1.0, на 2K ~1.33–1.5, на 4K — не больше UI_SCALE_MAX.
    Возвращает 1.0 на не-Windows платформах.
    """
    if sys.platform != "win32":
        return 1.0
    try:
        import ctypes
        # Делаем процесс DPI-aware, чтобы GetSystemMetrics вернул
        # ФИЗИЧЕСКИЕ координаты (не виртуализированные ОС).
        ctypes.windll.user32.SetProcessDPIAware()
        phys_h = ctypes.windll.user32.GetSystemMetrics(1)  # SM_CYSCREEN
        if phys_h <= 0:
            return 1.0
        return max(UI_SCALE_MIN, min(phys_h / UI_REFERENCE_HEIGHT, UI_SCALE_MAX))
    except Exception:
        return 1.0


# Масштаб интерфейса (должен быть задан ДО создания QApplication).
UI_SCALE = _compute_ui_scale()
if UI_SCALE > 1.0:
    os.environ["QT_SCALE_FACTOR"] = f"{UI_SCALE:.3f}"
os.environ.setdefault("QT_SCALE_FACTOR_ROUNDING_POLICY", "PassThrough")


# ============================================================
# ВАЖНО: torch импортируется строго ДО любых модулей PyQt5,
# чтобы избежать конфликтов event-loop на некоторых системах.
# ============================================================
import torch
import numpy as np

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import Qt

from core.logger import get_logger
from gui.main_window import MainWindow
from gui.activity_filter import install_activity_filter


def check_environment(logger) -> list[str]:
    """
    Проверяет минимальную пригодность окружения.
    Возвращает список предупреждений (пустой = всё ок).
    Не блокирует запуск, только логирует.
    """
    warnings = []

    if not torch.cuda.is_available():
        warnings.append(
            "CUDA недоступна — обучение будет на CPU (медленнее, но работает)."
        )

    major, minor, *_ = sys.version_info
    if (major, minor) < (3, 9):
        warnings.append(
            f"Python {major}.{minor} может быть слишком старым. "
            "Рекомендуется 3.9+."
        )

    for w in warnings:
        logger.warning(w)

    return warnings


def _get_gpu_memory_gb(device_id: int = 0) -> float:
    """
    Безопасно получает объём памяти GPU в гигабайтах.
    Защищено от изменений API PyTorch между версиями.
    Возвращает 0.0, если узнать не удалось.
    """
    try:
        props = torch.cuda.get_device_properties(device_id)
        # В PyTorch 2.x правильный атрибут — total_memory (байты).
        # Поддержка и старого (total_mem), и нового имён.
        mem_bytes = getattr(props, "total_memory", None)
        if mem_bytes is None:
            mem_bytes = getattr(props, "total_mem", 0)
        return float(mem_bytes) / 1e9
    except Exception:
        return 0.0


def main() -> int:
    """Инициализация приложения. Возвращает код выхода."""

    # === 1. Высокое разрешение (строго до QApplication) ===
    if sys.platform == "win32" and hasattr(Qt, "AA_DisableHighDpiScaling"):
        # Отключаем авто-детекцию DPI Qt (неверна на 2K) — масштаб задан
        # детерминированно через QT_SCALE_FACTOR (см. начало файла).
        QApplication.setAttribute(Qt.AA_DisableHighDpiScaling, True)
    else:
        if hasattr(Qt, "AA_EnableHighDpiScaling"):
            QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        if hasattr(Qt, "AA_UseHighDpiPixmaps"):
            QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    app.setApplicationVersion(__version__)

    # Глобальное логирование активности пользователя (мышь/клавиши) для
    # панели администратора. Устанавливается ДО создания главного окна.
    try:
        install_activity_filter(app)
    except Exception as e:
        print(f"Не удалось установить фильтр активности: {e}")

    # Базовый читаемый шрифт для всего приложения.
    # При QT_SCALE_FACTOR размер шрифта масштабируется автоматически.
    font = app.font()
    font.setPointSize(11)
    app.setFont(font)

    # === 2. Логирование старта ===
    logger = get_logger()
    logger.info("=" * 60)
    logger.info(f"  {APP_NAME} v{__version__}")
    logger.info("=" * 60)
    logger.info(f"Python      : {sys.version.split()[0]}")
    logger.info(f"PyTorch     : {torch.__version__}")
    logger.info(f"NumPy       : {np.__version__}")
    logger.info(f"CUDA avail. : {torch.cuda.is_available()}")
    logger.info(f"UI scale    : {UI_SCALE:.3f} (QT_SCALE_FACTOR)")

    if torch.cuda.is_available():
        try:
            logger.info(f"CUDA device : {torch.cuda.get_device_name(0)}")
            mem_gb = _get_gpu_memory_gb(0)
            if mem_gb > 0:
                logger.info(f"GPU memory  : {mem_gb:.1f} GB")
            else:
                logger.info("GPU memory  : unknown")
        except Exception as e:
            logger.warning(f"Не удалось получить свойства GPU: {e}")
    logger.info("-" * 60)

    env_warnings = check_environment(logger)

    # === 3. Создание главного окна ===
    try:
        window = MainWindow()

        # Если были предупреждения окружения — мягко сообщаем
        if env_warnings:
            msg = "\n".join(f"• {w}" for w in env_warnings)
            logger.info("Показываю предупреждения окружения пользователю.")
            QMessageBox.information(
                window,
                "Предупреждение окружения",
                f"Программа запущена, но есть замечания:\n\n{msg}",
            )

        window.show()
        return app.exec_()

    except Exception:
        error_text = traceback.format_exc()
        logger.critical("Критическая ошибка при запуске:\n" + error_text)

        # Если окно ещё не создано, показываем аварийный диалог
        QMessageBox.critical(
            None,
            "Авария при запуске",
            "Не удалось инициализировать приложение.\n\n"
            "Подробности в логе: data/logs/oracleai_latest.log\n\n"
            f"{error_text[:2000]}",
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())