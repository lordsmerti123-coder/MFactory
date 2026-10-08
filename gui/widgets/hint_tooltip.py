"""
gui/widgets/hint_tooltip.py
==============================
Подключается к любому элементу и показывает контекстную подсказку.
Работает через core/hint_engine.py (если доступен) или статические подсказки.
Не создаёт ключей в shared_state.

4 режима: "static" | "ai" | "hybrid" | "none"
"""

from typing import Optional, Dict, Any
from PyQt5.QtWidgets import QWidget, QToolTip, QApplication
from PyQt5.QtCore import Qt, QPoint, QEvent, QTimer
from PyQt5.QtGui import QHelpEvent


# Статические подсказки для типовых элементов (fallback если hint_engine недоступен)
DEFAULT_HINTS = {
    # Генератор
    "generator.task_combo": "Выбери тип задачи. От этого зависит, чему будет учиться нейросеть.",
    "generator.samples_spin": "Количество примеров. Для простых задач хватит 1000, для сложных нужно 20000+.",
    "generator.generate_btn": "Создать файл с парами «Вопрос-Ответ» для обучения.",

    # Данные
    "dataset.load_btn": "Загрузить файл с данными (.oai, .csv, .json).",
    "dataset.train_spin": "Доля данных для обучения. 80% — стандарт.",
    "dataset.val_spin": "Доля для проверки во время обучения. 10% — стандарт.",
    "dataset.test_spin": "Доля для финального экзамена. 10% — стандарт.",
    "dataset.vocab_limit_check": "Оставить только N самых частых символов. Редкие станут <UNK>.",
    "dataset.split_btn": "Разрезать данные на 3 части и построить словарь. Без этого обучение не запустится.",

    # Архитектура
    "arch.model_type": "Тип нейросети. Определяет, как модель обрабатывает данные.",
    "arch.tf_embed": "Размер вектора для каждого символа. 64 — для простых задач, 256 — для сложных.",
    "arch.tf_heads": "Сколько независимых потоков внимания ищут закономерности. 4–8 — стандарт.",
    "arch.dropout": "Доля нейронов, случайно отключаемых при обучении. Защита от зубрёжки. 0.1–0.3 — стандарт.",
    "arch.max_len": "Максимальная длина последовательности. Всё длиннее — обрезается.",

    # Гиперпараметры
    "params.lr": "Скорость обучения. Как сильно сеть корректирует веса после каждой ошибки. Начни с 0.001.",
    "params.epochs": "Сколько раз сеть просмотрит все данные. Больше = дольше, но точнее.",
    "params.batch_size": "Сколько примеров сеть видит за один шаг. 64 — стандарт.",
    "params.weight_decay": "Штраф за зубрёжку (L2). Не даёт сети запоминать примеры наизусть.",
    "params.teacher_forcing": "Подсказки учителя при генерации. 1.0 = всегда правильный ответ. 0.0 = всегда свой.",
    "params.clip": "Защита от взрыва градиента. Не даёт ошибке стать бесконечной.",

    # Обучение
    "training.start_btn": "Запустить процесс обучения модели.",
    "training.pause_btn": "Приостановить обучение (можно продолжить).",
    "training.stop_btn": "Остановить обучение. Лучшие веса уже сохранены.",
    "training.save_btn": "Сохранить веса модели в файл .pth.",
    "training.load_btn": "Загрузить ранее сохранённые веса.",

    # Мониторинг
    "monitoring.plot_loss": "График ошибки. Чем ниже — тем лучше. Синий = учёба, оранжевый = экзамен.",
    "monitoring.plot_metric": "График качества. Для текста — точность (выше = лучше).",

    # Анализ
    "analysis.test_input": "Введи свой пример, чтобы проверить чему научилась сеть.",
    "analysis.predict_btn": "Отправить пример модели и получить ответ.",
}


class HintTooltip:
    """
    Универсальный подключатель подсказок к виджетам.

    Не является QWidget — это хелпер, который перехватывает события.

    Использование:
        # В панели, после создания виджета:
        HintTooltip.attach(my_button, "training.start_btn", hint_mode_getter)

    Где hint_mode_getter — callable, возвращающий текущий режим ("static"|"ai"|"hybrid"|"none")
    """

    _instances = []  # предотвращаем GC

    @classmethod
    def attach(
        cls,
        widget: QWidget,
        hint_id: str,
        mode_getter=None,
        extra_context: Optional[Dict] = None,
    ):
        """
        Подключает подсказку к виджету.

        Аргументы:
            widget: виджет, к которому подключаем
            hint_id: идентификатор подсказки (ключ в словаре)
            mode_getter: функция, возвращающая текущий режим подсказок
            extra_context: дополнительный контекст для динамических подсказок
        """
        instance = cls(widget, hint_id, mode_getter, extra_context)
        cls._instances.append(instance)
        return instance

    def __init__(self, widget, hint_id, mode_getter, extra_context):
        self.widget = widget
        self.hint_id = hint_id
        self.mode_getter = mode_getter
        self.extra_context = extra_context or {}

        # Устанавливаем базовый tooltip
        hint_text = self._get_hint()
        if hint_text:
            widget.setToolTip(hint_text)

        # Перехватываем событие изменения для обновления подсказки
        widget.installEventFilter(self)

    def eventFilter(self, obj, event):
        """Обновляем подсказку при наведении (для динамических подсказок)."""
        if event.type() == QEvent.ToolTip and obj is self.widget:
            hint_text = self._get_hint()
            if hint_text:
                QToolTip.showText(event.globalPos(), hint_text, self.widget)
                return True
        return False

    def _get_hint(self) -> str:
        """Получает подсказку с учётом режима."""
        mode = "hybrid"
        if self.mode_getter and callable(self.mode_getter):
            try:
                mode = self.mode_getter()
            except Exception:
                mode = "hybrid"

        if mode == "none":
            return ""

        if mode == "ai":
            # Пытаемся получить ИИ-подсказку (заглушка)
            ai_hint = self._try_ai_hint()
            if ai_hint:
                return f"🤖 {ai_hint}"
            return ""

        if mode == "hybrid":
            ai_hint = self._try_ai_hint()
            if ai_hint:
                return f"🤖 {ai_hint}\n\n(Автоматическая подсказка)"
            # Фолбэк на статическую
            return self._static_hint()

        # mode == "static"
        return self._static_hint()

    def _static_hint(self) -> str:
        """Статическая подсказка из словаря."""
        # Пытаемся импортировать из hint_engine (если доступен)
        try:
            from core.hint_engine import HintEngine
            # Если есть полноценный движок — используем его
            return DEFAULT_HINTS.get(self.hint_id, "")
        except ImportError:
            pass
        return DEFAULT_HINTS.get(self.hint_id, "")

    def _try_ai_hint(self) -> str:
        """Заглушка: попытка получить подсказку от ИИ."""
        # В будущем здесь будет вызов локальной модели
        # Пока возвращаем пустую строку (ИИ недоступен)
        return ""

    def refresh(self):
        """Принудительно обновляет подсказку."""
        hint_text = self._get_hint()
        if hint_text:
            self.widget.setToolTip(hint_text)
        else:
            self.widget.setToolTip("")

    @classmethod
    def refresh_all(cls):
        """Обновляет все подсказки (при смене режима)."""
        for instance in cls._instances:
            instance.refresh()

    @classmethod
    def clear_all(cls):
        """Очищает все подключённые подсказки."""
        cls._instances.clear()