"""
core/logger.py
==============
Система логирования OracleAI Studio v2.0.

Ответственность модуля:
  • Консольный вывод (INFO+)
  • Файловый вывод последнего запуска (DEBUG+)
  • Сессионный файловый вывод (история сеансов)
  • Структурированные записи для ИИ-Куратора и Движка подсказок
  • Потокобезопасность (обучение работает в QThread)
  • Подписка на события логов (GUI-панель логов, Куратор, Оркестратор)
  • Обратная совместимость с существующими панелями GUI

Зависимости:
  • config.py  — LOGS_DIR, SESSIONS_DIR, LOG_FORMAT, LOG_DATE_FORMAT
  • НЕ импортирует core/session.py напрямую (избегаем циклических зависимостей).
    Вместо этого предоставляет set_session_context() для инъекции.

Порядок загрузки: после config.py, до core/session.py
"""

import logging
import sys
import json
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field, asdict
from enum import Enum

from config import LOGS_DIR, SESSIONS_DIR, LOG_FORMAT, LOG_DATE_FORMAT


# ============================================================
# КОНТРАКТЫ ЛОГИРОВАНИЯ (используются Куратором и Подсказками)
# ============================================================

class LogCategory(str, Enum):
    """Категория события для структурированной аналитики."""
    SYSTEM = "system"           # Запуск/остановка программы
    DATA = "data"               # Операции с данными
    MODEL = "model"             # Создание/загрузка модели
    TRAINING = "training"       # Процесс обучения
    MONITORING = "monitoring"   # Графики, метрики
    USER_ACTION = "user_action" # Действия пользователя в GUI
    ERROR = "error"             # Ошибки
    WARNING = "warning"         # Предупреждения
    EXPORT = "export"           # Экспорт/сохранение
    HINT = "hint"               # События системы подсказок
    CURATOR = "curator"         # Сообщения куратора
    ORACULUM = "oraculum"       # Действия ИИ-агента «Оракул»


@dataclass
class StructuredLogEntry:
    """
    Структурированная запись лога.
    Куратор и Движок подсказок читают именно эти объекты.
    """
    timestamp: str
    level: str
    category: LogCategory
    module: str
    message: str
    session_id: str = ""
    panel: str = ""
    action: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    numeric_values: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


# ============================================================
# ПОДПИСКА НА СОБЫТИЯ ЛОГОВ
# ============================================================

class LogSubscriber:
    """
    Интерфейс подписчика на логи.
    GUI-панель логов, Куратор, Оркестратор — все подписчики.
    """
    def on_log_entry(self, entry: StructuredLogEntry):
        raise NotImplementedError


class LogEventBus:
    """
    Потокобезопасная шина событий логов.
    Позволяет Куратору и другим модулям получать логи в реальном времени.
    """
    def __init__(self):
        self._subscribers: List[LogSubscriber] = []
        self._lock = threading.Lock()

    def subscribe(self, subscriber: LogSubscriber):
        """Регистрирует подписчика."""
        with self._lock:
            if subscriber not in self._subscribers:
                self._subscribers.append(subscriber)

    def unsubscribe(self, subscriber: LogSubscriber):
        """Удаляет подписчика."""
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

    def publish(self, entry: StructuredLogEntry):
        """Рассылает запись всем подписчикам. Вызывается из хендлера."""
        with self._lock:
            subscribers = list(self._subscribers)
        for sub in subscribers:
            try:
                sub.on_log_entry(entry)
            except Exception:
                pass  # Подписчик не должен ронять логгер


# Глобальная шина событий (единая точка для всего приложения)
log_event_bus = LogEventBus()


# ============================================================
# КОНТЕКСТ СЕАНСА (инъекция из session.py без циклического импорта)
# ============================================================

_session_context: Dict[str, str] = {
    "session_id": "",
    "user_level": "beginner",
}
_session_lock = threading.Lock()


def set_session_context(session_id: str, user_level: str = "beginner"):
    """
    Вызывается из core/session.py при инициализации сеанса.
    НЕ импортируем session.py здесь — избегаем циклической зависимости.
    """
    global _session_context
    with _session_lock:
        _session_context["session_id"] = session_id
        _session_context["user_level"] = user_level


def get_session_context() -> Dict[str, str]:
    with _session_lock:
        return dict(_session_context)


# ============================================================
# КАСТОМНЫЙ LOGGING HANDLER ДЛЯ ШИНЫ СОБЫТИЙ
# ============================================================

class EventBusHandler(logging.Handler):
    """
    Хендлер, который преобразует стандартные LogRecord в StructuredLogEntry
    и публикует в LogEventBus для Куратора и GUI.
    """
    def __init__(self, event_bus: LogEventBus):
        super().__init__()
        self.event_bus = event_bus

    def emit(self, record: logging.LogRecord):
        try:
            entry = self._record_to_structured(record)
            self.event_bus.publish(entry)
        except Exception:
            pass  # Никогда не роняем логгер

    def _record_to_structured(self, record: logging.LogRecord) -> StructuredLogEntry:
        """Преобразует стандартный LogRecord в структурированную запись."""
        # Извлекаем категорию из доп. поля, если есть
        category = getattr(record, "category", LogCategory.SYSTEM)
        if isinstance(category, str):
            try:
                category = LogCategory(category)
            except ValueError:
                category = LogCategory.SYSTEM

        panel = getattr(record, "panel", "")
        action = getattr(record, "action", "")
        context = getattr(record, "context", {})
        numeric_values = getattr(record, "numeric_values", {})

        ctx = get_session_context()

        return StructuredLogEntry(
            timestamp=datetime.fromtimestamp(record.created).strftime(LOG_DATE_FORMAT),
            level=record.levelname,
            category=category,
            module=record.name,
            message=record.getMessage(),
            session_id=ctx.get("session_id", ""),
            panel=panel,
            action=action,
            context=context if isinstance(context, dict) else {},
            numeric_values=numeric_values if isinstance(numeric_values, dict) else {},
        )


# ============================================================
# СЕССИОННЫЙ ФАЙЛОВЫЙ ХЕНДЛЕР
# ============================================================

class SessionFileHandler(logging.FileHandler):
    """
    Файловый хендлер, привязанный к конкретному сеансу.
    Файл: data/sessions/session_{id}_{date}.log
    """
    def __init__(self, session_id: str):
        self._session_id = session_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{session_id}_{timestamp}.log"
        filepath = SESSIONS_DIR / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(filepath, mode="a", encoding="utf-8")


# ============================================================
# РЕЕСТР ЛОГГЕРОВ И ИНИЦИАЛИЗАЦИЯ
# ============================================================

_loggers: Dict[str, logging.Logger] = {}
_loggers_lock = threading.Lock()
_initialized = False
_init_lock = threading.Lock()


def _ensure_initialized():
    """Однократная инициализация глобальной инфраструктуры логирования."""
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        _initialized = True


def setup_logger(name: str) -> logging.Logger:
    """
    Создаёт или возвращает существующий логгер.

    Хендлеры:
      1. Консоль (INFO+)
      2. Файл последнего запуска (DEBUG+)
      3. Шина событий (все уровни) — для Куратора и GUI

    Сессионный файловый хендлер добавляется отдельно через
    attach_session_handler() после инициализации сеанса.
    """
    _ensure_initialized()

    # Защита консоли от UnicodeEncodeError при выводе эмодзи на cp1251-терминал.
    # Без этого logging печатает «--- Logging error ---» при каждом эмодзи.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="replace")
        except Exception:
            pass

    with _loggers_lock:
        if name in _loggers:
            return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Очищаем старые хендлеры (защита от дублирования при перезагрузке)
    if logger.handlers:
        logger.handlers.clear()

    # --- 1. Консольный вывод ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    logger.addHandler(console_handler)

    # --- 2. Файл последнего запуска (перезаписывается каждый запуск) ---
    latest_log = LOGS_DIR / "oracleai_latest.log"
    file_handler = logging.FileHandler(latest_log, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    logger.addHandler(file_handler)

    # --- 3. Шина событий (для Куратора, Подсказок, GUI) ---
    event_handler = EventBusHandler(log_event_bus)
    event_handler.setLevel(logging.DEBUG)
    logger.addHandler(event_handler)

    # Не пробрасываем в root-логгер (избегаем дублирования)
    logger.propagate = False

    with _loggers_lock:
        _loggers[name] = logger

    return logger


def attach_session_handler(session_id: str):
    """
    Добавляет сессионный файловый хендлер ко ВСЕМ активным логгерам.
    Вызывается из core/session.py после создания/загрузки сеанса.
    """
    _ensure_initialized()
    session_handler = SessionFileHandler(session_id)
    session_handler.setLevel(logging.DEBUG)
    session_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))

    with _loggers_lock:
        for logger in _loggers.values():
            # Проверяем, не добавлен ли уже сессионный хендлер
            has_session = any(
                isinstance(h, SessionFileHandler) for h in logger.handlers
            )
            if not has_session:
                logger.addHandler(session_handler)


def detach_session_handlers():
    """
    Удаляет все сессионные хендлеры (при завершении сеанса).
    """
    with _loggers_lock:
        for logger in _loggers.values():
            logger.handlers = [
                h for h in logger.handlers
                if not isinstance(h, SessionFileHandler)
            ]


# ============================================================
# ПУБЛИЧНЫЙ API
# ============================================================

def get_logger(name: str = "OracleAI") -> logging.Logger:
    """
    Основная точка входа. Совместима со старым кодом.
    Использование:
        from core.logger import get_logger
        logger = get_logger()
        logger.info("Сообщение")
    """
    return setup_logger(name)


def log_structured(
    logger: logging.Logger,
    level: int,
    message: str,
    category: LogCategory = LogCategory.SYSTEM,
    panel: str = "",
    action: str = "",
    context: Optional[Dict[str, Any]] = None,
    numeric_values: Optional[Dict[str, float]] = None,
):
    """
    Записывает структурированное сообщение с метаданными.
    Используется панелями GUI и Куратором для аналитики.

    Пример:
        log_structured(
            logger, logging.INFO,
            "Пользователь нажал 'Старт Обучения'",
            category=LogCategory.USER_ACTION,
            panel="training_panel",
            action="start_training",
            context={"epochs": 50, "batch_size": 64},
            numeric_values={"learning_rate": 0.001},
        )
    """
    extra = {
        "category": category,
        "panel": panel,
        "action": action,
        "context": context or {},
        "numeric_values": numeric_values or {},
    }
    logger.log(level, message, extra=extra)


def log_user_action(
    panel: str,
    action: str,
    details: str = "",
    context: Optional[Dict[str, Any]] = None,
):
    """
    Упрощённый метод для логирования действий пользователя.
    Вызывается из GUI-панелей для истории сеансов.

    Пример:
        log_user_action("dataset_panel", "split_dataset", "Разбиение 80/10/10")
    """
    logger = get_logger("UserAction")
    msg = f"[{panel}] {action}"
    if details:
        msg += f": {details}"
    log_structured(
        logger,
        logging.INFO,
        msg,
        category=LogCategory.USER_ACTION,
        panel=panel,
        action=action,
        context=context or {},
    )


def log_training_event(
    event: str,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    context: Optional[Dict[str, Any]] = None,
):
    """
    Специализированный метод для событий обучения.
    Используется Trainer и MonitoringPanel.

    Пример:
        log_training_event("epoch_end", epoch=5, metrics={"train_loss": 0.23})
    """
    logger = get_logger("Training")
    log_structured(
        logger,
        logging.INFO,
        f"Training event: {event} (epoch={epoch})",
        category=LogCategory.TRAINING,
        panel="training_panel",
        action=event,
        context={"epoch": epoch, **(context or {})},
        numeric_values=metrics or {},
    )


def log_error_with_context(
    error: Exception,
    panel: str = "",
    action: str = "",
    context: Optional[Dict[str, Any]] = None,
):
    """
    Логирование ошибок с полным контекстом для Куратора.
    """
    logger = get_logger("Error")
    log_structured(
        logger,
        logging.ERROR,
        f"{type(error).__name__}: {error}",
        category=LogCategory.ERROR,
        panel=panel,
        action=action,
        context={"exception_type": type(error).__name__, **(context or {})},
    )


# ============================================================
# УТИЛИТЫ ДЛЯ КУРАТОРА
# ============================================================

class CuratorLogCollector(LogSubscriber):
    """
    Подписчик для ИИ-Куратора.
    Собирает структурированные логи в кольцевой буфер.
    Куратор анализирует последние N записей для генерации советов.
    """
    def __init__(self, buffer_size: int = 500):
        self._buffer: List[StructuredLogEntry] = []
        self._buffer_size = buffer_size
        self._lock = threading.Lock()

    def on_log_entry(self, entry: StructuredLogEntry):
        with self._lock:
            self._buffer.append(entry)
            if len(self._buffer) > self._buffer_size:
                self._buffer = self._buffer[-self._buffer_size:]

    def get_recent(self, n: int = 50) -> List[StructuredLogEntry]:
        """Возвращает последние n записей."""
        with self._lock:
            return list(self._buffer[-n:])

    def get_by_category(self, category: LogCategory, n: int = 50) -> List[StructuredLogEntry]:
        """Возвращает последние n записей конкретной категории."""
        with self._lock:
            filtered = [e for e in self._buffer if e.category == category]
            return filtered[-n:]

    def get_errors(self, n: int = 20) -> List[StructuredLogEntry]:
        """Возвращает последние ошибки."""
        return self.get_by_category(LogCategory.ERROR, n)

    def get_user_actions(self, n: int = 100) -> List[StructuredLogEntry]:
        """Возвращает последние действия пользователя."""
        return self.get_by_category(LogCategory.USER_ACTION, n)

    def get_training_metrics(self, n: int = 50) -> List[StructuredLogEntry]:
        """Возвращает последние события обучения с метриками."""
        with self._lock:
            training = [
                e for e in self._buffer
                if e.category == LogCategory.TRAINING and e.numeric_values
            ]
            return training[-n:]

    def clear(self):
        """Очищает буфер."""
        with self._lock:
            self._buffer.clear()


# Глобальный экземпляр коллектора для Куратора
curator_collector = CuratorLogCollector(buffer_size=1000)
log_event_bus.subscribe(curator_collector)


# ============================================================
# ЭКСПОРТ ЛОГОВ
# ============================================================

def export_session_log(session_id: str, output_path: Path) -> bool:
    """
    Экспортирует сессионный лог в файл для отчёта.
    Используется в Exporter.
    """
    _ensure_initialized()
    # Ищем файл сессии
    pattern = f"session_{session_id}_*.log"
    matches = list(SESSIONS_DIR.glob(pattern))
    if not matches:
        return False

    latest = max(matches, key=lambda p: p.stat().st_mtime)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import shutil
        shutil.copy2(latest, output_path)
        return True
    except Exception:
        return False


def get_log_summary() -> Dict[str, Any]:
    """
    Краткая сводка по текущему состоянию логирования.
    Для панели логов и отладки.
    """
    with _loggers_lock:
        logger_count = len(_loggers)
    ctx = get_session_context()
    return {
        "active_loggers": logger_count,
        "session_id": ctx.get("session_id", ""),
        "user_level": ctx.get("user_level", "unknown"),
        "subscribers_count": len(log_event_bus._subscribers),
        "curator_buffer_size": len(curator_collector._buffer),
    }