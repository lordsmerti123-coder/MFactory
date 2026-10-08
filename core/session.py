"""
core/session.py
===============
Модуль управления историей сеансов пользователя.

ОТВЕТСТВЕННОСТЬ:
  • Хранение и загрузка состояния сеанса (SessionState из contracts.py)
  • Подсчёт количества запусков, ошибок, успешных обучений
  • Логирование действий пользователя (для адаптивного куратора)
  • Определение уровня пользователя (beginner / intermediate / advanced)
  • Управление режимом подсказок (static / ai / hybrid / none)
  • Отметка завершённых шагов рабочего процесса

ФАЙЛ ДАННЫХ:
  data/sessions/session_state.json

ЗАВИСИМОСТИ:
  • core/contracts.py  → SessionState (dataclass)
  • config.py          → SESSIONS_DIR (Path)
  • Стандартная библиотека: json, uuid, datetime, pathlib, logging

КОНТРАКТ ДЛЯ ДРУГИХ МОДУЛЕЙ:
  Все модули, которым нужна история пользователя, создают или получают
  экземпляр SessionManager через shared_state["session_manager"].
  НИКОГДА не пишут в файл напрямую — только через методы этого класса.

ПРАВИЛА:
  • Файл сессии НЕ должен превышать ~200 КБ. Лог действий обрезается.
  • При повреждении файла сессии создаётся новая сессия без падения.
  • Все методы потокобезопасны для чтения. Запись — только через save().
"""

import json
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# ============================================================
# ИМПОРТЫ ИЗ ПРОЕКТА
# ============================================================
# КРИТИЧНО: типы берутся ТОЛЬКО из contracts.py
try:
    from core.contracts import SessionState
except ImportError:
    # Fallback на случай, если contracts.py ещё не написан.
    # В финальной сборке этот блок НЕ должен срабатывать.
    from dataclasses import dataclass, field

    @dataclass
    class SessionState:
        session_id: str = ""
        total_sessions: int = 0
        current_tab: str = ""
        completed_steps: List[str] = field(default_factory=list)
        user_actions_log: List[Dict] = field(default_factory=list)
        mistakes_count: int = 0
        successful_models: int = 0
        last_model_name: str = ""
        hint_mode: str = "hybrid"

try:
    from config import SESSIONS_DIR
except ImportError:
    # Fallback: если config.py ещё не обновлён
    SESSIONS_DIR = Path(__file__).parent.parent / "data" / "sessions"

# ============================================================
# КОНСТАНТЫ МОДУЛЯ
# ============================================================
SESSION_FILE_NAME = "session_state.json"
MAX_ACTION_LOG_SIZE = 500        # Максимум записей в логе действий
MAX_COMPLETED_STEPS = 100        # Максимум отметок завершённых шагов
FORMAT_VERSION = "2.0"           # Версия формата файла сессии

# Допустимые режимы подсказок (должны совпадать с config.py)
VALID_HINT_MODES = {"static", "ai", "hybrid", "none"}

# Пороги для определения уровня пользователя
BEGINNER_MAX_SESSIONS = 3
BEGINNER_MAX_SUCCESS = 2
INTERMEDIATE_MAX_SESSIONS = 15
INTERMEDIATE_MAX_SUCCESS = 10

logger = logging.getLogger("OracleAI")


class SessionManager:
    """
    Менеджер сессий пользователя.

    Хранит состояние между запусками программы в JSON-файле.
    Все панели и куратор обращаются к этому объекту для получения
    контекста (уровень пользователя, история, режим подсказок).

    Использование:
        session_mgr = SessionManager()
        session_mgr.increment_sessions()
        session_mgr.log_action("generator_panel", "generate", "addition, 20000 samples")
        session_mgr.save()
    """

    # ============================================================
    # ИНИЦИАЛИЗАЦИЯ
    # ============================================================
    def __init__(self, session_dir: Optional[Path] = None):
        """
        Создаёт менеджер сессий.

        Аргументы:
            session_dir: Путь к папке сессий. Если None — берётся из config.
        """
        self.session_dir = Path(session_dir) if session_dir else SESSIONS_DIR
        self.session_file = self.session_dir / SESSION_FILE_NAME
        self.state = SessionState()
        self._dirty = False  # Флаг: есть несохранённые изменения

        self._ensure_directory()
        self._load_or_create()

    def _ensure_directory(self):
        """Создаёт папку сессий, если её нет."""
        try:
            self.session_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(f"Не удалось создать папку сессий {self.session_dir}: {e}")

    # ============================================================
    # ЗАГРУЗКА / СОЗДАНИЕ
    # ============================================================
    def _load_or_create(self):
        """
        Загружает состояние из файла или создаёт новое.

        При повреждении файла — логирует ошибку и создаёт чистую сессию.
        Программа НЕ должна падать из-за битого файла сессии.
        """
        if self.session_file.exists():
            try:
                self._load_from_file()
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                logger.warning(
                    f"Файл сессии повреждён ({e}). Создаётся новая сессия."
                )
                self._create_new_session()
        else:
            logger.info("Файл сессии не найден. Создаётся новая сессия.")
            self._create_new_session()

    def _load_from_file(self):
        """Читает JSON и заполняет SessionState."""
        with open(self.session_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Проверка версии формата
        file_version = data.get("format_version", "1.0")
        if file_version != FORMAT_VERSION:
            logger.info(
                f"Миграция сессии: версия файла {file_version} → {FORMAT_VERSION}"
            )
            # В будущем здесь будет логика миграции.
            # Пока просто перезаписываем недостающие поля значениями по умолчанию.

        # Заполняем SessionState из словаря
        self.state.session_id = data.get("session_id", str(uuid.uuid4()))
        self.state.total_sessions = data.get("total_sessions", 0)
        self.state.current_tab = data.get("current_tab", "")
        self.state.completed_steps = data.get("completed_steps", [])
        self.state.user_actions_log = data.get("user_actions_log", [])
        self.state.mistakes_count = data.get("mistakes_count", 0)
        self.state.successful_models = data.get("successful_models", 0)
        self.state.last_model_name = data.get("last_model_name", "")
        self.state.hint_mode = data.get("hint_mode", "hybrid")

        # Валидация режима подсказок
        if self.state.hint_mode not in VALID_HINT_MODES:
            self.state.hint_mode = "hybrid"

        # Обрезаем лог, если он слишком большой (защита от разрастания)
        if len(self.state.user_actions_log) > MAX_ACTION_LOG_SIZE:
            self.state.user_actions_log = self.state.user_actions_log[-MAX_ACTION_LOG_SIZE:]

        logger.info(
            f"Сессия загружена: id={self.state.session_id[:8]}..., "
            f"запусков={self.state.total_sessions}"
        )

    def _create_new_session(self):
        """Создаёт чистую сессию с новым UUID."""
        self.state = SessionState()
        self.state.session_id = str(uuid.uuid4())
        self.state.total_sessions = 0
        self.state.hint_mode = "hybrid"
        self._dirty = True

    # ============================================================
    # СОХРАНЕНИЕ
    # ============================================================
    def save(self):
        """
        Сохраняет текущее состояние в JSON-файл.

        Вызывается:
          • При закрытии программы (в main_window)
          • После критических действий (обучение завершено, ошибка)
          • При смене режима подсказок

        Безопасность: пишет во временный файл, потом переименовывает.
        Это защищает от повреждения при внезапном отключении питания.
        """
        try:
            data = self._to_dict()
            tmp_file = self.session_file.with_suffix(".tmp")

            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # Атомарная замена (защита от повреждения)
            if self.session_file.exists():
                self.session_file.unlink()
            tmp_file.rename(self.session_file)

            self._dirty = False
            logger.debug("Сессия сохранена.")

        except OSError as e:
            logger.error(f"Не удалось сохранить сессию: {e}")
        except Exception as e:
            logger.error(f"Непредвиденная ошибка при сохранении сессии: {e}")

    def _to_dict(self) -> Dict[str, Any]:
        """Сериализует SessionState в словарь для JSON."""
        return {
            "format_version": FORMAT_VERSION,
            "session_id": self.state.session_id,
            "total_sessions": self.state.total_sessions,
            "current_tab": self.state.current_tab,
            "completed_steps": self.state.completed_steps,
            "user_actions_log": self.state.user_actions_log,
            "mistakes_count": self.state.mistakes_count,
            "successful_models": self.state.successful_models,
            "last_model_name": self.state.last_model_name,
            "hint_mode": self.state.hint_mode,
            "saved_at": datetime.now().isoformat(),
        }

    # ============================================================
    # УПРАВЛЕНИЕ СЧЁТЧИКАМИ
    # ============================================================
    def increment_sessions(self):
        """
        Вызывается ОДИН РАЗ при запуске программы (в main_window).
        Увеличивает счётчик запусков и генерирует новый session_id.
        """
        self.state.total_sessions += 1
        self.state.session_id = str(uuid.uuid4())
        self.state.current_tab = ""
        self.state.completed_steps = []  # Сбрасываем шаги нового сеанса
        self._dirty = True

        logger.info(
            f"Новый сеанс #{self.state.total_sessions} "
            f"(id={self.state.session_id[:8]}...)"
        )
        self.save()

    # ============================================================
    # ЛОГИРОВАНИЕ ДЕЙСТВИЙ
    # ============================================================
    def log_action(self, panel: str, action: str, details: str = ""):
        """
        Записывает действие пользователя в лог.

        Аргументы:
            panel:   Имя панели ("generator_panel", "dataset_panel", ...)
            action:  Действие ("generate", "split", "create_model", ...)
            details: Дополнительная информация (параметры, значения)

        Пример:
            session_mgr.log_action("generator_panel", "generate", "addition, 20000")
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "panel": panel,
            "action": action,
            "details": details,
        }
        self.state.user_actions_log.append(entry)

        # Обрезаем лог, если он слишком длинный
        if len(self.state.user_actions_log) > MAX_ACTION_LOG_SIZE:
            self.state.user_actions_log = self.state.user_actions_log[-MAX_ACTION_LOG_SIZE:]

        self._dirty = True

    def log_tab_change(self, tab_name: str):
        """Записывает переключение вкладки."""
        self.state.current_tab = tab_name
        self.log_action("navigation", "tab_change", tab_name)

    # ============================================================
    # ОТСЛЕЖИВАНИЕ ОШИБОК И УСПЕХОВ
    # ============================================================
    def record_mistake(self, context: str):
        """
        Записывает ошибку пользователя.

        Используется куратором для адаптации подсказок:
        если пользователь часто ошибается в одном месте — подсказки
        становятся подробнее.

        Аргументы:
            context: Описание контекста ("split_forget", "lr_too_high", ...)
        """
        self.state.mistakes_count += 1
        self.log_action("mistake", "user_error", context)
        self._dirty = True

        logger.debug(f"Ошибка пользователя #{self.state.mistakes_count}: {context}")

    def record_success(self, model_name: str):
        """
        Записывает успешное завершение обучения модели.

        Аргументы:
            model_name: Имя модели ("НейроВася-7", ...)
        """
        self.state.successful_models += 1
        self.state.last_model_name = model_name
        self.log_action("success", "training_complete", model_name)
        self._dirty = True

        logger.info(
            f"Успешное обучение #{self.state.successful_models}: '{model_name}'"
        )
        self.save()  # Сохраняем сразу — важное событие

    # ============================================================
    # ШАГИ РАБОЧЕГО ПРОЦЕССА
    # ============================================================
    def mark_step_completed(self, step: str):
        """
        Отмечает шаг рабочего процесса как завершённый.

        Аргументы:
            step: Идентификатор шага. Допустимые значения:
                  "project_created", "data_generated", "data_loaded",
                  "data_split", "model_created", "params_set",
                  "training_started", "training_finished",
                  "analysis_done", "export_done"
        """
        if step not in self.state.completed_steps:
            self.state.completed_steps.append(step)
            # Обрезаем, если слишком много
            if len(self.state.completed_steps) > MAX_COMPLETED_STEPS:
                self.state.completed_steps = self.state.completed_steps[-MAX_COMPLETED_STEPS:]
            self._dirty = True

    def is_step_completed(self, step: str) -> bool:
        """Проверяет, завершён ли шаг в текущем сеансе."""
        return step in self.state.completed_steps

    def get_next_suggested_step(self) -> str:
        """
        Возвращает следующий рекомендуемый шаг на основе завершённых.
        Используется куратором для подсказки «Что делать дальше».
        """
        workflow = [
            "project_created",
            "data_generated",
            "data_loaded",
            "data_split",
            "model_created",
            "params_set",
            "training_started",
            "training_finished",
            "analysis_done",
            "export_done",
        ]
        for step in workflow:
            if step not in self.state.completed_steps:
                return step
        return "all_done"

    # ============================================================
    # РЕЖИМ ПОДСКАЗОК
    # ============================================================
    def get_hint_mode(self) -> str:
        """Возвращает текущий режим подсказок."""
        return self.state.hint_mode

    def set_hint_mode(self, mode: str):
        """
        Устанавливает режим подсказок.

        Аргументы:
            mode: "static" | "ai" | "hybrid" | "none"

        При невалидном значении — оставляет текущий режим и логирует.
        """
        if mode not in VALID_HINT_MODES:
            logger.warning(
                f"Недопустимый режим подсказок: '{mode}'. "
                f"Допустимые: {VALID_HINT_MODES}. Оставляем '{self.state.hint_mode}'."
            )
            return
        self.state.hint_mode = mode
        self._dirty = True
        self.save()  # Сохраняем сразу — настройка пользователя

    # ============================================================
    # ОПРЕДЕЛЕНИЕ УРОВНЯ ПОЛЬЗОВАТЕЛЯ
    # ============================================================
    def get_user_level(self) -> str:
        """
        Определяет уровень пользователя на основе истории.

        Возвращает:
            "beginner"      — первые 3 сеанса или < 2 успешных моделей
            "intermediate"  — до 15 сеансов и < 10 успешных
            "advanced"      — всё остальное

        Используется hint_engine для адаптации подробности подсказок.
        """
        sessions = self.state.total_sessions
        success = self.state.successful_models

        if sessions <= BEGINNER_MAX_SESSIONS or success < BEGINNER_MAX_SUCCESS:
            return "beginner"
        elif sessions <= INTERMEDIATE_MAX_SESSIONS or success < INTERMEDIATE_MAX_SUCCESS:
            return "intermediate"
        else:
            return "advanced"

    # ============================================================
    # АНАЛИТИКА ДЛЯ КУРАТОРА
    # ============================================================
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Возвращает сводку текущего сеанса для куратора и отчётов.

        Используется:
          • core/curator.py — для генерации советов
          • gui/panels/analysis_panel.py — для финального отчёта
          • core/hint_engine.py — для контекстных подсказок
        """
        return {
            "session_id": self.state.session_id[:8],
            "total_sessions": self.state.total_sessions,
            "user_level": self.get_user_level(),
            "hint_mode": self.state.hint_mode,
            "current_tab": self.state.current_tab,
            "completed_steps": list(self.state.completed_steps),
            "next_step": self.get_next_suggested_step(),
            "mistakes_count": self.state.mistakes_count,
            "successful_models": self.state.successful_models,
            "last_model_name": self.state.last_model_name,
            "actions_in_session": len(self.state.user_actions_log),
        }

    def get_frequent_mistakes(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Возвращает топ-N самых частых типов ошибок.
        Используется куратором для проактивных подсказок.
        """
        mistake_entries = [
            e for e in self.state.user_actions_log
            if e.get("panel") == "mistake"
        ]
        if not mistake_entries:
            return []

        # Считаем частоту по контексту (details)
        freq: Dict[str, int] = {}
        for entry in mistake_entries:
            ctx = entry.get("details", "unknown")
            freq[ctx] = freq.get(ctx, 0) + 1

        sorted_mistakes = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [
            {"context": ctx, "count": count}
            for ctx, count in sorted_mistakes[:top_n]
        ]

    def get_recent_actions(self, n: int = 20) -> List[Dict[str, Any]]:
        """Возвращает последние N действий пользователя."""
        return self.state.user_actions_log[-n:]

    # ============================================================
    # СЛУЖЕБНЫЕ МЕТОДЫ
    # ============================================================
    def reset_session_steps(self):
        """
        Сбрасывает отметки шагов текущего сеанса.
        Вызывается при начале нового проекта внутри того же сеанса.
        """
        self.state.completed_steps = []
        self._dirty = True

    def has_unsaved_changes(self) -> bool:
        """Возвращает True, если есть несохранённые изменения."""
        return self._dirty

    def get_state(self) -> SessionState:
        """
        Возвращает объект SessionState.
        Используется для записи в shared_state["session"].
        """
        return self.state

    def __repr__(self) -> str:
        level = self.get_user_level()
        return (
            f"<SessionManager sessions={self.state.total_sessions} "
            f"level='{level}' hint_mode='{self.state.hint_mode}'>"
        )


# ============================================================
# АВТОТЕСТ ПРИ ЗАПУСКЕ КАК СКРИПТ
# ============================================================
if __name__ == "__main__":
    # Настройка минимального логирования для теста
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("АВТОТЕСТ core/session.py")
    print("=" * 60)

    # Создаём менеджер (используем временную папку)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        mgr = SessionManager(session_dir=Path(tmp))

        # Тест 1: инкремент сессий
        mgr.increment_sessions()
        assert mgr.state.total_sessions == 1, "Счётчик сессий должен быть 1"
        print("✅ increment_sessions работает")

        # Тест 2: логирование действий
        mgr.log_action("generator_panel", "generate", "addition, 20000")
        mgr.log_action("dataset_panel", "split", "80/10/10")
        assert len(mgr.state.user_actions_log) >= 2
        print("✅ log_action работает")

        # Тест 3: ошибки и успехи
        mgr.record_mistake("split_forget")
        mgr.record_mistake("lr_too_high")
        mgr.record_success("НейроВася-7")
        assert mgr.state.mistakes_count == 2
        assert mgr.state.successful_models == 1
        assert mgr.state.last_model_name == "НейроВася-7"
        print("✅ record_mistake / record_success работают")

        # Тест 4: шаги
        mgr.mark_step_completed("project_created")
        mgr.mark_step_completed("data_generated")
        assert mgr.is_step_completed("project_created")
        assert not mgr.is_step_completed("model_created")
        assert mgr.get_next_suggested_step() == "data_loaded"
        print("✅ mark_step_completed / get_next_suggested_step работают")

        # Тест 5: режим подсказок
        mgr.set_hint_mode("static")
        assert mgr.get_hint_mode() == "static"
        mgr.set_hint_mode("invalid_mode")  # Должен проигнорировать
        assert mgr.get_hint_mode() == "static"
        mgr.set_hint_mode("hybrid")
        assert mgr.get_hint_mode() == "hybrid"
        print("✅ set_hint_mode / get_hint_mode работают")

        # Тест 6: уровень пользователя
        level = mgr.get_user_level()
        assert level in ("beginner", "intermediate", "advanced")
        print(f"✅ get_user_level работает (уровень: {level})")

        # Тест 7: сводка
        summary = mgr.get_session_summary()
        assert "session_id" in summary
        assert "user_level" in summary
        assert "next_step" in summary
        print("✅ get_session_summary работает")

        # Тест 8: сохранение и перезагрузка
        mgr.save()
        mgr2 = SessionManager(session_dir=Path(tmp))
        assert mgr2.state.total_sessions == 1
        assert mgr2.state.successful_models == 1
        assert mgr2.state.last_model_name == "НейроВася-7"
        assert mgr2.state.hint_mode == "hybrid"
        print("✅ save / load работают")

        # Тест 9: повреждённый файл
        with open(Path(tmp) / SESSION_FILE_NAME, "w") as f:
            f.write("{broken json!!!")
        mgr3 = SessionManager(session_dir=Path(tmp))
        assert mgr3.state.total_sessions == 0  # Должна создаться новая
        print("✅ Повреждённый файл обрабатывается без падения")

    print("=" * 60)
    print("ВСЕ ТЕСТЫ ПРОЙДЕНЫ ✅")
    print("=" * 60)