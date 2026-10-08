"""
core/oraculum/agent.py
======================
Главный класс ИИ-агента «Оракул» (OraculumAgent).

ОТВЕТСТВЕННОСТЬ:
  • Точка входа для обработки запросов пользователя.
  • Управление историей чата.
  • Сбор контекста и классификация намерений.
  • Генерация ответов (локальные шаблоны → API).
  • Логирование в категорию HINT (панель «Логи»).
  • Контроль ресурсов через ResourceMonitor.

ИСПОЛЬЗОВАНИЕ:
    agent = OraculumAgent(shared_state)
    answer = agent.process_user_query("Как создать модель?")
    hint = agent.get_contextual_hint()
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.oraculum.context_builder import ContextBuilder
from core.oraculum.intent_classifier import IntentClassifier, Intent
from core.oraculum.knowledge_base import KnowledgeBase
from core.oraculum.response_generator import ResponseGenerator
from core.oraculum.api_adapter import APIAdapter
from core.oraculum.resource_monitor import ResourceMonitor
from core.oraculum.local_llm import LocalLLMBackend, SYSTEM_PROMPT
from core.oraculum.settings import load_settings, save_settings, build_system_prompt
from core.oraculum.tools import ToolRegistry, TOOL_INSTRUCTIONS, parse_tool_call, is_reply_object, FEW_SHOT_EXAMPLES
from core.logger import get_logger, LogCategory, log_structured

from config import (
    BASE_DIR,
    ORACULUM_ENABLE_LOCAL_MODEL,
    ORACULUM_LOCAL_MODEL_PATH,
    ORACULUM_GGUF_CANDIDATES,
    ORACULUM_GGUF_CTX_SIZE,
    ORACULUM_GGUF_N_GPU_LAYERS,
    ORACULUM_GGUF_MAX_TOKENS,
    ORACULUM_GGUF_TEMPERATURE,
    ORACULUM_GGUF_TOP_P,
    ORACULUM_GGUF_VERBOSE,
    ORACULUM_LOCAL_GEN_TIMEOUT,
)

logger = get_logger("Oraculum")


def resolve_local_model_path(explicit: Optional[str] = None) -> Optional[Path]:
    """Определяет путь к локальной GGUF-модели.

    Сначала явный путь (или ORACULUM_LOCAL_MODEL_PATH), затем авто-поиск
    кандидатов рядом с проектом. Возвращает None, если ничего не найдено.
    """
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    if ORACULUM_LOCAL_MODEL_PATH:
        p = Path(ORACULUM_LOCAL_MODEL_PATH)
        return p if p.exists() else None
    for name in ORACULUM_GGUF_CANDIDATES:
        p = Path(BASE_DIR) / name
        if p.exists():
            return p
    return None


@dataclass
class ChatMessage:
    """Сообщение в истории чата Оракула."""
    role: str  # "user" | "assistant" | "system"
    text: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "text": self.text, "timestamp": self.timestamp}


@dataclass
class ScreenAnalysis:
    """Структурированный анализ текущего экрана."""
    current_tab: str = ""
    scenario: str = ""
    model_exists: bool = False
    dataset_exists: bool = False
    has_split: bool = False
    model_trained: bool = False
    errors_last_hour: int = 0
    next_step: str = ""

    def to_text(self) -> str:
        lines = [
            f"Вкладка: {self.current_tab}",
            f"Сценарий: {self.scenario}",
            f"Модель: {'есть' if self.model_exists else 'нет'}",
            f"Данные: {'есть' if self.dataset_exists else 'нет'}",
            f"Разбиение: {'есть' if self.has_split else 'нет'}",
            f"Обучение: {'завершено' if self.model_trained else 'не проводилось'}",
            f"Ошибок за час: {self.errors_last_hour}",
            f"Следующий шаг: {self.next_step}",
        ]
        return "\n".join(lines)


class OraculumAgent:
    """ИИ-агент «Оракул»."""

    MAX_HISTORY = 100

    def __init__(self, shared_state: dict, resource_limits: Optional[dict] = None):
        self.shared_state = shared_state
        self.context_builder = ContextBuilder(shared_state)
        self.intent_classifier = IntentClassifier()
        self.knowledge = KnowledgeBase()
        self.response_generator = ResponseGenerator()
        self.resource_monitor = ResourceMonitor(resource_limits)
        self.api_adapter = APIAdapter(self.resource_monitor)
        self.chat_history: List[ChatMessage] = []
        self._last_context_hint = ""
        self._local_backend: Optional[LocalLLMBackend] = None
        self._local_model_path: Optional[Path] = None
        self.settings: Dict[str, Any] = {}
        self.tools = ToolRegistry()
        self._dispatcher: Optional[Callable[[str, Dict[str, Any]], str]] = None
        self._training_state: Dict[str, Any] = {
            "status": "idle",  # idle | running | finished | failed
            "last_epoch": 0,
            "last_metrics": {},
        }
        self._load_settings()
        self._register_builtin_tools()

        log_structured(
            logger, 20, "Оракул инициализирован",
            category=LogCategory.HINT, panel="oraculum", action="init",
        )

    # ============================================================
    # НАСТРОЙКИ (файл data/system/oraculum_settings.json + session)
    # ============================================================
    def _load_settings(self):
        # 1) Основной источник — файл настроек (редактируется админом)
        self.settings = load_settings()

        # 2) Legacy-настройки из сессии (перекрывают файл, если есть)
        session = self.shared_state.get("session", {}) or {}
        legacy = {}
        if hasattr(session, "preferences"):
            legacy = session.preferences.get("oraculum", {}) or {}
        elif isinstance(session, dict):
            legacy = session.get("oraculum", {}) or {}
        if legacy:
            self.settings.update(legacy)

        provider = self.settings.get("provider")
        if provider == "yandex":
            self.api_adapter.set_yandex_gpt(
                self.settings.get("api_key", ""),
                self.settings.get("folder_id", ""),
            )
        elif provider == "sber":
            self.api_adapter.set_sber_gpt(
                self.settings.get("client_id", ""),
                self.settings.get("client_secret", ""),
            )

    def get_settings(self) -> Dict[str, Any]:
        """Возвращает копию текущих настроек Оракула."""
        return dict(self.settings)

    def apply_settings(self, settings: Dict[str, Any]) -> bool:
        """Применяет и сохраняет настройки Оракула (для панели админа)."""
        self.settings.update(settings)
        ok = save_settings(self.settings)
        self._load_settings()  # перечитываем и применяем провайдера
        return ok

    def configure_api(self, provider: str, **kwargs) -> bool:
        """Настраивает внешний API (yandex/sber). Сохраняет в файл и сессию."""
        if provider == "yandex":
            self.api_adapter.set_yandex_gpt(kwargs.get("api_key", ""),
                                            kwargs.get("folder_id", ""))
        elif provider == "sber":
            self.api_adapter.set_sber_gpt(kwargs.get("client_id", ""),
                                          kwargs.get("client_secret", ""))
        elif provider == "local":
            self.api_adapter.disable()
        else:
            return False

        self.settings["provider"] = provider
        if provider == "yandex":
            self.settings["api_key"] = kwargs.get("api_key", "")
            self.settings["folder_id"] = kwargs.get("folder_id", "")
        elif provider == "sber":
            self.settings["client_id"] = kwargs.get("client_id", "")
            self.settings["client_secret"] = kwargs.get("client_secret", "")

        # Сохраняем в preferences сессии (обратная совместимость)
        session = self.shared_state.get("session")
        prefs = {}
        if hasattr(session, "preferences"):
            prefs = dict(session.preferences)
        elif isinstance(session, dict):
            prefs = dict(session.get("preferences", {}))
        prefs["oraculum"] = {
            "provider": provider,
            "api_key": kwargs.get("api_key", ""),
            "folder_id": kwargs.get("folder_id", ""),
            "client_id": kwargs.get("client_id", ""),
            "client_secret": kwargs.get("client_secret", ""),
        }
        try:
            if hasattr(session, "preferences"):
                session.preferences = prefs
            elif isinstance(session, dict):
                session["preferences"] = prefs
        except Exception:
            pass
        return True

    def is_api_available(self) -> bool:
        return self.api_adapter.is_available()

    def api_provider(self) -> Optional[str]:
        return self.api_adapter.provider

    def get_api_error(self) -> Optional[str]:
        return self.api_adapter.get_last_error()

    # ============================================================
    # ИНСТРУМЕНТЫ (tools) — агентские возможности
    # ============================================================
    def _register_builtin_tools(self):
        """Встроенные read-only инструменты (не требуют GUI)."""
        self.tools.register(
            "get_context", "Получить текущий контекст (сценарий, вкладка, проект, "
            "модель, данные, уровень пользователя).",
            handler=self._tool_get_context,
        )
        self.tools.register(
            "get_status", "Получить состояние Оракула (провайдер, модель, ресурсы).",
            handler=self._tool_get_status,
        )
        self.tools.register(
            "list_tools", "Получить список всех доступных инструментов.",
            handler=self._tool_list_tools,
        )
        self.tools.register(
            "model_info", "Получить информацию о текущей модели и данных.",
            handler=self._tool_model_info,
        )
        self.tools.register(
            "monitor_training", "Проверить состояние обучения (идёт/завершено, "
            "последняя эпоха, метрики).",
            handler=self._tool_monitor_training,
        )

    def register_tool(self, name: str, description: str,
                      parameters: Optional[Dict[str, str]] = None):
        """Регистрирует GUI-инструмент (исполняется через dispatcher)."""
        self.tools.register(name, description, parameters or {}, handler=None)

    def register_action(self, name: str, handler: Callable[..., Any],
                        description: str,
                        parameters: Optional[Dict[str, str]] = None):
        """Регистрирует инструмент с прямым обработчиком (не через dispatcher)."""
        self.tools.register(name, description, parameters or {}, handler=handler)

    def set_action_dispatcher(self, dispatcher: Callable[[str, Dict[str, Any]], str]):
        """
        Устанавливает исполнитель GUI-инструментов. Вызывается MainWindow.
        dispatcher(name, args_dict) -> str (результат в виде строки/JSON).
        Должен быть потокобезопасным (маршалит вызов в главный поток).
        """
        self._dispatcher = dispatcher

    def _tool_get_context(self) -> str:
        return self.build_context_text()

    def _tool_get_status(self) -> str:
        return json.dumps(self.get_status(), ensure_ascii=False, default=str)

    def _tool_list_tools(self) -> str:
        return self.tools.describe()

    def _tool_model_info(self) -> str:
        model = self.shared_state.get("model")
        config = self.shared_state.get("config")
        dataset = self.shared_state.get("dataset")
        history = self.shared_state.get("history")
        lines = []
        if model is None:
            lines.append("Модель не создана.")
        else:
            try:
                params = sum(p.numel() for p in model.parameters())
                arch = getattr(config, "architecture", "?")
                if hasattr(arch, "value"):
                    arch = arch.value
                name = getattr(config, "model_name", "Модель")
                lines.append(f"Модель '{name}' ({arch}): {params} параметров.")
            except Exception:
                lines.append("Модель создана (детали недоступны).")
        if dataset is not None:
            try:
                n = dataset.get_split_size("train") if hasattr(dataset, "get_split_size") else "?"
                lines.append(f"Данные загружены, train-выборка: {n}.")
            except Exception:
                lines.append("Данные загружены.")
        else:
            lines.append("Данные не загружены.")
        if history is not None:
            try:
                val = history.val_loss if hasattr(history, "val_loss") else history.get("val_loss", [])
                lines.append(f"Завершено эпох обучения: {len(val) if val else 0}.")
            except Exception:
                pass
        return " ".join(lines)

    def _tool_monitor_training(self) -> str:
        return json.dumps(self._training_state, ensure_ascii=False, default=str)

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        """Выполняет инструмент. Возвращает результат строкой (обычно JSON)."""
        tool = self.tools.get(name)
        if tool is None:
            return json.dumps({"error": f"неизвестный инструмент: {name}"},
                              ensure_ascii=False)
        try:
            if tool.handler is not None:
                result = tool.handler(**args) if args else tool.handler()
            elif self._dispatcher is not None:
                result = self._dispatcher(name, args or {})
            else:
                return json.dumps(
                    {"error": f"инструмент '{name}' недоступен (нет исполнителя)"},
                    ensure_ascii=False,
                )
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)

    def agent_mode_enabled(self) -> bool:
        """Включён ли агентский режим (tools)."""
        return bool(self.settings.get("agent_enabled", False))

    def _agent_backend_available(self) -> bool:
        return self.is_local_model_loaded() or self.api_adapter.is_available()

    def _build_agent_system_prompt(self) -> str:
        """Системный промпт агентского режима: роль + контекст + инструменты."""
        context_text = self.build_context_text()
        base = build_system_prompt(self.settings, context_text)
        parts = [
            base,
            TOOL_INSTRUCTIONS,
            "Доступные инструменты:",
            self.tools.describe(),
            FEW_SHOT_EXAMPLES,
        ]
        return "\n\n".join(parts)

    def _agent_chat(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Отправляет диалог в доступный бэкенд (локальная модель → API)."""
        temperature = float(self.settings.get("agent_temperature", 0.4))
        max_tokens = 512
        system = self._build_agent_system_prompt()
        if self.is_local_model_loaded() and self._local_backend is not None:
            return self._local_backend.generate_messages(
                [{"role": "system", "content": system}] + messages,
                max_tokens=max_tokens, temperature=temperature,
            )
        if self.api_adapter.is_available():
            return self.api_adapter.chat(
                messages, max_tokens=max_tokens, temperature=temperature, system=system,
            )
        return None

    def run_agent_loop(self, query: str, max_steps: Optional[int] = None) -> str:
        """
        Агентский цикл ReAct: модель решает, какие инструменты вызывать,
        чтобы выполнить просьбу пользователя (вплоть до нажатий «за него»).

        Возвращает итоговый текстовый ответ.
        """
        if not self._agent_backend_available():
            return (
                "Агентский режим недоступен: подключите Яндекс GPT или "
                "загрузите локальную модель в настройках."
            )
        steps = max_steps or int(self.settings.get("agent_max_steps", 5))
        steps = max(1, min(steps, 10))

        messages: List[Dict[str, str]] = [
            {"role": "user", "content": query},
        ]
        executed: List[Tuple[str, Dict[str, Any]]] = []

        for _ in range(steps):
            raw = self._agent_chat(messages)
            if raw is None:
                break
            action, args = parse_tool_call(raw)
            if action is not None:
                result = self._execute_tool(action, args or {})
                executed.append((action, args or {}))
                log_structured(
                    logger, 20,
                    f"Оракул-агент вызвал инструмент '{action}' args={args}",
                    category=LogCategory.ORACULUM, panel="oraculum", action="tool_call",
                    context={"tool": action},
                )
                # Подмешиваем результат в диалог. Если инструмент ошибся —
                # явно просим модель НЕ повторять его и сразу ответить.
                messages.append({"role": "assistant", "content": raw})
                if '"error"' in result:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Инструмент {action} НЕ выполнен: {result}. "
                            "Не вызывай этот инструмент снова. Если действие "
                            "невозможно выполнить сейчас, сразу ответь JSON: "
                            '{"reply": "краткое объяснение и что делать дальше"}'
                        ),
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": f"Результат инструмента {action}: {result}",
                    })
                continue
            # Проверяем JSON-ответ вида {"reply": "..."}
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    reply = is_reply_object(parsed)
                    if reply:
                        return reply
            except (json.JSONDecodeError, TypeError):
                pass
            # Модель вернула обычный текст (без JSON) — это финальный ответ
            return raw

        # Цикл исчерпан или бэкенд молчит — отвечаем по состоянию программы
        # и кратко перечисляем, что уже было выполнено.
        context = self.context_builder.build()
        base = self.response_generator.generate(
            self.intent_classifier.classify(query, context), context, query,
        )
        if executed:
            summary = ", ".join(
                f"«{name}»" + (f" ({', '.join(str(v) for v in args.values())})"
                               if args else "")
                for name, args in executed
            )
            return (
                f"Просьбу не удалось довести до конца, но уже выполнено: "
                f"{summary}. {base}"
            )
        return base

    # ============================================================
    # МОНИТОРИНГ ОБУЧЕНИЯ (Оракул следит за процессом)
    # ============================================================
    def on_training_started(self, is_text: bool = False):
        """Вызывается из MainWindow при старте обучения."""
        self._training_state = {
            "status": "running",
            "last_epoch": 0,
            "last_metrics": {},
        }
        log_structured(
            logger, 20, "Оракул начал следить за обучением",
            category=LogCategory.ORACULUM, panel="oraculum", action="training_started",
        )

    def on_training_epoch(self, epoch: int, logs: dict):
        """Вызывается каждую эпоху. logs: train_loss/val_loss/train_metric/val_metric."""
        self._training_state["status"] = "running"
        self._training_state["last_epoch"] = epoch
        self._training_state["last_metrics"] = {k: float(v) for k, v in (logs or {}).items()}

    def on_training_finished(self, history=None):
        """Вызывается по завершении обучения. history: dict/объект истории."""
        self._training_state["status"] = "finished"
        try:
            if hasattr(history, "to_dict"):
                h = history.to_dict()
            elif isinstance(history, dict):
                h = history
            else:
                h = {}
            self._training_state["best_val_loss"] = h.get("best_val_loss", "?")
            self._training_state["epochs_completed"] = h.get("epochs_completed", 0)
        except Exception:
            pass
        log_structured(
            logger, 20, "Оракул зафиксировал завершение обучения",
            category=LogCategory.ORACULUM, panel="oraculum", action="training_finished",
        )

    # ============================================================
    # ДИАГНОСТИКА СОСТОЯНИЯ (конкретные ответы «что происходит»)
    # ============================================================
    def diagnose(self) -> List[str]:
        """
        Возвращает список конкретных проблем/статусов в порядке важности.
        Используется для ответов на «что происходит?», «почему не работает?».
        Пример: «Вы пытаетесь обучить модель, но датасет не загружен».
        """
        context = self.context_builder.build()
        tab = context.current_tab or "Проект"
        findings: List[str] = []
        in_training_tab = "Обучение" in tab

        # 1. Обучение без датасета — самый критичный кейс
        if in_training_tab and not context.dataset_exists:
            findings.append(
                "Вы пытаетесь обучить модель, но датасет не загружен. "
                "Сначала создайте данные во вкладке «Генератор» или "
                "загрузите их во вкладке «Данные»."
            )
        elif not context.dataset_exists:
            findings.append(
                "Данные не загружены. Создайте их во вкладке «Генератор» "
                "или загрузите во вкладке «Данные»."
            )
        elif not context.has_split:
            findings.append(
                "Данные загружены, но нет разбиения на train/val/test. "
                "Примените разбиение во вкладке «Данные»."
            )

        if context.dataset_exists and context.has_split and not context.model_exists:
            findings.append(
                "Модель ещё не создана. Перейдите во вкладку «Архитектура» "
                "и нажмите «Создать модель»."
            )

        # 2. Состояние обучения (следит Оракул через сигналы training_panel)
        tstate = self._training_state.get("status", "idle")
        if tstate == "running":
            epoch = self._training_state.get("last_epoch", 0)
            metrics = self._training_state.get("last_metrics", {}) or {}
            loss = metrics.get("train_loss", metrics.get("val_loss", None))
            if isinstance(loss, (int, float)):
                findings.append(
                    f"Сейчас идёт обучение: эпоха {epoch}, loss={loss:.4f}. "
                    "Графики обновляются во вкладке «Мониторинг»."
                )
            else:
                findings.append(
                    f"Сейчас идёт обучение: эпоха {epoch}. "
                    "Графики обновляются во вкладке «Мониторинг»."
                )

        if context.model_exists and context.has_split and context.model_trained:
            findings.append(
                "Обучение завершено. Посмотрите графики во вкладке «Мониторинг» "
                "и результаты в «Анализ»."
            )

        if context.errors_last_hour > 0:
            findings.append(
                f"За последний час было {context.errors_last_hour} ошибок — "
                "проверьте вкладку «Логи»."
            )

        # 3. Всё готово к обучению — позитивный статус
        ready = (context.dataset_exists and context.has_split
                 and context.model_exists and not context.model_trained
                 and tstate != "running")
        if ready:
            findings.append(
                "Всё готово к обучению: данные разбиты, модель создана. "
                "Запустите обучение во вкладке «Обучение»."
            )

        # 4. Ничего критичного — краткая сводка
        if not findings:
            model_name = context.last_model_name or "не создана"
            findings.append(
                f"Вы во вкладке «{tab}», сценарий: {context.scenario}. "
                f"Модель: {model_name}. {context.suggest_step or 'Продолжайте работу.'}"
            )
        return findings

    def build_status_answer(self) -> str:
        """Полный ответ на «что происходит?» на основе diagnose()."""
        findings = self.diagnose()
        lines = ["Вот что сейчас происходит:"]
        lines.extend(f"• {f}" for f in findings)
        return "\n".join(lines)

    def _build_general_answer(self, findings: List[str]) -> str:
        """Полезный ответ на нераспознанный вопрос (по состоянию программы)."""
        lines = ["Сейчас в программе вот что:"]
        lines.extend(f"• {f}" for f in findings)
        lines.append("")
        lines.append(
            "Могу помочь: создать модель, обучить её, объяснить loss/эпохи/"
            "переобучение, подсказать следующий шаг или показать состояние. "
            "Что нужно?"
        )
        return "\n".join(lines)

    @staticmethod
    def _is_identity_answer(text: str) -> bool:
        """
        Определяет, что ответ модели — это лишь самоописание («кто я такая»),
        а не содержательный ответ на вопрос. Такие ответы отбрасываются.
        """
        t = (text or "").strip()
        if not t or len(t) > 200:
            return False
        low = t.lower()
        markers = (
            "я — модель", "я модель", "я - модель",
            "я искусственный интеллект", "я искусственного интеллекта",
            "я — оракул", "я оракул", "я - оракул",
            "я — большая языковая", "я большая языковая",
        )
        for m in markers:
            if low.startswith(m):
                return True
            # короткий ответ, целиком состоящий из самоописания
            if m in low and len(t) < 100:
                return True
        return False

    # ============================================================
    # ОБРАБОТКА ЗАПРОСА
    # ============================================================
    def process_user_query(self, query: str) -> str:
        """Основной метод: обрабатывает запрос пользователя."""
        query = (query or "").strip()
        if not query:
            return "Напишите что-нибудь — и я помогу."

        # 1. Контроль ресурсов
        if not self.resource_monitor.can_process():
            # Памяти мало — выгружаем локальную модель (освобождаем RAM/VRAM)
            self._unload_low_memory()
            if not self.resource_monitor.can_process():
                return "Извините, сейчас я перегружен. Попробуйте чуть позже."

        # 1.5 Агентский режим: Оракул может вызывать инструменты в программе
        # (создавать модель, запускать обучение, переключать вкладки и т.д.)
        if self.agent_mode_enabled() and self._agent_backend_available():
            # Вопросы о статусе/диагностике отвечаем детерминированно —
            # чтобы модель не «нажимала лишнего» на простой вопрос.
            context = self.context_builder.build()
            intent = self.intent_classifier.classify(query, context)
            if intent == Intent.STATUS:
                response = self.build_status_answer()
                self.chat_history.append(ChatMessage("user", query))
                self.chat_history.append(ChatMessage("assistant", response))
                return response
            if intent == Intent.TROUBLESHOOT:
                findings = self.diagnose()
                response = (
                    "Похоже, вот в чём дело:\n" + "\n".join(
                        f"• {f}" for f in findings
                    )
                    if findings else self.build_status_answer()
                )
                self.chat_history.append(ChatMessage("user", query))
                self.chat_history.append(ChatMessage("assistant", response))
                return response

            response = self.run_agent_loop(query)
            if self._is_identity_answer(response):
                response = self.build_status_answer()
            self.chat_history.append(ChatMessage("user", query))
            self.chat_history.append(ChatMessage("assistant", response))
            if len(self.chat_history) > self.MAX_HISTORY:
                self.chat_history = self.chat_history[-self.MAX_HISTORY:]
            return response

        # 2. Контекст и намерение
        context = self.context_builder.build()
        intent = self.intent_classifier.classify(query, context)

        # 2.5 Статус и диагностика — конкретный ответ о состоянии программы
        # (например: «Вы пытаетесь обучить модель, но датасет не загружен»)
        if intent == Intent.STATUS:
            response = self.build_status_answer()
            self.chat_history.append(ChatMessage("user", query))
            self.chat_history.append(ChatMessage("assistant", response))
            return response
        if intent == Intent.TROUBLESHOOT:
            findings = self.diagnose()
            if findings:
                response = "Похоже, вот в чём дело:\n" + "\n".join(
                    f"• {f}" for f in findings
                )
                self.chat_history.append(ChatMessage("user", query))
                self.chat_history.append(ChatMessage("assistant", response))
                return response

        # 2.55 База знаний Оракула — локальный ответ, направляющий к действию.
        #     Отвечаем сразу из базы (без шаблонов и нейросети), если есть
        #     подходящая запись. Это даёт конкретный ответ вместо отписки.
        kb_answer = self.knowledge.search(query)
        if kb_answer:
            self.chat_history.append(ChatMessage("user", query))
            self.chat_history.append(ChatMessage("assistant", kb_answer))
            return kb_answer

        # 2.6 Общий вопрос (классификатор не распознал) — отвечаем по состоянию
        #     программы, а не шаблонным «кто я такая».
        if intent == Intent.GENERAL:
            response = self._build_general_answer(self.diagnose())
            self.chat_history.append(ChatMessage("user", query))
            self.chat_history.append(ChatMessage("assistant", response))
            return response

        # 3. Генерация ответа (шаблоны — базовая линия, работают всегда)
        response = self.response_generator.generate(intent, context, query)

        # Системный промпт с автоконтекстом (сценарий + данные пользователя)
        context_text = self.build_context_text()
        system = build_system_prompt(self.settings, context_text)

        # 4. Локальная модель (GGUF), если уместно и она загружена
        if self._should_use_local_model(intent, context, response):
            local = self._generate_with_local_model(query, context, system=system)
            if local and not self._is_identity_answer(local):
                response = local

        # 5. Внешний API (Яндекс/Сбер) — «гуру» для открытых вопросов.
        #    Шаблоны дают быстрый ответ, но открытые вопросы лучше отвечать
        #    через внешний ИИ (если он настроен).
        open_intents = (
            Intent.GENERAL, Intent.EXPLAIN, Intent.HELP,
            Intent.ASK_MODEL, Intent.ASK_DATA, Intent.ASK_TRAINING,
        )
        if self.api_adapter.is_available() and (
            intent in open_intents or response.startswith("Я пока не знаю")
        ):
            api_response = self.api_adapter.send_request(query, system=system)
            if api_response and not self._is_identity_answer(api_response):
                response = api_response

        # 5. История
        self.chat_history.append(ChatMessage("user", query))
        self.chat_history.append(ChatMessage("assistant", response))
        if len(self.chat_history) > self.MAX_HISTORY:
            self.chat_history = self.chat_history[-self.MAX_HISTORY:]

        log_structured(
            logger, 20, f"Оракул ответил на '{query[:40]}' (намерение: {intent})",
            category=LogCategory.HINT, panel="oraculum", action="answer",
        )
        return response

    # ============================================================
    # ПОДСКАЗКИ И АНАЛИЗ
    # ============================================================
    def get_contextual_hint(self) -> str:
        """Ненавязчивая подсказка без запроса пользователя."""
        context = self.context_builder.build()
        hint = self.response_generator.generate_hint(context)
        self._last_context_hint = hint
        return hint

    def get_workflow_hint(self) -> str:
        context = self.context_builder.build()
        return self.response_generator.generate_workflow_hint(context)

    def analyze_screen(self) -> ScreenAnalysis:
        """Структурированный анализ текущего экрана."""
        context = self.context_builder.build()
        return ScreenAnalysis(
            current_tab=context.current_tab,
            scenario=context.scenario,
            model_exists=context.model_exists,
            dataset_exists=context.dataset_exists,
            has_split=context.has_split,
            model_trained=context.model_trained,
            errors_last_hour=context.errors_last_hour,
            next_step=context.suggest_step,
        )

    # ============================================================
    # ЛОКАЛЬНАЯ МОДЕЛЬ
    # ============================================================
    def load_local_model(self, model_path=None, model_type: str = "gguf") -> bool:
        """
        Загружает локальную GGUF-модель с контролем ресурсов.

        model_path: путь к файлу. Если None — авто-поиск (resolve_local_model_path).
        model_type: оставлен для совместимости (пока поддерживается только "gguf").
        """
        if self.is_local_model_loaded():
            return True
        if not ORACULUM_ENABLE_LOCAL_MODEL:
            logger.info(
                "Локальная модель отключена в конфиге "
                "(ORACULUM_ENABLE_LOCAL_MODEL=False)"
            )
            return False

        path = resolve_local_model_path(str(model_path) if model_path else None)
        if path is None:
            logger.warning("Локальная GGUF-модель не найдена")
            return False
        if not self.resource_monitor.can_load_llm(path):
            logger.warning("Недостаточно ресурсов для загрузки GGUF-модели")
            return False

        self._local_backend = LocalLLMBackend(
            path,
            ctx_size=ORACULUM_GGUF_CTX_SIZE,
            n_gpu_layers=ORACULUM_GGUF_N_GPU_LAYERS,
            max_tokens=ORACULUM_GGUF_MAX_TOKENS,
            temperature=ORACULUM_GGUF_TEMPERATURE,
            top_p=ORACULUM_GGUF_TOP_P,
            verbose=ORACULUM_GGUF_VERBOSE,
            gen_timeout=ORACULUM_LOCAL_GEN_TIMEOUT,
        )
        if not self._local_backend.load():
            self._local_backend = None
            return False

        self._local_model_path = path
        self.resource_monitor.mark_model_loaded(
            path.stat().st_size / (1024 * 1024)
        )
        return True

    def ensure_local_model(self, model_path=None) -> bool:
        """Ленивая загрузка локальной модели (безопасно вызывать повторно)."""
        return self.load_local_model(model_path)

    def unload_local_model(self):
        if self._local_backend is not None:
            self._local_backend.unload()
            self._local_backend = None
        self._local_model_path = None
        self.resource_monitor.mark_model_unloaded()
        logger.info("Локальная модель выгружена")

    def is_local_model_loaded(self) -> bool:
        return self._local_backend is not None and self._local_backend.is_loaded()

    def _unload_low_memory(self):
        """Выгружает локальную модель, если памяти/GPU мало.

        Оракул «выгружает себя»: освобождает RAM/VRAM, чтобы не вешать
        обучение и саму программу. Вызывается при нехватке ресурсов.
        """
        if not self.is_local_model_loaded():
            return
        try:
            import psutil
            vm = psutil.virtual_memory()
            low_ram = vm.percent > 88 or vm.available < 1024 * 1024 * 1024
        except Exception:
            low_ram = False
        if low_ram:
            logger.warning(
                f"Мало оперативной памяти ({vm.percent:.0f}%) — выгружаю "
                "локальную модель, чтобы освободить ресурсы."
            )
            self.unload_local_model()
            return
        # Проверяем видеопамять (если CUDA)
        try:
            import torch
            if torch.cuda.is_available():
                total = torch.cuda.get_device_properties(0).total_memory
                free = total - torch.cuda.memory_allocated(0)
                if free < 512 * 1024 * 1024:  # меньше 512 МБ свободно
                    logger.warning("Мало видеопамяти — выгружаю локальную модель.")
                    self.unload_local_model()
        except Exception:
            pass

    def local_model_error(self) -> Optional[str]:
        if self._local_backend is not None:
            return self._local_backend.get_last_error()
        return None

    def generate_free_text(self, prompt: str) -> Optional[str]:
        """
        Генерирует короткий текст через доступный бэкенд (локальную GGUF-модель
        или внешний API). Используется для ИИ-подсказок (hint_engine).
        Возвращает None, если ни один бэкенд недоступен.
        """
        if self.is_local_model_loaded():
            try:
                return self._local_backend.generate(
                    prompt,
                    system="Ты — помощник OracleAI Studio. Отвечай кратко, "
                           "понятно и по-русски.",
                )
            except Exception:
                return None
        if self.api_adapter.is_available():
            try:
                return self.api_adapter.send_request(prompt)
            except Exception:
                return None
        return None

    def _should_use_local_model(self, intent: str, context, response: str) -> bool:
        """Определяет, стоит ли подключать локальную модель к этому запросу."""
        if self._local_backend is None or not self._local_backend.is_loaded():
            return False
        mode = getattr(context, "hint_mode", "hybrid")
        if mode == "ai":
            return True
        if mode in ("hybrid", "mixed"):
            # В смешанном режиме локальная модель отвечает на общие вопросы
            # или когда шаблон не дал содержательного ответа.
            if response.startswith("Я пока не знаю"):
                return True
            return intent in (Intent.GENERAL, Intent.EXPLAIN)
        return False

    def _generate_with_local_model(self, query: str, context,
                                   system: Optional[str] = None) -> Optional[str]:
        """Генерирует ответ локальной GGUF-моделью с учётом контекста экрана."""
        if self._local_backend is None:
            return None
        if not system:
            system = SYSTEM_PROMPT + " " + self._context_summary(context)
        log_structured(
            logger, 20, f"Запрос к локальной модели: {query[:60]}...",
            category=LogCategory.HINT, panel="oraculum", action="local_model_request",
        )
        result = self._local_backend.generate(query, system=system)
        if result is None and self._local_backend.is_hung():
            # Модель зависла — выгружаем, чтобы не мешать дальнейшей работе.
            logger.warning("Локальная модель зависла; выгружаю")
            self.unload_local_model()
        return result

    # ============================================================
    # АВТОКОНТЕКСТ (сценарий + данные пользователя)
    # ============================================================
    def build_context_text(self) -> str:
        """Формирует текстовый блок контекста для подстановки в системный промпт."""
        context = self.context_builder.build()
        parts = [f"Сценарий: {getattr(context, 'scenario', 'new') or 'new'}."]
        tab = getattr(context, "current_tab", "") or "Проект"
        parts.append(f"Текущая вкладка: {tab}.")

        project = self.shared_state.get("project", {}) or {}
        pname = project.get("name", "")
        mname = project.get("model_name", "") or getattr(context, "last_model_name", "")
        if pname:
            parts.append(f"Проект: {pname}.")
        if mname:
            parts.append(f"Модель: {mname}.")

        user = self.shared_state.get("user", {}) or {}
        username = user.get("username", "")
        level = user.get("level", "beginner")
        if username:
            parts.append(f"Пользователь: {username} (уровень {level}).")

        if getattr(context, "model_exists", False):
            parts.append("Модель создана.")
        if getattr(context, "dataset_exists", False):
            parts.append("Данные загружены.")
        if getattr(context, "has_split", False):
            parts.append("Разбиение применено.")
        if getattr(context, "model_trained", False):
            parts.append("Обучение завершено.")

        suggest = getattr(context, "suggest_step", "")
        if suggest:
            parts.append(f"Следующий шаг: {suggest}.")

        recent = getattr(context, "recent_actions", []) or []
        if recent:
            actions = "; ".join(
                f"{a.get('panel', '?')}:{a.get('action', '?')}" for a in recent[:5]
            )
            parts.append(f"Последние действия: {actions}.")

        return " ".join(parts)

    @staticmethod
    def _context_summary(context) -> str:
        parts = [
            f"Сценарий: {getattr(context, 'scenario', 'new')}.",
            f"Вкладка: {getattr(context, 'current_tab', '') or 'Проект'}.",
        ]
        if getattr(context, "model_exists", False):
            parts.append("Модель создана.")
        if getattr(context, "dataset_exists", False):
            parts.append("Данные загружены.")
        if getattr(context, "has_split", False):
            parts.append("Разбиение применено.")
        if getattr(context, "model_trained", False):
            parts.append("Обучение завершено.")
        return " ".join(parts)

    # ============================================================
    # ИСТОРИЯ
    # ============================================================
    def get_chat_history(self) -> List[ChatMessage]:
        return list(self.chat_history)

    def get_chat_history_dicts(self) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self.chat_history]

    def clear_history(self):
        self.chat_history.clear()

    # ============================================================
    # СЛУЖЕБНОЕ
    # ============================================================
    def get_usage_report(self) -> Dict[str, Any]:
        return self.resource_monitor.get_usage_report()

    def get_status(self) -> Dict[str, Any]:
        """Полная сводка состояния Оракула для панели администратора."""
        usage = self.resource_monitor.get_usage_report()
        local_loaded = self.is_local_model_loaded()
        settings = self.get_settings()
        # Маскируем ключи, чтобы не светить секреты в UI
        api_key = settings.get("api_key", "")
        masked_key = (api_key[:4] + "…" + api_key[-4:]) if len(api_key) > 8 else (
            "•••" if api_key else ""
        )
        return {
            "provider": self.api_adapter.provider or settings.get("provider") or "local",
            "api_available": self.api_adapter.is_available(),
            "api_key_masked": masked_key,
            "api_error": self.api_adapter.get_last_error(),
            "local_model_loaded": local_loaded,
            "local_model_path": str(self._local_model_path) if self._local_model_path else "",
            "local_model_error": self.local_model_error(),
            "auto_context": bool(settings.get("auto_context", True)),
            "system_prompt": settings.get("system_prompt", ""),
            "history_len": len(self.chat_history),
            "usage": usage,
            "agent_enabled": self.agent_mode_enabled(),
            "agent_max_steps": int(settings.get("agent_max_steps", 5)),
            "tools_count": len(self.tools.names()),
            "training_status": self._training_state.get("status", "idle"),
        }

    def __repr__(self) -> str:
        return (
            f"<OraculumAgent api={self.api_adapter.provider or 'offline'} "
            f"history={len(self.chat_history)}>"
        )
