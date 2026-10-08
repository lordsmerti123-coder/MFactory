"""
core/oraculum/tools.py
======================
Инструменты (tools) для ИИ-агента «Оракул».

ОТВЕТСТВЕННОСТЬ:
  • Реестр инструментов: имя, описание, параметры, обработчик.
  • Формирование текстовых инструкций для LLM (как вызывать инструменты).
  • Разбор tool-call из ответа модели (JSON: {"action": ..., "args": {...}}).

ИНСТРУМЕНТЫ ДЕЛЯТСЯ НА ДВА ТИПА:
  1. Встроенные (read-only) — выполняются самим агентом без GUI:
     get_context, get_status, list_tools, model_info.
  2. GUI-инструменты — регистрируются MainWindow (переключение вкладок,
     создание модели, запуск обучения и т.д.) и исполняются в главном потоке
     через dispatcher (см. OraculumAgent.set_action_dispatcher).

ФОРМАТ TOOL-CALL (соглашение с LLM):
    {"action": "имя_инструмента", "args": {"параметр": "значение"}}
    {"reply": "обычный текстовый ответ"}

ЗАВИСИМОСТИ:
  • Стандартная библиотека (json, re, dataclasses, typing).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class Tool:
    """Описание одного инструмента."""
    name: str
    description: str
    parameters: Dict[str, str] = field(default_factory=dict)
    handler: Optional[Callable[..., Any]] = None  # None = исполняется через dispatcher

    def to_prompt_line(self) -> str:
        params = ""
        if self.parameters:
            params = "; ".join(f"{k}: {v}" for k, v in self.parameters.items())
            params = f" (параметры: {params})"
        return f"- {self.name}: {self.description}{params}"


class ToolRegistry:
    """Реестр инструментов агента."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, name: str, description: str,
                 parameters: Optional[Dict[str, str]] = None,
                 handler: Optional[Callable[..., Any]] = None):
        self._tools[name] = Tool(
            name=name,
            description=description,
            parameters=parameters or {},
            handler=handler,
        )

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def describe(self) -> str:
        """Текстовое описание всех инструментов для системного промпта."""
        if not self._tools:
            return "(инструменты недоступны)"
        return "\n".join(t.to_prompt_line() for t in self._tools.values())


# ============================================================
# ИНСТРУКЦИИ ДЛЯ LLM
# ============================================================

TOOL_INSTRUCTIONS = (
    "Ты работаешь в режиме ИИ-агента. Ты можешь выполнять действия в программе "
    "через инструменты. Отвечай СТРОГО в одном из двух форматов JSON:\n"
    "1) Чтобы выполнить действие: {\"action\": \"имя_инструмента\", \"args\": {…}}\n"
    "2) Чтобы просто ответить текстом: {\"reply\": \"твой ответ\"}\n"
    "Не выводи ничего кроме JSON. Выполняй инструменты по одному за шаг. "
    "Сначала получи контекст (get_context), если его не хватает."
)

# Few-shot примеры: локальная модель Яндекса намного лучше следует формату,
# когда видит конкретные пары «команда пользователя → JSON» перед ответом.
FEW_SHOT_EXAMPLES = (
    "Примеры. Пользователь пишет команду, ты отвечаешь ТОЛЬКО JSON:\n"
    "Пользователь: переключи вкладку на Обучение\n"
    "Ты: {\"action\": \"switch_tab\", \"args\": {\"tab\": \"Обучение\"}}\n"
    "Пользователь: создай проект МояСеть\n"
    "Ты: {\"action\": \"set_project\", \"args\": {\"name\": \"МояСеть\", "
    "\"model_name\": \"МояСеть-1\"}}\n"
    "Пользователь: сгенерируй данные: сложение от 0 до 10, 200 примеров\n"
    "Ты: {\"action\": \"generate_dataset\", \"args\": {\"task\": \"addition\", "
    "\"num_samples\": 200, \"min\": 0, \"max\": 10}}\n"
    "Пользователь: разбей данные\n"
    "Ты: {\"action\": \"split_dataset\", \"args\": {}}\n"
    "Пользователь: создай трансформер\n"
    "Ты: {\"action\": \"set_architecture\", \"args\": {\"arch\": "
    "\"transformer_seq2seq\"}}\n"
    "Пользователь: задай эпохи 5 и скорость обучения 0.001\n"
    "Ты: {\"action\": \"set_hyperparams\", \"args\": {\"epochs\": 5, \"lr\": 0.001}}\n"
    "Пользователь: запусти обучение\n"
    "Ты: {\"action\": \"start_training\", \"args\": {}}\n"
    "Пользователь: что сейчас в программе?\n"
    "Ты: {\"action\": \"get_state\", \"args\": {}}\n"
    "Пользователь: какая модель лучше для текста?\n"
    "Ты: {\"reply\": \"Для текста лучше Transformer.\"}\n"
    "Не пиши объяснений, поздравлений или текста вокруг JSON."
)


def parse_tool_call(text: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Разбирает ответ модели в поисках tool-call.

    Возвращает (action_name, args) или (None, None), если это обычный текст.
    Поддерживает JSON целиком, markdown-блоки ```json … ``` и JSON с «шумом»
    вокруг (ищет первый сбалансированный объект {}). Оборванный JSON
    (модель может обрезать вывод) достраивается недостающими закрывающими
    скобками.
    """
    if not text:
        return None, None

    s = text.strip()
    # Снимаем markdown-ограждения
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", s, re.DOTALL)
    if fence:
        s = fence.group(1).strip()

    obj = _try_parse_object(s)
    if obj is None:
        obj = _extract_first_json_object(s)

    if not isinstance(obj, dict):
        return None, None

    # Разные именования действия
    action = (
        obj.get("action")
        or obj.get("tool")
        or obj.get("function")
        or obj.get("name")
    )
    if action:
        args = obj.get("args") or obj.get("arguments") or {}
        if not isinstance(args, dict):
            args = {}
        return str(action), args

    return None, None


def _try_parse_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Пытается распарсить JSON-объект из текста. Если объект «обрезан»
    (не хватает закрывающих фигурных скобок) — достраивает их.
    """
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    start = text.find("{")
    if start == -1:
        return None
    end = text.rfind("}")
    if end == -1:
        end = len(text)
    candidate = text[start:end + 1] if end != len(text) else text[start:]

    # Считаем баланс фигурных скобок (вне строк)
    depth = 0
    in_str = False
    esc = False
    for ch in candidate:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
    if depth > 0:
        candidate += "}" * depth

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        return None
    return None


def is_reply_object(obj: Dict[str, Any]) -> Optional[str]:
    """Если объект — обычный ответ, возвращает его текст."""
    if "reply" in obj:
        return str(obj.get("reply", ""))
    if "answer" in obj:
        return str(obj.get("answer", ""))
    if "message" in obj and isinstance(obj.get("message"), str):
        return str(obj["message"])
    return None


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Извлекает первый сбалансированный JSON-объект из строки."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                    except (json.JSONDecodeError, TypeError):
                        pass
                    break
        start = text.find("{", start + 1)
    return None
