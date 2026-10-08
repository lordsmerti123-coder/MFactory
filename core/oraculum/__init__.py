"""
core/oraculum
=============
Пакет ИИ-агента «Оракул» (Oraculum).

Состав:
  • agent               — главный класс OraculumAgent
  • context_builder     — сбор контекста из shared_state и логов
  • intent_classifier   — определение намерения пользователя
  • response_generator  — генерация ответов по шаблонам
  • api_adapter         — адаптер для Яндекс GPT / Сбер GigaChat
  • resource_monitor    — контроль памяти и CPU
  • templates           — локальные шаблоны ответов

Приоритет генерации ответа:
  1. Локальные шаблоны (всегда доступны, без сети)
  2. Локальная модель (GGUF/ONNX), если подключена
  3. Внешние API (Яндекс/Сбер), если заданы ключи
"""

from core.oraculum.agent import OraculumAgent
from core.oraculum.context_builder import AgentContext, ContextBuilder
from core.oraculum.intent_classifier import Intent, IntentClassifier
from core.oraculum.resource_monitor import ResourceMonitor
from core.oraculum.local_llm import LocalLLMBackend

__all__ = [
    "OraculumAgent",
    "AgentContext",
    "ContextBuilder",
    "Intent",
    "IntentClassifier",
    "ResourceMonitor",
    "LocalLLMBackend",
]
