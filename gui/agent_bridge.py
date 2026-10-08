"""
gui/agent_bridge.py
===================
Мост между ИИ-агентом Оракулом (core) и GUI-инструментами.

ОТВЕТСТВЕННОСТЬ:
  • Потокобезопасное исполнение инструментов в ГЛАВНОМ потоке GUI.
  • Оракул может выполняться в фоновом QThread (локальная модель) —
    а действия с интерфейсом (создать модель, запустить обучение и т.д.)
    обязаны выполняться в главном потоке.

МЕХАНИЗМ:
  Фоновый поток эмитит сигнал request → главный поток исполняет слот
  _on_request (обычный QueuedConnection), результат возвращается через
  threading.Event + общий словарь (надёжно в PyQt5, в отличие от
  QMetaObject.invokeMethod с BlockingQueuedConnection и возвращаемым аргументом,
  который в PyQt5 зависает).

ИСПОЛЬЗОВАНИЕ (в MainWindow):
    from gui.agent_bridge import AgentBridge
    self.agent_bridge = AgentBridge(self._execute_agent_action)
    self.oraculum_agent.set_action_dispatcher(self.agent_bridge.dispatch)

ЗАВИСИМОСТИ:
  • PyQt5 (QtCore: QObject, pyqtSignal, pyqtSlot, Qt, QThread)
  • Стандартная библиотека (json, threading)
"""

from __future__ import annotations

import json
import threading
from typing import Any, Callable, Dict, Optional

from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal, pyqtSlot

REQUEST_TIMEOUT_SECONDS = 30


class _AgentActionRunner(QObject):
    """Слот в главном потоке, исполняющий инструмент."""

    request = pyqtSignal(str, str, str)  # (request_id, name, args_json)

    def __init__(self, handler: Callable[[str, Dict[str, Any]], str],
                 parent: Optional[QObject] = None):
        super().__init__(parent)
        self._handler = handler
        self.bridge: Optional["AgentBridge"] = None

    @pyqtSlot(str, str, str)
    def _on_request(self, request_id: str, name: str, args_json: str):
        try:
            args = json.loads(args_json or "{}") or {}
        except (json.JSONDecodeError, TypeError):
            args = {}
        try:
            result = self._handler(name, args)
        except Exception as e:  # инструмент не должен ронять приложение
            result = json.dumps({"error": f"{type(e).__name__}: {e}"},
                                ensure_ascii=False)
        if self.bridge is not None:
            self.bridge._finish(request_id, result)


class AgentBridge:
    """Раздаёт вызовы инструментов в главный поток GUI."""

    def __init__(self, handler: Callable[[str, Dict[str, Any]], str]):
        self._handler = handler
        self._runner = _AgentActionRunner(handler)
        self._runner.bridge = self
        # QueuedConnection: если сигнал испускается из другого потока,
        # слот выполнится в главном потоке через его event loop.
        self._runner.request.connect(self._runner._on_request, Qt.QueuedConnection)

        self._lock = threading.Lock()
        self._pending: Dict[str, tuple] = {}  # request_id -> (Event, holder)
        self._seq = 0

    def dispatch(self, name: str, args: Dict[str, Any]) -> str:
        """
        Выполняет инструмент с именем name и аргументами args.
        Потокобезопасно: из фонового потока вызов маршалится в главный.
        """
        args_json = json.dumps(args or {}, ensure_ascii=False)

        # Из главного потока — исполняем напрямую.
        if self._runner.thread() == QThread.currentThread():
            return self._handler(name, args or {})

        # Из фонового потока — через сигнал + ожидание результата.
        with self._lock:
            self._seq += 1
            request_id = f"req{self._seq}"
            event = threading.Event()
            holder: Dict[str, Any] = {"result": None}
            self._pending[request_id] = (event, holder)

        self._runner.request.emit(request_id, name, args_json)

        if not event.wait(timeout=REQUEST_TIMEOUT_SECONDS):
            with self._lock:
                self._pending.pop(request_id, None)
            return json.dumps(
                {"error": f"таймаут исполнения инструмента '{name}'"},
                ensure_ascii=False,
            )
        return holder.get("result") or "{}"

    def _finish(self, request_id: str, result: str):
        """Вызывается из главного потока после исполнения инструмента."""
        with self._lock:
            entry = self._pending.get(request_id)
        if entry is not None:
            entry[1]["result"] = result
            entry[0].set()
