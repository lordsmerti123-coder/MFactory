"""
core/oraculum/local_llm.py
==========================
Локальная LLM-модель (GGUF) для Оракула через llama-cpp-python.

ОТВЕТСТВЕННОСТЬ:
  • Ленивая загрузка GGUF-модели (Llama из llama_cpp).
  • Генерация ответа через create_chat_completion.
  • Таймаут генерации (защита от «зависания» инференса).
  • Контроль повторного входа (нельзя генерировать параллельно).

БЕЗОПАСНОСТЬ:
  • llama_cpp импортируется только при загрузке модели.
  • Если llama_cpp нет или модель отсутствует — load() возвращает False,
    исключения наружу не выбрасываются.
  • Генерация выполняется в отдельном потоке с таймаутом; при превышении
    модель помечается «зависшей», и последующие вызовы отклоняются до выгрузки.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.logger import get_logger, LogCategory, log_structured

logger = get_logger("Oraculum")

# Системный промпт — роль Оракула в чате.
SYSTEM_PROMPT = (
    "Ты Оракул — помощник по обучению нейросетей в программе OracleAI Studio. "
    "Отвечай кратко, понятно и по-русски. Помогай новичкам."
)


class LocalLLMBackend:
    """Обёртка над локальной GGUF-моделью (llama.cpp)."""

    def __init__(self, model_path, ctx_size: int = 2048, n_gpu_layers: int = -1,
                 max_tokens: int = 256, temperature: float = 0.3, top_p: float = 0.9,
                 verbose: bool = False, gen_timeout: float = 90.0):
        self.model_path = Path(model_path)
        self.ctx_size = ctx_size
        self.n_gpu_layers = n_gpu_layers
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.verbose = verbose
        self.gen_timeout = gen_timeout

        self._llm = None
        self._loaded = False
        self._hung = False
        self._last_error: Optional[str] = None
        self._lock = threading.Lock()

    # ============================================================
    # ЗАГРУЗКА / ВЫГРУЗКА
    # ============================================================
    def is_loaded(self) -> bool:
        return self._loaded and self._llm is not None

    def load(self) -> bool:
        """Загружает модель. Идемпотентно, потокобезопасно."""
        with self._lock:
            if self.is_loaded():
                return True
            if not self.model_path.exists():
                self._last_error = f"Файл модели не найден: {self.model_path}"
                logger.warning(self._last_error)
                return False

            try:
                from llama_cpp import Llama
            except ImportError as e:
                self._last_error = f"llama_cpp не установлен: {e}"
                logger.warning(self._last_error)
                return False

            try:
                logger.info(
                    f"Загрузка GGUF-модели {self.model_path.name} "
                    f"(ctx={self.ctx_size}, gpu_layers={self.n_gpu_layers})..."
                )
                self._llm = Llama(
                    model_path=str(self.model_path),
                    n_gpu_layers=self.n_gpu_layers,
                    n_ctx=self.ctx_size,
                    verbose=self.verbose,
                )
                self._loaded = True
                self._hung = False
                self._last_error = None
                log_structured(
                    logger, 20, "Локальная GGUF-модель загружена",
                    category=LogCategory.HINT, panel="oraculum", action="local_model_loaded",
                )
                return True
            except Exception as e:
                self._last_error = f"Не удалось загрузить модель: {e}"
                self._llm = None
                self._loaded = False
                logger.error(self._last_error)
                return False

    def unload(self):
        """Выгружает модель и освобождает память."""
        with self._lock:
            if self._llm is not None:
                try:
                    del self._llm
                except Exception:
                    pass
            self._llm = None
            self._loaded = False
            self._hung = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("Локальная GGUF-модель выгружена")

    # ============================================================
    # ГЕНЕРАЦИЯ
    # ============================================================
    def generate(self, prompt: str,
                 system: Optional[str] = None,
                 max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> Optional[str]:
        """
        Генерирует ответ. Возвращает текст или None при ошибке/таймауте.
        Не бросает исключений наружу.
        """
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.generate_messages(messages, max_tokens, temperature)

    def generate_messages(self, messages: List[Dict[str, str]],
                          max_tokens: Optional[int] = None,
                          temperature: Optional[float] = None) -> Optional[str]:
        """
        Генерирует ответ по полному списку сообщений (для агентского цикла).
        Возвращает текст или None при ошибке/таймауте.
        """
        if not self.is_loaded():
            self._last_error = "Модель не загружена"
            return None
        if self._hung:
            self._last_error = "Модель зависла при предыдущей генерации; требуется перезагрузка"
            logger.warning(self._last_error)
            return None

        mt = max_tokens if max_tokens is not None else self.max_tokens
        tp = temperature if temperature is not None else self.temperature

        try:
            # YandexGPT-5-Lite: нестандартный шаблон чата — системное сообщение
            # при применении шаблона теряется, и модель отвечает как «обычный
            # помощник», игнорируя инструкции и инструменты. Поэтому системный
            # промпт встраиваем прямо в user-сообщение (проверено эмпирически:
            # так модель корректно вызывает инструменты).
            chat_messages = self._fold_system_into_user(messages)
            # Генерация в отдельном потоке с таймаутом — защита от зависания.
            # shutdown(wait=False): при таймауте не ждём «зависший» поток.
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(self._run_completion, chat_messages, mt, tp)
            try:
                return future.result(timeout=self.gen_timeout)
            except FutureTimeout:
                self._hung = True
                self._last_error = f"Генерация превысила таймаут {self.gen_timeout}с"
                logger.error(self._last_error)
                return None
            except Exception as e:
                self._last_error = f"Ошибка генерации: {e}"
                logger.error(self._last_error)
                return None
            finally:
                executor.shutdown(wait=False)
        except Exception as e:
            self._last_error = f"Ошибка запуска генерации: {e}"
            logger.error(self._last_error)
            return None

    @staticmethod
    def _fold_system_into_user(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Встраивает системные сообщения в последнее user-сообщение.

        Шаблон YandexGPT-5-Lite теряет системные сообщения. Чтобы инструкции
        и описание инструментов дошли до модели, они приклеиваются к тексту
        пользователя (перед самим вопросом).
        """
        system_parts = [m.get("content", "") for m in messages
                        if m.get("role") == "system"]
        if not system_parts:
            return messages
        result = [dict(m) for m in messages if m.get("role") != "system"]
        for i in range(len(result) - 1, -1, -1):
            if result[i].get("role") == "user":
                result[i] = dict(result[i])
                result[i]["content"] = (
                    "\n\n".join(part for part in system_parts if part)
                    + "\n\n" + result[i]["content"]
                )
                break
        return result

    def _run_completion(self, messages: List[Dict[str, str]],
                        max_tokens: int, temperature: float) -> Optional[str]:
        """Собственно вызов llama_cpp (выполняется в рабочем потоке)."""
        out = self._llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            top_p=self.top_p,
            max_tokens=max_tokens,
        )
        choices = out.get("choices") or []
        if not choices:
            return None
        return (choices[0].get("message") or {}).get("content")

    # ============================================================
    # СЛУЖЕБНОЕ
    # ============================================================
    def get_last_error(self) -> Optional[str]:
        return self._last_error

    def is_hung(self) -> bool:
        return self._hung
