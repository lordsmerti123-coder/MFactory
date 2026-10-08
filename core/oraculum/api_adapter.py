"""
core/oraculum/api_adapter.py
============================
Адаптер внешних API для Оракула: Яндекс GPT и Сбер GigaChat.

ОТВЕТСТВЕННОСТЬ:
  • Хранение конфигурации провайдера (ключи, endpoint).
  • Отправка запросов с таймаутами и повторными попытками.
  • Безопасное поведение без сети: is_available() == False, send_request() -> None.
  • Логирование в категорию ORACULUM.

ДОКУМЕНТАЦИЯ (без интернета):
  Яндекс GPT:  POST https://llm.api.cloud.yandex.net/foundationModels/v1/completion
               Header: Authorization: Api-Key <key>
               Body: {"modelUri": "gpt://<folder_id>/yandexgpt/latest",
                      "completionOptions": {"stream": false, "temperature": 0.6, "maxTokens": 200},
                      "messages": [{"role": "system", "text": "..."}, {"role": "user", "text": "..."}]}
               Ответ: result.alternatives[0].message.text

  Сбер GigaChat:
               Шаг 1: POST https://ngw.devices.sberbank.ru:9443/api/v2/oauth
                      Header: Authorization: Basic base64(client_id:client_secret)
                      Body: scope=GIGACHAT_API_PERS
                      Ответ: access_token, expires_in
               Шаг 2: POST https://gigachat.devices.sberbank.ru/api/v1/chat/completions
                      Header: Authorization: Bearer <token>
                      Body: {"model": "GigaChat:latest", "messages": [...], ...}
                      Ответ: choices[0].message.content
"""

from __future__ import annotations

import base64
import time
from datetime import datetime, timedelta
from typing import Optional

from core.oraculum.resource_monitor import ResourceMonitor
from core.logger import get_logger, LogCategory, log_structured

logger = get_logger("Oraculum")

YANDEX_ENDPOINT = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
SBER_AUTH_ENDPOINT = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
SBER_CHAT_ENDPOINT = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

SYSTEM_PROMPT = (
    "Ты Оракул — помощник по обучению нейросетей в программе OracleAI Studio. "
    "Отвечай кратко, понятно и по-русски. Помогай новичкам."
)


class APIAdapter:
    """Адаптер для внешних LLM API."""

    def __init__(self, resource_monitor: Optional[ResourceMonitor] = None):
        self.provider = None  # "yandex" | "sber" | None
        self.api_key = None
        self.folder_id = None
        self.client_id = None
        self.client_secret = None
        self.endpoint = None
        self.timeout = 10
        self.max_retries = 2
        self._last_error = None
        self.resource_monitor = resource_monitor or ResourceMonitor()
        self._sber_token = None
        self._sber_token_expiry = None

    # ============================================================
    # КОНФИГУРАЦИЯ ПРОВАЙДЕРА
    # ============================================================
    def set_yandex_gpt(self, api_key: str, folder_id: str):
        self.provider = "yandex"
        self.api_key = api_key
        self.folder_id = folder_id
        self.client_id = None
        self.client_secret = None
        self.endpoint = YANDEX_ENDPOINT
        logger.info("Настроен провайдер: Яндекс GPT")

    def set_sber_gpt(self, client_id: str, client_secret: str):
        self.provider = "sber"
        self.client_id = client_id
        self.client_secret = client_secret
        self.api_key = None
        self.endpoint = SBER_CHAT_ENDPOINT
        logger.info("Настроен провайдер: Сбер GigaChat")

    def disable(self):
        self.provider = None
        self.api_key = None
        self.client_id = None
        self.client_secret = None

    def is_available(self) -> bool:
        if self.provider == "yandex":
            return bool(self.api_key and self.folder_id)
        if self.provider == "sber":
            return bool(self.client_id and self.client_secret)
        return False

    def get_last_error(self) -> Optional[str]:
        return self._last_error

    # ============================================================
    # ОТПРАВКА ЗАПРОСА
    # ============================================================
    def send_request(self, prompt: str, max_tokens: int = 100,
                     temperature: float = 0.6,
                     system: Optional[str] = None) -> Optional[str]:
        """Отправляет запрос к API. Возвращает текст ответа или None."""
        if not self.is_available():
            self._last_error = "Провайдер не настроен"
            return None
        if not self.resource_monitor.can_use_api():
            self._last_error = "Превышен лимит запросов к API за минуту"
            return None

        log_structured(
            logger, 20,
            f"Запрос к {self.provider}: {prompt[:60]}...",
            category=LogCategory.HINT, panel="oraculum", action="api_request",
        )

        try:
            if self.provider == "yandex":
                return self._send_yandex(prompt, max_tokens, temperature, system)
            if self.provider == "sber":
                return self._send_sber(prompt, max_tokens, temperature, system)
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"Ошибка API {self.provider}: {e}")
        return None

    def chat(self, messages, max_tokens: int = 200,
             temperature: float = 0.4,
             system: Optional[str] = None) -> Optional[str]:
        """
        Многошаговый диалог (для агентского цикла tools).

        messages — список {"role": "user"/"assistant", "content": "..."}.
        Возвращает текст ответа или None.
        """
        if not self.is_available():
            self._last_error = "Провайдер не настроен"
            return None
        if not self.resource_monitor.can_use_api():
            self._last_error = "Превышен лимит запросов к API за минуту"
            return None

        sys_msg = system or SYSTEM_PROMPT
        try:
            if self.provider == "yandex":
                return self._send_yandex_messages(messages, sys_msg, max_tokens, temperature)
            if self.provider == "sber":
                return self._send_sber_messages(messages, sys_msg, max_tokens, temperature)
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"Ошибка API {self.provider}: {e}")
        return None

    def _send_yandex_messages(self, messages, system: str,
                              max_tokens: int, temperature: float) -> Optional[str]:
        import requests
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json",
        }
        body_messages = [{"role": "system", "text": system}]
        for m in messages:
            body_messages.append({"role": m.get("role", "user"), "text": m.get("content", "")})
        data = {
            "modelUri": f"gpt://{self.folder_id}/yandexgpt/latest",
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": max_tokens,
            },
            "messages": body_messages,
        }
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(self.endpoint, headers=headers,
                                         json=data, timeout=self.timeout)
                if response.status_code == 200:
                    result = response.json()
                    return result["result"]["alternatives"][0]["message"]["text"]
                self._last_error = f"Yandex API {response.status_code}: {response.text[:200]}"
            except Exception as e:
                self._last_error = str(e)
            if attempt < self.max_retries:
                time.sleep(0.5 * (attempt + 1))
        return None

    def _send_sber_messages(self, messages, system: str,
                            max_tokens: int, temperature: float) -> Optional[str]:
        import requests
        token = self._get_sber_token()
        if token is None:
            return None
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        body_messages = [{"role": "system", "content": system}]
        for m in messages:
            body_messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})
        data = {
            "model": "GigaChat:latest",
            "messages": body_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(self.endpoint, headers=headers,
                                         json=data, timeout=self.timeout)
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                self._last_error = f"Sber API {response.status_code}: {response.text[:200]}"
            except Exception as e:
                self._last_error = str(e)
            if attempt < self.max_retries:
                time.sleep(0.5 * (attempt + 1))
        return None

    def _send_yandex(self, prompt: str, max_tokens: int, temperature: float,
                     system: Optional[str] = None) -> Optional[str]:
        import requests
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "modelUri": f"gpt://{self.folder_id}/yandexgpt/latest",
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": max_tokens,
            },
            "messages": [
                {"role": "system", "text": system or SYSTEM_PROMPT},
                {"role": "user", "text": prompt},
            ],
        }
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(self.endpoint, headers=headers,
                                         json=data, timeout=self.timeout)
                if response.status_code == 200:
                    result = response.json()
                    text = result["result"]["alternatives"][0]["message"]["text"]
                    return text
                self._last_error = f"Yandex API {response.status_code}: {response.text[:200]}"
            except Exception as e:
                self._last_error = str(e)
            if attempt < self.max_retries:
                time.sleep(0.5 * (attempt + 1))
        return None

    def _send_sber(self, prompt: str, max_tokens: int, temperature: float,
                   system: Optional[str] = None) -> Optional[str]:
        import requests
        token = self._get_sber_token()
        if token is None:
            return None
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "GigaChat:latest",
            "messages": [
                {"role": "system", "content": system or SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(self.endpoint, headers=headers,
                                         json=data, timeout=self.timeout)
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                self._last_error = f"Sber API {response.status_code}: {response.text[:200]}"
            except Exception as e:
                self._last_error = str(e)
            if attempt < self.max_retries:
                time.sleep(0.5 * (attempt + 1))
        return None

    def _get_sber_token(self) -> Optional[str]:
        """Возвращает валидный токен, обновляя его при необходимости."""
        if self._sber_token and self._sber_token_expiry and \
                datetime.now() < self._sber_token_expiry:
            return self._sber_token
        import requests
        auth_str = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()
        headers = {
            "Authorization": f"Basic {auth_str}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {"scope": "GIGACHAT_API_PERS"}
        try:
            response = requests.post(SBER_AUTH_ENDPOINT, headers=headers,
                                     data=data, timeout=5, verify=False)
            if response.status_code != 200:
                self._last_error = f"Sber auth {response.status_code}: {response.text[:200]}"
                return None
            token_data = response.json()
            self._sber_token = token_data["access_token"]
            expires_in = token_data.get("expires_in", 1800)
            self._sber_token_expiry = datetime.now() + timedelta(seconds=expires_in)
            return self._sber_token
        except Exception as e:
            self._last_error = str(e)
            return None
