"""
core/user_profile.py
====================
Профиль пользователя OracleAI Studio v2.0.

ОТВЕТСТВЕННОСТЬ:
  • Описание пользователя: имя, id, пароль, даты, настройки.
  • Сериализация/десериализация в JSON.
  • Расчёт занимаемого места в хранилище.

ЗАВИСИМОСТИ:
  • core/contracts.py  → HintMode, UserLevel
  • Стандартная библиотека.

ПРАВИЛА:
  • Типы берутся ТОЛЬКО из core/contracts.py.
  • Пароли хранятся только в виде хеша (sha256 + соль).
"""

from __future__ import annotations

import uuid
import hashlib
import secrets
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict

from core.contracts import HintMode, UserLevel


def hash_password(password: str, salt: str = "") -> str:
    """
    Хеширует пароль. Если соль не передана — генерируется новая.
    Формат: "sha256$<salt>$<hash>".
    """
    if not salt:
        salt = secrets.token_hex(8)
    digest = hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()
    return f"sha256${salt}${digest}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Проверяет пароль против сохранённого хеша."""
    try:
        _, salt, expected = stored_hash.split("$")
    except (ValueError, AttributeError):
        return False
    return hash_password(password, salt) == stored_hash


@dataclass
class UserProfile:
    """Профиль пользователя локальной системы."""
    username: str = ""
    user_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    password_hash: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_login: str = ""
    hint_mode: HintMode = HintMode.HYBRID
    user_level: UserLevel = UserLevel.BEGINNER
    total_projects: int = 0
    total_models: int = 0
    total_datasets: int = 0
    storage_usage_mb: float = 0.0
    preferences: Dict[str, Any] = field(default_factory=dict)
    is_guest: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["hint_mode"] = self.hint_mode.value
        d["user_level"] = self.user_level.value
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "UserProfile":
        return UserProfile(
            username=d.get("username", ""),
            user_id=d.get("user_id", str(uuid.uuid4())[:12]),
            password_hash=d.get("password_hash", ""),
            created_at=d.get("created_at", datetime.now().isoformat()),
            last_login=d.get("last_login", ""),
            hint_mode=HintMode.from_str(d.get("hint_mode", "hybrid")),
            user_level=UserLevel(d.get("user_level", "beginner"))
            if d.get("user_level", "beginner") in (
                "beginner", "intermediate", "advanced"
            ) else UserLevel.BEGINNER,
            total_projects=d.get("total_projects", 0),
            total_models=d.get("total_models", 0),
            total_datasets=d.get("total_datasets", 0),
            storage_usage_mb=d.get("storage_usage_mb", 0.0),
            preferences=d.get("preferences", {}),
            is_guest=d.get("is_guest", False),
        )

    def __repr__(self) -> str:
        return (
            f"<UserProfile username='{self.username}' "
            f"level='{self.user_level.value}' guest={self.is_guest}>"
        )
