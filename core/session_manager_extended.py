"""
core/session_manager_extended.py
================================
Расширенный менеджер сессий с поддержкой пользователей.

ОТВЕТСТВЕННОСТЬ:
  • Хранение сессий в директории конкретного пользователя
    (data/users/{user_id}/sessions/session_state.json).
  • Доступ к проектам/моделям/датасетам пользователя.
  • Сохранение проектов в хранилище пользователя.

ЗАВИСИМОСТИ:
  • core/session.py       → SessionManager
  • core/user_manager.py  → UserManager
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from core.session import SessionManager
from core.user_manager import UserManager
from core.logger import get_logger

logger = get_logger(__name__)


class ExtendedSessionManager(SessionManager):
    """SessionManager, привязанный к конкретному пользователю."""

    def __init__(self, user_manager: UserManager, username: str):
        self.user_manager = user_manager
        self.username = username
        profile = user_manager.get_profile(username)
        if profile is None:
            raise ValueError(f"Пользователь '{username}' не найден")
        self.profile = profile

        subdirs = user_manager.get_user_subdirs(username)
        self.user_dir = subdirs["projects"].parent
        super().__init__(session_dir=subdirs["sessions"])

        # Храним данные профиля в состоянии сессии
        self.state.total_sessions += 0  # не трогаем счётчик здесь
        self._user_extra = {
            "username": username,
            "user_id": profile.user_id,
            "user_level": profile.user_level.value,
            "is_guest": profile.is_guest,
        }

    # ============================================================
    # ДОСТУП К ДАННЫМ ПОЛЬЗОВАТЕЛЯ
    # ============================================================
    def get_user_projects(self) -> List[Path]:
        return self.user_manager.get_user_projects(self.username)

    def get_user_models(self) -> List[Path]:
        return self.user_manager.get_user_models(self.username)

    def get_user_datasets(self) -> List[Path]:
        return self.user_manager.get_user_datasets(self.username)

    def get_user_subdirs(self) -> Dict[str, Path]:
        return self.user_manager.get_user_subdirs(self.username)

    def save_project_to_user(self, project_data: dict, name: str) -> Optional[Path]:
        return self.user_manager.save_project_to_user(
            self.username, project_data, name
        )

    def get_storage_usage_mb(self) -> float:
        return self.user_manager.get_storage_usage(self.username)

    def count_objects(self) -> Dict[str, int]:
        return self.user_manager.count_user_objects(self.username)

    def get_user_summary(self) -> dict:
        """Сводка для shared_state['user']."""
        counts = self.count_objects()
        return {
            "username": self.username,
            "user_id": self.profile.user_id,
            "level": self.profile.user_level.value,
            "is_guest": self.profile.is_guest,
            "is_admin": self.user_manager.is_admin(self.username),
            "storage_usage_mb": self.get_storage_usage_mb(),
            **counts,
        }

    def _to_dict(self) -> dict:
        """Дополняет сериализацию сессии данными пользователя."""
        d = super()._to_dict()
        d["user"] = self._user_extra
        return d
