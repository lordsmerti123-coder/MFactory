"""
core/user_manager.py
====================
Менеджер пользователей OracleAI Studio v2.0.

ОТВЕТСТВЕННОСТЬ:
  • Регистрация / авторизация / удаление пользователей (локально, без интернета).
  • Создание пользовательских директорий (projects, models, datasets, exports, sessions).
  • Индекс всех пользователей в data/system/users_index.json.
  • Разделение данных по пользователям.

СТРУКТУРА ДИРЕКТОРИЙ:
  data/
  ├── users/
  │   ├── {user_id}/
  │   │   ├── profile.json
  │   │   ├── projects/
  │   │   ├── models/
  │   │   ├── datasets/
  │   │   ├── exports/
  │   │   └── sessions/
  ├── shared/
  │   └── pretrained_models/
  └── system/
      └── users_index.json

ЗАВИСИМОСТИ:
  • core/user_profile.py  → UserProfile, hash_password, verify_password
  • config.py             → DATA_DIR
  • Стандартная библиотека.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from core.user_profile import UserProfile, hash_password, verify_password
from core.logger import get_logger
from config import DATA_DIR, ADMIN_USERNAME

logger = get_logger(__name__)

# ============================================================
# КОНСТАНТЫ МОДУЛЯ
# ============================================================
USERS_DIR = DATA_DIR / "users"
SHARED_DIR = DATA_DIR / "shared"
SYSTEM_DIR = DATA_DIR / "system"
USERS_INDEX_FILE = SYSTEM_DIR / "users_index.json"
PROFILE_FILE_NAME = "profile.json"

MAX_PROJECTS_PER_USER = 100
GUEST_PREFIX = "guest"
USERNAME_RE = re.compile(r"^[a-zA-Z0-9_а-яА-ЯёЁ\-]{3,32}$")


def _slugify(name: str) -> str:
    """Приводит имя пользователя к безопасному виду для имён папок."""
    s = re.sub(r"[^a-zA-Z0-9а-яА-ЯёЁ_\-]", "_", name).strip("_")
    return s or "user"


class UserManager:
    """Управление локальными пользователями."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.users_dir = self.data_dir / "users"
        self.system_dir = self.data_dir / "system"
        self.index_file = self.system_dir / USERS_INDEX_FILE.name
        self.users: Dict[str, UserProfile] = {}  # user_id -> profile
        self._by_name: Dict[str, str] = {}       # username -> user_id
        self.current_user_id: Optional[str] = None

        self._ensure_directories()
        self._load_index()

    # ============================================================
    # ДИРЕКТОРИИ
    # ============================================================
    def _ensure_directories(self):
        (self.users_dir).mkdir(parents=True, exist_ok=True)
        (self.system_dir).mkdir(parents=True, exist_ok=True)
        (self.data_dir / "shared" / "pretrained_models").mkdir(
            parents=True, exist_ok=True
        )

    def get_user_dir(self, username: str) -> Path:
        """Директория пользователя (создаётся при необходимости)."""
        profile = self.get_profile(username)
        uid = profile.user_id if profile else _slugify(username)
        user_dir = self.users_dir / uid
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir

    def get_user_subdirs(self, username: str) -> Dict[str, Path]:
        """Возвращает служебные поддиректории пользователя."""
        base = self.get_user_dir(username)
        subdirs = {
            "projects": base / "projects",
            "models": base / "models",
            "datasets": base / "datasets",
            "exports": base / "exports",
            "sessions": base / "sessions",
        }
        for p in subdirs.values():
            p.mkdir(parents=True, exist_ok=True)
        return subdirs

    # ============================================================
    # ИНДЕКС ПОЛЬЗОВАТЕЛЕЙ
    # ============================================================
    def _load_index(self):
        self.users.clear()
        self._by_name.clear()

        # 1) Сканируем папки пользователей (профили)
        for profile_path in self.users_dir.glob(f"*/{PROFILE_FILE_NAME}"):
            try:
                with open(profile_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                profile = UserProfile.from_dict(data)
                if profile.username:
                    self.users[profile.user_id] = profile
                    self._by_name[profile.username] = profile.user_id
            except (json.JSONDecodeError, KeyError, OSError) as e:
                logger.warning(f"Не удалось прочитать профиль {profile_path}: {e}")

        # 2) Индексный файл (список имён для быстрого отображения)
        if self.index_file.exists():
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    index = json.load(f)
                for entry in index.get("users", []):
                    uid = entry.get("user_id", "")
                    name = entry.get("username", "")
                    if uid and name:
                        self._by_name.setdefault(name, uid)
                        if uid not in self.users:
                            self.users[uid] = UserProfile.from_dict(entry)
            except (json.JSONDecodeError, OSError):
                logger.warning(f"Не удалось прочитать индекс {self.index_file}")

        logger.info(f"Загружено пользователей: {len(self.users)}")

    def _save_index(self):
        entries = []
        for uid, profile in self.users.items():
            entries.append({
                "user_id": uid,
                "username": profile.username,
                "created_at": profile.created_at,
                "last_login": profile.last_login,
                "is_guest": profile.is_guest,
                "storage_usage_mb": profile.storage_usage_mb,
            })
        data = {"format_version": "2.0", "users": entries}
        try:
            self.index_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.index_file.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            if self.index_file.exists():
                self.index_file.unlink()
            tmp.rename(self.index_file)
        except OSError as e:
            logger.error(f"Не удалось сохранить индекс пользователей: {e}")

    # ============================================================
    # СОЗДАНИЕ / АВТОРИЗАЦИЯ / УДАЛЕНИЕ
    # ============================================================
    def create_user(self, username: str, password_hash: str = "",
                    is_guest: bool = False) -> Optional[UserProfile]:
        """
        Создаёт пользователя. Возвращает профиль или None при ошибке.
        Пароль принимается УЖЕ хешированным (hash_password).
        """
        username = username.strip()
        if not is_guest and not USERNAME_RE.match(username):
            logger.warning(f"Недопустимое имя пользователя: '{username}'")
            return None
        if username in self._by_name:
            logger.warning(f"Пользователь '{username}' уже существует")
            return None

        profile = UserProfile(
            username=username,
            password_hash=password_hash,
            is_guest=is_guest,
            created_at=datetime.now().isoformat(),
            last_login=datetime.now().isoformat(),
        )
        # Проверка уникальности id
        while profile.user_id in self.users:
            profile.user_id = UserProfile().user_id

        user_dir = self.users_dir / profile.user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("projects", "models", "datasets", "exports", "sessions"):
            (user_dir / sub).mkdir(parents=True, exist_ok=True)

        self.users[profile.user_id] = profile
        self._by_name[username] = profile.user_id
        self._save_profile(profile)
        self._save_index()

        logger.info(f"Создан пользователь '{username}' (id={profile.user_id})")
        return profile

    def create_guest(self) -> UserProfile:
        """Создаёт/возвращает гостевого пользователя."""
        existing = self.get_profile("guest")
        if existing:
            return existing
        profile = self.create_user("guest", is_guest=True)
        return profile

    def authenticate(self, username: str, password: str) -> bool:
        """Проверяет имя и пароль (открытый пароль)."""
        profile = self.get_profile(username)
        if profile is None:
            return False
        if profile.is_guest:
            return True  # гость без пароля
        if not profile.password_hash:
            return False
        return verify_password(password, profile.password_hash)

    def switch_user(self, username: str) -> bool:
        """Переключает текущего пользователя. Обновляет last_login."""
        profile = self.get_profile(username)
        if profile is None:
            return False
        self.current_user_id = profile.user_id
        profile.last_login = datetime.now().isoformat()
        self._save_profile(profile)
        self._save_index()
        logger.info(f"Текущий пользователь: '{username}'")
        return True

    def delete_user(self, username: str) -> bool:
        """Удаляет пользователя вместе с его данными."""
        profile = self.get_profile(username)
        if profile is None or profile.is_guest:
            logger.warning(f"Нельзя удалить '{username}' (не найден или гость)")
            return False
        try:
            user_dir = self.users_dir / profile.user_id
            if user_dir.exists():
                shutil.rmtree(user_dir)
            self.users.pop(profile.user_id, None)
            self._by_name.pop(username, None)
            if self.current_user_id == profile.user_id:
                self.current_user_id = None
            self._save_index()
            logger.info(f"Пользователь '{username}' удалён")
            return True
        except OSError as e:
            logger.error(f"Ошибка удаления '{username}': {e}")
            return False

    # ============================================================
    # ДОСТУП К ПРОФИЛЯМ
    # ============================================================
    def get_profile(self, username_or_id: str) -> Optional[UserProfile]:
        """Ищет профиль по имени или user_id."""
        if not username_or_id:
            return None
        # По id
        if username_or_id in self.users:
            return self.users[username_or_id]
        # По имени
        uid = self._by_name.get(username_or_id)
        if uid:
            return self.users.get(uid)
        return None

    def get_current_user(self) -> Optional[UserProfile]:
        if self.current_user_id:
            return self.users.get(self.current_user_id)
        return None

    def get_all_users(self) -> List[UserProfile]:
        return sorted(self.users.values(), key=lambda p: p.username)

    def list_usernames(self) -> List[str]:
        return [p.username for p in self.get_all_users()]

    def get_active_user_count(self) -> int:
        return len(self.users)

    # ============================================================
    # АДМИНИСТРАТОР
    # ============================================================
    def is_admin(self, username_or_id: str) -> bool:
        """Возвращает True, если пользователь — администратор."""
        profile = self.get_profile(username_or_id)
        if profile is None:
            return False
        return profile.username.strip().lower() == ADMIN_USERNAME.strip().lower()

    def is_current_admin(self) -> bool:
        """Возвращает True, если текущий пользователь — администратор."""
        return self.is_admin(self.current_user_id or "")


    # ============================================================
    # ПРОФИЛЬ
    # ============================================================
    def _save_profile(self, profile: UserProfile):
        path = self.users_dir / profile.user_id / PROFILE_FILE_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(profile.to_dict(), f, ensure_ascii=False, indent=2)
        if path.exists():
            path.unlink()
        tmp.rename(path)

    def update_profile(self, username: str, **kwargs) -> bool:
        """Обновляет поля профиля (hint_mode, user_level, preferences и т.д.)."""
        profile = self.get_profile(username)
        if profile is None:
            return False
        for key, value in kwargs.items():
            if key in ("username", "user_id", "password_hash"):
                continue  # запрещено менять напрямую
            if hasattr(profile, key):
                setattr(profile, key, value)
        self._save_profile(profile)
        return True

    def change_password(self, username: str, new_password: str) -> bool:
        profile = self.get_profile(username)
        if profile is None:
            return False
        profile.password_hash = hash_password(new_password)
        self._save_profile(profile)
        return True

    def set_user_level(self, username: str, level: str) -> bool:
        return self.update_profile(username, user_level=level)

    def get_storage_usage(self, username: str) -> float:
        """Размер всех файлов пользователя в МБ."""
        user_dir = self.get_user_dir(username)
        total = 0.0
        for p in user_dir.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
        return round(total / (1024 * 1024), 2)

    def count_user_objects(self, username: str) -> Dict[str, int]:
        """Считает проекты/модели/датасеты пользователя."""
        subdirs = self.get_user_subdirs(username)
        return {
            "projects": len(list(subdirs["projects"].glob("*.oai"))),
            "models": len(list(subdirs["models"].glob("*.pth")))
                      + len(list(subdirs["models"].glob("*.pt"))),
            "datasets": len(list(subdirs["datasets"].glob("*.oai"))),
        }

    def save_project_to_user(self, username: str, project_data: dict,
                             name: str, extension: str = ".oai") -> Optional[Path]:
        """Сохраняет проект/датасет в хранилище пользователя."""
        profile = self.get_profile(username)
        if profile is None:
            return None
        if profile.total_projects >= MAX_PROJECTS_PER_USER:
            logger.warning(f"Лимит проектов ({MAX_PROJECTS_PER_USER}) для '{username}'")
            return None
        subdirs = self.get_user_subdirs(username)
        safe_name = _slugify(name) or "project"
        path = subdirs["projects"] / f"{safe_name}{extension}"
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(project_data, f, ensure_ascii=False, indent=2)
        profile.total_projects += 1
        self._save_profile(profile)
        self._save_index()
        return path

    def get_user_projects(self, username: str) -> List[Path]:
        return sorted(self.get_user_subdirs(username)["projects"].glob("*.oai"))

    def get_user_models(self, username: str) -> List[Path]:
        d = self.get_user_subdirs(username)["models"]
        return sorted(list(d.glob("*.pth")) + list(d.glob("*.pt")))

    def get_user_datasets(self, username: str) -> List[Path]:
        return sorted(self.get_user_subdirs(username)["datasets"].glob("*.oai"))

    # ============================================================
    # ПЕРЕНОС СУЩЕСТВУЮЩИХ ДАННЫХ (миграция)
    # ============================================================
    def migrate_legacy_data(self, username: str = "guest") -> bool:
        """
        Переносит старые файлы из data/datasets, data/models, data/exports
        в хранилище пользователя. Возвращает True, если что-то перенесено.
        """
        subdirs = self.get_user_subdirs(username)
        moved = False
        legacy_map = [
            (DATA_DIR / "datasets", subdirs["datasets"]),
            (DATA_DIR / "models", subdirs["models"]),
            (DATA_DIR / "exports", subdirs["exports"]),
        ]
        for src, dst in legacy_map:
            if not src.exists():
                continue
            for f in src.glob("*"):
                if f.is_file():
                    try:
                        shutil.copy2(f, dst / f.name)
                        moved = True
                    except OSError:
                        pass
        return moved
