"""
gui/login_dialog.py
===================
Диалог входа / регистрации пользователя.

ОТВЕТСТВЕННОСТЬ:
  • Выбор существующего пользователя.
  • Регистрация нового пользователя.
  • Вход как гость.
  • Возврат имени выбранного пользователя.

ЗАВИСИМОСТИ:
  • core/user_manager.py → UserManager, hash_password, verify_password
"""

from __future__ import annotations

from typing import Dict, Any, Optional

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QListWidget, QListWidgetItem, QMessageBox,
    QGroupBox, QFormLayout, QDialogButtonBox,
)
from PyQt5.QtCore import Qt

from core.user_manager import UserManager, hash_password
from core.logger import get_logger

logger = get_logger(__name__)


class LoginDialog(QDialog):
    """Окно входа в OracleAI Studio."""

    def __init__(self, user_manager: UserManager, parent=None):
        super().__init__(parent)
        self.user_manager = user_manager
        self.selected_username: Optional[str] = None
        self.setWindowTitle("Вход в OracleAI Studio")
        self.setMinimumWidth(520)
        self.init_ui()
        self.refresh_user_list()

    def init_ui(self):
        layout = QVBoxLayout(self)

        header = QLabel("👤 Вход в OracleAI Studio")
        header.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #cc7832; padding: 4px;"
        )
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

        # ---------- Список пользователей ----------
        users_group = QGroupBox("Существующие пользователи")
        users_layout = QVBoxLayout()

        self.user_list = QListWidget()
        self.user_list.itemDoubleClicked.connect(self._on_enter)
        users_layout.addWidget(self.user_list)

        login_row = QHBoxLayout()
        self.password_edit = QLineEdit()
        self.password_edit.setPlaceholderText("Пароль")
        self.password_edit.setEchoMode(QLineEdit.Password)
        self.password_edit.returnPressed.connect(self._on_login)
        login_row.addWidget(self.password_edit, stretch=1)

        self.btn_login = QPushButton("🔓 Войти")
        self.btn_login.clicked.connect(self._on_login)
        login_row.addWidget(self.btn_login)

        self.btn_guest = QPushButton("👻 Гость")
        self.btn_guest.setToolTip("Войти без регистрации. Данные сохранятся под «гость».")
        self.btn_guest.clicked.connect(self._on_guest)
        login_row.addWidget(self.btn_guest)

        users_layout.addLayout(login_row)
        users_group.setLayout(users_layout)
        layout.addWidget(users_group)

        # ---------- Регистрация ----------
        register_group = QGroupBox("Новый пользователь")
        form = QFormLayout()

        self.new_username = QLineEdit()
        self.new_username.setPlaceholderText("3–32 символа: буквы, цифры, _-")
        self.new_username.setMaximumWidth(260)
        form.addRow("Имя:", self.new_username)

        self.new_password = QLineEdit()
        self.new_password.setPlaceholderText("Пароль (можно пустой)")
        self.new_password.setEchoMode(QLineEdit.Password)
        self.new_password.setMaximumWidth(260)
        form.addRow("Пароль:", self.new_password)

        self.btn_register = QPushButton("➕ Зарегистрировать")
        self.btn_register.clicked.connect(self._on_register)
        form.addRow("", self.btn_register)

        register_group.setLayout(form)
        layout.addWidget(register_group)

        # ---------- Справка ----------
        info = QLabel(
            "ℹ️ Данные хранятся локально, без интернета. "
            "Пароль сохраняется в виде хеша."
        )
        info.setStyleSheet("color: #777; font-size: 11px;")
        info.setWordWrap(True)
        layout.addWidget(info)

    def refresh_user_list(self):
        self.user_list.clear()
        users = self.user_manager.get_all_users()
        for profile in users:
            label = profile.username
            if profile.is_guest:
                label += " (гость)"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, profile.username)
            self.user_list.addItem(item)
        if self.user_list.count() > 0:
            self.user_list.setCurrentRow(0)

    # ============================================================
    # ДЕЙСТВИЯ
    # ============================================================
    def _on_login(self):
        item = self.user_list.currentItem()
        if item is None:
            QMessageBox.warning(self, "Выбор", "Выберите пользователя из списка.")
            return
        username = item.data(Qt.UserRole)
        password = self.password_edit.text()

        profile = self.user_manager.get_profile(username)
        if profile is None:
            return
        if profile.is_guest:
            self.selected_username = username
            self.accept()
            return
        if self.user_manager.authenticate(username, password):
            self.selected_username = username
            self.accept()
        else:
            QMessageBox.warning(self, "Ошибка", "Неверный пароль.")

    def _on_guest(self):
        profile = self.user_manager.create_guest()
        self.selected_username = profile.username
        self.accept()

    def _on_register(self):
        username = self.new_username.text().strip()
        password = self.new_password.text()

        if self.user_manager.get_profile(username) is not None:
            QMessageBox.warning(self, "Ошибка", "Такой пользователь уже есть.")
            return

        password_hash = hash_password(password) if password else ""
        profile = self.user_manager.create_user(username, password_hash)
        if profile is None:
            QMessageBox.warning(
                self, "Ошибка",
                "Имя не подходит. Используйте 3–32 символа: буквы, цифры, _-",
            )
            return

        self.selected_username = username
        self.user_manager.switch_user(username)
        QMessageBox.information(
            self, "Готово",
            f"Пользователь «{username}» создан. Теперь вы вошли.",
        )
        self.accept()

    def _on_enter(self, item):
        """Двойной клик по пользователю → вход с пустым паролем (если нет пароля)."""
        username = item.data(Qt.UserRole)
        profile = self.user_manager.get_profile(username)
        if profile is not None and not profile.password_hash:
            self.selected_username = username
            self.accept()

    def get_selected_username(self) -> Optional[str]:
        return self.selected_username
