"""
core/user_storage_analyzer.py
=============================
Анализ хранилища пользователя.

ОТВЕТСТВЕННОСТЬ:
  • Анализ проектов, моделей, датасетов пользователя.
  • Подсчёт размера хранилища.
  • Генерация рекомендаций (что удалить, что дообучить, что продолжить).
  • Формирование StorageReport.

ЗАВИСИМОСТИ:
  • core/user_manager.py → UserManager
  • core/contracts.py   → ProjectState, ModelConfig
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.contracts import ProjectState
from core.user_manager import UserManager
from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class StorageItem:
    """Один файл в хранилище."""
    name: str
    path: str
    size_kb: float
    modified_at: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StorageReport:
    """Полный отчёт о хранилище пользователя."""
    username: str = ""
    projects: List[StorageItem] = field(default_factory=list)
    models: List[StorageItem] = field(default_factory=list)
    datasets: List[StorageItem] = field(default_factory=list)
    exports: List[StorageItem] = field(default_factory=list)
    total_size_mb: float = 0.0
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        def items(i):
            return {
                "name": i.name, "path": i.path, "size_kb": round(i.size_kb, 1),
                "modified_at": i.modified_at, "meta": i.meta,
            }
        return {
            "username": self.username,
            "projects": [items(p) for p in self.projects],
            "models": [items(m) for m in self.models],
            "datasets": [items(d) for d in self.datasets],
            "exports": [items(e) for e in self.exports],
            "total_size_mb": round(self.total_size_mb, 2),
            "recommendations": self.recommendations,
            "generated_at": datetime.now().isoformat(),
        }


class UserStorageAnalyzer:
    """Анализирует хранилище пользователя и выдаёт рекомендации."""

    def __init__(self, user_manager: UserManager, username: str):
        self.user_manager = user_manager
        self.username = username
        self.user_dir = user_manager.get_user_dir(username)

    # ============================================================
    # АНАЛИЗ
    # ============================================================
    def analyze(self) -> StorageReport:
        report = StorageReport(username=self.username)
        subdirs = self.user_manager.get_user_subdirs(self.username)

        report.projects = self._scan_dir(subdirs["projects"], "*.oai", self._read_project_meta)
        report.models = self._scan_dir(subdirs["models"], "*.pth")
        report.models += self._scan_dir(subdirs["models"], "*.pt")
        report.datasets = self._scan_dir(subdirs["datasets"], "*.oai")
        report.exports = self._scan_dir(subdirs["exports"], "*")

        total_bytes = 0
        for group in (report.projects, report.models, report.datasets, report.exports):
            total_bytes += sum(i.size_kb * 1024 for i in group)
        report.total_size_mb = total_bytes / (1024 * 1024)

        report.recommendations = self._generate_recommendations(report)
        return report

    # ============================================================
    # СКАНИРОВАНИЕ
    # ============================================================
    def _scan_dir(self, directory: Path, pattern: str,
                  meta_reader=None) -> List[StorageItem]:
        items = []
        if not directory.exists():
            return items
        for path in sorted(directory.glob(pattern)):
            if not path.is_file():
                continue
            meta = {}
            if meta_reader is not None:
                try:
                    meta = meta_reader(path) or {}
                except Exception:
                    meta = {}
            items.append(StorageItem(
                name=path.stem,
                path=str(path),
                size_kb=path.stat().st_size / 1024,
                modified_at=datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                meta=meta,
            ))
        return items

    def _read_project_meta(self, path: Path) -> Dict[str, Any]:
        """Читает метаданные .oai проекта без полной загрузки данных."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            config = data.get("config", {}) or {}
            history = data.get("history", {}) or {}
            trained = bool(history.get("val_loss"))
            return {
                "scenario": data.get("scenario", "new"),
                "model_name": config.get("model_name", ""),
                "architecture": config.get("type", ""),
                "trained": trained,
                "epochs_completed": history.get("epochs_completed", 0),
            }
        except (json.JSONDecodeError, OSError):
            return {}

    # ============================================================
    # РЕКОМЕНДАЦИИ
    # ============================================================
    def _generate_recommendations(self, report: StorageReport) -> List[str]:
        recs: List[str] = []

        # 1. Много проектов → предложить архивировать старые
        if len(report.projects) >= 10:
            recs.append(
                f"У вас {len(report.projects)} проектов. Старые незавершённые "
                "можно перенести в папку «Экспорт» или удалить."
            )

        # 2. Незавершённые проекты → продолжить
        unfinished = [
            p for p in report.projects
            if not p.meta.get("trained") and p.meta.get("epochs_completed", 0) > 0
        ]
        if unfinished:
            names = ", ".join(p.name for p in unfinished[:3])
            recs.append(f"Есть незавершённые проекты: {names}. Возможно, стоит продолжить обучение.")

        # 3. Обученные, но не экспортированные модели
        trained_not_exported = [
            p for p in report.projects
            if p.meta.get("trained") and not report.exports
        ]
        if trained_not_exported and not report.exports:
            recs.append(
                "У вас есть обученные модели, но нет экспортов. "
                "Экспортируйте модель на флешку или в HTML, чтобы показать результат."
            )

        # 4. Большое хранилище
        if report.total_size_mb > 500:
            recs.append(
                f"Хранилище занимает {report.total_size_mb:.0f} МБ. "
                "Проверьте датасеты — они могут занимать больше всего места."
            )

        # 5. Мало данных для обучения
        if report.datasets and not report.projects:
            recs.append(
                "Датасеты загружены, но проектов нет. Создайте проект и свяжите его с данными."
            )

        # 6. Всё отлично
        if not recs:
            recs.append("Хранилище в порядке. Продолжайте экспериментировать!")

        return recs

    # ============================================================
    # ТЕКСТОВОЕ ПРЕДСТАВЛЕНИЕ
    # ============================================================
    def format_report_text(self, report: Optional[StorageReport] = None) -> str:
        report = report or self.analyze()
        lines = [
            f"📊 Хранилище пользователя «{report.username}»",
            f"Всего: {report.total_size_mb:.2f} МБ",
            "",
            f"📁 Проекты ({len(report.projects)}):",
        ]
        for p in report.projects:
            trained = "✓ обучен" if p.meta.get("trained") else "✗ не обучен"
            lines.append(f"  • {p.name} — {p.size_kb:.0f} КБ ({trained})")
        lines.append(f"🧠 Модели ({len(report.models)}):")
        for m in report.models:
            lines.append(f"  • {m.name} — {m.size_kb:.0f} КБ")
        lines.append(f"💾 Датасеты ({len(report.datasets)}):")
        for d in report.datasets:
            lines.append(f"  • {d.name} — {d.size_kb:.0f} КБ")
        lines.append(f"🚀 Экспорты ({len(report.exports)}):")
        for e in report.exports:
            lines.append(f"  • {e.name} — {e.size_kb:.0f} КБ")
        lines.append("")
        lines.append("💡 Рекомендации:")
        for r in report.recommendations:
            lines.append(f"  • {r}")
        return "\n".join(lines)
