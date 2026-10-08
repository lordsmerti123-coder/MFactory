"""
core/pretrained_model_info.py
=============================
Описание предобученной модели для манифеста.

ОТВЕТСТВЕННОСТЬ:
  • Dataclass PretrainedModelInfo — метаданные готовой модели.
  • Сериализация/десериализация.

ЗАВИСИМОСТИ:
  • core/contracts.py  → DataType, ArchitectureType
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict

from core.contracts import DataType, ArchitectureType


@dataclass
class PretrainedModelInfo:
    """Метаданные предобученной модели из манифеста."""
    model_id: str = ""
    name: str = ""
    description: str = ""
    data_type: DataType = DataType.TEXT
    architecture: ArchitectureType = ArchitectureType.TRANSFORMER_SEQ2SEQ
    params_count: int = 0
    size_mb: float = 0.0
    source_url: str = ""
    sha256_hash: str = ""
    requires_license: bool = False
    license_text: str = ""
    category: str = "text"  # text | music | image | translation | audio
    installed: bool = False
    local_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["data_type"] = self.data_type.value
        d["architecture"] = self.architecture.value
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PretrainedModelInfo":
        return PretrainedModelInfo(
            model_id=d.get("model_id", d.get("id", "")),
            name=d.get("name", ""),
            description=d.get("description", ""),
            data_type=DataType.from_str(d.get("data_type", "text")),
            architecture=ArchitectureType.from_str(d.get("architecture", "transformer_seq2seq")),
            params_count=d.get("params_count", 0),
            size_mb=d.get("size_mb", 0.0),
            source_url=d.get("source_url", ""),
            sha256_hash=d.get("sha256_hash", ""),
            requires_license=d.get("requires_license", False),
            license_text=d.get("license_text", ""),
            category=d.get("category", "text"),
            installed=d.get("installed", False),
            local_path=d.get("local_path", ""),
        )

    def __repr__(self) -> str:
        state = "✓" if self.installed else "✗"
        return f"<PretrainedModelInfo {self.model_id} [{state}] {self.size_mb}МБ>"
