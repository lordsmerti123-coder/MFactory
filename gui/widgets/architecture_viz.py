"""
gui/widgets/architecture_viz.py
================================
Рисует блок-схему слоёв, показывает количество параметров.
Не создаёт ключей в shared_state.
"""

from typing import Optional, Dict, Any, List
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox,
    QScrollArea, QFrame, QGridLayout, QSizePolicy
)
from PyQt5.QtCore import Qt, QRect, QPoint, QSize
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPainterPath


# Цвета для разных типов слоёв
LAYER_COLORS = {
    "embedding": QColor("#4a88c7"),
    "positional": QColor("#5c9fd6"),
    "linear": QColor("#cc7832"),
    "conv2d": QColor("#e06c75"),
    "lstm": QColor("#a6e3a1"),
    "gru": QColor("#98c379"),
    "rnn": QColor("#7ec87e"),
    "transformer_encoder": QColor("#c678dd"),
    "transformer_decoder": QColor("#b45fc4"),
    "attention": QColor("#d19a66"),
    "dropout": QColor("#777777"),
    "batchnorm": QColor("#56b6c2"),
    "layernorm": QColor("#56b6c2"),
    "activation": QColor("#e5c07b"),
    "pooling": QColor("#be5046"),
    "flatten": QColor("#888888"),
    "upsample": QColor("#61afef"),
    "convtranspose": QColor("#e06c75"),
    "output": QColor("#385a3a"),
    "input": QColor("#4a88c7"),
    "unknown": QColor("#555555"),
}

LAYER_LABELS_RU = {
    "embedding": "Эмбеддинг",
    "positional": "Позиц. код.",
    "linear": "Линейный",
    "conv2d": "Свёртка 2D",
    "lstm": "LSTM",
    "gru": "GRU",
    "rnn": "RNN",
    "transformer_encoder": "Трансф. Энкодер",
    "transformer_decoder": "Трансф. Декодер",
    "attention": "Внимание",
    "dropout": "Dropout",
    "batchnorm": "BatchNorm",
    "layernorm": "LayerNorm",
    "activation": "Активация",
    "pooling": "Pooling",
    "flatten": "Flatten",
    "upsample": "Upsample",
    "convtranspose": "ConvTranspose",
    "output": "Выход",
    "input": "Вход",
    "unknown": "Слой",
}


def _classify_layer(layer) -> str:
    """Определяет тип слоя PyTorch для раскраски."""
    import torch.nn as nn
    type_map = {
        nn.Embedding: "embedding",
        nn.Linear: "linear",
        nn.Conv2d: "conv2d",
        nn.ConvTranspose2d: "convtranspose",
        nn.LSTM: "lstm",
        nn.GRU: "gru",
        nn.RNN: "rnn",
        nn.Dropout: "dropout",
        nn.Dropout2d: "dropout",
        nn.BatchNorm1d: "batchnorm",
        nn.BatchNorm2d: "batchnorm",
        nn.LayerNorm: "layernorm",
        nn.ReLU: "activation",
        nn.GELU: "activation",
        nn.Tanh: "activation",
        nn.Sigmoid: "activation",
        nn.LeakyReLU: "activation",
        nn.MaxPool2d: "pooling",
        nn.AvgPool2d: "pooling",
        nn.AdaptiveAvgPool2d: "pooling",
        nn.Flatten: "flatten",
        nn.Upsample: "upsample",
        nn.TransformerEncoder: "transformer_encoder",
        nn.TransformerDecoder: "transformer_decoder",
        nn.MultiheadAttention: "attention",
    }
    for cls, name in type_map.items():
        if isinstance(layer, cls):
            return name
    # Для nn.Sequential и подобных
    if isinstance(layer, nn.Sequential):
        return "unknown"
    return "unknown"


def _count_layer_params(layer) -> int:
    """Подсчёт параметров одного слоя."""
    return sum(p.numel() for p in layer.parameters())


def _format_params(n: int) -> str:
    """Форматирование числа параметров."""
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    elif n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


class LayerBlock:
    """Один блок на схеме."""
    def __init__(self, name: str, layer_type: str, params: int, detail: str = ""):
        self.name = name
        self.layer_type = layer_type
        self.params = params
        self.detail = detail


class ArchitectureVizWidget(QWidget):
    """
    Виджет визуализации архитектуры.
    Показывает блок-схему и статистику параметров.

    Использование:
        viz = ArchitectureVizWidget()
        viz.set_model(model)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layers: List[LayerBlock] = []
        self.total_params = 0
        self.model_name = ""
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # Заголовок
        self.title_label = QLabel("🏗 Архитектура модели")
        self.title_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #cc7832;")
        layout.addWidget(self.title_label)

        # Сводка
        self.summary_label = QLabel("Модель не загружена")
        self.summary_label.setStyleSheet("color: #a9b7c6; font-size: 12px;")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        # Область прокрутки для блок-схемы
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumHeight(200)
        self.scroll_area.setStyleSheet(
            "QScrollArea { background-color: #1e1e1e; border: 1px solid #555; }"
        )

        self.blocks_container = QWidget()
        self.blocks_layout = QVBoxLayout(self.blocks_container)
        self.blocks_layout.setAlignment(Qt.AlignTop)
        self.blocks_layout.setSpacing(2)
        self.scroll_area.setWidget(self.blocks_container)
        layout.addWidget(self.scroll_area)

        # Итого
        self.total_label = QLabel("")
        self.total_label.setStyleSheet(
            "color: #4a88c7; font-size: 13px; font-weight: bold; padding: 4px;"
        )
        layout.addWidget(self.total_label)

    def set_model(self, model, model_name: str = "Модель"):
        """
        Принимает torch.nn.Module и строит визуализацию.
        Безопасно: если model=None, показывает заглушку.
        """
        self.model_name = model_name
        self._clear_blocks()

        if model is None:
            self.summary_label.setText("Модель не загружена. Создай её во вкладке «Архитектура».")
            self.total_label.setText("")
            return

        # Извлекаем слои
        self.layers = self._extract_layers(model)
        self.total_params = sum(p.numel() for p in model.parameters())

        # Сводка
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = self.total_params - trainable
        self.summary_label.setText(
            f"Модель: <b>{model_name}</b> | "
            f"Всего параметров: <b>{_format_params(self.total_params)}</b> | "
            f"Обучаемых: {_format_params(trainable)} | "
            f"Замороженных: {_format_params(frozen)}"
        )

        # Рисуем блоки
        for i, block in enumerate(self.layers):
            widget = self._create_block_widget(block, i)
            self.blocks_layout.addWidget(widget)

        # Итого
        mem_mb = self.total_params * 4 / (1024 * 1024)  # float32
        self.total_label.setText(
            f"📊 Итого: {self.total_params:,} параметров (~{mem_mb:.2f} МБ в float32) | "
            f"Слоёв: {len(self.layers)}"
        )

    def _extract_layers(self, model) -> List[LayerBlock]:
        """Рекурсивно извлекает слои из модели."""
        blocks = []
        self._extract_recursive(model, blocks, depth=0, max_depth=3)
        if not blocks:
            blocks.append(LayerBlock("model", "unknown", self.total_params))
        return blocks

    def _extract_recursive(self, module, blocks: List[LayerBlock], depth: int, max_depth: int):
        import torch.nn as nn
        if depth > max_depth:
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                blocks.append(LayerBlock(
                    module.__class__.__name__,
                    _classify_layer(module),
                    params
                ))
            return

        children = list(module.named_children())
        if not children:
            # Листовой слой
            params = sum(p.numel() for p in module.parameters())
            layer_type = _classify_layer(module)
            name = module.__class__.__name__
            detail = ""
            if isinstance(module, nn.Linear):
                detail = f"{module.in_features}→{module.out_features}"
            elif isinstance(module, nn.Embedding):
                detail = f"vocab={module.num_embeddings}, dim={module.embedding_dim}"
            elif isinstance(module, nn.Conv2d):
                detail = f"{module.in_channels}→{module.out_channels}, k={module.kernel_size}"
            elif isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
                detail = f"hidden={module.hidden_size}, layers={module.num_layers}"

            blocks.append(LayerBlock(name, layer_type, params, detail))
        else:
            for name, child in children:
                params = sum(p.numel() for p in child.parameters())
                if isinstance(child, nn.Sequential):
                    self._extract_recursive(child, blocks, depth + 1, max_depth)
                elif params > 0 or isinstance(child, (nn.Dropout, nn.Flatten)):
                    blocks.append(LayerBlock(
                        name or child.__class__.__name__,
                        _classify_layer(child),
                        params
                    ))
                else:
                    self._extract_recursive(child, blocks, depth + 1, max_depth)

    def _create_block_widget(self, block: LayerBlock, index: int) -> QWidget:
        """Создаёт визуальный блок для одного слоя."""
        widget = QFrame()
        color = LAYER_COLORS.get(block.layer_type, LAYER_COLORS["unknown"])
        widget.setStyleSheet(
            f"QFrame {{ background-color: #2b2b2b; border-left: 4px solid {color.name()}; "
            f"border: 1px solid #444; border-left: 4px solid {color.name()}; "
            f"padding: 4px; margin: 1px; }}"
        )

        layout = QHBoxLayout(widget)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # Номер
        num_label = QLabel(f"{index + 1}.")
        num_label.setStyleSheet("color: #777; font-size: 11px; min-width: 24px;")
        layout.addWidget(num_label)

        # Цветной индикатор типа
        type_label = QLabel(LAYER_LABELS_RU.get(block.layer_type, block.layer_type))
        type_label.setStyleSheet(f"color: {color.name()}; font-weight: bold; font-size: 12px;")
        layout.addWidget(type_label)

        # Имя класса
        name_label = QLabel(f"({block.name})")
        name_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(name_label)

        # Детали
        if block.detail:
            detail_label = QLabel(block.detail)
            detail_label.setStyleSheet("color: #a9b7c6; font-size: 11px;")
            layout.addWidget(detail_label)

        layout.addStretch()

        # Параметры
        if block.params > 0:
            params_label = QLabel(f"{_format_params(block.params)} пар.")
            params_label.setStyleSheet("color: #4a88c7; font-size: 11px; font-weight: bold;")
            layout.addWidget(params_label)

        return widget

    def _clear_blocks(self):
        """Удаляет все блоки из контейнера."""
        while self.blocks_layout.count():
            child = self.blocks_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def set_empty(self, message: str = "Модель не создана"):
        """Показывает заглушку."""
        self._clear_blocks()
        self.summary_label.setText(message)
        self.total_label.setText("")