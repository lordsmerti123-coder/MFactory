"""
gui/widgets/__init__.py
=======================
Все виджеты работают независимо и не создают ключей в shared_state.
Типы данных импортируются ТОЛЬКО из core/contracts.py.
"""

from gui.widgets.canvas_widget import CanvasWidget
from gui.widgets.architecture_viz import ArchitectureVizWidget
from gui.widgets.vocab_editor import VocabEditorDialog
from gui.widgets.audio_player import AudioPlayerWidget
from gui.widgets.model_name_generator import ModelNameGeneratorWidget
from gui.widgets.data_flow_widget import DataFlowWidget
from gui.widgets.hint_tooltip import HintTooltip

__all__ = [
    "CanvasWidget",
    "ArchitectureVizWidget",
    "VocabEditorDialog",
    "AudioPlayerWidget",
    "ModelNameGeneratorWidget",
    "DataFlowWidget",
    "HintTooltip",
]