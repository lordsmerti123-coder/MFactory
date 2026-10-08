"""
gui/panels/architecture_panel.py
================================
Панель выбора и настройки архитектуры нейросети из зоопарка.

Зависимости (строго по контрактам):
  - core/contracts.py       → ArchitectureType, DataType, TaskType, ModelConfig
  - core/model_factory.py   → ModelFactory
  - gui/hint_widget.py      → HintWidget (если готов; иначе тултипы)

Сигналы:
  model_created = pyqtSignal(object, object)
    # arg1: nn.Module (модель)
    # arg2: ModelConfig (контракт)

Правила:
  - Dropout свободный: 0.0 – 1.0, без ограничений.
  - Архитектуры сгруппированы: Базовые / Генеративные / Продвинутые / Экспериментальные.
  - Каждый элемент имеет tooltip.
  - refresh() вызывается при изменении датасета.
  - Валидация совместимости архитектура × тип_данных ДО создания.

ИСПРАВЛЕНИЯ v2.0.1:
  - Убран преждевременный вызов _on_arch_changed из _populate_arch_combo
  - _on_arch_changed вызывается только в конце _init_ui после создания всех виджетов
  - Добавлены защитные hasattr-проверки во все callback-методы
"""

import torch
import torch.nn as nn
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QFormLayout, QComboBox, QSpinBox, QDoubleSpinBox,
    QPushButton, QLabel, QStackedWidget, QMessageBox,
    QTextEdit, QScrollArea, QCheckBox, QGridLayout,
    QFrame, QSizePolicy
)
from PyQt5.QtCore import pyqtSignal, Qt
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from core.contracts import (
    ArchitectureType, DataType, TaskType, ModelConfig, FORMAT_VERSION,
    dataset_to_legacy_dict,
)
from core.model_factory import ModelFactory
from core.logger import get_logger

logger = get_logger("ArchitecturePanel")

# ============================================================
# ОПИСАНИЯ АРХИТЕКТУР ДЛЯ ШКОЛЬНИКОВ
# ============================================================
ARCH_DESCRIPTIONS: Dict[str, str] = {
    ArchitectureType.MLP.value: (
        "<b>Многослойный перцептрон (MLP)</b> — самая простая нейросеть. "
        "Состоит из слоёв нейронов, каждый связан с каждым. "
        "<i>Идеально для таблиц и чисел. Не умеет читать текст и картинки.</i>"
    ),
    ArchitectureType.CNN.value: (
        "<b>Свёрточная нейросеть (CNN)</b> — смотрит на картинку маленькими окошками "
        "и находит в них узоры: края, углы, текстуры. "
        "<i>Идеально для распознавания фигур, цифр, изображений.</i>"
    ),
    ArchitectureType.RNN.value: (
        "<b>Рекуррентная сеть (RNN)</b> — читает текст по одному символу слева направо, "
        "запоминая прочитанное. <i>Минус: быстро забывает начало длинных строк.</i>"
    ),
    ArchitectureType.LSTM.value: (
        "<b>LSTM (Долгая краткосрочная память)</b> — улучшенная RNN с 'блокнотом'. "
        "Записывает важное и стирает ненужное. <i>Хорошо для математики и шифров.</i>"
    ),
    ArchitectureType.GRU.value: (
        "<b>GRU</b> — упрощённая версия LSTM. Работает быстрее, "
        "но чуть хуже запоминает очень длинные последовательности."
    ),
    ArchitectureType.TRANSFORMER_SEQ2SEQ.value: (
        "<b>Трансформер (Секвенция → Секвенция)</b> — основа ChatGPT. "
        "Читает ВХОД целиком и генерирует ОТВЕТ целиком. "
        "Механизм 'внимания' находит связи между любыми символами. "
        "<i>Лучший выбор для математики, шифров, перевода!</i>"
    ),
    ArchitectureType.TRANSFORMER_ENCODER.value: (
        "<b>Трансформер-Энкодер</b> — только 'читает' текст и выдаёт ответ-класс. "
        "<i>Подходит для классификации текста: спам/не спам, тональность отзыва.</i>"
    ),
    ArchitectureType.AUTOENCODER.value: (
        "<b>Автоэнкодер</b> — учится сжимать данные и восстанавливать их. "
        "<i>Используется для очистки шума, сжатия, поиска аномалий.</i>"
    ),
    ArchitectureType.VAE.value: (
        "<b>Вариационный автоэнкодер (VAE)</b> — автоэнкодер, который умеет "
        "ГЕНЕРИРОВАТЬ новые данные, похожие на обучающие. "
        "<i>Например: нарисовать новые цифры, похожие на ваши.</i>"
    ),
    ArchitectureType.CVAE.value: (
        "<b>Условный VAE (CVAE)</b> — VAE с подсказкой: 'нарисуй мне именно тройку'. "
        "<i>Генерирует данные заданного класса.</i>"
    ),
    ArchitectureType.GAN.value: (
        "<b>Генеративно-состязательная сеть (GAN)</b> — две нейросети играют в игру: "
        "Генератор рисует подделки, Дискриминатор пытается отличить их от настоящих. "
        "<i>Создаёт реалистичные изображения.</i>"
    ),
    ArchitectureType.DCGAN.value: (
        "<b>DCGAN</b> — GAN со свёрточными слоями. Стабильнее в обучении. "
        "<i>Классика для генерации изображений.</i>"
    ),
    ArchitectureType.DIFFUSION.value: (
        "<b>Диффузионная модель (DDPM)</b> — учится постепенно убирать шум из картинки. "
        "Начинает с чистого шума и за много шагов получает чёткое изображение. "
        "<i>Основа Stable Diffusion и Midjourney. Требует много данных и времени!</i>"
    ),
    ArchitectureType.LATENT_DIFFUSION.value: (
        "<b>Латентная диффузия (LDM)</b> — диффузия не в пикселях, а в сжатом пространстве. "
        "<i>Быстрее и легче. Требует автоэнкодер + диффузию. Экспериментально.</i>"
    ),
    ArchitectureType.VIT.value: (
        "<b>Vision Transformer (ViT)</b> — режет картинку на квадратики и подаёт их "
        "в Трансформер как слова. <i>Мощно для больших наборов изображений.</i>"
    ),
    ArchitectureType.RESNET.value: (
        "<b>ResNet (Остаточная сеть)</b> — глубокая CNN с 'обходными путями' (skip-connections). "
        "Может быть очень глубокой (50+ слоёв) без потери качества. "
        "<i>Стандарт для классификации изображений.</i>"
    ),
    ArchitectureType.CAPSNET.value: (
        "<b>Капсульная сеть (CapsNet)</b> — экспериментальная архитектура Хаффа. "
        "Каждая 'капсула' распознаёт объект и его ориентацию. "
        "<i>Менее изучена, но интересна для исследований.</i>"
    ),
    ArchitectureType.GNN.value: (
        "<b>Графовая нейросеть (GNN)</b> — работает с графами: узлы и связи между ними. "
        "<i>Для социальных сетей, молекул, маршрутов.</i>"
    ),
    ArchitectureType.GCN.value: (
        "<b>GCN</b> — графовая свёрточная сеть. Самый простой тип GNN."
    ),
    ArchitectureType.GAT.value: (
        "<b>GAT</b> — графовая сеть с вниманием. Некоторые связи важнее других."
    ),
    ArchitectureType.RBFN.value: (
        "<b>Сеть радиально-базисных функций (RBFN)</b> — использует 'колокольчики' "
        "(гауссовы функции) вместо обычных нейронов. "
        "<i>Простая и быстрая для регрессии.</i>"
    ),
    ArchitectureType.SOM.value: (
        "<b>Карта Кохонена (SOM)</b> — сеть без учителя. Раскладывает данные по карте "
        "так, что похожие оказываются рядом. <i>Для кластеризации и визуализации.</i>"
    ),
    ArchitectureType.LIQUID_NN.value: (
        "<b>Жидкая нейросеть (Liquid NN)</b> — нейроны меняют свои свойства со временем. "
        "<i>Экспериментально. Для временных рядов и робототехники.</i>"
    ),
    ArchitectureType.SPIKING_NN.value: (
        "<b>Импульсная нейросеть (SNN)</b> — имитирует мозг: нейроны 'стреляют' импульсами. "
        "<i>Очень энергоэффективна. Экспериментально.</i>"
    ),
    ArchitectureType.PINN.value: (
        "<b>PINN</b> — нейросеть, которая знает законы физики и не нарушает их. "
        "<i>Для моделирования жидкостей, тепла, гравитации.</i>"
    ),
    ArchitectureType.NEURAL_ODE.value: (
        "<b>Neural ODE</b> — нейросеть как дифференциальное уравнение. "
        "Не дискретные слои, а непрерывный поток. <i>Экспериментально.</i>"
    ),
    ArchitectureType.CONV_AE.value: (
        "<b>ConvAE (Свёрточный автоэнкодер)</b> — автоэнкодер со свёртками. "
        "<i>Очистка шума с изображений, сжатие.</i>"
    ),
}

# ============================================================
# ГРУППИРОВКА АРХИТЕКТУР ДЛЯ ВЫПАДАЮЩЕГО СПИСКА
# ============================================================
ARCH_GROUPS: Dict[str, List[Tuple[str, str]]] = {
    "🟢 Базовые (для начала)": [
        (ArchitectureType.MLP.value, "MLP — Многослойный перцептрон"),
        (ArchitectureType.CNN.value, "CNN — Свёрточная сеть"),
        (ArchitectureType.RNN.value, "RNN — Рекуррентная сеть"),
        (ArchitectureType.LSTM.value, "LSTM — Долгая память"),
        (ArchitectureType.GRU.value, "GRU — Быстрая рекуррентная"),
        (ArchitectureType.TRANSFORMER_SEQ2SEQ.value, "Transformer Seq2Seq"),
        (ArchitectureType.TRANSFORMER_ENCODER.value, "Transformer Encoder"),
    ],
    "🎨 Генеративные": [
        (ArchitectureType.AUTOENCODER.value, "Автоэнкодер"),
        (ArchitectureType.VAE.value, "VAE — Вариационный автоэнкодер"),
        (ArchitectureType.CVAE.value, "CVAE — Условный VAE"),
        (ArchitectureType.GAN.value, "GAN — Генеративно-состязательная"),
        (ArchitectureType.DCGAN.value, "DCGAN — Свёрточная GAN"),
        (ArchitectureType.DIFFUSION.value, "Diffusion (DDPM)"),
        (ArchitectureType.LATENT_DIFFUSION.value, "Latent Diffusion (LDM)"),
        (ArchitectureType.CONV_AE.value, "ConvAE — Свёрточный автоэнкодер"),
    ],
    "🔬 Продвинутые": [
        (ArchitectureType.VIT.value, "ViT — Vision Transformer"),
        (ArchitectureType.RESNET.value, "ResNet — Остаточная сеть"),
        (ArchitectureType.CAPSNET.value, "CapsNet — Капсульная"),
        (ArchitectureType.GNN.value, "GNN — Графовая сеть"),
        (ArchitectureType.GCN.value, "GCN — Графовая свёрточная"),
        (ArchitectureType.GAT.value, "GAT — Графовое внимание"),
        (ArchitectureType.NEURAL_ODE.value, "Neural ODE"),
        (ArchitectureType.PINN.value, "PINN — Физическая"),
    ],
    "🧪 Экспериментальные": [
        (ArchitectureType.RBFN.value, "RBFN — Радиально-базисная"),
        (ArchitectureType.SOM.value, "SOM — Карта Кохонена"),
        (ArchitectureType.LIQUID_NN.value, "Liquid NN — Жидкая"),
        (ArchitectureType.SPIKING_NN.value, "Spiking NN — Импульсная"),
    ],
}

# Матрица совместимости: архитектура → допустимые типы данных
ARCH_DATA_COMPAT: Dict[str, List[DataType]] = {
    ArchitectureType.MLP.value: [DataType.NUMERIC],
    ArchitectureType.CNN.value: [DataType.IMAGE, DataType.AUDIO],
    ArchitectureType.RNN.value: [DataType.TEXT, DataType.TIME_SERIES],
    ArchitectureType.LSTM.value: [DataType.TEXT, DataType.TIME_SERIES],
    ArchitectureType.GRU.value: [DataType.TEXT, DataType.TIME_SERIES],
    ArchitectureType.TRANSFORMER_SEQ2SEQ.value: [DataType.TEXT],
    ArchitectureType.TRANSFORMER_ENCODER.value: [DataType.TEXT],
    ArchitectureType.AUTOENCODER.value: [DataType.NUMERIC, DataType.IMAGE],
    ArchitectureType.VAE.value: [DataType.NUMERIC, DataType.IMAGE],
    ArchitectureType.CVAE.value: [DataType.NUMERIC, DataType.IMAGE],
    ArchitectureType.GAN.value: [DataType.IMAGE],
    ArchitectureType.DCGAN.value: [DataType.IMAGE],
    ArchitectureType.DIFFUSION.value: [DataType.IMAGE],
    ArchitectureType.LATENT_DIFFUSION.value: [DataType.IMAGE],
    ArchitectureType.VIT.value: [DataType.IMAGE],
    ArchitectureType.RESNET.value: [DataType.IMAGE],
    ArchitectureType.CAPSNET.value: [DataType.IMAGE],
    ArchitectureType.GNN.value: [DataType.GRAPH],
    ArchitectureType.GCN.value: [DataType.GRAPH],
    ArchitectureType.GAT.value: [DataType.GRAPH],
    ArchitectureType.RBFN.value: [DataType.NUMERIC],
    ArchitectureType.SOM.value: [DataType.NUMERIC],
    ArchitectureType.LIQUID_NN.value: [DataType.TIME_SERIES],
    ArchitectureType.SPIKING_NN.value: [DataType.NUMERIC, DataType.TIME_SERIES],
    ArchitectureType.PINN.value: [DataType.NUMERIC],
    ArchitectureType.NEURAL_ODE.value: [DataType.TIME_SERIES],
    ArchitectureType.CONV_AE.value: [DataType.IMAGE],
}

# Архитектуры-заглушки (экспериментальные, предупреждаем)
EXPERIMENTAL_ARCHS = {
    ArchitectureType.DIFFUSION.value,
    ArchitectureType.LATENT_DIFFUSION.value,
    ArchitectureType.CAPSNET.value,
    ArchitectureType.GNN.value,
    ArchitectureType.GCN.value,
    ArchitectureType.GAT.value,
    ArchitectureType.LIQUID_NN.value,
    ArchitectureType.SPIKING_NN.value,
    ArchitectureType.PINN.value,
    ArchitectureType.NEURAL_ODE.value,
}


class ArchitecturePanel(QWidget):
    """Панель выбора и настройки архитектуры нейросети."""

    # Сигнал: модель создана. Передаёт (nn.Module, ModelConfig)
    model_created = pyqtSignal(object, object)

    def __init__(self, shared_state: dict, parent=None):
        super().__init__(parent)
        self.shared_state = shared_state
        self._current_arch: str = ArchitectureType.MLP.value
        self._init_ui()

    # ================================================================
    # ИНИЦИАЛИЗАЦИЯ ИНТЕРФЕЙСА
    # ================================================================
    def _init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setSpacing(8)

        # Прокручиваемая область для всего контента
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        content_widget = QWidget()
        self.content_layout = QVBoxLayout(content_widget)
        self.content_layout.setSpacing(8)
        scroll.setWidget(content_widget)
        root_layout.addWidget(scroll)

        # ─── 1. Описание архитектуры (СНАЧАЛА, чтобы desc_box был готов) ───
        self._build_description_box()
        # ─── 2. Выбор архитектуры (без вызова _on_arch_changed!) ───
        self._build_arch_selector()
        # ─── 3. Общие параметры (активация, dropout, инициализация) ───
        self._build_common_params()
        # ─── 4. Специфичные настройки (QStackedWidget) ───
        self._build_arch_specific_settings()
        # ─── 5. Визуализация архитектуры ───
        self._build_arch_visualization()
        # ─── 6. Индикатор сложности ───
        self._build_complexity_indicator()
        # ─── 7. Кнопка создания ───
        self._build_create_button()
        # ─── 8. Отчёт ───
        self._build_report_box()

        # ─── 9. НАЧАЛЬНОЕ СОСТОЯНИЕ (все виджеты уже созданы!) ───
        if self.arch_combo.count() > 0:
            self._on_arch_changed(0)

    # ────────────────────────────────────────────────────────────────
    # 1. ОПИСАНИЕ АРХИТЕКТУРЫ (теперь создаётся ПЕРВЫМ)
    # ────────────────────────────────────────────────────────────────
    def _build_description_box(self):
        self.desc_box = QTextEdit()
        self.desc_box.setReadOnly(True)
        self.desc_box.setMaximumHeight(90)
        self.desc_box.setStyleSheet(
            "background-color: #242424; color: #a6e3a1; "
            "font-size: 13px; padding: 8px; border: 1px solid #555;"
        )
        self.desc_box.setToolTip("Описание выбранной архитектуры и для чего она подходит.")
        self.desc_box.setText("Выберите архитектуру выше...")
        self.content_layout.addWidget(self.desc_box)

    # ────────────────────────────────────────────────────────────────
    # 2. ВЫБОР АРХИТЕКТУРЫ
    # ────────────────────────────────────────────────────────────────
    def _build_arch_selector(self):
        group = QGroupBox("🏗 Выбор архитектуры")
        group.setToolTip(
            "Выберите тип нейросети. Каждая архитектура подходит для своих задач.\n"
            "Читайте подсказки ниже — они помогут не ошибиться."
        )
        layout = QHBoxLayout(group)

        # Группа (выпадающий список групп)
        layout.addWidget(QLabel("Группа:"))
        self.group_combo = QComboBox()
        self.group_combo.addItems(list(ARCH_GROUPS.keys()))
        self.group_combo.setToolTip("Архитектуры сгруппированы по сложности и назначению.")
        self.group_combo.currentIndexChanged.connect(self._on_group_changed)
        layout.addWidget(self.group_combo)

        # Конкретная архитектура
        layout.addWidget(QLabel("Архитектура:"))
        self.arch_combo = QComboBox()
        self.arch_combo.setToolTip("Конкретная архитектура. Описание появится ниже.")
        self.arch_combo.currentIndexChanged.connect(self._on_arch_changed)
        layout.addWidget(self.arch_combo)

        layout.addStretch()
        self.content_layout.addWidget(group)

        # Заполняем первую группу (БЕЗ вызова _on_arch_changed!)
        self._populate_arch_combo(0)

    def _populate_arch_combo(self, group_index: int):
        """Заполняет список архитектур для выбранной группы.
        
        ВАЖНО: НЕ вызывает _on_arch_changed здесь.
        Начальное состояние устанавливается в конце _init_ui.
        При смене группы пользователем — _on_arch_changed вызовется
        через сигнал currentIndexChanged от arch_combo.
        """
        if not hasattr(self, 'arch_combo'):
            return

        self.arch_combo.blockSignals(True)
        self.arch_combo.clear()
        group_names = list(ARCH_GROUPS.keys())
        if 0 <= group_index < len(group_names):
            group_name = group_names[group_index]
            for arch_value, arch_label in ARCH_GROUPS[group_name]:
                self.arch_combo.addItem(arch_label, arch_value)
        self.arch_combo.blockSignals(False)

        # Если пользователь сменил группу вручную — обновляем описание
        # Но только если все виджеты уже созданы (не в __init__)
        if hasattr(self, 'desc_box') and hasattr(self, 'settings_stack'):
            if self.arch_combo.count() > 0:
                self._on_arch_changed(0)

    def _on_group_changed(self, index: int):
        self._populate_arch_combo(index)

    def _on_arch_changed(self, index: int):
        """При смене архитектуры: обновляем описание, панель настроек, совместимость.
        
        ЗАЩИТА: если виджеты ещё не созданы — выходим.
        """
        # ─── Защитные проверки ───
        if not hasattr(self, 'arch_combo') or not hasattr(self, 'desc_box'):
            return
        if not hasattr(self, 'settings_stack'):
            return

        if index < 0 or self.arch_combo.count() == 0:
            return

        arch_value = self.arch_combo.currentData()
        if arch_value is None:
            return

        self._current_arch = arch_value

        # Обновляем описание
        desc = ARCH_DESCRIPTIONS.get(arch_value, "Описание отсутствует.")
        is_exp = arch_value in EXPERIMENTAL_ARCHS
        exp_warn = (
            "<br><span style='color:#e06c75;'>⚠️ Экспериментальная архитектура. "
            "Обучение может быть нестабильным.</span>" if is_exp else ""
        )
        self.desc_box.setHtml(f"{desc}{exp_warn}")

        # Переключаем панель настроек
        stack_index = self._arch_to_stack_index(arch_value)
        if 0 <= stack_index < self.settings_stack.count():
            self.settings_stack.setCurrentIndex(stack_index)

        # Проверяем совместимость с текущими данными
        self._check_data_compatibility()

        # Обновляем оценку сложности
        self._estimate_complexity()

        # Обновляем визуализацию
        self._update_arch_viz()

    def _arch_to_stack_index(self, arch_value: str) -> int:
        """Маппинг архитектуры → индекс в QStackedWidget."""
        mapping = {
            ArchitectureType.MLP.value: 0,
            ArchitectureType.CNN.value: 1,
            ArchitectureType.RNN.value: 2,
            ArchitectureType.LSTM.value: 2,
            ArchitectureType.GRU.value: 2,
            ArchitectureType.TRANSFORMER_SEQ2SEQ.value: 3,
            ArchitectureType.TRANSFORMER_ENCODER.value: 3,
            ArchitectureType.AUTOENCODER.value: 4,
            ArchitectureType.VAE.value: 5,
            ArchitectureType.CVAE.value: 5,
            ArchitectureType.GAN.value: 6,
            ArchitectureType.DCGAN.value: 6,
            ArchitectureType.DIFFUSION.value: 7,
            ArchitectureType.LATENT_DIFFUSION.value: 7,
            ArchitectureType.VIT.value: 8,
            ArchitectureType.RESNET.value: 9,
            ArchitectureType.CAPSNET.value: 10,
            ArchitectureType.GNN.value: 11,
            ArchitectureType.GCN.value: 11,
            ArchitectureType.GAT.value: 11,
            ArchitectureType.RBFN.value: 12,
            ArchitectureType.SOM.value: 13,
            ArchitectureType.LIQUID_NN.value: 14,
            ArchitectureType.SPIKING_NN.value: 14,
            ArchitectureType.PINN.value: 15,
            ArchitectureType.NEURAL_ODE.value: 15,
            ArchitectureType.CONV_AE.value: 4,
        }
        return mapping.get(arch_value, 0)

    # ────────────────────────────────────────────────────────────────
    # 3. ОБЩИЕ ПАРАМЕТРЫ
    # ────────────────────────────────────────────────────────────────
    def _build_common_params(self):
        group = QGroupBox("⚙️ Общие параметры")
        group.setToolTip("Эти настройки применяются ко всем архитектурам.")
        layout = QFormLayout(group)

        # Функция активации
        self.activation_combo = QComboBox()
        self.activation_combo.addItems(["relu", "gelu", "tanh", "sigmoid", "elu", "leaky_relu"])
        self.activation_combo.setToolTip(
            "Функция активации определяет, как нейрон 'решает', передавать ли сигнал дальше.\n"
            "• ReLU — быстрая и надёжная (по умолчанию)\n"
            "• GELU — плавнее, для Трансформеров\n"
            "• Sigmoid — для бинарных выходов (0 или 1)\n"
            "• Tanh — для рекуррентных сетей"
        )
        layout.addRow("Функция активации:", self.activation_combo)

        # Dropout — СВОБОДНЫЙ ввод 0.0–1.0
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 1.0)
        self.dropout_spin.setValue(0.1)
        self.dropout_spin.setSingleStep(0.05)
        self.dropout_spin.setDecimals(3)
        self.dropout_spin.setToolTip(
            "Dropout — процент нейронов, которые случайно 'выключаются' при обучении.\n"
            "Это защита от зубрёжки: сеть не может полагаться на одни и те же нейроны.\n"
            "• 0.0 — без защиты (риск переобучения)\n"
            "• 0.1–0.3 — стандарт для большинства задач\n"
            "• 0.5 — агрессивная защита (для маленьких датасетов)\n"
            "• 1.0 — выключить ВСЁ (сеть не будет учиться!)\n"
            "Вы можете поставить ЛЮБОЕ значение от 0 до 1."
        )
        layout.addRow("Dropout (отсев нейронов):", self.dropout_spin)

        # Инициализация весов
        self.init_combo = QComboBox()
        self.init_combo.addItems(["xavier", "he", "normal", "uniform"])
        self.init_combo.setToolTip(
            "Как инициализировать веса перед обучением:\n"
            "• Xavier — для sigmoid/tanh\n"
            "• He — для ReLU (рекомендуется)\n"
            "• Normal — случайные из нормального распределения\n"
            "• Uniform — равномерное распределение"
        )
        layout.addRow("Инициализация весов:", self.init_combo)

        self.content_layout.addWidget(group)

    # ────────────────────────────────────────────────────────────────
    # 4. СПЕЦИФИЧНЫЕ НАСТРОЙКИ АРХИТЕКТУР (QStackedWidget)
    # ────────────────────────────────────────────────────────────────
    def _build_arch_specific_settings(self):
        self.settings_stack = QStackedWidget()

        # Индекс 0: MLP
        self.settings_stack.addWidget(self._create_mlp_settings())
        # Индекс 1: CNN
        self.settings_stack.addWidget(self._create_cnn_settings())
        # Индекс 2: RNN/LSTM/GRU
        self.settings_stack.addWidget(self._create_rnn_settings())
        # Индекс 3: Transformer
        self.settings_stack.addWidget(self._create_transformer_settings())
        # Индекс 4: Autoencoder / ConvAE
        self.settings_stack.addWidget(self._create_autoencoder_settings())
        # Индекс 5: VAE / CVAE
        self.settings_stack.addWidget(self._create_vae_settings())
        # Индекс 6: GAN / DCGAN
        self.settings_stack.addWidget(self._create_gan_settings())
        # Индекс 7: Diffusion / LDM
        self.settings_stack.addWidget(self._create_diffusion_settings())
        # Индекс 8: ViT
        self.settings_stack.addWidget(self._create_vit_settings())
        # Индекс 9: ResNet
        self.settings_stack.addWidget(self._create_resnet_settings())
        # Индекс 10: CapsNet
        self.settings_stack.addWidget(self._create_stub_settings("CapsNet"))
        # Индекс 11: GNN / GCN / GAT
        self.settings_stack.addWidget(self._create_gnn_settings())
        # Индекс 12: RBFN
        self.settings_stack.addWidget(self._create_rbfn_settings())
        # Индекс 13: SOM
        self.settings_stack.addWidget(self._create_som_settings())
        # Индекс 14: Liquid NN / Spiking NN
        self.settings_stack.addWidget(self._create_stub_settings("Liquid/Spiking NN"))
        # Индекс 15: PINN / Neural ODE
        self.settings_stack.addWidget(self._create_stub_settings("PINN/Neural ODE"))

        self.content_layout.addWidget(self.settings_stack)

    # ---------- MLP ----------
    def _create_mlp_settings(self) -> QWidget:
        widget = QGroupBox("Настройки MLP (табличная сеть)")
        widget.setToolTip("Настройки для многослойного перцептрона.")
        main_layout = QVBoxLayout(widget)

        form = QFormLayout()

        self.mlp_num_layers = QSpinBox()
        self.mlp_num_layers.setRange(1, 20)
        self.mlp_num_layers.setValue(3)
        self.mlp_num_layers.setToolTip(
            "Количество скрытых слоёв.\n"
            "• 1–2 слоя: простые задачи (линейная регрессия)\n"
            "• 3–4 слоя: средний уровень (математика)\n"
            "• 5+: сложные задачи, но риск переобучения"
        )
        self.mlp_num_layers.valueChanged.connect(self._update_mlp_layer_spins)
        form.addRow("Скрытые слои:", self.mlp_num_layers)

        self.mlp_batchnorm = QCheckBox("BatchNorm (ускоряет обучение)")
        self.mlp_batchnorm.setToolTip(
            "BatchNorm нормализует выходы каждого слоя.\n"
            "Обучение идёт быстрее и стабильнее.\n"
            "Рекомендуется включать для сетей с 2+ слоями."
        )
        form.addRow("", self.mlp_batchnorm)

        main_layout.addLayout(form)

        # Динамические спинбоксы для нейронов в каждом слое
        self.mlp_layers_container = QVBoxLayout()
        self.mlp_layer_spins: List[QSpinBox] = []
        main_layout.addLayout(self.mlp_layers_container)
        self._update_mlp_layer_spins(3)

        return widget

    def _update_mlp_layer_spins(self, count: int):
        """Пересоздаёт спинбоксы количества нейронов для MLP."""
        if not hasattr(self, 'mlp_layers_container'):
            return
        for i in reversed(range(self.mlp_layers_container.count())):
            w = self.mlp_layers_container.itemAt(i).widget()
            if w:
                w.setParent(None)
        self.mlp_layer_spins.clear()

        defaults = [64, 128, 256, 128, 64]
        for i in range(count):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"  Слой {i + 1}:"))
            spin = QSpinBox()
            spin.setRange(4, 8192)
            spin.setSingleStep(32)
            spin.setValue(defaults[i] if i < len(defaults) else 64)
            spin.setToolTip(
                f"Количество нейронов в слое {i + 1}.\n"
                "Больше нейронов = умнее сеть, но медленнее обучение\n"
                "и больше риск переобучения на малых данных."
            )
            row.addWidget(spin)
            row.addStretch()
            self.mlp_layers_container.addLayout(row)
            self.mlp_layer_spins.append(spin)

    # ---------- CNN ----------
    def _create_cnn_settings(self) -> QWidget:
        widget = QGroupBox("Настройки CNN (свёрточная сеть)")
        widget.setToolTip("Настройки для обработки изображений и спектрограмм.")
        layout = QFormLayout(widget)

        self.cnn_num_layers = QSpinBox()
        self.cnn_num_layers.setRange(1, 8)
        self.cnn_num_layers.setValue(3)
        self.cnn_num_layers.setToolTip(
            "Количество свёрточных блоков.\n"
            "Каждый блок: свёртка → активация → pooling.\n"
            "• 2–3: простые фигуры (круг/квадрат)\n"
            "• 4–6: средние задачи (цифры, буквы)\n"
            "• 7+: сложные (нужно много данных)"
        )
        self.cnn_num_layers.valueChanged.connect(self._update_cnn_layer_spins)
        layout.addRow("Свёрточные блоки:", self.cnn_num_layers)

        self.cnn_kernel_size = QComboBox()
        self.cnn_kernel_size.addItems(["3×3", "5×5", "7×7"])
        self.cnn_kernel_size.setToolTip(
            "Размер ядра (окошка), которым сеть сканирует изображение.\n"
            "• 3×3 — стандарт, находит мелкие детали\n"
            "• 5×5 — находит более крупные паттерны\n"
            "• 7×7 — для первого слоя (быстрый захват больших областей)"
        )
        layout.addRow("Размер ядра (по умолчанию):", self.cnn_kernel_size)

        self.cnn_pool_combo = QComboBox()
        self.cnn_pool_combo.addItems(["max_pool", "avg_pool", "none"])
        self.cnn_pool_combo.setToolTip(
            "Pooling (сжатие) после каждой свёртки:\n"
            "• Max Pool — берёт максимум из окна (находит яркие признаки)\n"
            "• Avg Pool — берёт среднее (сглаживает)\n"
            "• None — без сжатия (больше параметров)"
        )
        layout.addRow("Pooling:", self.cnn_pool_combo)

        self.cnn_global_pool = QCheckBox("Global Average Pooling в конце")
        self.cnn_global_pool.setChecked(True)
        self.cnn_global_pool.setToolTip(
            "Вместо того чтобы 'вытянуть' всю карту признаков в длинный вектор (Flatten),\n"
            "усредняем каждый канал до одного числа.\n"
            "Меньше параметров, меньше переобучение. Рекомендуется."
        )
        layout.addRow("", self.cnn_global_pool)

        # Контейнер для фильтров на каждый слой
        self.cnn_filters_container = QVBoxLayout()
        self.cnn_filter_spins: List[QSpinBox] = []
        layout.addRow(QLabel("Фильтры на блок:"))
        layout.addRow(self.cnn_filters_container)
        self._update_cnn_layer_spins(3)

        return widget

    def _update_cnn_layer_spins(self, count: int):
        if not hasattr(self, 'cnn_filters_container'):
            return
        for i in reversed(range(self.cnn_filters_container.count())):
            w = self.cnn_filters_container.itemAt(i).widget()
            if w:
                w.setParent(None)
        self.cnn_filter_spins.clear()
        defaults = [16, 32, 64, 128, 256, 256, 512, 512]
        for i in range(count):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"  Блок {i + 1}:"))
            spin = QSpinBox()
            spin.setRange(4, 1024)
            spin.setSingleStep(8)
            spin.setValue(defaults[i] if i < len(defaults) else 32)
            spin.setToolTip(
                f"Количество фильтров (каналов) в блоке {i + 1}.\n"
                "Каждый фильтр ищет свой узор.\n"
                "Обычно удваивают с каждым блоком: 16→32→64→128."
            )
            row.addWidget(spin)
            row.addStretch()
            self.cnn_filters_container.addLayout(row)
            self.cnn_filter_spins.append(spin)

    # ---------- RNN / LSTM / GRU ----------
    def _create_rnn_settings(self) -> QWidget:
        widget = QGroupBox("Настройки рекуррентной сети (RNN/LSTM/GRU)")
        widget.setToolTip("Настройки для последовательных данных: текст, временные ряды.")
        layout = QFormLayout(widget)

        self.rnn_hidden = QSpinBox()
        self.rnn_hidden.setRange(16, 2048)
        self.rnn_hidden.setValue(128)
        self.rnn_hidden.setSingleStep(16)
        self.rnn_hidden.setToolTip(
            "Размер 'памяти' сети (скрытый вектор).\n"
            "Чем больше, тем больше информации сеть может запомнить.\n"
            "• 64–128: простые задачи (сложение)\n"
            "• 256–512: средние (шифры)\n"
            "• 512+: сложные (длинные тексты)"
        )
        layout.addRow("Размер памяти (hidden):", self.rnn_hidden)

        self.rnn_layers = QSpinBox()
        self.rnn_layers.setRange(1, 6)
        self.rnn_layers.setValue(2)
        self.rnn_layers.setToolTip(
            "Количество рекуррентных слоёв друг за другом.\n"
            "• 1 слой: простые задачи, быстрое обучение\n"
            "• 2–3 слоя: стандарт для большинства задач\n"
            "• 4+ слоёв: только для очень длинных последовательностей"
        )
        layout.addRow("Количество слоёв:", self.rnn_layers)

        self.rnn_bidirectional = QCheckBox("Двунаправленная (читает слева и справа)")
        self.rnn_bidirectional.setToolTip(
            "Если включено: сеть читает текст слева→направо И справа→налево.\n"
            "Понимает контекст с обеих сторон. Удваивает размер памяти.\n"
            "Рекомендуется для классификации, НЕ для генерации."
        )
        layout.addRow("", self.rnn_bidirectional)

        self.rnn_embed = QSpinBox()
        self.rnn_embed.setRange(8, 512)
        self.rnn_embed.setValue(64)
        self.rnn_embed.setSingleStep(8)
        self.rnn_embed.setToolTip(
            "Размер эмбеддинга (векторное представление каждого символа).\n"
            "Для текстовых задач. Для числовых рядов игнорируется.\n"
            "• 32–64: маленький словарь (< 30 символов)\n"
            "• 128–256: средний словарь"
        )
        layout.addRow("Эмбеддинг (для текста):", self.rnn_embed)

        return widget

    # ---------- Transformer ----------
    def _create_transformer_settings(self) -> QWidget:
        widget = QGroupBox("Настройки Трансформера")
        widget.setToolTip("Настройки для архитектуры на основе механизма внимания.")
        layout = QFormLayout(widget)

        self.tf_embed = QSpinBox()
        self.tf_embed.setRange(16, 2048)
        self.tf_embed.setValue(128)
        self.tf_embed.setSingleStep(16)
        self.tf_embed.setToolTip(
            "Размер эмбеддинга (d_model).\n"
            "Каждый символ превращается в вектор этого размера.\n"
            "• 64: простые задачи (сложение однозначных)\n"
            "• 128: стандарт (математика, шифры)\n"
            "• 256: сложные задачи (уравнения)\n"
            "• 512+: продвинутый уровень, нужен GPU"
        )
        self.tf_embed.valueChanged.connect(self._estimate_complexity)
        layout.addRow("Эмбеддинг (d_model):", self.tf_embed)

        self.tf_heads = QSpinBox()
        self.tf_heads.setRange(1, 32)
        self.tf_heads.setValue(8)
        self.tf_heads.setToolTip(
            "Количество голов внимания.\n"
            "Каждая голова ищет СВОИ связи между символами.\n"
            "• 4: минимум для простых задач\n"
            "• 8: стандарт\n"
            "• 16+: для длинных и сложных последовательностей"
        )
        self.tf_heads.valueChanged.connect(self._estimate_complexity)
        layout.addRow("Головы внимания:", self.tf_heads)

        self.tf_enc_layers = QSpinBox()
        self.tf_enc_layers.setRange(1, 24)
        self.tf_enc_layers.setValue(2)
        self.tf_enc_layers.setToolTip(
            "Слои Энкодера (чтение вопроса).\n"
            "Каждый слой: Внимание + Feed-Forward сеть.\n"
            "• 2: простые задачи (сложение)\n"
            "• 3–4: средние (шифры, уравнения)\n"
            "• 6+: сложные (многошаговые)"
        )
        self.tf_enc_layers.valueChanged.connect(self._estimate_complexity)
        layout.addRow("Слои Энкодера:", self.tf_enc_layers)

        self.tf_dec_layers = QSpinBox()
        self.tf_dec_layers.setRange(1, 24)
        self.tf_dec_layers.setValue(2)
        self.tf_dec_layers.setToolTip(
            "Слои Декодера (генерация ответа).\n"
            "Можно сделать меньше, чем в энкодере, для ускорения.\n"
            "• 2: стандарт\n"
            "• 3–4: для длинных ответов"
        )
        self.tf_dec_layers.valueChanged.connect(self._estimate_complexity)
        layout.addRow("Слои Декодера:", self.tf_dec_layers)

        self.tf_ffn_dim = QSpinBox()
        self.tf_ffn_dim.setRange(64, 8192)
        self.tf_ffn_dim.setValue(512)
        self.tf_ffn_dim.setSingleStep(64)
        self.tf_ffn_dim.setToolTip(
            "Размер скрытого слоя Feed-Forward внутри каждого слоя.\n"
            "Обычно в 4 раза больше эмбеддинга.\n"
            "Пример: d_model=128 → FFN=512."
        )
        self.tf_ffn_dim.valueChanged.connect(self._estimate_complexity)
        layout.addRow("Feed-Forward (FFN):", self.tf_ffn_dim)

        self.tf_pos_encoding = QComboBox()
        self.tf_pos_encoding.addItems(["sinusoidal", "learned"])
        self.tf_pos_encoding.setToolTip(
            "Позиционное кодирование — говорит модели, ГДЕ стоит символ.\n"
            "• Sinusoidal — фиксированные формулы (стандарт)\n"
            "• Learned — модель учит позиции сама (для длинных текстов)"
        )
        layout.addRow("Позиционное кодирование:", self.tf_pos_encoding)

        return widget

    # ---------- Autoencoder / ConvAE ----------
    def _create_autoencoder_settings(self) -> QWidget:
        widget = QGroupBox("Настройки Автоэнкодера / ConvAE")
        widget.setToolTip("Автоэнкодер сжимает данные в скрытое представление и восстанавливает их.")
        layout = QFormLayout(widget)

        self.ae_latent_dim = QSpinBox()
        self.ae_latent_dim.setRange(2, 512)
        self.ae_latent_dim.setValue(32)
        self.ae_latent_dim.setToolTip(
            "Размер латентного (сжатого) представления.\n"
            "Это 'бутылочное горлышко' — через него проходит вся информация.\n"
            "• Маленький (2–16): сильное сжатие, грубое восстановление.\n"
            "• Средний (32–128): баланс.\n"
            "• Большой (128+): почти без сжатия."
        )
        layout.addRow("Латентный размер (сжатие):", self.ae_latent_dim)

        self.ae_enc_layers = QSpinBox()
        self.ae_enc_layers.setRange(1, 8)
        self.ae_enc_layers.setValue(3)
        self.ae_enc_layers.setToolTip(
            "Сколько слоёв сжимают данные (энкодер).\n"
            "Столько же слоёв будет в декодере (зеркально)."
        )
        layout.addRow("Слои энкодера:", self.ae_enc_layers)

        self.ae_hidden = QSpinBox()
        self.ae_hidden.setRange(16, 2048)
        self.ae_hidden.setValue(128)
        self.ae_hidden.setSingleStep(16)
        self.ae_hidden.setToolTip(
            "Количество нейронов в скрытых слоях энкодера.\n"
            "Каждый следующий слой уменьшается к латентному размеру."
        )
        layout.addRow("Скрытый размер:", self.ae_hidden)

        return widget

    # ---------- VAE / CVAE ----------
    def _create_vae_settings(self) -> QWidget:
        widget = QGroupBox("Настройки VAE / CVAE")
        widget.setToolTip(
            "Вариационный автоэнкодер учится генерировать новые данные.\n"
            "CVAE — условная версия: можно попросить 'сгенерируй именно тройку'."
        )
        layout = QFormLayout(widget)

        self.vae_latent_dim = QSpinBox()
        self.vae_latent_dim.setRange(2, 512)
        self.vae_latent_dim.setValue(16)
        self.vae_latent_dim.setToolTip(
            "Размер латентного пространства.\n"
            "Из этого пространства генерируются новые данные.\n"
            "• Маленький (2–8): простые данные.\n"
            "• Средний (16–64): стандарт.\n"
            "• Большой: для сложных изображений."
        )
        layout.addRow("Латентный размер:", self.vae_latent_dim)

        self.vae_enc_layers = QSpinBox()
        self.vae_enc_layers.setRange(1, 8)
        self.vae_enc_layers.setValue(3)
        self.vae_enc_layers.setToolTip("Слои энкодера (сжатие в латентное пространство).")
        layout.addRow("Слои энкодера:", self.vae_enc_layers)

        self.vae_hidden = QSpinBox()
        self.vae_hidden.setRange(16, 2048)
        self.vae_hidden.setValue(256)
        self.vae_hidden.setSingleStep(32)
        self.vae_hidden.setToolTip("Скрытые нейроны в энкодере и декодере.")
        layout.addRow("Скрытый размер:", self.vae_hidden)

        self.vae_kl_weight = QDoubleSpinBox()
        self.vae_kl_weight.setRange(0.0, 10.0)
        self.vae_kl_weight.setValue(0.001)
        self.vae_kl_weight.setSingleStep(0.001)
        self.vae_kl_weight.setDecimals(4)
        self.vae_kl_weight.setToolTip(
            "Вес KL-дивергенции (штраф за 'неправильное' латентное пространство).\n"
            "• Маленький (0.0001–0.001): сеть лучше реконструирует, но хуже генерирует.\n"
            "• Средний (0.001–0.01): баланс.\n"
            "• Большой (0.1+): латентное пространство упорядочено, но реконструкция хуже."
        )
        layout.addRow("Вес KL-штрафа (β):", self.vae_kl_weight)

        self.vae_conditional = QCheckBox("Условный режим (CVAE)")
        self.vae_conditional.setToolTip(
            "Если включено: генерация по метке класса.\n"
            "Например: 'сгенерируй изображение цифры 7'.\n"
            "Требует размеченных данных с классами."
        )
        layout.addRow("", self.vae_conditional)

        return widget

    # ---------- GAN / DCGAN ----------
    def _create_gan_settings(self) -> QWidget:
        widget = QGroupBox("Настройки GAN / DCGAN")
        widget.setToolTip(
            "Две сети играют в игру: Генератор рисует подделки,\n"
            "Дискриминатор пытается отличить подделку от настоящих данных."
        )
        layout = QFormLayout(widget)

        self.gan_latent_dim = QSpinBox()
        self.gan_latent_dim.setRange(8, 512)
        self.gan_latent_dim.setValue(100)
        self.gan_latent_dim.setToolTip(
            "Размер вектора шума (латентного входа генератора).\n"
            "Из этого случайного шума генератор создаёт данные.\n"
            "• 50–100: стандарт.\n"
            "• 128–256: для сложных изображений."
        )
        layout.addRow("Размер шума (latent):", self.gan_latent_dim)

        self.gan_gen_filters = QSpinBox()
        self.gan_gen_filters.setRange(16, 512)
        self.gan_gen_filters.setValue(64)
        self.gan_gen_filters.setSingleStep(16)
        self.gan_gen_filters.setToolTip(
            "Базовое количество фильтров генератора.\n"
            "Каждый следующий слой уменьшает вдвое."
        )
        layout.addRow("Фильтры генератора:", self.gan_gen_filters)

        self.gan_disc_filters = QSpinBox()
        self.gan_disc_filters.setRange(16, 512)
        self.gan_disc_filters.setValue(64)
        self.gan_disc_filters.setSingleStep(16)
        self.gan_disc_filters.setToolTip(
            "Базовое количество фильтров дискриминатора.\n"
            "Каждый следующий слой увеличивает вдвое."
        )
        layout.addRow("Фильтры дискриминатора:", self.gan_disc_filters)

        self.gan_gen_layers = QSpinBox()
        self.gan_gen_layers.setRange(2, 8)
        self.gan_gen_layers.setValue(4)
        self.gan_gen_layers.setToolTip(
            "Количество слоёв генератора.\n"
            "• 3–4: для небольших изображений (28×28, 64×64).\n"
            "• 5–6: для больших (128×128)."
        )
        layout.addRow("Слои генератора:", self.gan_gen_layers)

        self.gan_disc_layers = QSpinBox()
        self.gan_disc_layers.setRange(2, 8)
        self.gan_disc_layers.setValue(4)
        self.gan_disc_layers.setToolTip("Количество слоёв дискриминатора.")
        layout.addRow("Слои дискриминатора:", self.gan_disc_layers)

        return widget

    # ---------- Diffusion / LDM ----------
    def _create_diffusion_settings(self) -> QWidget:
        widget = QGroupBox("Настройки Диффузионной модели")
        widget.setToolTip(
            "Модель учится постепенно убирать шум из данных.\n"
            "⚠️ Экспериментально: требует много данных и времени!"
        )
        layout = QFormLayout(widget)

        self.diff_steps = QSpinBox()
        self.diff_steps.setRange(10, 2000)
        self.diff_steps.setValue(1000)
        self.diff_steps.setSingleStep(100)
        self.diff_steps.setToolTip(
            "Количество шагов диффузии (сколько раз добавить шум при обучении).\n"
            "• 100–500: быстрое обучение, ниже качество.\n"
            "• 1000: стандарт.\n"
            "• 2000: максимальное качество, очень долго."
        )
        layout.addRow("Шаги диффузии (T):", self.diff_steps)

        self.diff_schedule = QComboBox()
        self.diff_schedule.addItems(["linear", "cosine"])
        self.diff_schedule.setToolTip(
            "Расписание добавления шума:\n"
            "• Linear — равномерно от малого к большому.\n"
            "• Cosine — плавно в начале и конце, агрессивно в середине.\n"
            "Cosine обычно даёт лучшие результаты."
        )
        layout.addRow("Расписание шума:", self.diff_schedule)

        self.diff_channels = QSpinBox()
        self.diff_channels.setRange(8, 512)
        self.diff_channels.setValue(64)
        self.diff_channels.setSingleStep(16)
        self.diff_channels.setToolTip(
            "Базовое количество каналов в U-Net.\n"
            "Больше каналов = мощнее модель = больше VRAM."
        )
        layout.addRow("Каналы U-Net:", self.diff_channels)

        self.diff_attn_res = QCheckBox("Внимание на средних разрешениях")
        self.diff_attn_res.setChecked(True)
        self.diff_attn_res.setToolTip(
            "Добавляет механизм внимания на промежуточных слоях U-Net.\n"
            "Улучшает глобальную согласованность изображения.\n"
            "Увеличивает время обучения на ~20%."
        )
        layout.addRow("", self.diff_attn_res)

        return widget

    # ---------- ViT ----------
    def _create_vit_settings(self) -> QWidget:
        widget = QGroupBox("Настройки Vision Transformer (ViT)")
        widget.setToolTip(
            "Режет изображение на квадратики (патчи) и обрабатывает их как слова в Трансформере."
        )
        layout = QFormLayout(widget)

        self.vit_patch_size = QSpinBox()
        self.vit_patch_size.setRange(4, 32)
        self.vit_patch_size.setValue(8)
        self.vit_patch_size.setSingleStep(2)
        self.vit_patch_size.setToolTip(
            "Размер одного патча (квадратика).\n"
            "• Маленький (4): больше патчей, детальнее, но тяжелее.\n"
            "• Средний (8): стандарт для 64×64.\n"
            "• Большой (16): для 224×224."
        )
        layout.addRow("Размер патча:", self.vit_patch_size)

        self.vit_embed = QSpinBox()
        self.vit_embed.setRange(32, 1024)
        self.vit_embed.setValue(128)
        self.vit_embed.setSingleStep(32)
        self.vit_embed.setToolTip(
            "Размер эмбеддинга каждого патча (d_model).\n"
            "Как и в текстовом Трансформере."
        )
        layout.addRow("Эмбеддинг (d_model):", self.vit_embed)

        self.vit_heads = QSpinBox()
        self.vit_heads.setRange(1, 16)
        self.vit_heads.setValue(8)
        self.vit_heads.setToolTip("Количество голов внимания.")
        layout.addRow("Головы внимания:", self.vit_heads)

        self.vit_layers = QSpinBox()
        self.vit_layers.setRange(1, 24)
        self.vit_layers.setValue(6)
        self.vit_layers.setToolTip(
            "Количество блоков Трансформера.\n"
            "• 4–6: для небольших датасетов.\n"
            "• 12+: для больших наборов изображений."
        )
        layout.addRow("Слои Трансформера:", self.vit_layers)

        return widget

    # ---------- ResNet ----------
    def _create_resnet_settings(self) -> QWidget:
        widget = QGroupBox("Настройки ResNet")
        widget.setToolTip(
            "Остаточная сеть с 'обходными путями' (skip-connections).\n"
            "Позволяет строить очень глубокие сети без деградации."
        )
        layout = QFormLayout(widget)

        self.resnet_depth = QComboBox()
        self.resnet_depth.addItems(["18", "34", "50"])
        self.resnet_depth.setToolTip(
            "Глубина сети (количество слоёв):\n"
            "• 18: лёгкая, быстрая, для небольших данных.\n"
            "• 34: средняя.\n"
            "• 50: тяжёлая, требует много данных и VRAM."
        )
        layout.addRow("Глубина (вариант):", self.resnet_depth)

        self.resnet_filters = QSpinBox()
        self.resnet_filters.setRange(8, 128)
        self.resnet_filters.setValue(32)
        self.resnet_filters.setSingleStep(8)
        self.resnet_filters.setToolTip(
            "Базовое количество фильтров.\n"
            "Каждый следующий блок увеличивает вдвое."
        )
        layout.addRow("Базовые фильтры:", self.resnet_filters)

        return widget

    # ---------- GNN / GCN / GAT ----------
    def _create_gnn_settings(self) -> QWidget:
        widget = QGroupBox("Настройки графовой нейросети")
        widget.setToolTip("Работает с графами: узлы, рёбра, признаки.")
        layout = QFormLayout(widget)

        self.gnn_hidden = QSpinBox()
        self.gnn_hidden.setRange(16, 512)
        self.gnn_hidden.setValue(64)
        self.gnn_hidden.setSingleStep(16)
        self.gnn_hidden.setToolTip(
            "Размер скрытого представления каждого узла графа.\n"
            "• 32–64: для небольших графов.\n"
            "• 128+: для сложных."
        )
        layout.addRow("Скрытый размер узла:", self.gnn_hidden)

        self.gnn_layers = QSpinBox()
        self.gnn_layers.setRange(1, 8)
        self.gnn_layers.setValue(3)
        self.gnn_layers.setToolTip(
            "Количество слоёв распространения сообщений.\n"
            "Каждый слой 'видит' соседей на 1 шаг дальше.\n"
            "• 2: видит соседей соседей.\n"
            "• 3–4: стандарт."
        )
        layout.addRow("Слои сообщения:", self.gnn_layers)

        self.gnn_heads = QSpinBox()
        self.gnn_heads.setRange(1, 16)
        self.gnn_heads.setValue(4)
        self.gnn_heads.setToolTip(
            "Головы внимания (только для GAT).\n"
            "Для GCN и GNN игнорируются."
        )
        layout.addRow("Головы внимания (GAT):", self.gnn_heads)

        return widget

    # ---------- RBFN ----------
    def _create_rbfn_settings(self) -> QWidget:
        widget = QGroupBox("Настройки RBFN (Радиально-базисная сеть)")
        widget.setToolTip("Использует гауссовы 'колокольчики' вместо обычных нейронов.")
        layout = QFormLayout(widget)

        self.rbfn_centers = QSpinBox()
        self.rbfn_centers.setRange(4, 512)
        self.rbfn_centers.setValue(32)
        self.rbfn_centers.setSingleStep(8)
        self.rbfn_centers.setToolTip(
            "Количество радиально-базисных функций (центров).\n"
            "Каждый 'колокольчик' — один центр.\n"
            "• 8–32: простые задачи.\n"
            "• 64–128: средние."
        )
        layout.addRow("Количество центров:", self.rbfn_centers)

        self.rbfn_sigma = QDoubleSpinBox()
        self.rbfn_sigma.setRange(0.01, 10.0)
        self.rbfn_sigma.setValue(1.0)
        self.rbfn_sigma.setSingleStep(0.1)
        self.rbfn_sigma.setToolTip(
            "Ширина гауссовой функции (σ).\n"
            "• Маленькая: каждый центр 'видит' только близкие точки.\n"
            "• Большая: центры перекрываются, более гладкая аппроксимация."
        )
        layout.addRow("Ширина (σ):", self.rbfn_sigma)

        return widget

    # ---------- SOM ----------
    def _create_som_settings(self) -> QWidget:
        widget = QGroupBox("Настройки SOM (Карта Кохонена)")
        widget.setToolTip(
            "Самоорганизующаяся карта: раскладывает данные по сетке нейронов.\n"
            "Похожие данные оказываются рядом. Обучение без учителя."
        )
        layout = QFormLayout(widget)

        self.som_grid_x = QSpinBox()
        self.som_grid_x.setRange(2, 50)
        self.som_grid_x.setValue(10)
        self.som_grid_x.setToolTip("Количество нейронов по горизонтали.")
        layout.addRow("Сетка (ширина):", self.som_grid_x)

        self.som_grid_y = QSpinBox()
        self.som_grid_y.setRange(2, 50)
        self.som_grid_y.setValue(10)
        self.som_grid_y.setToolTip("Количество нейронов по вертикали.")
        layout.addRow("Сетка (высота):", self.som_grid_y)

        self.som_sigma = QDoubleSpinBox()
        self.som_sigma.setRange(0.5, 10.0)
        self.som_sigma.setValue(3.0)
        self.som_sigma.setSingleStep(0.5)
        self.som_sigma.setToolTip(
            "Радиус соседства: как далеко от 'победившего' нейрона обновляются соседи.\n"
            "Обычно уменьшается со временем обучения."
        )
        layout.addRow("Радиус соседства (σ):", self.som_sigma)

        return widget

    # ---------- Заглушка для экспериментальных ----------
    def _create_stub_settings(self, name: str) -> QWidget:
        widget = QGroupBox(f"Настройки {name} (экспериментально)")
        widget.setToolTip(
            f"Архитектура '{name}' находится в экспериментальном статусе.\n"
            "Настройки базовые. Обучение может быть нестабильным."
        )
        layout = QVBoxLayout(widget)
        stub_label = QLabel(
            f"⚠️ {name} — экспериментальная архитектура.\n"
            "Используются базовые параметры из блока «Общие настройки».\n"
            "Специфичные настройки будут добавлены в будущих версиях."
        )
        stub_label.setWordWrap(True)
        stub_label.setStyleSheet("color: #e5c07b; font-size: 13px; padding: 10px;")
        layout.addWidget(stub_label)

        # Базовый hidden size для экспериментальных
        self.stub_hidden = QSpinBox()
        self.stub_hidden.setRange(16, 1024)
        self.stub_hidden.setValue(64)
        self.stub_hidden.setToolTip("Базовый скрытый размер (для экспериментальных архитектур).")
        layout.addWidget(QLabel("Скрытый размер:"))
        layout.addWidget(self.stub_hidden)
        layout.addStretch()
        return widget

    # ────────────────────────────────────────────────────────────────
    # 5. ВИЗУАЛИЗАЦИЯ АРХИТЕКТУРЫ
    # ────────────────────────────────────────────────────────────────
    def _build_arch_visualization(self):
        group = QGroupBox("📐 Схема архитектуры")
        group.setToolTip(
            "Упрощённая блок-схема выбранной архитектуры.\n"
            "Показывает, как данные проходят через слои."
        )
        layout = QVBoxLayout(group)
        self.viz_box = QTextEdit()
        self.viz_box.setReadOnly(True)
        self.viz_box.setMaximumHeight(120)
        self.viz_box.setStyleSheet(
            "background-color: #1e1e1e; color: #a9b7c6; "
            "font-family: 'Consolas', monospace; font-size: 12px; "
            "padding: 6px; border: 1px solid #555;"
        )
        self.viz_box.setToolTip("Текстовая схема прохождения данных через слои модели.")
        self.viz_box.setPlainText("Выберите архитектуру...")
        layout.addWidget(self.viz_box)
        self.content_layout.addWidget(group)

    def _update_arch_viz(self):
        """Обновляет текстовую визуализацию архитектуры."""
        if not hasattr(self, 'viz_box'):
            return

        arch = self._current_arch
        viz_map = {
            ArchitectureType.MLP.value: self._viz_mlp,
            ArchitectureType.CNN.value: self._viz_cnn,
            ArchitectureType.RNN.value: self._viz_rnn,
            ArchitectureType.LSTM.value: self._viz_rnn,
            ArchitectureType.GRU.value: self._viz_rnn,
            ArchitectureType.TRANSFORMER_SEQ2SEQ.value: self._viz_transformer,
            ArchitectureType.TRANSFORMER_ENCODER.value: self._viz_transformer_enc,
            ArchitectureType.AUTOENCODER.value: self._viz_autoencoder,
            ArchitectureType.CONV_AE.value: self._viz_autoencoder,
            ArchitectureType.VAE.value: self._viz_vae,
            ArchitectureType.CVAE.value: self._viz_vae,
            ArchitectureType.GAN.value: self._viz_gan,
            ArchitectureType.DCGAN.value: self._viz_gan,
            ArchitectureType.DIFFUSION.value: self._viz_diffusion,
            ArchitectureType.LATENT_DIFFUSION.value: self._viz_diffusion,
            ArchitectureType.VIT.value: self._viz_vit,
            ArchitectureType.RESNET.value: self._viz_resnet,
            ArchitectureType.GNN.value: self._viz_gnn,
            ArchitectureType.GCN.value: self._viz_gnn,
            ArchitectureType.GAT.value: self._viz_gnn,
            ArchitectureType.RBFN.value: self._viz_rbfn,
            ArchitectureType.SOM.value: self._viz_som,
        }
        func = viz_map.get(arch)
        if func:
            try:
                self.viz_box.setPlainText(func())
            except Exception as e:
                self.viz_box.setPlainText(f"[Ошибка визуализации: {e}]")
        else:
            self.viz_box.setPlainText(f"[{arch}]\nСхема будет доступна в следующей версии.")

    def _viz_mlp(self) -> str:
        layers = [s.value() for s in self.mlp_layer_spins] if hasattr(self, 'mlp_layer_spins') and self.mlp_layer_spins else [64]
        act = self.activation_combo.currentText().upper() if hasattr(self, 'activation_combo') else "RELU"
        parts = ["Вход"] + [f"Linear({h}) → {act}" for h in layers] + ["Выход"]
        return " → ".join(parts)

    def _viz_cnn(self) -> str:
        n = self.cnn_num_layers.value() if hasattr(self, 'cnn_num_layers') else 3
        blocks = []
        for i in range(n):
            f = self.cnn_filter_spins[i].value() if hasattr(self, 'cnn_filter_spins') and i < len(self.cnn_filter_spins) else 32
            blocks.append(f"Conv({f})→ReLU→Pool")
        return "Изображение → " + " → ".join(blocks) + " → FC → Класс"

    def _viz_rnn(self) -> str:
        h = self.rnn_hidden.value() if hasattr(self, 'rnn_hidden') else 128
        n = self.rnn_layers.value() if hasattr(self, 'rnn_layers') else 2
        bi = " ↔ " if hasattr(self, 'rnn_bidirectional') and self.rnn_bidirectional.isChecked() else " → "
        return f"Символы{bi}[{self._current_arch.upper()} x{n} (h={h})]{bi}FC → Ответ"

    def _viz_transformer(self) -> str:
        d = self.tf_embed.value() if hasattr(self, 'tf_embed') else 128
        heads = self.tf_heads.value() if hasattr(self, 'tf_heads') else 8
        enc = self.tf_enc_layers.value() if hasattr(self, 'tf_enc_layers') else 2
        dec = self.tf_dec_layers.value() if hasattr(self, 'tf_dec_layers') else 2
        return (
            f"Вход → Embedding({d}) → Позиции →\n"
            f"  Энкодер [{enc}× (Attention({heads}) + FFN)] →\n"
            f"  Декодер [{dec}× (MaskedAttn + CrossAttn + FFN)] →\n"
            f"  FC → Словарь"
        )

    def _viz_transformer_enc(self) -> str:
        d = self.tf_embed.value() if hasattr(self, 'tf_embed') else 128
        heads = self.tf_heads.value() if hasattr(self, 'tf_heads') else 8
        enc = self.tf_enc_layers.value() if hasattr(self, 'tf_enc_layers') else 2
        return (
            f"Вход → Embedding({d}) → Позиции →\n"
            f"  Энкодер [{enc}× (Attention({heads}) + FFN)] →\n"
            f"  [CLS] → FC → Класс"
        )

    def _viz_autoencoder(self) -> str:
        lat = self.ae_latent_dim.value() if hasattr(self, 'ae_latent_dim') else 32
        n = self.ae_enc_layers.value() if hasattr(self, 'ae_enc_layers') else 3
        return f"Данные → Энкодер[{n} слоёв] → Латент({lat}) → Декодер[{n} слоёв] → Реконструкция"

    def _viz_vae(self) -> str:
        lat = self.vae_latent_dim.value() if hasattr(self, 'vae_latent_dim') else 16
        cond = " + Метка класса" if hasattr(self, 'vae_conditional') and self.vae_conditional.isChecked() else ""
        return (
            f"Данные{cond} → Энкодер → μ, σ →\n"
            f"  Латент({lat}) [сэмплирование] →\n"
            f"  Декодер → Реконструкция / Генерация"
        )

    def _viz_gan(self) -> str:
        lat = self.gan_latent_dim.value() if hasattr(self, 'gan_latent_dim') else 100
        return (
            f"Шум({lat}) → Генератор → Подделка ─┐\n"
            f"                                    ├→ Дискриминатор → Реально/Фальшиво\n"
            f"Реальные данные ───────────────────┘"
        )

    def _viz_diffusion(self) -> str:
        t = self.diff_steps.value() if hasattr(self, 'diff_steps') else 1000
        return (
            f"Данные → [+шум × {t} шагов] → Чистый шум (обучение)\n"
            f"Шум → [−шум × {t} шагов] → Сгенерированные данные (инференс)"
        )

    def _viz_vit(self) -> str:
        p = self.vit_patch_size.value() if hasattr(self, 'vit_patch_size') else 8
        d = self.vit_embed.value() if hasattr(self, 'vit_embed') else 128
        n = self.vit_layers.value() if hasattr(self, 'vit_layers') else 6
        return (
            f"Изображение → Патчи({p}×{p}) → Embedding({d}) →\n"
            f"  [CLS] + Позиции → Трансформер [{n}× Attention + FFN] →\n"
            f"  [CLS] → FC → Класс"
        )

    def _viz_resnet(self) -> str:
        depth = self.resnet_depth.currentText() if hasattr(self, 'resnet_depth') else "18"
        return f"Изображение → Stem → ResBlock ×{depth} (skip-connections) → GAP → FC → Класс"

    def _viz_gnn(self) -> str:
        h = self.gnn_hidden.value() if hasattr(self, 'gnn_hidden') else 64
        n = self.gnn_layers.value() if hasattr(self, 'gnn_layers') else 3
        return f"Узлы графа → [{n}× Сообщение от соседей → Обновление({h})] → Пул → Класс"

    def _viz_rbfn(self) -> str:
        c = self.rbfn_centers.value() if hasattr(self, 'rbfn_centers') else 32
        return f"Вход → {c} гауссовых центров (φ) → Линейная комбинация → Выход"

    def _viz_som(self) -> str:
        gx = self.som_grid_x.value() if hasattr(self, 'som_grid_x') else 10
        gy = self.som_grid_y.value() if hasattr(self, 'som_grid_y') else 10
        return f"Вход → Сетка нейронов ({gx}×{gy}) → Победитель (лучший) → Обновление соседей"

    # ────────────────────────────────────────────────────────────────
    # 6. ИНДИКАТОР СЛОЖНОСТИ
    # ────────────────────────────────────────────────────────────────
    def _build_complexity_indicator(self):
        group = QGroupBox("🚦 Оценка сложности")
        group.setToolTip("Примерная оценка размера модели и требуемых ресурсов.")
        layout = QHBoxLayout(group)
        self.complexity_label = QLabel("⚪ Выберите архитектуру")
        self.complexity_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(self.complexity_label)
        layout.addStretch()
        self.params_label = QLabel("Параметров: —")
        self.params_label.setStyleSheet("color: #888; font-size: 13px;")
        self.params_label.setToolTip("Примерное количество обучаемых параметров модели.")
        layout.addWidget(self.params_label)
        self.content_layout.addWidget(group)

    def _estimate_complexity(self):
        """Оценивает примерное число параметров для светофора."""
        if not hasattr(self, 'complexity_label') or not hasattr(self, 'params_label'):
            return

        # Если кнопка создания отключена — complexity_label уже показывает
        # ПРИЧИНУ (нет данных / несовместимость). Не затираем её оценкой сложности.
        if hasattr(self, 'create_btn') and not self.create_btn.isEnabled():
            return

        dataset = dataset_to_legacy_dict(self.shared_state.get("dataset"))
        vocab_size = len(dataset.get("vocab", {})) if dataset and dataset.get("vocab") else 100
        arch = self._current_arch
        approx = 0

        try:
            if arch == ArchitectureType.MLP.value:
                layers = [s.value() for s in self.mlp_layer_spins] if hasattr(self, 'mlp_layer_spins') and self.mlp_layer_spins else [64]
                input_dim = 1
                if dataset:
                    input_dim = dataset.get("meta", {}).get("input_dim", 1)
                prev = input_dim
                for h in layers:
                    approx += prev * h + h
                    prev = h
                output_dim = dataset.get("meta", {}).get("output_dim", 1) if dataset else 1
                approx += prev * output_dim + output_dim

            elif arch == ArchitectureType.CNN.value:
                n = self.cnn_num_layers.value() if hasattr(self, 'cnn_num_layers') else 3
                for i in range(n):
                    f = self.cnn_filter_spins[i].value() if hasattr(self, 'cnn_filter_spins') and i < len(self.cnn_filter_spins) else 32
                    k = 3
                    in_ch = 1 if i == 0 else (self.cnn_filter_spins[i-1].value() if hasattr(self, 'cnn_filter_spins') and i > 0 else 1)
                    approx += k * k * in_ch * f + f
                approx += 256 * 10  # примерный FC

            elif arch in (ArchitectureType.RNN.value, ArchitectureType.LSTM.value, ArchitectureType.GRU.value):
                h = self.rnn_hidden.value() if hasattr(self, 'rnn_hidden') else 128
                n = self.rnn_layers.value() if hasattr(self, 'rnn_layers') else 2
                emb = self.rnn_embed.value() if hasattr(self, 'rnn_embed') and dataset and dataset.get("meta", {}).get("type") == "text" else 1
                approx = vocab_size * emb + n * 4 * (emb + h) * h

            elif arch in (ArchitectureType.TRANSFORMER_SEQ2SEQ.value, ArchitectureType.TRANSFORMER_ENCODER.value):
                d = self.tf_embed.value() if hasattr(self, 'tf_embed') else 128
                heads = self.tf_heads.value() if hasattr(self, 'tf_heads') else 8
                enc = self.tf_enc_layers.value() if hasattr(self, 'tf_enc_layers') else 2
                dec = self.tf_dec_layers.value() if hasattr(self, 'tf_dec_layers') else 2
                ffn = self.tf_ffn_dim.value() if hasattr(self, 'tf_ffn_dim') else 512
                layers = enc + dec
                approx = vocab_size * d + layers * (4 * d * d + 2 * d * ffn) + vocab_size * d

            elif arch in (ArchitectureType.AUTOENCODER.value, ArchitectureType.CONV_AE.value):
                h = self.ae_hidden.value() if hasattr(self, 'ae_hidden') else 128
                lat = self.ae_latent_dim.value() if hasattr(self, 'ae_latent_dim') else 32
                n = self.ae_enc_layers.value() if hasattr(self, 'ae_enc_layers') else 3
                approx = n * h * h + h * lat + lat * h + n * h * h

            elif arch in (ArchitectureType.VAE.value, ArchitectureType.CVAE.value):
                h = self.vae_hidden.value() if hasattr(self, 'vae_hidden') else 256
                lat = self.vae_latent_dim.value() if hasattr(self, 'vae_latent_dim') else 16
                n = self.vae_enc_layers.value() if hasattr(self, 'vae_enc_layers') else 3
                approx = 2 * (n * h * h + h * lat * 2 + lat * h + n * h * h)

            elif arch in (ArchitectureType.GAN.value, ArchitectureType.DCGAN.value):
                gf = self.gan_gen_filters.value() if hasattr(self, 'gan_gen_filters') else 64
                df = self.gan_disc_filters.value() if hasattr(self, 'gan_disc_filters') else 64
                gl = self.gan_gen_layers.value() if hasattr(self, 'gan_gen_layers') else 4
                dl = self.gan_disc_layers.value() if hasattr(self, 'gan_disc_layers') else 4
                approx = gl * gf * gf * 16 + dl * df * df * 16

            elif arch in (ArchitectureType.DIFFUSION.value, ArchitectureType.LATENT_DIFFUSION.value):
                c = self.diff_channels.value() if hasattr(self, 'diff_channels') else 64
                approx = c * c * 4 * 8  # грубая оценка U-Net

            elif arch == ArchitectureType.VIT.value:
                d = self.vit_embed.value() if hasattr(self, 'vit_embed') else 128
                n = self.vit_layers.value() if hasattr(self, 'vit_layers') else 6
                approx = n * (4 * d * d + 2 * d * d * 4) + vocab_size * d

            elif arch == ArchitectureType.RESNET.value:
                f = self.resnet_filters.value() if hasattr(self, 'resnet_filters') else 32
                depth = int(self.resnet_depth.currentText()) if hasattr(self, 'resnet_depth') else 18
                approx = depth * f * f * 9

            elif arch in (ArchitectureType.GNN.value, ArchitectureType.GCN.value, ArchitectureType.GAT.value):
                h = self.gnn_hidden.value() if hasattr(self, 'gnn_hidden') else 64
                n = self.gnn_layers.value() if hasattr(self, 'gnn_layers') else 3
                approx = n * 2 * h * h

            elif arch == ArchitectureType.RBFN.value:
                c = self.rbfn_centers.value() if hasattr(self, 'rbfn_centers') else 32
                approx = c * 2 + c  # центры + веса + смещения

            elif arch == ArchitectureType.SOM.value:
                gx = self.som_grid_x.value() if hasattr(self, 'som_grid_x') else 10
                gy = self.som_grid_y.value() if hasattr(self, 'som_grid_y') else 10
                approx = gx * gy

            else:
                approx = 100_000  # заглушка для экспериментальных

        except Exception as e:
            logger.warning(f"Ошибка оценки сложности: {e}")
            approx = 100_000

        self.params_label.setText(f"Параметров: ~{approx:,}")

        if approx < 1_000_000:
            self.complexity_label.setText(f"🟢 Лёгкая (~{approx/1e6:.1f}M) — секунды на ноутбуке")
            self.complexity_label.setStyleSheet("color: #a6e3a1; font-size: 14px; font-weight: bold;")
        elif approx < 10_000_000:
            self.complexity_label.setText(f"🟡 Средняя (~{approx/1e6:.1f}M) — видеокарта, 5–15 минут")
            self.complexity_label.setStyleSheet("color: #e5c07b; font-size: 14px; font-weight: bold;")
        elif approx < 100_000_000:
            self.complexity_label.setText(f"🔴 Тяжёлая (~{approx/1e6:.1f}M) — нужен GPU с 4+ ГБ")
            self.complexity_label.setStyleSheet("color: #e06c75; font-size: 14px; font-weight: bold;")
        else:
            self.complexity_label.setText(f"💀 Экстремальная (~{approx/1e6:.1f}M) — может не влезть в память!")
            self.complexity_label.setStyleSheet("color: #ff0000; font-size: 14px; font-weight: bold;")

    # ────────────────────────────────────────────────────────────────
    # 7. КНОПКА СОЗДАНИЯ
    # ────────────────────────────────────────────────────────────────
    def _build_create_button(self):
        btn_layout = QHBoxLayout()
        self.create_btn = QPushButton("🛠 Создать модель")
        self.create_btn.setToolTip(
            "Создаёт нейросеть по выбранным параметрам.\n"
            "Модель появится в памяти программы и будет готова к обучению.\n"
            "Если модель уже существует — она будет заменена."
        )
        self.create_btn.setStyleSheet(
            "QPushButton { font-size: 16px; font-weight: bold; padding: 12px; "
            "background-color: #385a3a; border-color: #4a7a4c; }"
            "QPushButton:hover { background-color: #4a7a4c; }"
        )
        self.create_btn.clicked.connect(self.create_model)
        btn_layout.addWidget(self.create_btn)
        btn_layout.addStretch()
        self.content_layout.addLayout(btn_layout)

    # ────────────────────────────────────────────────────────────────
    # 8. ОТЧЁТ
    # ────────────────────────────────────────────────────────────────
    def _build_report_box(self):
        group = QGroupBox("📋 Отчёт о создании модели")
        group.setToolTip("Результат последней попытки создать модель.")
        layout = QVBoxLayout(group)
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setMaximumHeight(150)
        self.report_text.setStyleSheet(
            "background-color: #1e1e1e; color: #a9b7c6; "
            "font-family: 'Consolas', monospace; font-size: 12px; "
            "padding: 6px; border: 1px solid #555;"
        )
        self.report_text.setToolTip("Лог создания модели: параметры, размер, структура.")
        self.report_text.setText("Модель ещё не создана.")
        layout.addWidget(self.report_text)
        self.content_layout.addWidget(group)

    # ────────────────────────────────────────────────────────────────
    # ПРОВЕРКА СОВМЕСТИМОСТИ С ДАННЫМИ
    # ────────────────────────────────────────────────────────────────
    def _check_data_compatibility(self):
        """Проверяет совместимость выбранной архитектуры с текущими данными."""
        if not hasattr(self, 'complexity_label') or not hasattr(self, 'create_btn'):
            return

        dataset = dataset_to_legacy_dict(self.shared_state.get("dataset"))
        if not dataset:
            self.complexity_label.setText("⚪ Загрузите датасет во вкладке 'Данные'")
            self.complexity_label.setStyleSheet("color: #888; font-size: 14px; font-weight: bold;")
            self.create_btn.setEnabled(False)
            return

        # Датасет загружен, но ещё не разбит — модель без разбиения бесполезна
        if not dataset.get("train_inputs"):
            self.complexity_label.setText(
                "⚠️ Датасет загружен, но НЕ разбит!\n"
                "Вкладка 'Данные' → «Применить разбиение»."
            )
            self.complexity_label.setStyleSheet("color: #e5c07b; font-size: 14px; font-weight: bold;")
            self.create_btn.setEnabled(False)
            return

        meta = dataset.get("meta", {})
        data_type_str = meta.get("type", "numeric")

        # Маппинг строки типа данных из датасета → DataType enum
        type_map = {
            "text": DataType.TEXT,
            "numeric": DataType.NUMERIC,
            "image": DataType.IMAGE,
            "audio": DataType.AUDIO,
            "graph": DataType.GRAPH,
            "time_series": DataType.TIME_SERIES,
        }
        data_type = type_map.get(data_type_str, DataType.NUMERIC)
        compatible_types = ARCH_DATA_COMPAT.get(self._current_arch, [])

        if compatible_types and data_type not in compatible_types:
            allowed = ", ".join([t.value for t in compatible_types])
            self.complexity_label.setText(
                f"⚠️ НЕСОВМЕСТИМО! {self._current_arch} не работает с '{data_type_str}'. "
                f"Допустимо: {allowed}"
            )
            self.complexity_label.setStyleSheet("color: #e06c75; font-size: 14px; font-weight: bold;")
            self.create_btn.setEnabled(False)
        else:
            self.create_btn.setEnabled(True)
            self._estimate_complexity()

    # ────────────────────────────────────────────────────────────────
    # ВАЛИДАЦИЯ ПЕРЕД СОЗДАНИЕМ
    # ────────────────────────────────────────────────────────────────
    def _validate_before_create(self) -> str:
        """Возвращает текст ошибки или пустую строку, если всё ок."""
        dataset = dataset_to_legacy_dict(self.shared_state.get("dataset"))
        arch = self._current_arch

        if not dataset:
            return "Сначала загрузите датасет во вкладке 'Данные'."

        if "train_inputs" not in dataset:
            return "Датасет НЕ разбит! Вкладка 'Данные' → 'Применить разбиение'."

        meta = dataset.get("meta", {})
        data_type_str = meta.get("type", "numeric")

        # Проверка совместимости
        type_map = {
            "text": DataType.TEXT, "numeric": DataType.NUMERIC,
            "image": DataType.IMAGE, "audio": DataType.AUDIO,
            "graph": DataType.GRAPH, "time_series": DataType.TIME_SERIES,
        }
        data_type = type_map.get(data_type_str, DataType.NUMERIC)
        compatible = ARCH_DATA_COMPAT.get(arch, [])
        if compatible and data_type not in compatible:
            allowed = ", ".join([t.value for t in compatible])
            return (
                f"Архитектура '{arch}' несовместима с типом данных '{data_type_str}'.\n"
                f"Допустимые типы: {allowed}"
            )

        # Специфичные проверки
        if data_type_str == "text":
            vocab = dataset.get("vocab", {})
            if not vocab or len(vocab) < 4:
                return "Словарь пуст. Примените разбиение данных."

        if arch in (ArchitectureType.TRANSFORMER_SEQ2SEQ.value, ArchitectureType.TRANSFORMER_ENCODER.value):
            embed = self.tf_embed.value() if hasattr(self, 'tf_embed') else 128
            heads = self.tf_heads.value() if hasattr(self, 'tf_heads') else 8
            if heads <= 0:
                return "Количество голов должно быть больше 0."
            if embed % heads != 0:
                suggested = heads * (embed // heads + 1)
                return (
                    f"Эмбеддинг ({embed}) должен делиться на головы ({heads}) без остатка.\n"
                    f"Попробуйте установить эмбеддинг = {suggested}."
                )

        return ""

    # ────────────────────────────────────────────────────────────────
    # СОЗДАНИЕ МОДЕЛИ (ГЛАВНЫЙ МЕТОД)
    # ────────────────────────────────────────────────────────────────
    # Ключи гиперпараметров, которые НЕ должны сбрасываться при создании модели.
    # Пользователь настраивает их во вкладке «Гиперпараметры», и пересоздание
    # модели (смена архитектуры) не должно их затирать — иначе программа
    # «наказывает» пользователя за смену архитектуры.
    _PRESERVE_HYPERPARAM_KEYS = (
        "optimizer", "learning_rate", "weight_decay", "gradient_clipping",
        "epochs", "batch_size", "loss_function", "scheduler",
        "scheduler_step", "scheduler_gamma", "scheduler_params",
        "early_stopping", "patience", "min_delta", "tf_ratio", "grad_accum",
        "nesterov", "momentum", "beta1", "beta2", "eps",
        "rmsprop_alpha", "rho",
    )

    def _preserve_hyperparams(self, new_config: dict) -> dict:
        """Переносит гиперпараметры из старого конфига в новый.

        Новый конфиг (от ModelConfig) содержит гиперпараметры ПО УМОЛЧАНИЮ
        (epochs=50, lr=0.001 и т.д.), поэтому их нужно ПЕРЕЗАПИСАТЬ
        значениями, которые пользователь уже выставил во вкладке
        «Гиперпараметры» — иначе смена архитектуры сбросит настройки обучения.
        """
        old = self.shared_state.get("config")
        if old is None:
            return new_config
        try:
            old_dict = old.to_dict() if hasattr(old, "to_dict") else old
        except Exception:
            old_dict = old
        if not isinstance(old_dict, dict):
            return new_config
        for key in self._PRESERVE_HYPERPARAM_KEYS:
            if key in old_dict:
                new_config[key] = old_dict[key]
        return new_config

    def create_model_preset(self, arch: str, arch_params: dict = None) -> str:
        """Создаёт модель по пресету без модальных диалогов (для ИИ-агента «Мастер»).
        Возвращает отчёт или строку ошибки. Гиперпараметры сохраняются."""
        dataset = self.shared_state.get("dataset")
        if dataset is None:
            return "данные не загружены — сначала создайте/загрузите датасет"
        d = dataset_to_legacy_dict(dataset)
        meta = d.get("meta", {}) or {}
        data_type_str = meta.get("type", "numeric")
        vocab = d.get("vocab", {}) or {}

        config = ModelConfig()
        project = self.shared_state.get("project", {}) or {}
        config.model_name = project.get("model_name", "Модель")
        config.architecture = ArchitectureType(arch)
        config.data_type = (DataType(data_type_str)
                            if data_type_str in [t.value for t in DataType]
                            else DataType.NUMERIC)
        config.task_type = (TaskType(meta.get("task_type", "regression"))
                            if meta.get("task_type") in [t.value for t in TaskType]
                            else TaskType.REGRESSION)
        config.input_dim = meta.get("input_dim", 1)
        config.output_dim = meta.get("output_dim", 1)
        config.vocab_size = len(vocab) if vocab else 0
        config.num_classes = meta.get("num_classes", 0)
        config.image_shape = (tuple(meta.get("image_shape", []))
                              if meta.get("image_shape") else None)
        config.arch_params.update(arch_params or {})

        # Совместимость текстовой модели со словарём
        if config.data_type == DataType.TEXT and vocab:
            config.arch_params.setdefault("pad_idx", vocab.get("<PAD>", 0))
            config.arch_params.setdefault("bos_idx", vocab.get("<BOS>", 2))
            config.arch_params.setdefault("eos_idx", vocab.get("<EOS>", 3))

        errors = ModelFactory.validate_compatibility(config)
        if errors:
            return "; ".join(errors)

        try:
            factory_config = config.to_dict()
            model = ModelFactory.create_model(factory_config)
            self.shared_state["model"] = model
            factory_config = self._preserve_hyperparams(factory_config)
            self.shared_state["config"] = factory_config
            params = ModelFactory.get_param_count(model)
            mem = ModelFactory.estimate_memory(model)
            report = (
                f"✅ Модель «{config.model_name}» ({arch}) создана: "
                f"{params:,} параметров, {mem:.2f} MB"
            )
            if hasattr(self, "report_text"):
                self.report_text.setText(report)
            self._select_arch_in_ui(arch)
            # Эмитим ОБЪЕДИНЁННЫЙ конфиг (с сохранёнными гиперпараметрами),
            # чтобы смена архитектуры не сбрасывала настройки обучения.
            self.model_created.emit(model, factory_config)
            return report
        except Exception as e:
            logger.error(f"Ошибка создания модели по пресету: {e}")
            return f"ошибка создания модели: {e}"

    def _select_arch_in_ui(self, arch_value: str) -> bool:
        """Показывает выбранную архитектуру в комбо (best-effort, без диалогов)."""
        try:
            for gi, group_name in enumerate(ARCH_GROUPS.keys()):
                self._populate_arch_combo(gi)
                for ai in range(self.arch_combo.count()):
                    if self.arch_combo.itemData(ai) == arch_value:
                        self.group_combo.setCurrentIndex(gi)
                        self.arch_combo.setCurrentIndex(ai)
                        return True
        except Exception:
            pass
        return False

    def create_model(self):
        """Создаёт модель и отправляет сигнал model_created."""
        error = self._validate_before_create()
        if error:
            QMessageBox.warning(self, "Проверка не пройдена", error)
            logger.warning(f"Валидация архитектуры не пройдена: {error}")
            return

        dataset = dataset_to_legacy_dict(self.shared_state["dataset"])
        meta = dataset.get("meta", {})
        data_type_str = meta.get("type", "numeric")
        vocab = dataset.get("vocab", {})
        arch = self._current_arch

        # ─── Формируем ModelConfig ───
        config = ModelConfig()
        config.model_name = self.shared_state.get("project", {}).get("model_name", "Модель")
        config.architecture = ArchitectureType(arch)
        config.data_type = DataType(data_type_str) if data_type_str in [t.value for t in DataType] else DataType.NUMERIC
        config.task_type = TaskType(meta.get("task_type", "regression")) if meta.get("task_type") in [t.value for t in TaskType] else TaskType.REGRESSION
        config.input_dim = meta.get("input_dim", 1)
        config.output_dim = meta.get("output_dim", 1)
        config.vocab_size = len(vocab) if vocab else 0
        config.num_classes = meta.get("num_classes", 0)
        config.image_shape = tuple(meta.get("image_shape", [])) if meta.get("image_shape") else None

        # Общие параметры
        config.arch_params["activation"] = self.activation_combo.currentText() if hasattr(self, 'activation_combo') else "relu"
        config.arch_params["dropout"] = self.dropout_spin.value() if hasattr(self, 'dropout_spin') else 0.1
        config.arch_params["init_method"] = self.init_combo.currentText() if hasattr(self, 'init_combo') else "he"

        # Специфичные параметры по архитектуре
        if arch == ArchitectureType.MLP.value:
            config.arch_params["hidden_layers"] = [s.value() for s in self.mlp_layer_spins] if hasattr(self, 'mlp_layer_spins') else [64, 128, 64]
            config.arch_params["batch_norm"] = self.mlp_batchnorm.isChecked() if hasattr(self, 'mlp_batchnorm') else False

        elif arch == ArchitectureType.CNN.value:
            config.arch_params["num_conv_layers"] = self.cnn_num_layers.value() if hasattr(self, 'cnn_num_layers') else 3
            config.arch_params["filters"] = [s.value() for s in self.cnn_filter_spins] if hasattr(self, 'cnn_filter_spins') else [16, 32, 64]
            config.arch_params["kernel_size"] = int(self.cnn_kernel_size.currentText().split("×")[0]) if hasattr(self, 'cnn_kernel_size') else 3
            config.arch_params["pooling"] = self.cnn_pool_combo.currentText() if hasattr(self, 'cnn_pool_combo') else "max_pool"
            config.arch_params["global_pool"] = self.cnn_global_pool.isChecked() if hasattr(self, 'cnn_global_pool') else True

        elif arch in (ArchitectureType.RNN.value, ArchitectureType.LSTM.value, ArchitectureType.GRU.value):
            config.arch_params["hidden_size"] = self.rnn_hidden.value() if hasattr(self, 'rnn_hidden') else 128
            config.arch_params["num_layers"] = self.rnn_layers.value() if hasattr(self, 'rnn_layers') else 2
            config.arch_params["bidirectional"] = self.rnn_bidirectional.isChecked() if hasattr(self, 'rnn_bidirectional') else False
            config.arch_params["rnn_type"] = arch
            if data_type_str == "text":
                config.arch_params["embedding_dim"] = self.rnn_embed.value() if hasattr(self, 'rnn_embed') else 64

        elif arch in (ArchitectureType.TRANSFORMER_SEQ2SEQ.value, ArchitectureType.TRANSFORMER_ENCODER.value):
            config.arch_params["embedding_dim"] = self.tf_embed.value() if hasattr(self, 'tf_embed') else 128
            config.arch_params["num_heads"] = self.tf_heads.value() if hasattr(self, 'tf_heads') else 8
            config.arch_params["num_encoder_layers"] = self.tf_enc_layers.value() if hasattr(self, 'tf_enc_layers') else 2
            config.arch_params["num_decoder_layers"] = self.tf_dec_layers.value() if hasattr(self, 'tf_dec_layers') else 2
            config.arch_params["dim_feedforward"] = self.tf_ffn_dim.value() if hasattr(self, 'tf_ffn_dim') else 512
            config.arch_params["pos_encoding"] = self.tf_pos_encoding.currentText() if hasattr(self, 'tf_pos_encoding') else "sinusoidal"
            # Совместимость со старым форматом
            config.arch_params["pad_idx"] = vocab.get("<PAD>", 0) if vocab else 0
            config.arch_params["bos_idx"] = vocab.get("<BOS>", 2) if vocab else 2
            config.arch_params["eos_idx"] = vocab.get("<EOS>", 3) if vocab else 3

        elif arch in (ArchitectureType.AUTOENCODER.value, ArchitectureType.CONV_AE.value):
            config.arch_params["latent_dim"] = self.ae_latent_dim.value() if hasattr(self, 'ae_latent_dim') else 32
            config.arch_params["num_layers"] = self.ae_enc_layers.value() if hasattr(self, 'ae_enc_layers') else 3
            config.arch_params["hidden_size"] = self.ae_hidden.value() if hasattr(self, 'ae_hidden') else 128

        elif arch in (ArchitectureType.VAE.value, ArchitectureType.CVAE.value):
            config.arch_params["latent_dim"] = self.vae_latent_dim.value() if hasattr(self, 'vae_latent_dim') else 16
            config.arch_params["num_layers"] = self.vae_enc_layers.value() if hasattr(self, 'vae_enc_layers') else 3
            config.arch_params["hidden_size"] = self.vae_hidden.value() if hasattr(self, 'vae_hidden') else 256
            config.arch_params["kl_weight"] = self.vae_kl_weight.value() if hasattr(self, 'vae_kl_weight') else 0.001
            config.arch_params["conditional"] = self.vae_conditional.isChecked() if hasattr(self, 'vae_conditional') else False

        elif arch in (ArchitectureType.GAN.value, ArchitectureType.DCGAN.value):
            config.arch_params["latent_dim"] = self.gan_latent_dim.value() if hasattr(self, 'gan_latent_dim') else 100
            config.arch_params["gen_filters"] = self.gan_gen_filters.value() if hasattr(self, 'gan_gen_filters') else 64
            config.arch_params["disc_filters"] = self.gan_disc_filters.value() if hasattr(self, 'gan_disc_filters') else 64
            config.arch_params["gen_layers"] = self.gan_gen_layers.value() if hasattr(self, 'gan_gen_layers') else 4
            config.arch_params["disc_layers"] = self.gan_disc_layers.value() if hasattr(self, 'gan_disc_layers') else 4

        elif arch in (ArchitectureType.DIFFUSION.value, ArchitectureType.LATENT_DIFFUSION.value):
            config.arch_params["timesteps"] = self.diff_steps.value() if hasattr(self, 'diff_steps') else 1000
            config.arch_params["schedule"] = self.diff_schedule.currentText() if hasattr(self, 'diff_schedule') else "linear"
            config.arch_params["base_channels"] = self.diff_channels.value() if hasattr(self, 'diff_channels') else 64
            config.arch_params["attention"] = self.diff_attn_res.isChecked() if hasattr(self, 'diff_attn_res') else True

        elif arch == ArchitectureType.VIT.value:
            config.arch_params["patch_size"] = self.vit_patch_size.value() if hasattr(self, 'vit_patch_size') else 8
            config.arch_params["embedding_dim"] = self.vit_embed.value() if hasattr(self, 'vit_embed') else 128
            config.arch_params["num_heads"] = self.vit_heads.value() if hasattr(self, 'vit_heads') else 8
            config.arch_params["num_layers"] = self.vit_layers.value() if hasattr(self, 'vit_layers') else 6

        elif arch == ArchitectureType.RESNET.value:
            config.arch_params["depth"] = int(self.resnet_depth.currentText()) if hasattr(self, 'resnet_depth') else 18
            config.arch_params["base_filters"] = self.resnet_filters.value() if hasattr(self, 'resnet_filters') else 32

        elif arch in (ArchitectureType.GNN.value, ArchitectureType.GCN.value, ArchitectureType.GAT.value):
            config.arch_params["hidden_size"] = self.gnn_hidden.value() if hasattr(self, 'gnn_hidden') else 64
            config.arch_params["num_layers"] = self.gnn_layers.value() if hasattr(self, 'gnn_layers') else 3
            config.arch_params["num_heads"] = self.gnn_heads.value() if hasattr(self, 'gnn_heads') else 4
            config.arch_params["gnn_type"] = arch

        elif arch == ArchitectureType.RBFN.value:
            config.arch_params["num_centers"] = self.rbfn_centers.value() if hasattr(self, 'rbfn_centers') else 32
            config.arch_params["sigma"] = self.rbfn_sigma.value() if hasattr(self, 'rbfn_sigma') else 1.0

        elif arch == ArchitectureType.SOM.value:
            config.arch_params["grid_x"] = self.som_grid_x.value() if hasattr(self, 'som_grid_x') else 10
            config.arch_params["grid_y"] = self.som_grid_y.value() if hasattr(self, 'som_grid_y') else 10
            config.arch_params["sigma"] = self.som_sigma.value() if hasattr(self, 'som_sigma') else 3.0

        else:
            # Экспериментальные: базовый hidden
            config.arch_params["hidden_size"] = self.stub_hidden.value() if hasattr(self, 'stub_hidden') else 64

        # ─── Предупреждение для экспериментальных ───
        if arch in EXPERIMENTAL_ARCHS:
            reply = QMessageBox.question(
                self, "Экспериментальная архитектура",
                f"'{arch}' — экспериментальная архитектура.\n"
                "Обучение может быть нестабильным.\n"
                "Продолжить?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        # ─── Создаём модель через ModelFactory ───
        try:
            # Передаём config как словарь для совместимости с ModelFactory
            factory_config = config.to_dict() if hasattr(config, 'to_dict') else {
                "type": arch,
                "data_type": data_type_str,
                "input_dim": config.input_dim,
                "output_dim": config.output_dim,
                "vocab_size": config.vocab_size,
                "arch_params": config.arch_params,
                **config.arch_params,  # для обратной совместимости со старым форматом
            }
            model = ModelFactory.create_model(factory_config)

            # Сохраняем в shared_state
            self.shared_state["model"] = model
            factory_config = self._preserve_hyperparams(factory_config)
            self.shared_state["config"] = factory_config

            # Подсчёт параметров и память
            params = ModelFactory.get_param_count(model)
            mem = ModelFactory.estimate_memory(model)

            # Отчёт
            report = (
                f"✅ УСПЕХ: Модель '{config.model_name}' создана!\n"
                f"{'=' * 50}\n"
                f"Архитектура: {arch.upper()}\n"
                f"Тип данных: {data_type_str}\n"
                f"Параметров: {params:,}\n"
                f"Память: {mem:.2f} MB\n"
            )
            if data_type_str == "text" and vocab:
                report += f"Словарь: {len(vocab)} токенов\n"
            if config.image_shape:
                report += f"Изображение: {config.image_shape}\n"
            report += f"{'=' * 50}\n"

            if hasattr(self, 'report_text'):
                self.report_text.setText(report)
            logger.info(f"Модель '{config.model_name}' ({arch}) создана: {params:,} параметров, {mem:.2f} MB")

            # Отправляем сигнал с ОБЪЕДИНЁННЫМ конфигом (гиперпараметры сохранены)
            self.model_created.emit(model, factory_config)

        except Exception as e:
            logger.error(f"Ошибка создания модели: {e}")
            if hasattr(self, 'report_text'):
                self.report_text.setText(f"❌ ОШИБКА: {e}")
            QMessageBox.critical(self, "Критическая ошибка", str(e))

    # ────────────────────────────────────────────────────────────────
    # REFRESH — вызывается из main_window при обновлении датасета
    # ────────────────────────────────────────────────────────────────
    def refresh(self):
        """
        Вызывается при изменении датасета или переключении на вкладку.
        Обновляет: словарь, совместимость, подсказки, оценку сложности.
        """
        dataset = dataset_to_legacy_dict(self.shared_state.get("dataset"))
        if not dataset:
            if hasattr(self, 'complexity_label'):
                self.complexity_label.setText("⚪ Датасет не загружен")
                self.complexity_label.setStyleSheet("color: #888; font-size: 14px; font-weight: bold;")
            if hasattr(self, 'params_label'):
                self.params_label.setText("Параметров: —")
            if hasattr(self, 'report_text'):
                self.report_text.setText("Датасет не загружен. Перейдите во вкладку 'Данные'.")
            if hasattr(self, 'create_btn'):
                self.create_btn.setEnabled(False)
            return

        if hasattr(self, 'create_btn'):
            self.create_btn.setEnabled(True)

        meta = dataset.get("meta", {})
        data_type = meta.get("type", "numeric")
        vocab = dataset.get("vocab", {})

        # Обновляем видимость эмбеддинга для RNN
        if data_type == "text" and hasattr(self, 'rnn_embed'):
            self.rnn_embed.setVisible(True)

        # Проверяем совместимость
        self._check_data_compatibility()

        # Обновляем визуализацию
        self._update_arch_viz()

        # Обновляем оценку
        self._estimate_complexity()

        logger.debug(f"ArchitecturePanel.refresh(): data_type={data_type}, vocab={len(vocab)}")