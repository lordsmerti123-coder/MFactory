"""
core/generator.py
=================
Генератор синтетических датасетов для зоопарка архитектур.
Поддерживает 30+ задач: текст, изображения, звук, временные ряды, графы, таблицы.

Все публичные методы возвращают DatasetContainer из core/contracts.py.
Формат сохранения: .oai (JSON) с обязательным полем format_version.

Импортирует типы только из core/contracts.py.
"""

import random
import json
import math
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Callable

from core.contracts import (
    DatasetContainer,
    DatasetMeta,
    DataType,
    TaskType,
    FORMAT_VERSION,
)
from core.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
#  РЕЕСТР ЗАДАЧ (единая точка регистрации)
# ============================================================================

TASK_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ======================== МАТЕМАТИКА ========================
    "addition": {
        "method": "_gen_addition",
        "data_type": DataType.TEXT,
        "task_type": TaskType.SEQ2SEQ,
        "category": "math",
        "display_name": "➕ Сложение",
        "description": "Сложение чисел в заданном диапазоне.",
    },
    "subtraction": {
        "method": "_gen_subtraction",
        "data_type": DataType.TEXT,
        "task_type": TaskType.SEQ2SEQ,
        "category": "math",
        "display_name": "➖ Вычитание",
        "description": "Вычитание чисел в заданном диапазоне.",
    },
    "multiplication": {
        "method": "_gen_multiplication",
        "data_type": DataType.TEXT,
        "task_type": TaskType.SEQ2SEQ,
        "category": "math",
        "display_name": "✖️ Умножение",
        "description": "Умножение чисел.",
    },
    "division": {
        "method": "_gen_division",
        "data_type": DataType.TEXT,
        "task_type": TaskType.SEQ2SEQ,
        "category": "math",
        "display_name": "➗ Деление нацело",
        "description": "Деление без остатка.",
    },
    "mixed_math": {
        "method": "_gen_mixed_math",
        "data_type": DataType.TEXT,
        "task_type": TaskType.SEQ2SEQ,
        "category": "math",
        "display_name": "🧮 Смешанные операции",
        "description": "Выражения с приоритетом операций.",
    },
    "linear_equation": {
        "method": "_gen_linear_equation",
        "data_type": DataType.TEXT,
        "task_type": TaskType.SEQ2SEQ,
        "category": "math",
        "display_name": "📐 Линейные уравнения",
        "description": "Решение линейных уравнений ax + b = c.",
    },
    "quadratic": {
        "method": "_gen_quadratic",
        "data_type": DataType.TEXT,
        "task_type": TaskType.SEQ2SEQ,
        "category": "math",
        "display_name": "📈 Квадратные уравнения",
        "description": "Нахождение корней квадратного уравнения.",
    },
    "modular_arithmetic": {
        "method": "_gen_modular_arithmetic",
        "data_type": DataType.TEXT,
        "task_type": TaskType.SEQ2SEQ,
        "category": "math",
        "display_name": "🔄 Модульная арифметика",
        "description": "Операции по модулю.",
    },
    "prime_check": {
        "method": "_gen_prime_check",
        "data_type": DataType.TEXT,
        "task_type": TaskType.SEQ2SEQ,
        "category": "math",
        "display_name": "🔍 Простое число?",
        "description": "Определение, является ли число простым.",
    },
    "fibonacci": {
        "method": "_gen_fibonacci",
        "data_type": DataType.TEXT,
        "task_type": TaskType.SEQ2SEQ,
        "category": "math",
        "display_name": "🐚 Число Фибоначчи",
        "description": "Вычисление n-го числа Фибоначчи.",
    },
    "number_sequence": {
        "method": "_gen_number_sequence",
        "data_type": DataType.TEXT,
        "task_type": TaskType.SEQ2SEQ,
        "category": "math",
        "display_name": "🔢 Продолжи последовательность",
        "description": "Продолжение арифметической или геометрической прогрессии.",
    },
    # ======================== КРИПТОГРАФИЯ / ТЕКСТ ========================
    "cipher_caesar": {
        "method": "_gen_cipher_caesar",
        "data_type": DataType.TEXT,
        "task_type": TaskType.SEQ2SEQ,
        "category": "crypto",
        "display_name": "🔐 Шифр Цезаря",
        "description": "Шифрование сдвигом по алфавиту.",
    },
    "cipher_atbash": {
        "method": "_gen_cipher_atbash",
        "data_type": DataType.TEXT,
        "task_type": TaskType.SEQ2SEQ,
        "category": "crypto",
        "display_name": "🔏 Шифр Атбаш",
        "description": "Отражение алфавита (А↔Я).",
    },
    "cipher_vigenere": {
        "method": "_gen_cipher_vigenere",
        "data_type": DataType.TEXT,
        "task_type": TaskType.SEQ2SEQ,
        "category": "crypto",
        "display_name": "🗝 Шифр Виженера",
        "description": "Полиалфавитный шифр с ключевым словом.",
    },
    "reverse_string": {
        "method": "_gen_reverse_string",
        "data_type": DataType.TEXT,
        "task_type": TaskType.SEQ2SEQ,
        "category": "crypto",
        "display_name": "🔁 Перевернуть строку",
        "description": "Инвертирование последовательности символов.",
    },
    "palindrome_check": {
        "method": "_gen_palindrome_check",
        "data_type": DataType.TEXT,
        "task_type": TaskType.SEQ2SEQ,
        "category": "crypto",
        "display_name": "🪞 Палиндром?",
        "description": "Определение, является ли слово палиндромом.",
    },
    "word_sort": {
        "method": "_gen_word_sort",
        "data_type": DataType.TEXT,
        "task_type": TaskType.SEQ2SEQ,
        "category": "crypto",
        "display_name": "🔤 Сортировка букв",
        "description": "Сортировка символов слова в алфавитном порядке.",
    },
    "translation_pairs": {
        "method": "_gen_translation_pairs",
        "data_type": DataType.TEXT,
        "task_type": TaskType.SEQ2SEQ,
        "category": "crypto",
        "display_name": "🌍 Пары переводов",
        "description": "Синтетические пары слов для перевода (заглушка под ИИ).",
    },
    # ======================== ИЗОБРАЖЕНИЯ ========================
    "shapes_2d": {
        "method": "_gen_shapes_2d",
        "data_type": DataType.IMAGE,
        "task_type": TaskType.IMAGE_CLASSIFICATION,
        "category": "image",
        "display_name": "🔷 2D фигуры",
        "description": "Классификация простых геометрических фигур.",
    },
    "shapes_3d": {
        "method": "_gen_shapes_3d",
        "data_type": DataType.IMAGE,
        "task_type": TaskType.IMAGE_CLASSIFICATION,
        "category": "image",
        "display_name": "🧊 3D проекции",
        "description": "Проекции трёхмерных объектов (куб, сфера, пирамида).",
    },
    "mnist_like": {
        "method": "_gen_mnist_like",
        "data_type": DataType.IMAGE,
        "task_type": TaskType.IMAGE_CLASSIFICATION,
        "category": "image",
        "display_name": "✍️ Рукописные цифры",
        "description": "Синтетические рукописные цифры 0–9.",
    },
    "noise_denoise": {
        "method": "_gen_noise_denoise",
        "data_type": DataType.IMAGE,
        "task_type": TaskType.ANOMALY_DETECTION,
        "category": "image",
        "display_name": "🔇 Очистка шума (изображения)",
        "description": "Пары: зашумлённое → чистое изображение.",
    },
    "color_classification": {
        "method": "_gen_color_classification",
        "data_type": DataType.IMAGE,
        "task_type": TaskType.IMAGE_CLASSIFICATION,
        "category": "image",
        "display_name": "🎨 Классификация цвета",
        "description": "Определение доминирующего цвета изображения.",
    },
    "pattern_generation": {
        "method": "_gen_pattern_generation",
        "data_type": DataType.IMAGE,
        "task_type": TaskType.IMAGE_GENERATION,
        "category": "image",
        "display_name": "🌀 Генерация паттернов",
        "description": "Синтетические паттерны для GAN/Diffusion.",
    },
    # ======================== ЗВУК / АУДИО ========================
    "tone_classification": {
        "method": "_gen_tone_classification",
        "data_type": DataType.AUDIO,
        "task_type": TaskType.AUDIO_CLASSIFICATION,
        "category": "audio",
        "display_name": "🎵 Угадай тон",
        "description": "Классификация звуковых тонов по частоте.",
    },
    "chord_recognition": {
        "method": "_gen_chord_recognition",
        "data_type": DataType.AUDIO,
        "task_type": TaskType.AUDIO_CLASSIFICATION,
        "category": "audio",
        "display_name": "🎶 Распознавание аккордов",
        "description": "Определение типа аккорда (мажор/минор).",
    },
    "noise_denoise_audio": {
        "method": "_gen_noise_denoise_audio",
        "data_type": DataType.AUDIO,
        "task_type": TaskType.ANOMALY_DETECTION,
        "category": "audio",
        "display_name": "🔇 Очистка шума (звук)",
        "description": "Пары: зашумлённый → чистый звук.",
    },
    "frequency_regression": {
        "method": "_gen_frequency_regression",
        "data_type": DataType.AUDIO,
        "task_type": TaskType.REGRESSION,
        "category": "audio",
        "display_name": "📻 Предсказание частоты",
        "description": "Регрессия: звук → его частота в Гц.",
    },
    "audio_generation": {
        "method": "_gen_audio_generation",
        "data_type": DataType.AUDIO,
        "task_type": TaskType.AUDIO_GENERATION,
        "category": "audio",
        "display_name": "🎼 Генерация мелодий",
        "description": "Простые синтетические мелодии для генеративных моделей.",
    },
    # ======================== ВРЕМЕННЫЕ РЯДЫ ========================
    "sine_wave": {
        "method": "_gen_sine_wave",
        "data_type": DataType.TIME_SERIES,
        "task_type": TaskType.REGRESSION,
        "category": "time_series",
        "display_name": "📈 Синусоида",
        "description": "Предсказание следующего значения синусоиды.",
    },
    "trend_prediction": {
        "method": "_gen_trend_prediction",
        "data_type": DataType.TIME_SERIES,
        "task_type": TaskType.REGRESSION,
        "category": "time_series",
        "display_name": "📊 Предсказание тренда",
        "description": "Линейный или квадратичный тренд с шумом.",
    },
    "anomaly_detection": {
        "method": "_gen_anomaly_detection",
        "data_type": DataType.TIME_SERIES,
        "task_type": TaskType.ANOMALY_DETECTION,
        "category": "time_series",
        "display_name": "🚨 Аномалии в ряду",
        "description": "Поиск аномальных точек во временном ряду.",
    },
    "seasonal_pattern": {
        "method": "_gen_seasonal_pattern",
        "data_type": DataType.TIME_SERIES,
        "task_type": TaskType.REGRESSION,
        "category": "time_series",
        "display_name": "🗓 Сезонные паттерны",
        "description": "Ряды с сезонной компонентой.",
    },
    # ======================== ГРАФЫ ========================
    "graph_connectivity": {
        "method": "_gen_graph_connectivity",
        "data_type": DataType.GRAPH,
        "task_type": TaskType.CLASSIFICATION,
        "category": "graph",
        "display_name": "🕸 Связность графа",
        "description": "Определение, является ли граф связным.",
    },
    "shortest_path": {
        "method": "_gen_shortest_path",
        "data_type": DataType.GRAPH,
        "task_type": TaskType.REGRESSION,
        "category": "graph",
        "display_name": "🗺 Кратчайший путь",
        "description": "Длина кратчайшего пути между двумя узлами.",
    },
    "node_classification": {
        "method": "_gen_node_classification",
        "data_type": DataType.GRAPH,
        "task_type": TaskType.CLASSIFICATION,
        "category": "graph",
        "display_name": "🏷 Классификация узлов",
        "description": "Определение класса центрального узла графа.",
    },
    # ======================== ТАБЛИЦЫ / ЧИСЛА ========================
    "xor_problem": {
        "method": "_gen_xor_problem",
        "data_type": DataType.NUMERIC,
        "task_type": TaskType.CLASSIFICATION,
        "category": "tabular",
        "display_name": "⊕ XOR",
        "description": "Классическая задача XOR для нейросетей.",
    },
    "function_approximation": {
        "method": "_gen_function_approximation",
        "data_type": DataType.NUMERIC,
        "task_type": TaskType.REGRESSION,
        "category": "tabular",
        "display_name": "📐 Аппроксимация функции",
        "description": "Аппроксимация математической функции.",
    },
    "clustering_blobs": {
        "method": "_gen_clustering_blobs",
        "data_type": DataType.NUMERIC,
        "task_type": TaskType.CLUSTERING,
        "category": "tabular",
        "display_name": "🫧 Кластеризация",
        "description": "Гауссовы кластеры для задач кластеризации.",
    },
    "regression_linear": {
        "method": "_gen_regression_linear",
        "data_type": DataType.NUMERIC,
        "task_type": TaskType.REGRESSION,
        "category": "tabular",
        "display_name": "📏 Линейная регрессия",
        "description": "Линейная зависимость с шумом.",
    },
}


# ============================================================================
#  ОСНОВНОЙ КЛАСС ГЕНЕРАТОРА
# ============================================================================

class DatasetGenerator:
    """
    Генератор синтетических датасетов для зоопарка архитектур.

    Использование:
        gen = DatasetGenerator(seed=42)
        dataset = gen.generate_task("addition", {"num_range": (1, 99)}, 5000)
        gen.save_dataset(dataset, Path("dataset_addition.oai"))

    Все методы возвращают DatasetContainer из core/contracts.py.
    """

    # Алфавит для текстовых задач
    RUSSIAN_ALPHABET = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    ENGLISH_ALPHABET = "abcdefghijklmnopqrstuvwxyz"

    # Словари для генерации слов
    RUSSIAN_WORDS = [
        "нейросеть", "алгоритм", "программист", "матрица", "интеллект",
        "школа", "градиент", "тензор", "эпоха", "функция",
        "данные", "модель", "вектор", "слой", "ядро",
        "код", "бит", "байт", "процесс", "память",
        "экран", "кнопка", "файл", "папка", "сеть",
    ]
    ENGLISH_WORDS = [
        "neural", "network", "gradient", "tensor", "epoch",
        "model", "data", "layer", "kernel", "code",
        "byte", "bit", "file", "path", "node",
        "tree", "graph", "loss", "train", "test",
    ]

    def __init__(self, seed: Optional[int] = None):
        """
        Инициализация генератора.

        Аргументы:
            seed: зерно случайности. Если None — случайное.
        """
        self.seed = seed if seed is not None else random.randint(0, 2**31)
        random.seed(self.seed)
        np.random.seed(self.seed)
        logger.info(f"DatasetGenerator инициализирован (seed={self.seed})")

    # ================================================================
    #  ЕДИНАЯ ТОЧКА ВХОДА
    # ================================================================

    def generate_task(
        self,
        task_type: str,
        params: Optional[Dict[str, Any]] = None,
        num_samples: int = 1000,
    ) -> DatasetContainer:
        """
        Генерирует датасет для указанной задачи.

        Аргументы:
            task_type: идентификатор задачи из TASK_REGISTRY
            params: словарь параметров генерации (зависит от задачи)
            num_samples: количество примеров

        Возвращает:
            DatasetContainer с заполненными meta, raw_inputs, raw_outputs

        Исключения:
            ValueError: если task_type не найден в реестре
        """
        if task_type not in TASK_REGISTRY:
            available = ", ".join(sorted(TASK_REGISTRY.keys()))
            raise ValueError(
                f"Неизвестный тип задачи: '{task_type}'. "
                f"Доступные: {available}"
            )

        if params is None:
            params = {}

        if num_samples < 1:
            raise ValueError("Количество примеров должно быть >= 1")

        registry_entry = TASK_REGISTRY[task_type]
        method_name = registry_entry["method"]
        gen_method = getattr(self, method_name)

        logger.info(
            f"Генерация датасета: {task_type} "
            f"({num_samples} примеров, params={params})"
        )

        container = gen_method(params, num_samples)

        # Заполняем обязательные поля мета
        container.meta.generator_task = task_type
        container.meta.num_samples = num_samples
        container.meta.format_version = FORMAT_VERSION

        logger.info(
            f"Датасет сгенерирован: {num_samples} примеров, "
            f"тип={container.meta.data_type.value}"
        )
        return container

    def create_and_save_dataset(
        self,
        task_type: str,
        num_samples: int,
        save_path: Path,
        params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Генерирует датасет и сохраняет в файл .oai.
        Совместимо со старым интерфейсом.

        Аргументы:
            task_type: идентификатор задачи
            num_samples: количество примеров
            save_path: путь для сохранения
            params: параметры генерации

        Возвращает:
            Строка с объяснением задачи (для отображения в GUI)
        """
        container = self.generate_task(task_type, params, num_samples)
        self.save_dataset(container, save_path)

        explanation = self.get_task_explanation(task_type, container)
        return explanation

    def save_dataset(self, container: DatasetContainer, path: Path) -> None:
        """
        Сохраняет DatasetContainer в файл .oai (JSON).

        Формат обратно совместим со старым data_loader.py:
        содержит ключи meta, raw_inputs, raw_outputs.
        Добавлено поле format_version.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = container.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Датасет сохранён: {path} ({path.stat().st_size / 1024:.1f} КБ)")

    def get_task_explanation(
        self, task_type: str, container: Optional[DatasetContainer] = None
    ) -> str:
        """
        Возвращает HTML-объяснение задачи для отображения в GUI.

        Аргументы:
            task_type: идентификатор задачи
            container: сгенерированный датасет (для примеров)

        Возвращает:
            Строка с HTML-разметкой объяснения
        """
        entry = TASK_REGISTRY.get(task_type)
        if not entry:
            return f"<p>Неизвестная задача: {task_type}</p>"

        title = entry["display_name"]
        desc = entry["description"]
        data_type = entry["data_type"].value
        task_t = entry["task_type"].value

        html = f"<h3>🔹 Задача: {title}</h3>"
        html += f"<p>{desc}</p>"
        html += f"<p><b>Тип данных:</b> {data_type} | <b>Тип задачи:</b> {task_t}</p>"

        if container and container.raw_inputs:
            sample_in = str(container.raw_inputs[0])
            sample_out = str(container.raw_outputs[0])
            # Обрезаем длинные примеры (изображения, звук)
            if len(sample_in) > 200:
                sample_in = sample_in[:200] + "..."
            if len(sample_out) > 200:
                sample_out = sample_out[:200] + "..."
            html += f"<p><b>Пример входа:</b> <code>{sample_in}</code><br>"
            html += f"<b>Пример выхода:</b> <code>{sample_out}</code></p>"

        # Специфичные объяснения по типу данных
        if data_type == "text":
            html += (
                "<p><b>🧠 Чему учится нейросеть:</b><br>"
                "Нейросеть видит только символы (текст). "
                "Ей предстоит самой открыть закономерности в данных.</p>"
            )
        elif data_type == "image":
            html += (
                "<p><b>🧠 Чему учится нейросеть:</b><br>"
                "Нейросеть получает изображение как матрицу чисел (пикселей). "
                "Она учится находить визуальные закономерности.</p>"
            )
        elif data_type == "audio":
            html += (
                "<p><b>🧠 Чему учится нейросеть:</b><br>"
                "Нейросеть получает звуковую волну как последовательность чисел. "
                "Она учится распознавать паттерны в звуке.</p>"
            )
        elif data_type == "time_series":
            html += (
                "<p><b>🧠 Чему учится нейросеть:</b><br>"
                "Нейросеть видит последовательность значений во времени "
                "и учится предсказывать следующие точки.</p>"
            )
        elif data_type == "graph":
            html += (
                "<p><b>🧠 Чему учится нейросеть:</b><br>"
                "Нейросеть получает граф (узлы и связи) "
                "и учится находить структурные закономерности.</p>"
            )
        elif data_type == "numeric":
            html += (
                "<p><b>🧠 Чему учится нейросеть:</b><br>"
                "Нейросеть получает числовой вектор "
                "и учится находить математические зависимости.</p>"
            )

        return html

    def get_available_tasks(self, category: Optional[str] = None) -> List[Dict]:
        """
        Возвращает список доступных задач.

        Аргументы:
            category: фильтр по категории (math, crypto, image, audio,
                      time_series, graph, tabular). Если None — все.

        Возвращает:
            Список словарей с информацией о задачах
        """
        result = []
        for task_id, entry in TASK_REGISTRY.items():
            if category and entry["category"] != category:
                continue
            result.append({
                "id": task_id,
                "display_name": entry["display_name"],
                "description": entry["description"],
                "category": entry["category"],
                "data_type": entry["data_type"].value,
                "task_type": entry["task_type"].value,
            })
        return result

    @staticmethod
    def get_default_params(task_type: str) -> Dict[str, Any]:
        """
        Возвращает параметры по умолчанию для задачи.

        Аргументы:
            task_type: идентификатор задачи

        Возвращает:
            Словарь параметров по умолчанию
        """
        defaults: Dict[str, Dict[str, Any]] = {
            # Математика
            "addition": {"num_range": (1, 99), "num_operands": 2, "allow_negative": False},
            "subtraction": {"num_range": (1, 99), "allow_negative": False},
            "multiplication": {"num_range": (1, 20)},
            "division": {"num_range": (1, 20)},
            "mixed_math": {"num_range": (1, 20), "operations": ["+", "-", "*"]},
            "linear_equation": {"coeff_range": (1, 15), "x_range": (-50, 50)},
            "quadratic": {"root_range": (-15, 15)},
            "modular_arithmetic": {"modulus": 10, "operations": ["+"]},
            "prime_check": {"num_range": (2, 1000)},
            "fibonacci": {"index_range": (1, 30)},
            "number_sequence": {"seq_type": "arithmetic", "length": 5},
            # Криптография
            "cipher_caesar": {"shift": 3, "word_length": (3, 10), "alphabet": "russian"},
            "cipher_atbash": {"word_length": (3, 10), "alphabet": "russian"},
            "cipher_vigenere": {"key_length": 3, "word_length": (3, 10)},
            "reverse_string": {"word_length": (3, 12)},
            "palindrome_check": {"word_length": (3, 8)},
            "word_sort": {"word_length": (3, 10)},
            "translation_pairs": {"num_pairs": 50},
            # Изображения
            "shapes_2d": {
                "image_size": 28, "color_mode": "grayscale",
                "shapes": ["circle", "square", "triangle"],
                "noise_level": 0.0, "rotation": False,
            },
            "shapes_3d": {
                "image_size": 64, "color_mode": "grayscale",
                "shapes": ["cube", "sphere", "pyramid"],
                "noise_level": 0.0,
            },
            "mnist_like": {"image_size": 28, "noise_level": 0.05},
            "noise_denoise": {"image_size": 28, "noise_level": 0.3},
            "color_classification": {"image_size": 16, "num_colors": 5},
            "pattern_generation": {"image_size": 32, "pattern_type": "stripes"},
            # Звук
            "tone_classification": {
                "sample_rate": 16000, "duration": 0.5,
                "frequencies": [262, 294, 330, 349, 392, 440, 494],
                "noise_level": 0.0,
            },
            "chord_recognition": {
                "sample_rate": 16000, "duration": 0.5,
                "chord_types": ["major", "minor"],
            },
            "noise_denoise_audio": {
                "sample_rate": 16000, "duration": 0.5, "noise_level": 0.3,
            },
            "frequency_regression": {
                "sample_rate": 16000, "duration": 0.5,
                "freq_range": (100, 1000),
            },
            "audio_generation": {
                "sample_rate": 16000, "duration": 1.0,
                "num_notes": 8,
            },
            # Временные ряды
            "sine_wave": {"sequence_length": 50, "noise": 0.05},
            "trend_prediction": {"sequence_length": 30, "trend_type": "linear", "noise": 0.1},
            "anomaly_detection": {"sequence_length": 100, "anomaly_ratio": 0.05},
            "seasonal_pattern": {"sequence_length": 60, "period": 12, "noise": 0.1},
            # Графы
            "graph_connectivity": {"num_nodes": 10, "edge_probability": 0.3},
            "shortest_path": {"num_nodes": 8, "edge_probability": 0.4},
            "node_classification": {"num_nodes": 10, "num_classes": 3},
            # Таблицы
            "xor_problem": {"num_features": 2},
            "function_approximation": {"function": "sin", "x_range": (-3.14, 3.14)},
            "clustering_blobs": {"num_clusters": 3, "num_features": 2},
            "regression_linear": {"num_features": 3, "noise": 0.1},
        }
        return defaults.get(task_type, {})

    # ================================================================
    #  МАТЕМАТИЧЕСКИЕ ГЕНЕРАТОРЫ
    # ================================================================

    def _gen_addition(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация примеров на сложение."""
        lo, hi = params.get("num_range", (1, 99))
        num_operands = params.get("num_operands", 2)
        allow_negative = params.get("allow_negative", False)

        inputs, outputs = [], []
        for _ in range(n):
            nums = [random.randint(lo, hi) for _ in range(num_operands)]
            result = sum(nums)
            if not allow_negative:
                # Сдвигаем числа чтобы результат был положительным
                if result < 0:
                    nums = [abs(x) for x in nums]
                    result = sum(nums)
            expr = "+".join(str(x) for x in nums)
            inputs.append(expr)
            outputs.append(str(result))

        return self._build_text_container(inputs, outputs, TaskType.SEQ2SEQ)

    def _gen_subtraction(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация примеров на вычитание."""
        lo, hi = params.get("num_range", (1, 99))
        allow_negative = params.get("allow_negative", False)

        inputs, outputs = [], []
        for _ in range(n):
            a = random.randint(lo, hi)
            b = random.randint(lo, hi)
            if not allow_negative and b > a:
                a, b = b, a
            inputs.append(f"{a}-{b}")
            outputs.append(str(a - b))

        return self._build_text_container(inputs, outputs, TaskType.SEQ2SEQ)

    def _gen_multiplication(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация примеров на умножение."""
        lo, hi = params.get("num_range", (1, 20))

        inputs, outputs = [], []
        for _ in range(n):
            a, b = random.randint(lo, hi), random.randint(lo, hi)
            inputs.append(f"{a}*{b}")
            outputs.append(str(a * b))

        return self._build_text_container(inputs, outputs, TaskType.SEQ2SEQ)

    def _gen_division(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация примеров на деление нацело."""
        lo, hi = params.get("num_range", (1, 20))

        inputs, outputs = [], []
        for _ in range(n):
            b = random.randint(lo, hi)  # ответ
            a = random.randint(lo, hi)  # делитель
            c = a * b                    # делимое
            inputs.append(f"{c}/{a}")
            outputs.append(str(b))

        return self._build_text_container(inputs, outputs, TaskType.SEQ2SEQ)

    def _gen_mixed_math(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация смешанных выражений с приоритетом операций."""
        lo, hi = params.get("num_range", (1, 20))
        operations = params.get("operations", ["+", "-", "*"])

        inputs, outputs = [], []
        for _ in range(n):
            a = random.randint(lo, hi)
            b = random.randint(lo, hi)
            c = random.randint(lo, hi)
            op1 = random.choice(operations)
            op2 = random.choice(operations)
            expr = f"{a}{op1}{b}{op2}{c}"
            try:
                ans = eval(expr)
            except Exception:
                ans = a + b + c  # fallback
            inputs.append(expr)
            outputs.append(str(ans))

        return self._build_text_container(inputs, outputs, TaskType.SEQ2SEQ)

    def _gen_linear_equation(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация линейных уравнений ax + b = c."""
        coeff_lo, coeff_hi = params.get("coeff_range", (1, 15))
        x_lo, x_hi = params.get("x_range", (-50, 50))

        inputs, outputs = [], []
        for _ in range(n):
            x = random.randint(x_lo, x_hi)
            a = random.randint(coeff_lo, coeff_hi) * random.choice([-1, 1])
            b = random.randint(-50, 50)
            c = a * x + b
            b_str = f"+{b}" if b >= 0 else str(b)
            inputs.append(f"{a}x{b_str}={c}")
            outputs.append(str(x))

        return self._build_text_container(inputs, outputs, TaskType.SEQ2SEQ)

    def _gen_quadratic(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация квадратных уравнений."""
        root_lo, root_hi = params.get("root_range", (-15, 15))

        inputs, outputs = [], []
        for _ in range(n):
            x1 = random.randint(root_lo, root_hi)
            x2 = random.randint(root_lo, root_hi)
            a = random.choice([-3, -2, -1, 1, 2, 3])
            b = -a * (x1 + x2)
            c = a * (x1 * x2)
            b_str = f"+{b}" if b >= 0 else str(b)
            c_str = f"+{c}" if c >= 0 else str(c)
            inputs.append(f"{a}x^2{b_str}x{c_str}=0")
            roots = sorted([x1, x2])
            outputs.append(f"{roots[0]},{roots[1]}")

        return self._build_text_container(inputs, outputs, TaskType.SEQ2SEQ)

    def _gen_modular_arithmetic(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация модульной арифметики."""
        modulus = params.get("modulus", 10)
        operations = params.get("operations", ["+"])

        inputs, outputs = [], []
        for _ in range(n):
            a = random.randint(0, modulus * 3)
            b = random.randint(0, modulus * 3)
            op = random.choice(operations)
            if op == "+":
                result = (a + b) % modulus
            elif op == "-":
                result = (a - b) % modulus
            elif op == "*":
                result = (a * b) % modulus
            else:
                result = (a + b) % modulus
            inputs.append(f"({a}{op}{b})%{modulus}")
            outputs.append(str(result))

        return self._build_text_container(inputs, outputs, TaskType.SEQ2SEQ)

    def _gen_prime_check(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация задачи определения простого числа."""
        lo, hi = params.get("num_range", (2, 1000))

        def is_prime(num):
            if num < 2:
                return False
            for i in range(2, int(num**0.5) + 1):
                if num % i == 0:
                    return False
            return True

        inputs, outputs = [], []
        for _ in range(n):
            num = random.randint(lo, hi)
            inputs.append(str(num))
            outputs.append("1" if is_prime(num) else "0")

        return self._build_text_container(inputs, outputs, TaskType.SEQ2SEQ)

    def _gen_fibonacci(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация задачи вычисления числа Фибоначчи."""
        idx_lo, idx_hi = params.get("index_range", (1, 30))

        # Предвычисляем числа Фибоначчи
        fib = [0, 1]
        for i in range(2, idx_hi + 1):
            fib.append(fib[-1] + fib[-2])

        inputs, outputs = [], []
        for _ in range(n):
            idx = random.randint(idx_lo, idx_hi)
            inputs.append(f"fib({idx})")
            outputs.append(str(fib[idx]))

        return self._build_text_container(inputs, outputs, TaskType.SEQ2SEQ)

    def _gen_number_sequence(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация задачи продолжения последовательности."""
        seq_type = params.get("seq_type", "arithmetic")
        length = params.get("length", 5)

        inputs, outputs = [], []
        for _ in range(n):
            if seq_type == "arithmetic":
                start = random.randint(1, 50)
                step = random.randint(1, 10)
                seq = [start + i * step for i in range(length)]
                next_val = start + length * step
            elif seq_type == "geometric":
                start = random.randint(1, 5)
                ratio = random.randint(2, 4)
                seq = [start * (ratio ** i) for i in range(length)]
                next_val = start * (ratio ** length)
            else:
                start = random.randint(1, 50)
                step = random.randint(1, 10)
                seq = [start + i * step for i in range(length)]
                next_val = start + length * step

            inputs.append(",".join(str(x) for x in seq))
            outputs.append(str(next_val))

        return self._build_text_container(inputs, outputs, TaskType.SEQ2SEQ)

    # ================================================================
    #  КРИПТОГРАФИЧЕСКИЕ / ТЕКСТОВЫЕ ГЕНЕРАТОРЫ
    # ================================================================

    def _get_alphabet(self, params: Dict) -> str:
        """Возвращает алфавит по параметрам."""
        alphabet_type = params.get("alphabet", "russian")
        if alphabet_type == "russian":
            return self.RUSSIAN_ALPHABET
        elif alphabet_type == "english":
            return self.ENGLISH_ALPHABET
        return self.RUSSIAN_ALPHABET

    def _get_random_word(self, params: Dict) -> str:
        """Возвращает случайное слово."""
        alphabet = self._get_alphabet(params)
        if alphabet == self.RUSSIAN_ALPHABET:
            words = self.RUSSIAN_WORDS
        else:
            words = self.ENGLISH_WORDS
        return random.choice(words)

    def _gen_cipher_caesar(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация шифра Цезаря."""
        shift = params.get("shift", 3)
        word_len_lo, word_len_hi = params.get("word_length", (3, 10))
        alphabet = self._get_alphabet(params)

        inputs, outputs = [], []
        for _ in range(n):
            word = self._get_random_word(params) + str(random.randint(10, 99))
            shifted = "".join(
                alphabet[(alphabet.index(c) + shift) % len(alphabet)]
                if c in alphabet else c
                for c in word
            )
            inputs.append(shifted)
            outputs.append(word)

        return self._build_text_container(inputs, outputs, TaskType.SEQ2SEQ)

    def _gen_cipher_atbash(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация шифра Атбаш."""
        alphabet = self._get_alphabet(params)
        reversed_alphabet = alphabet[::-1]

        inputs, outputs = [], []
        for _ in range(n):
            word = self._get_random_word(params) + str(random.randint(100, 999))
            shifted = "".join(
                reversed_alphabet[alphabet.index(c)]
                if c in alphabet else c
                for c in word
            )
            inputs.append(shifted)
            outputs.append(word)

        return self._build_text_container(inputs, outputs, TaskType.SEQ2SEQ)

    def _gen_cipher_vigenere(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация шифра Виженера."""
        key_length = params.get("key_length", 3)
        alphabet = self._get_alphabet(params)

        inputs, outputs = [], []
        for _ in range(n):
            word = self._get_random_word(params)
            key = "".join(random.choice(alphabet) for _ in range(key_length))
            # Шифруем
            encrypted = ""
            for i, c in enumerate(word):
                if c in alphabet:
                    shift = alphabet.index(key[i % len(key)])
                    encrypted += alphabet[(alphabet.index(c) + shift) % len(alphabet)]
                else:
                    encrypted += c
            # Формат: зашифрованное слово + ключ → оригинал
            inputs.append(f"{encrypted}|{key}")
            outputs.append(word)

        return self._build_text_container(inputs, outputs, TaskType.SEQ2SEQ)

    def _gen_reverse_string(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация задачи переворота строки."""
        word_len_lo, word_len_hi = params.get("word_length", (3, 12))

        inputs, outputs = [], []
        for _ in range(n):
            word = self._get_random_word(params)
            # Добавляем цифры для разнообразия
            word += str(random.randint(10, 99))
            inputs.append(word[::-1])
            outputs.append(word)

        return self._build_text_container(inputs, outputs, TaskType.SEQ2SEQ)

    def _gen_palindrome_check(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация задачи проверки палиндрома."""
        word_len_lo, word_len_hi = params.get("word_length", (3, 8))

        inputs, outputs = [], []
        for _ in range(n):
            if random.random() < 0.5:
                # Генерируем палиндром
                half = "".join(
                    random.choice(self.RUSSIAN_ALPHABET)
                    for _ in range(random.randint(2, 4))
                )
                word = half + half[::-1]
            else:
                # Генерируем не-палиндром
                word = self._get_random_word(params)
                while word == word[::-1]:
                    word = self._get_random_word(params)
            inputs.append(word)
            outputs.append("1" if word == word[::-1] else "0")

        return self._build_text_container(inputs, outputs, TaskType.SEQ2SEQ)

    def _gen_word_sort(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация задачи сортировки букв в слове."""
        inputs, outputs = [], []
        for _ in range(n):
            word = self._get_random_word(params)
            sorted_word = "".join(sorted(word))
            inputs.append(word)
            outputs.append(sorted_word)

        return self._build_text_container(inputs, outputs, TaskType.SEQ2SEQ)

    def _gen_translation_pairs(self, params: Dict, n: int) -> DatasetContainer:
        """
        Генерация синтетических пар перевода.
        Заглушка: в будущем будет генерироваться через ИИ-бэкенд.
        Сейчас создаём простые пары из фиксированного словаря.
        """
        # Синтетический мини-словарь
        pairs = [
            ("кошка", "cat"), ("собака", "dog"), ("дом", "house"),
            ("книга", "book"), ("вода", "water"), ("огонь", "fire"),
            ("земля", "earth"), ("воздух", "air"), ("солнце", "sun"),
            ("луна", "moon"), ("звезда", "star"), ("дерево", "tree"),
            ("цветок", "flower"), ("рыба", "fish"), ("птица", "bird"),
            ("город", "city"), ("школа", "school"), ("друг", "friend"),
            ("время", "time"), ("свет", "light"), ("тень", "shadow"),
            ("снег", "snow"), ("дождь", "rain"), ("ветер", "wind"),
            ("гора", "mountain"), ("река", "river"), ("море", "sea"),
            ("небо", "sky"), ("облако", "cloud"), ("камень", "stone"),
        ]

        inputs, outputs = [], []
        for _ in range(n):
            pair = random.choice(pairs)
            if random.random() < 0.5:
                inputs.append(pair[0])
                outputs.append(pair[1])
            else:
                inputs.append(pair[1])
                outputs.append(pair[0])

        return self._build_text_container(inputs, outputs, TaskType.SEQ2SEQ)

    # ================================================================
    #  ГЕНЕРАТОРЫ ИЗОБРАЖЕНИЙ
    # ================================================================

    def _gen_shapes_2d(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация 2D фигур для классификации."""
        size = params.get("image_size", 28)
        color_mode = params.get("color_mode", "grayscale")
        shapes = params.get("shapes", ["circle", "square", "triangle"])
        noise_level = params.get("noise_level", 0.0)
        rotation = params.get("rotation", False)

        channels = 1 if color_mode == "grayscale" else 3
        inputs, outputs = [], []

        for _ in range(n):
            shape = random.choice(shapes)
            img = self._draw_shape_2d(shape, size, channels)

            # Добавляем шум
            if noise_level > 0:
                noise = np.random.normal(0, noise_level, img.shape)
                img = np.clip(img + noise, 0, 1)

            # Поворот (простой поворот на 90/180/270)
            if rotation and random.random() < 0.5:
                k = random.choice([1, 2, 3])
                img = np.rot90(img, k, axes=(0, 1))

            inputs.append(img.astype(np.float32).tolist())
            outputs.append(str(shapes.index(shape)))

        meta = self._build_image_meta(n, size, channels, len(shapes))
        container = DatasetContainer(meta=meta, raw_inputs=inputs, raw_outputs=outputs)
        return container

    def _gen_shapes_3d(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация 3D проекций для классификации."""
        size = params.get("image_size", 64)
        color_mode = params.get("color_mode", "grayscale")
        shapes = params.get("shapes", ["cube", "sphere", "pyramid"])
        noise_level = params.get("noise_level", 0.0)

        channels = 1 if color_mode == "grayscale" else 3
        inputs, outputs = [], []

        for _ in range(n):
            shape = random.choice(shapes)
            img = self._draw_shape_3d(shape, size, channels)

            if noise_level > 0:
                noise = np.random.normal(0, noise_level, img.shape)
                img = np.clip(img + noise, 0, 1)

            inputs.append(img.astype(np.float32).tolist())
            outputs.append(str(shapes.index(shape)))

        meta = self._build_image_meta(n, size, channels, len(shapes))
        container = DatasetContainer(meta=meta, raw_inputs=inputs, raw_outputs=outputs)
        return container

    def _gen_mnist_like(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация синтетических рукописных цифр."""
        size = params.get("image_size", 28)
        noise_level = params.get("noise_level", 0.05)
        num_classes = 10

        inputs, outputs = [], []
        for _ in range(n):
            digit = random.randint(0, 9)
            img = self._draw_digit(digit, size)

            if noise_level > 0:
                noise = np.random.normal(0, noise_level, img.shape)
                img = np.clip(img + noise, 0, 1)

            # Небольшой сдвиг для реалистичности
            if random.random() < 0.5:
                dx, dy = random.randint(-2, 2), random.randint(-2, 2)
                img = np.roll(img, shift=(dy, dx), axis=(0, 1))

            inputs.append(img.astype(np.float32).tolist())
            outputs.append(str(digit))

        meta = self._build_image_meta(n, size, 1, num_classes)
        container = DatasetContainer(meta=meta, raw_inputs=inputs, raw_outputs=outputs)
        return container

    def _gen_noise_denoise(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация пар зашумлённое → чистое изображение."""
        size = params.get("image_size", 28)
        noise_level = params.get("noise_level", 0.3)

        inputs, outputs = [], []
        for _ in range(n):
            # Генерируем чистое изображение (простая фигура)
            shape = random.choice(["circle", "square", "triangle"])
            clean = self._draw_shape_2d(shape, size, 1)

            # Добавляем шум
            noise = np.random.normal(0, noise_level, clean.shape)
            noisy = np.clip(clean + noise, 0, 1)

            inputs.append(noisy.astype(np.float32).tolist())
            outputs.append(clean.astype(np.float32).tolist())

        meta = self._build_image_meta(n, size, 1, 0)
        meta.task_type = TaskType.ANOMALY_DETECTION
        container = DatasetContainer(meta=meta, raw_inputs=inputs, raw_outputs=outputs)
        return container

    def _gen_color_classification(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация изображений для классификации цвета."""
        size = params.get("image_size", 16)
        num_colors = params.get("num_colors", 5)

        # Определяем цвета
        base_colors = [
            [1.0, 0.0, 0.0],  # красный
            [0.0, 1.0, 0.0],  # зелёный
            [0.0, 0.0, 1.0],  # синий
            [1.0, 1.0, 0.0],  # жёлтый
            [1.0, 0.0, 1.0],  # пурпурный
            [0.0, 1.0, 1.0],  # бирюзовый
            [1.0, 0.5, 0.0],  # оранжевый
            [0.5, 0.0, 1.0],  # фиолетовый
        ]
        colors = base_colors[:num_colors]

        inputs, outputs = [], []
        for _ in range(n):
            color_idx = random.randint(0, num_colors - 1)
            color = colors[color_idx]
            # Создаём изображение с доминирующим цветом + шум
            img = np.zeros((size, size, 3), dtype=np.float32)
            for c in range(3):
                img[:, :, c] = color[c] + np.random.normal(0, 0.1, (size, size))
            img = np.clip(img, 0, 1)

            inputs.append(img.tolist())
            outputs.append(str(color_idx))

        meta = self._build_image_meta(n, size, 3, num_colors)
        container = DatasetContainer(meta=meta, raw_inputs=inputs, raw_outputs=outputs)
        return container

    def _gen_pattern_generation(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация паттернов для GAN/Diffusion."""
        size = params.get("image_size", 32)
        pattern_type = params.get("pattern_type", "stripes")

        inputs, outputs = [], []
        for _ in range(n):
            if pattern_type == "stripes":
                img = self._gen_stripes(size)
            elif pattern_type == "checkerboard":
                img = self._gen_checkerboard(size)
            elif pattern_type == "circles":
                img = self._gen_concentric_circles(size)
            else:
                img = self._gen_stripes(size)

            inputs.append(img.astype(np.float32).tolist())
            outputs.append("0")  # для генеративных моделей выход не важен

        meta = self._build_image_meta(n, size, 1, 0)
        meta.task_type = TaskType.IMAGE_GENERATION
        container = DatasetContainer(meta=meta, raw_inputs=inputs, raw_outputs=outputs)
        return container

    # ================================================================
    #  ГЕНЕРАТОРЫ ЗВУКА / АУДИО
    # ================================================================

    def _gen_tone_classification(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация тонов для классификации по частоте."""
        sample_rate = params.get("sample_rate", 16000)
        duration = params.get("duration", 0.5)
        frequencies = params.get("frequencies", [262, 294, 330, 349, 392, 440, 494])
        noise_level = params.get("noise_level", 0.0)

        num_samples_audio = int(sample_rate * duration)
        inputs, outputs = [], []

        for _ in range(n):
            freq_idx = random.randint(0, len(frequencies) - 1)
            freq = frequencies[freq_idx]

            # Генерируем синусоиду
            t = np.linspace(0, duration, num_samples_audio, dtype=np.float32)
            audio = np.sin(2 * np.pi * freq * t)

            if noise_level > 0:
                noise = np.random.normal(0, noise_level, audio.shape)
                audio = audio + noise

            inputs.append(audio.tolist())
            outputs.append(str(freq_idx))

        meta = DatasetMeta(
            format_version=FORMAT_VERSION,
            data_type=DataType.AUDIO,
            task_type=TaskType.AUDIO_CLASSIFICATION,
            num_samples=n,
            audio_sample_rate=sample_rate,
            audio_channels=1,
            num_classes=len(frequencies),
            input_dim=(num_samples_audio,),
            output_dim=1,
        )
        container = DatasetContainer(meta=meta, raw_inputs=inputs, raw_outputs=outputs)
        return container

    def _gen_chord_recognition(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация аккордов для классификации."""
        sample_rate = params.get("sample_rate", 16000)
        duration = params.get("duration", 0.5)
        chord_types = params.get("chord_types", ["major", "minor"])

        num_samples_audio = int(sample_rate * duration)
        # Базовые частоты нот (до-ре-ми-фа-соль-ля-си)
        note_freqs = [262, 294, 330, 349, 392, 440, 494]

        inputs, outputs = [], []
        for _ in range(n):
            chord_type = random.choice(chord_types)
            root_idx = random.randint(0, 6)
            root_freq = note_freqs[root_idx]

            t = np.linspace(0, duration, num_samples_audio, dtype=np.float32)

            if chord_type == "major":
                # Мажорный аккорд: корень + большая терция + квинта
                freqs = [root_freq, root_freq * 1.25, root_freq * 1.5]
            else:
                # Минорный аккорд: корень + малая терция + квинта
                freqs = [root_freq, root_freq * 1.2, root_freq * 1.5]

            audio = np.zeros(num_samples_audio, dtype=np.float32)
            for f in freqs:
                audio += np.sin(2 * np.pi * f * t) / len(freqs)

            inputs.append(audio.tolist())
            outputs.append(str(chord_types.index(chord_type)))

        meta = DatasetMeta(
            format_version=FORMAT_VERSION,
            data_type=DataType.AUDIO,
            task_type=TaskType.AUDIO_CLASSIFICATION,
            num_samples=n,
            audio_sample_rate=sample_rate,
            audio_channels=1,
            num_classes=len(chord_types),
            input_dim=(num_samples_audio,),
            output_dim=1,
        )
        return DatasetContainer(meta=meta, raw_inputs=inputs, raw_outputs=outputs)

    def _gen_noise_denoise_audio(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация пар зашумлённый → чистый звук."""
        sample_rate = params.get("sample_rate", 16000)
        duration = params.get("duration", 0.5)
        noise_level = params.get("noise_level", 0.3)

        num_samples_audio = int(sample_rate * duration)
        inputs, outputs = [], []

        for _ in range(n):
            freq = random.uniform(200, 800)
            t = np.linspace(0, duration, num_samples_audio, dtype=np.float32)
            clean = np.sin(2 * np.pi * freq * t)

            noise = np.random.normal(0, noise_level, clean.shape)
            noisy = clean + noise

            inputs.append(noisy.tolist())
            outputs.append(clean.tolist())

        meta = DatasetMeta(
            format_version=FORMAT_VERSION,
            data_type=DataType.AUDIO,
            task_type=TaskType.ANOMALY_DETECTION,
            num_samples=n,
            audio_sample_rate=sample_rate,
            audio_channels=1,
            input_dim=(num_samples_audio,),
            output_dim=(num_samples_audio,),
        )
        return DatasetContainer(meta=meta, raw_inputs=inputs, raw_outputs=outputs)

    def _gen_frequency_regression(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация звука для регрессии частоты."""
        sample_rate = params.get("sample_rate", 16000)
        duration = params.get("duration", 0.5)
        freq_lo, freq_hi = params.get("freq_range", (100, 1000))

        num_samples_audio = int(sample_rate * duration)
        inputs, outputs = [], []

        for _ in range(n):
            freq = random.uniform(freq_lo, freq_hi)
            t = np.linspace(0, duration, num_samples_audio, dtype=np.float32)
            audio = np.sin(2 * np.pi * freq * t)

            inputs.append(audio.tolist())
            outputs.append(str(round(freq, 2)))

        meta = DatasetMeta(
            format_version=FORMAT_VERSION,
            data_type=DataType.AUDIO,
            task_type=TaskType.REGRESSION,
            num_samples=n,
            audio_sample_rate=sample_rate,
            audio_channels=1,
            input_dim=(num_samples_audio,),
            output_dim=1,
        )
        return DatasetContainer(meta=meta, raw_inputs=inputs, raw_outputs=outputs)

    def _gen_audio_generation(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация простых мелодий для генеративных моделей."""
        sample_rate = params.get("sample_rate", 16000)
        duration = params.get("duration", 1.0)
        num_notes = params.get("num_notes", 8)

        num_samples_audio = int(sample_rate * duration)
        note_duration = duration / num_notes
        # Пентатоника
        scale_freqs = [262, 294, 330, 392, 440, 523, 587, 659]

        inputs, outputs = [], []
        for _ in range(n):
            t = np.linspace(0, duration, num_samples_audio, dtype=np.float32)
            audio = np.zeros(num_samples_audio, dtype=np.float32)

            for note_idx in range(num_notes):
                freq = random.choice(scale_freqs)
                start = int(note_idx * note_duration * sample_rate)
                end = int((note_idx + 1) * note_duration * sample_rate)
                if end > num_samples_audio:
                    end = num_samples_audio
                note_t = t[start:end]
                # Простая огибающая (затухание)
                envelope = np.exp(-3 * (note_t - note_t[0]) / max(note_duration, 0.01))
                audio[start:end] += np.sin(2 * np.pi * freq * note_t) * envelope

            # Нормализация
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))

            inputs.append(audio.tolist())
            outputs.append("0")  # для генеративных моделей

        meta = DatasetMeta(
            format_version=FORMAT_VERSION,
            data_type=DataType.AUDIO,
            task_type=TaskType.AUDIO_GENERATION,
            num_samples=n,
            audio_sample_rate=sample_rate,
            audio_channels=1,
            input_dim=(num_samples_audio,),
            output_dim=(num_samples_audio,),
        )
        return DatasetContainer(meta=meta, raw_inputs=inputs, raw_outputs=outputs)

    # ================================================================
    #  ГЕНЕРАТОРЫ ВРЕМЕННЫХ РЯДОВ
    # ================================================================

    def _gen_sine_wave(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация синусоиды для предсказания."""
        seq_len = params.get("sequence_length", 50)
        noise = params.get("noise", 0.05)

        inputs, outputs = [], []
        for _ in range(n):
            freq = random.uniform(0.5, 3.0)
            phase = random.uniform(0, 2 * np.pi)
            t = np.linspace(0, 4 * np.pi, seq_len + 1)
            values = np.sin(freq * t + phase) + np.random.normal(0, noise, seq_len + 1)

            # Вход: первые seq_len значений, выход: следующее значение
            inputs.append(values[:seq_len].tolist())
            outputs.append(str(round(values[seq_len], 6)))

        meta = DatasetMeta(
            format_version=FORMAT_VERSION,
            data_type=DataType.TIME_SERIES,
            task_type=TaskType.REGRESSION,
            num_samples=n,
            input_dim=(seq_len,),
            output_dim=1,
        )
        return DatasetContainer(meta=meta, raw_inputs=inputs, raw_outputs=outputs)

    def _gen_trend_prediction(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация трендов для предсказания."""
        seq_len = params.get("sequence_length", 30)
        trend_type = params.get("trend_type", "linear")
        noise = params.get("noise", 0.1)

        inputs, outputs = [], []
        for _ in range(n):
            t = np.arange(seq_len + 1, dtype=np.float32)
            if trend_type == "linear":
                slope = random.uniform(-2, 2)
                intercept = random.uniform(-10, 10)
                values = slope * t + intercept
            elif trend_type == "quadratic":
                a = random.uniform(-0.5, 0.5)
                b = random.uniform(-2, 2)
                c = random.uniform(-10, 10)
                values = a * t**2 + b * t + c
            else:
                slope = random.uniform(-2, 2)
                intercept = random.uniform(-10, 10)
                values = slope * t + intercept

            values += np.random.normal(0, noise, seq_len + 1)

            inputs.append(values[:seq_len].tolist())
            outputs.append(str(round(values[seq_len], 6)))

        meta = DatasetMeta(
            format_version=FORMAT_VERSION,
            data_type=DataType.TIME_SERIES,
            task_type=TaskType.REGRESSION,
            num_samples=n,
            input_dim=(seq_len,),
            output_dim=1,
        )
        return DatasetContainer(meta=meta, raw_inputs=inputs, raw_outputs=outputs)

    def _gen_anomaly_detection(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация временных рядов с аномалиями."""
        seq_len = params.get("sequence_length", 100)
        anomaly_ratio = params.get("anomaly_ratio", 0.05)

        inputs, outputs = [], []
        for _ in range(n):
            # Нормальный ряд (синусоида + шум)
            t = np.linspace(0, 4 * np.pi, seq_len)
            values = np.sin(t) + np.random.normal(0, 0.1, seq_len)

            # Добавляем аномалии
            num_anomalies = max(1, int(seq_len * anomaly_ratio))
            anomaly_positions = np.random.choice(seq_len, num_anomalies, replace=False)
            labels = np.zeros(seq_len, dtype=np.float32)
            for pos in anomaly_positions:
                values[pos] += random.choice([-1, 1]) * random.uniform(2, 5)
                labels[pos] = 1.0

            inputs.append(values.tolist())
            outputs.append(labels.tolist())

        meta = DatasetMeta(
            format_version=FORMAT_VERSION,
            data_type=DataType.TIME_SERIES,
            task_type=TaskType.ANOMALY_DETECTION,
            num_samples=n,
            input_dim=(seq_len,),
            output_dim=(seq_len,),
        )
        return DatasetContainer(meta=meta, raw_inputs=inputs, raw_outputs=outputs)

    def _gen_seasonal_pattern(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация сезонных паттернов."""
        seq_len = params.get("sequence_length", 60)
        period = params.get("period", 12)
        noise = params.get("noise", 0.1)

        inputs, outputs = [], []
        for _ in range(n):
            t = np.arange(seq_len + 1, dtype=np.float32)
            trend = random.uniform(-0.1, 0.1) * t
            seasonal = np.sin(2 * np.pi * t / period)
            values = trend + seasonal + np.random.normal(0, noise, seq_len + 1)

            inputs.append(values[:seq_len].tolist())
            outputs.append(str(round(values[seq_len], 6)))

        meta = DatasetMeta(
            format_version=FORMAT_VERSION,
            data_type=DataType.TIME_SERIES,
            task_type=TaskType.REGRESSION,
            num_samples=n,
            input_dim=(seq_len,),
            output_dim=1,
        )
        return DatasetContainer(meta=meta, raw_inputs=inputs, raw_outputs=outputs)

    # ================================================================
    #  ГЕНЕРАТОРЫ ГРАФОВ
    # ================================================================

    def _gen_graph_connectivity(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация графов для задачи связности."""
        num_nodes = params.get("num_nodes", 10)
        edge_prob = params.get("edge_probability", 0.3)

        inputs, outputs = [], []
        for _ in range(n):
            adj = self._random_graph(num_nodes, edge_prob)
            is_connected = self._is_connected(adj, num_nodes)

            graph_data = {
                "adjacency_matrix": adj,
                "num_nodes": num_nodes,
                "node_features": [[1.0] * num_nodes],  # единичные признаки
            }
            inputs.append(graph_data)
            outputs.append("1" if is_connected else "0")

        meta = DatasetMeta(
            format_version=FORMAT_VERSION,
            data_type=DataType.GRAPH,
            task_type=TaskType.CLASSIFICATION,
            num_samples=n,
            input_dim=(num_nodes, num_nodes),
            output_dim=1,
            num_classes=2,
        )
        return DatasetContainer(meta=meta, raw_inputs=inputs, raw_outputs=outputs)

    def _gen_shortest_path(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация графов для задачи кратчайшего пути."""
        num_nodes = params.get("num_nodes", 8)
        edge_prob = params.get("edge_probability", 0.4)

        inputs, outputs = [], []
        for _ in range(n):
            adj = self._random_graph(num_nodes, edge_prob)
            src = random.randint(0, num_nodes - 1)
            dst = random.randint(0, num_nodes - 1)
            while dst == src:
                dst = random.randint(0, num_nodes - 1)

            path_len = self._bfs_shortest_path(adj, num_nodes, src, dst)

            graph_data = {
                "adjacency_matrix": adj,
                "num_nodes": num_nodes,
                "source": src,
                "destination": dst,
                "node_features": [[1.0] * num_nodes],
            }
            inputs.append(graph_data)
            outputs.append(str(path_len if path_len >= 0 else -1))

        meta = DatasetMeta(
            format_version=FORMAT_VERSION,
            data_type=DataType.GRAPH,
            task_type=TaskType.REGRESSION,
            num_samples=n,
            input_dim=(num_nodes, num_nodes),
            output_dim=1,
        )
        return DatasetContainer(meta=meta, raw_inputs=inputs, raw_outputs=outputs)

    def _gen_node_classification(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация графов для классификации узлов."""
        num_nodes = params.get("num_nodes", 10)
        num_classes = params.get("num_classes", 3)

        inputs, outputs = [], []
        for _ in range(n):
            adj = self._random_graph(num_nodes, 0.3)
            # Класс центрального узла зависит от степени
            center = num_nodes // 2
            degree = sum(adj[center])
            node_class = degree % num_classes

            # Признаки узлов: степень каждого узла
            features = [[float(sum(adj[i])) / max(num_nodes - 1, 1)] for i in range(num_nodes)]

            graph_data = {
                "adjacency_matrix": adj,
                "num_nodes": num_nodes,
                "center_node": center,
                "node_features": features,
            }
            inputs.append(graph_data)
            outputs.append(str(node_class))

        meta = DatasetMeta(
            format_version=FORMAT_VERSION,
            data_type=DataType.GRAPH,
            task_type=TaskType.CLASSIFICATION,
            num_samples=n,
            input_dim=(num_nodes, num_nodes),
            output_dim=1,
            num_classes=num_classes,
        )
        return DatasetContainer(meta=meta, raw_inputs=inputs, raw_outputs=outputs)

    # ================================================================
    #  ГЕНЕРАТОРЫ ТАБЛИЦ / ЧИСЕЛ
    # ================================================================

    def _gen_xor_problem(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация задачи XOR."""
        num_features = params.get("num_features", 2)

        inputs, outputs = [], []
        for _ in range(n):
            if num_features == 2:
                x = [float(random.randint(0, 1)), float(random.randint(0, 1))]
                y = float(int(x[0]) ^ int(x[1]))
            else:
                # Обобщённый XOR: чётность количества единиц
                x = [float(random.randint(0, 1)) for _ in range(num_features)]
                y = float(sum(int(v) for v in x) % 2)
            inputs.append(x)
            outputs.append(y)

        meta = DatasetMeta(
            format_version=FORMAT_VERSION,
            data_type=DataType.NUMERIC,
            task_type=TaskType.CLASSIFICATION,
            num_samples=n,
            input_dim=num_features,
            output_dim=1,
            num_classes=2,
        )
        return DatasetContainer(meta=meta, raw_inputs=inputs, raw_outputs=outputs)

    def _gen_function_approximation(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация данных для аппроксимации функции."""
        func_name = params.get("function", "sin")
        x_lo, x_hi = params.get("x_range", (-3.14, 3.14))

        inputs, outputs = [], []
        for _ in range(n):
            x = random.uniform(x_lo, x_hi)
            if func_name == "sin":
                y = math.sin(x)
            elif func_name == "cos":
                y = math.cos(x)
            elif func_name == "square":
                y = x ** 2
            elif func_name == "abs":
                y = abs(x)
            elif func_name == "exp":
                y = math.exp(x * 0.5)  # масштабируем чтобы не улететь
            else:
                y = math.sin(x)
            inputs.append([x])
            outputs.append(y)

        meta = DatasetMeta(
            format_version=FORMAT_VERSION,
            data_type=DataType.NUMERIC,
            task_type=TaskType.REGRESSION,
            num_samples=n,
            input_dim=1,
            output_dim=1,
        )
        return DatasetContainer(meta=meta, raw_inputs=inputs, raw_outputs=outputs)

    def _gen_clustering_blobs(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация гауссовых кластеров."""
        num_clusters = params.get("num_clusters", 3)
        num_features = params.get("num_features", 2)

        # Генерируем центры кластеров
        centers = np.random.uniform(-5, 5, (num_clusters, num_features))

        inputs, outputs = [], []
        for _ in range(n):
            cluster_id = random.randint(0, num_clusters - 1)
            point = centers[cluster_id] + np.random.normal(0, 0.5, num_features)
            inputs.append(point.tolist())
            outputs.append(float(cluster_id))

        meta = DatasetMeta(
            format_version=FORMAT_VERSION,
            data_type=DataType.NUMERIC,
            task_type=TaskType.CLUSTERING,
            num_samples=n,
            input_dim=num_features,
            output_dim=1,
            num_classes=num_clusters,
        )
        return DatasetContainer(meta=meta, raw_inputs=inputs, raw_outputs=outputs)

    def _gen_regression_linear(self, params: Dict, n: int) -> DatasetContainer:
        """Генерация данных для линейной регрессии."""
        num_features = params.get("num_features", 3)
        noise = params.get("noise", 0.1)

        # Случайные коэффициенты
        weights = np.random.uniform(-2, 2, num_features)
        bias = random.uniform(-5, 5)

        inputs, outputs = [], []
        for _ in range(n):
            x = np.random.uniform(-5, 5, num_features)
            y = np.dot(x, weights) + bias + np.random.normal(0, noise)
            inputs.append(x.tolist())
            outputs.append(float(y))

        meta = DatasetMeta(
            format_version=FORMAT_VERSION,
            data_type=DataType.NUMERIC,
            task_type=TaskType.REGRESSION,
            num_samples=n,
            input_dim=num_features,
            output_dim=1,
        )
        return DatasetContainer(meta=meta, raw_inputs=inputs, raw_outputs=outputs)

    # ================================================================
    #  ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ (РИСОВАНИЕ)
    # ================================================================

    @staticmethod
    def _draw_shape_2d(shape: str, size: int, channels: int) -> np.ndarray:
        """Рисует 2D фигуру на массиве."""
        img = np.zeros((size, size, channels), dtype=np.float32)
        cx, cy = size // 2, size // 2

        if shape == "circle":
            r = size // 3
            for y in range(size):
                for x in range(size):
                    if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                        img[y, x, :] = 1.0

        elif shape == "square":
            s = size // 3
            for y in range(cy - s, cy + s):
                for x in range(cx - s, cx + s):
                    if 0 <= y < size and 0 <= x < size:
                        img[y, x, :] = 1.0

        elif shape == "triangle":
            h = size // 2
            for row in range(h):
                width = int(row * (size / 2) / h)
                for x in range(cx - width, cx + width):
                    y = cy - h // 2 + row
                    if 0 <= y < size and 0 <= x < size:
                        img[y, x, :] = 1.0

        elif shape == "cross":
            w = max(size // 6, 1)
            for y in range(size):
                for x in range(cx - w, cx + w):
                    if 0 <= x < size:
                        img[y, x, :] = 1.0
            for x in range(size):
                for y in range(cy - w, cy + w):
                    if 0 <= y < size:
                        img[y, x, :] = 1.0

        elif shape == "star":
            # Упрощённая звезда (крест + диагонали)
            w = max(size // 8, 1)
            for y in range(size):
                for x in range(size):
                    if abs(x - cx) < w or abs(y - cy) < w:
                        img[y, x, :] = 1.0
                    if abs(x - y) < w or abs(x + y - size) < w:
                        img[y, x, :] = 0.7

        else:
            # По умолчанию круг
            r = size // 3
            for y in range(size):
                for x in range(size):
                    if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                        img[y, x, :] = 1.0

        return img

    @staticmethod
    def _draw_shape_3d(shape: str, size: int, channels: int) -> np.ndarray:
        """Рисует 3D проекцию на массиве."""
        img = np.zeros((size, size, channels), dtype=np.float32)
        cx, cy = size // 2, size // 2

        if shape == "cube":
            # Проекция куба: два квадрата + рёбра
            s = size // 4
            offset = size // 8
            # Передний квадрат
            for y in range(cy - s, cy + s):
                for x in range(cx - s, cx + s):
                    if 0 <= y < size and 0 <= x < size:
                        if (x == cx - s or x == cx + s - 1 or
                                y == cy - s or y == cy + s - 1):
                            img[y, x, :] = 1.0
            # Задний квадрат (смещённый)
            for y in range(cy - s + offset, cy + s + offset):
                for x in range(cx - s + offset, cx + s + offset):
                    if 0 <= y < size and 0 <= x < size:
                        if (x == cx - s + offset or x == cx + s + offset - 1 or
                                y == cy - s + offset or y == cy + s + offset - 1):
                            img[y, x, :] = 0.6

        elif shape == "sphere":
            # Сфера: круг с градиентом
            r = size // 3
            for y in range(size):
                for x in range(size):
                    dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                    if dist <= r:
                        intensity = 1.0 - (dist / r) * 0.7
                        img[y, x, :] = intensity

        elif shape == "pyramid":
            # Пирамида: треугольник с внутренними линиями
            h = size // 2
            for row in range(h):
                width = int(row * (size / 2) / h)
                for x in range(cx - width, cx + width):
                    y = cy - h // 2 + row
                    if 0 <= y < size and 0 <= x < size:
                        if abs(x - (cx - width)) < 2 or abs(x - (cx + width)) < 2:
                            img[y, x, :] = 1.0
                        elif row < 3:
                            img[y, x, :] = 0.8

        else:
            # По умолчанию сфера
            r = size // 3
            for y in range(size):
                for x in range(size):
                    dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                    if dist <= r:
                        img[y, x, :] = 1.0 - (dist / r) * 0.5

        return img

    @staticmethod
    def _draw_digit(digit: int, size: int) -> np.ndarray:
        """Рисует синтетическую цифру на массиве."""
        img = np.zeros((size, size, 1), dtype=np.float32)
        # Определяем области для каждой цифры (упрощённо)
        cx, cy = size // 2, size // 2
        r = size // 3

        if digit == 0:
            # Овал
            for y in range(size):
                for x in range(size):
                    dist = math.sqrt(((x - cx) / (r * 0.6)) ** 2 + ((y - cy) / r) ** 2)
                    if 0.7 <= dist <= 1.0:
                        img[y, x, 0] = 1.0
        elif digit == 1:
            # Вертикальная линия
            for y in range(cy - r, cy + r):
                for x in range(cx - 1, cx + 2):
                    if 0 <= y < size and 0 <= x < size:
                        img[y, x, 0] = 1.0
        elif digit == 2:
            # Дуга сверху + линия снизу
            for y in range(cy - r, cy):
                for x in range(cx - r, cx + r):
                    dist = math.sqrt((x - cx) ** 2 + (y - (cy - r // 2)) ** 2)
                    if r * 0.5 <= dist <= r * 0.8:
                        img[y, x, 0] = 1.0
            for x in range(cx - r, cx + r):
                if 0 <= x < size:
                    img[cy + r - 1, x, 0] = 1.0
        elif digit == 3:
            # Две дуги
            for y in range(cy - r, cy + r):
                for x in range(cx, cx + r):
                    dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                    if r * 0.5 <= dist <= r * 0.8:
                        img[y, x, 0] = 1.0
        elif digit == 4:
            # Вертикальная + горизонтальная + диагональ
            for y in range(cy - r, cy + r):
                if 0 <= y < size and 0 <= cx + r // 2 < size:
                    img[y, cx + r // 2, 0] = 1.0
            for x in range(cx - r, cx + r):
                if 0 <= x < size and 0 <= cy < size:
                    img[cy, x, 0] = 1.0
        elif digit == 5:
            # Горизонтальная сверху + дуга снизу
            for x in range(cx - r, cx + r):
                if 0 <= x < size:
                    img[cy - r, x, 0] = 1.0
            for y in range(cy, cy + r):
                if 0 <= y < size:
                    img[y, cx - r, 0] = 1.0
        elif digit == 6:
            # Круг снизу + линия сверху
            for y in range(size):
                for x in range(size):
                    dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                    if r * 0.5 <= dist <= r * 0.8 and y >= cy:
                        img[y, x, 0] = 1.0
        elif digit == 7:
            # Горизонтальная сверху + диагональ
            for x in range(cx - r, cx + r):
                if 0 <= x < size:
                    img[cy - r, x, 0] = 1.0
            for i in range(r * 2):
                x = cx + r - i
                y = cy - r + i
                if 0 <= x < size and 0 <= y < size:
                    img[y, x, 0] = 1.0
        elif digit == 8:
            # Два круга
            for y in range(size):
                for x in range(size):
                    dist_top = math.sqrt((x - cx) ** 2 + (y - cy + r // 2) ** 2)
                    dist_bot = math.sqrt((x - cx) ** 2 + (y - cy - r // 2) ** 2)
                    if (r * 0.3 <= dist_top <= r * 0.5 or
                            r * 0.3 <= dist_bot <= r * 0.5):
                        img[y, x, 0] = 1.0
        elif digit == 9:
            # Круг сверху + линия снизу
            for y in range(size):
                for x in range(size):
                    dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                    if r * 0.5 <= dist <= r * 0.8 and y <= cy:
                        img[y, x, 0] = 1.0

        return img

    @staticmethod
    def _gen_stripes(size: int) -> np.ndarray:
        """Генерация полосок."""
        img = np.zeros((size, size, 1), dtype=np.float32)
        stripe_width = max(size // 8, 1)
        for x in range(0, size, stripe_width * 2):
            img[:, x:x + stripe_width, :] = 1.0
        return img

    @staticmethod
    def _gen_checkerboard(size: int) -> np.ndarray:
        """Генерация шахматной доски."""
        img = np.zeros((size, size, 1), dtype=np.float32)
        cell = max(size // 8, 1)
        for y in range(size):
            for x in range(size):
                if ((x // cell) + (y // cell)) % 2 == 0:
                    img[y, x, 0] = 1.0
        return img

    @staticmethod
    def _gen_concentric_circles(size: int) -> np.ndarray:
        """Генерация концентрических кругов."""
        img = np.zeros((size, size, 1), dtype=np.float32)
        cx, cy = size // 2, size // 2
        for y in range(size):
            for x in range(size):
                dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                ring = int(dist / (size // 8))
                if ring % 2 == 0:
                    img[y, x, 0] = 1.0
        return img

    # ================================================================
    #  ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ (ГРАФЫ)
    # ================================================================

    @staticmethod
    def _random_graph(num_nodes: int, edge_prob: float) -> List[List[int]]:
        """Генерирует случайный граф (Эрдёш-Реньи)."""
        adj = [[0] * num_nodes for _ in range(num_nodes)]
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if random.random() < edge_prob:
                    adj[i][j] = 1
                    adj[j][i] = 1
        return adj

    @staticmethod
    def _is_connected(adj: List[List[int]], num_nodes: int) -> bool:
        """Проверяет связность графа (BFS)."""
        if num_nodes == 0:
            return True
        visited = [False] * num_nodes
        queue = [0]
        visited[0] = True
        count = 1
        while queue:
            node = queue.pop(0)
            for neighbor in range(num_nodes):
                if adj[node][neighbor] == 1 and not visited[neighbor]:
                    visited[neighbor] = True
                    count += 1
                    queue.append(neighbor)
        return count == num_nodes

    @staticmethod
    def _bfs_shortest_path(
        adj: List[List[int]], num_nodes: int, src: int, dst: int
    ) -> int:
        """Находит кратчайший путь (BFS). Возвращает -1 если пути нет."""
        if src == dst:
            return 0
        visited = [False] * num_nodes
        queue = [(src, 0)]
        visited[src] = True
        while queue:
            node, dist = queue.pop(0)
            for neighbor in range(num_nodes):
                if adj[node][neighbor] == 1 and not visited[neighbor]:
                    if neighbor == dst:
                        return dist + 1
                    visited[neighbor] = True
                    queue.append((neighbor, dist + 1))
        return -1

    # ================================================================
    #  ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ (КОНТЕЙНЕРЫ)
    # ================================================================

    @staticmethod
    def _build_text_container(
        inputs: List[str], outputs: List[str], task_type: TaskType
    ) -> DatasetContainer:
        """Строит DatasetContainer для текстовых данных."""
        max_seq_len = max(
            max((len(str(x)) for x in inputs), default=0),
            max((len(str(y)) for y in outputs), default=0),
        )
        meta = DatasetMeta(
            format_version=FORMAT_VERSION,
            data_type=DataType.TEXT,
            task_type=task_type,
            num_samples=len(inputs),
            vocab_size=0,  # будет заполнен при разбиении
            max_sequence_length=max_seq_len,
            input_dim=max_seq_len,
            output_dim=max_seq_len,
        )
        return DatasetContainer(meta=meta, raw_inputs=inputs, raw_outputs=outputs)

    @staticmethod
    def _build_image_meta(
        n: int, size: int, channels: int, num_classes: int
    ) -> DatasetMeta:
        """Строит DatasetMeta для изображений."""
        return DatasetMeta(
            format_version=FORMAT_VERSION,
            data_type=DataType.IMAGE,
            task_type=TaskType.IMAGE_CLASSIFICATION,
            num_samples=n,
            image_shape=(size, size, channels),
            input_dim=(size, size, channels),
            output_dim=1,
            num_classes=num_classes,
        )