"""
config.py
=========

Глобальная конфигурация OracleAI Trainer v2.0.

Этот файл является первым и базовым модулем новой архитектуры.
Он НЕ должен импортировать ничего из core/, gui/, torch, PyQt5 или numpy.
Разрешены только стандартная библиотека и константы.

Назначение:
- пути к данным, моделям, логам, пресетам, ИИ-моделям и профилю пользователя;
- дефолтные параметры обучения;
- настройки GUI;
- режимы подсказок;
- заглушки для будущего ИИ;
- лимиты безопасности;
- создание необходимых директорий при старте программы.

Важно для интеграции:
- все модули читают константы отсюда;
- запрещается создавать эти же директории в других модулях;
- запрещается хранить здесь изменяемое состояние проекта;
- для состояния проекта будет использоваться core/project_state.py.
"""

import os
from pathlib import Path
from enum import Enum


# ============================================================
# 1. Общая информация о приложении
# ============================================================

__version__ = "2.4.1"
APP_NAME = "OracleAI Trainer"
ORG_NAME = "OracleAI"

PROJECT_EXTENSION = ".oai"
MODEL_EXTENSION = ".pth"
TOKENIZER_EXTENSION = ".json"


# ============================================================
# 2. Пути к директориям
# ============================================================

# Корень проекта. Используем resolve(), чтобы стабильно работать
# независимо от того, откуда запустили программу.
BASE_DIR = Path(__file__).resolve().parent

# Основная папка данных
DATA_DIR = BASE_DIR / "data"

# Данные пользователя и проекты
DATASETS_DIR = DATA_DIR / "datasets"
MODELS_DIR = DATA_DIR / "models"
TOKENIZERS_DIR = DATA_DIR / "tokenizers"
LOGS_DIR = DATA_DIR / "logs"
DOCS_DIR = BASE_DIR / "docs"

# Новые директории v2.0
PRESETS_DIR = DATA_DIR / "presets"
AI_MODELS_DIR = DATA_DIR / "ai_models"
AI_CACHE_DIR = DATA_DIR / "ai_cache"
USER_DIR = DATA_DIR / "user"
EXPORT_DIR = DATA_DIR / "exports"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
TEMP_DIR = DATA_DIR / "temp"

# v2.0 FIX: Директория для истории сеансов пользователя.
# Используется в core/session.py и импортируется в core/logger.py.
SESSIONS_DIR = DATA_DIR / "sessions"

# v2.0: Директории многопользовательской системы.
# Используются core/user_manager.py, core/pretrained_manager.py,
# core/user_storage_analyzer.py.
USERS_DIR = DATA_DIR / "users"
SHARED_DIR = DATA_DIR / "shared"
SYSTEM_DIR = DATA_DIR / "system"
PRETRAINED_MODELS_DIR = SHARED_DIR / "pretrained_models"
INSTALLED_MODELS_FILE = DATA_DIR / "models" / "installed_models.json"

# Конкретные служебные файлы
SETTINGS_FILE = USER_DIR / "settings.json"
SESSION_PROFILE_FILE = USER_DIR / "session_profile.json"
LATEST_LOG_FILE = LOGS_DIR / "oracleai_latest.log"


# ============================================================
# 3. Режимы подсказок
# ============================================================

class HintMode(str, Enum):
    """
    Режимы работы подсказок.

    RULES_ONLY:
        Только встроенные правила и шаблоны.
        Основной режим, полностью рабочий без ИИ.

    AI_ONLY:
        Программа пытается использовать ИИ.
        Если ИИ недоступен, показывает автоматическую правила-подсказку
        с пометкой, что ИИ недоступен.

    MIXED:
        Сначала быстрые правила, затем, если ИИ доступен,
        ИИ может дополнить или перефразировать подсказку.

    OFF:
        Подсказки полностью выключены.
    """

    RULES_ONLY = "rules"
    AI_ONLY = "ai"
    MIXED = "mixed"
    OFF = "off"


# Значение по умолчанию.
# Важно: программа должна быть полностью работоспособна без ИИ.
DEFAULT_HINT_MODE = HintMode.RULES_ONLY.value

# Список допустимых значений для валидации в UI и конфиге.
ALLOWED_HINT_MODES = tuple(mode.value for mode in HintMode)


# ============================================================
# 4. Настройки обучения по умолчанию
# ============================================================

DEFAULT_LEARNING_RATE = 0.001
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 50
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_GRADIENT_CLIP = 1.0
DEFAULT_TEACHER_FORCING_RATIO = 0.5
DEFAULT_DROPOUT = 0.1

DEFAULT_TRAIN_RATIO = 0.80
DEFAULT_VAL_RATIO = 0.10
DEFAULT_TEST_RATIO = 0.10

# Ранняя остановка по умолчанию.
# В первой версии можно держать выключенной, но контракт должен быть.
DEFAULT_EARLY_STOPPING = {
    "enabled": False,
    "patience": 5,
    "min_delta": 0.001,
}

# Планировщик скорости обучения по умолчанию.
DEFAULT_SCHEDULER = "none"

# Допустимые значения, чтобы GUI и валидаторы могли на них опираться.
ALLOWED_OPTIMIZERS = ("adam", "adamw", "sgd", "rmsprop")
ALLOWED_LOSS_FUNCTIONS_NUMERIC = ("mse", "mae")
ALLOWED_LOSS_FUNCTIONS_TEXT = ("cross_entropy",)
ALLOWED_SCHEDULERS = ("none", "steplr", "reduce_on_plateau", "cosine")


# ============================================================
# 5. Настройки GUI
# ============================================================

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800

# Минимальные размеры окна, чтобы интерфейс не ломался.
WINDOW_MIN_WIDTH = 1000
WINDOW_MIN_HEIGHT = 700

# Базовый размер шрифта.
# Можно использовать в main.py до создания QApplication.
DEFAULT_FONT_POINT_SIZE = 11

# Сколько примеров показывать в предпросмотре датасета.
DATASET_PREVIEW_ROWS = 10

# Сколько примеров показывать в карточках пресетов.
PRESET_PREVIEW_SAMPLES = 5


# ============================================================
# 5b. Масштабирование интерфейса (HighDPI / 2K-экраны)
# ============================================================
#
# Qt 5.15 с AA_EnableHighDpiScaling неверно определяет масштаб на 2K
# (например dpr=2.0 при реальном 1.5), из-за чего интерфейс «плывёт».
# Поэтому масштаб задаётся детерминированно через QT_SCALE_FACTOR:
#   scale = физическая высота экрана / UI_REFERENCE_HEIGHT
# (см. main.py — _compute_ui_scale).

UI_REFERENCE_HEIGHT = 1152       # эталонная высота экрана (уменьшает масштаб на 2K)
UI_SCALE_MIN = 1.0               # минимум масштаба
UI_SCALE_MAX = 1.5               # максимум масштаба (не раздуваем на 4K+)

# Доля экрана, которую занимает главное окно (0.0–1.0)
WINDOW_SCREEN_FRACTION = 0.80


# ============================================================
# 6. Логирование
# ============================================================

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s.%(funcName)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Минимальный уровень для файла и консоли можно разделить в logger.py,
# но сами форматы должны быть централизованы здесь.
CONSOLE_LOG_LEVEL = "INFO"
FILE_LOG_LEVEL = "DEBUG"


# ============================================================
# 7. ИИ-заглушки
# ============================================================
#
# ВАЖНО:
# Сейчас полноценный ИИ не реализуется.
# Эти настройки нужны, чтобы будущий ИИ можно было внедрить без перелома
# архитектуры.
#
# Никаких проверок лицензий и лицензионных фильтров в текущей версии нет.
# Это сознательное требование проекта.

AI_ENABLED = False
AI_PROVIDER = "null"  # null | gguf | local | future

# Режим ИИ-функций в интерфейсе.
# Если AI_ENABLED=False, все ИИ-пункты должны быть отключены или помечены
# как недоступные.
AI_ALLOW_DOWNLOAD = False
AI_AUTO_LOAD_MODEL = False

# Расположение локальных ИИ-моделей и кэша.
AI_MODELS_MANIFEST_FILE = AI_MODELS_DIR / "models_manifest.json"
AI_DOWNLOADED_MODELS_INDEX_FILE = AI_MODELS_DIR / "installed_models.json"

# Ограничения ресурсов для будущего ИИ.
AI_MAX_RAM_MB = 2048
AI_MAX_CPU_THREADS = 2
AI_TIMEOUT_SECONDS = 30
AI_MAX_CONTEXT_TOKENS = 1024
AI_MAX_GENERATED_TOKENS = 256

# Будущие возможности ИИ.
# Хранить как список строк, чтобы можно было расширять без перелома кода.
AI_CAPABILITIES_TEXT = (
    "TEXT_HINTS",
    "TEXT_GENERATION",
    "TRANSLATION_PAIRS",
    "PARAPHRASE_PAIRS",
    "DATASET_GENERATION",
)

AI_CAPABILITIES_MULTIMODAL = (
    "IMAGE_CAPTIONING",
    "AUDIO_TRANSCRIPTION",
    "SPEECH_SYNTHESIS",
    "IMAGE_GENERATION",
)


# ============================================================
# 7b. Ограничения для ИИ-агента «Оракул»
# ============================================================
#
# Агент работает локально (шаблоны) и может подключаться к внешним API
# (Яндекс GPT / Сбер GigaChat) по ключам пользователя. Эти лимиты защищают
# систему от перегрузки.

ORACULUM_MAX_RAM_MB = 2048
ORACULUM_MAX_MODEL_SIZE_MB = 512
ORACULUM_MAX_CPU_PERCENT = 80
ORACULUM_API_CALLS_PER_MINUTE = 10
ORACULUM_ENABLE_LOCAL_MODEL = True
ORACULUM_LOCAL_MODEL_PATH = None

# Файл настроек Оракула (API-ключи, системный промпт, автоконтекст).
# Заполняется администратором из панели администратора и хранится в data/system/.
ORACULUM_SETTINGS_FILE = SYSTEM_DIR / "oraculum_settings.json"

# Лимиты на количество проектов / размер хранилища пользователя.
MAX_PROJECTS_PER_USER = 100
MAX_USER_STORAGE_MB = 1024


# ============================================================
# 7c. Локальная GGUF-модель Оракула (llama.cpp)
# ============================================================
#
# Оракул может использовать локальную GGUF-модель (например,
# YandexGPT-5-Lite-8B-instruct через llama-cpp-python) для ответов
# в мини-чате. Модель загружается ЛЕНИВО: только при выборе режима
# «ИИ (локальная модель)» и при первом запросе — это защищает систему
# от лишней нагрузки при обычной работе на шаблонах.

# Кандидаты на локальную модель. Ищутся в BASE_DIR (рядом с config.py),
# если ORACULUM_LOCAL_MODEL_PATH не задан явно.
ORACULUM_GGUF_CANDIDATES = (
    "YandexGPT-5-Lite-8B-instruct-Q4_K_M.gguf",
)

ORACULUM_GGUF_CTX_SIZE = 2048          # малый контекст → меньше риск зависания

# Сколько слоёв модели отдать на видеокарту:
#   -1 — все слои на GPU (быстро, нужна видеокарта NVIDIA с CUDA);
#    0 — все слои на CPU (медленно, но работает на любом компьютере).
# Значение выбирается автоматически при старте по наличию CUDA
# (см. блок «Автоопределение GPU» в конце этого файла). Чтобы задать
# режим вручную, раскомментируйте строку ниже и укажите число.
ORACULUM_GGUF_N_GPU_LAYERS = None      # None = определить автоматически

ORACULUM_GGUF_MAX_TOKENS = 256
ORACULUM_GGUF_TEMPERATURE = 0.3
ORACULUM_GGUF_TOP_P = 0.9
ORACULUM_GGUF_VERBOSE = False
ORACULUM_LOCAL_GEN_TIMEOUT = 90        # сек, таймаут генерации (защита от зависания)

# GGUF-модели больше 512 МБ (лимит ORACULUM_MAX_MODEL_SIZE_MB), поэтому
# для LLM-бэкенда действует отдельный, более щедрый лимит.
ORACULUM_MAX_LLM_MODEL_SIZE_MB = 8192


# ============================================================
# 7d. Администратор системы
# ============================================================
#
# Пользователь с этим ником (без учёта регистра) получает роль
# администратора: доступ к панели дебага, всем ресурсам и настройкам.

ADMIN_USERNAME = "admin"

# ============================================================
# 8. Безопасность и лимиты
# ============================================================
#
# Программа должна позволять эксперименты, но не должна падать
# или убивать компьютер пользователя.

# Максимальный размер загружаемого датасета в мегабайтах.
MAX_DATASET_SIZE_MB = 500

# Максимальное количество примеров в датасете.
MAX_DATASET_SAMPLES = 1_000_000

# Максимальная длина строки при текстовой загрузке.
# Всё длиннее считается аномалией и может быть удалено с предупреждением.
MAX_TEXT_SEQUENCE_LENGTH = 1500

# Максимальное количество параметров модели.
# Это мягкий лимит для защиты от случайного создания слишком больших сетей.
MAX_MODEL_PARAMS = 100_000_000

# Максимальное время обучения в часах.
# Используется как предохранитель.
MAX_TRAINING_HOURS = 12

# Как часто сохранять чекпоинт обучения.
AUTOSAVE_CHECKPOINT_EVERY = 5

# Максимальный размер изображения в пикселях по каждой стороне.
# Заглушка для будущей работы с изображениями.
MAX_IMAGE_SIDE = 1024

# Максимальная длительность аудио в секундах.
# Заглушка для будущей работы со звуком.
MAX_AUDIO_DURATION_SECONDS = 60

# Разрешить опасные эксперименты.
# Если False, программа блокирует только критически опасные действия.
# Если True, пользователь может делать почти всё, но программа предупреждает.
ALLOW_EXPERIMENTS = True


# ============================================================
# 9. Поддерживаемые типы данных
# ============================================================
#
# Это контракт для всех модулей данных.
# Сейчас реально используются только numeric и text.
# Остальные типы являются заготовкой под зоопарк моделей.

SUPPORTED_DATA_TYPES = (
    "numeric",
    "text",
    "image",
    "audio",
    "graph",
    "multimodal",
)

ACTIVE_DATA_TYPES = (
    "numeric",
    "text",
)

FUTURE_DATA_TYPES = (
    "image",
    "audio",
    "graph",
    "multimodal",
)


# ============================================================
# 10. Поддерживаемые типы задач
# ============================================================
#
# Используется генератором, датасетами и будущими пресетами.

SUPPORTED_TASK_TYPES = (
    # Математика
    "addition",
    "subtraction",
    "multiplication",
    "division",
    "mixed_math",

    # Алгебра
    "linear_equation",
    "quadratic",

    # Шифры
    "cipher_caesar",
    "cipher_atbash",

    # Текст и будущие ИИ-задачи
    "translation",
    "classification",

    # Изображения, аудио, графы, временные ряды - будущие заготовки
    "image_classification",
    "image_denoising",
    "image_generation",
    "image_captioning",

    "audio_classification",
    "audio_denoising",
    "audio_transcription",

    "graph_classification",

    "time_series_forecast",

    # Общее
    "custom",
)


# ============================================================
# 11. Пресеты генератора
# ============================================================
#
# Пресеты хранятся как рецепты, а не как тяжёлые готовые файлы.
# В будущем каждый пресет может быть JSON-файлом в PRESETS_DIR.

# Реестр встроенных пресетов.
# Каждый пресет: словарик с ключами
#   id, name, description, task, params,
#   recommended_architecture, recommended_config.
# core/preset_registry.py читает этот список и дополняет
# JSON-файлами из PRESETS_DIR при наличии.

BUILTIN_PRESETS = [
    {
        "id": "math_addition_easy",
        "name": "Сложение (однозначные)",
        "description": "Сложение чисел от 1 до 9. Идеально для первого запуска.",
        "task": "addition",
        "params": {"num_range": [1, 9], "num_samples": 5000},
        "recommended_architecture": "transformer",
        "recommended_config": {"embedding_dim": 64, "num_heads": 4, "epochs": 20},
    },
    {
        "id": "math_addition_hard",
        "name": "Сложение (двузначные)",
        "description": "Сложение чисел до 99. Сеть учится переносить разряды.",
        "task": "addition",
        "params": {"num_range": [1, 99], "num_samples": 20000},
        "recommended_architecture": "transformer",
        "recommended_config": {"embedding_dim": 128, "num_heads": 8, "epochs": 40},
    },
    {
        "id": "cipher_caesar_basic",
        "name": "Шифр Цезаря",
        "description": "Расшифровка сдвига на 3 позиции.",
        "task": "cipher_caesar",
        "params": {"num_samples": 10000},
        "recommended_architecture": "transformer",
        "recommended_config": {"embedding_dim": 64, "epochs": 30},
    },
]


# ============================================================
# 12. Создание директорий при старте
# ============================================================
#
# ВАЖНО: это ЕДИНСТВЕННОЕ место, где создаются директории.
# Другие модули НЕ должны вызывать mkdir для этих путей.

_ALL_DIRS = [
    DATA_DIR,
    DATASETS_DIR,
    MODELS_DIR,
    TOKENIZERS_DIR,
    LOGS_DIR,
    DOCS_DIR,
    PRESETS_DIR,
    AI_MODELS_DIR,
    AI_CACHE_DIR,
    USER_DIR,
    EXPORT_DIR,
    CHECKPOINTS_DIR,
    TEMP_DIR,
    SESSIONS_DIR,          # v2.0 FIX: добавлена в цикл создания
    USERS_DIR,
    SHARED_DIR,
    SYSTEM_DIR,
    PRETRAINED_MODELS_DIR,
]

for _d in _ALL_DIRS:
    _d.mkdir(parents=True, exist_ok=True)


# ============================================================
# 13. Автоопределение видеокарты для Оракула
# ============================================================
#
# Наличие CUDA проверяется без импорта torch и llama_cpp: этот файл
# обязан оставаться лёгким (см. правило в начале). Ищем библиотеку
# CUDA-рантайма в стандартных местах установки.
#
# Если библиотека не найдена — считаем, что видеокарты нет, и
# отправляем все слои модели на процессор. Программа продолжит
# работать: Оракул будет отвечать медленнее.

def _detect_cuda() -> bool:
    """
    Проверяет, доступна ли CUDA на этой машине.

    Ищет библиотеку `cudart64_*.dll` в каталогах CUDA Toolkit и в
    каталоге пакета llama_cpp. Импорт тяжёлых библиотек не выполняется.

    @returns: True, если найдена библиотека CUDA-рантайма.
    """
    if os.name != "nt":
        # На Linux/macOS проверка не нужна: llama.cpp сам выберет бэкенд.
        return True

    search_dirs = []

    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        search_dirs.append(Path(cuda_path) / "bin")

    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    cuda_root = Path(program_files) / "NVIDIA GPU Computing Toolkit" / "CUDA"
    if cuda_root.is_dir():
        try:
            search_dirs.extend(sorted(cuda_root.iterdir(), reverse=True))
        except OSError:
            # Каталог CUDA недоступен для чтения — считаем, что CUDA нет.
            pass

    # Рантайм, который кладётся рядом с llama_cpp при установке.
    try:
        import llama_cpp  # noqa: F401  (нужен только путь к пакету)
        search_dirs.append(Path(llama_cpp.__file__).parent / "lib")
    except Exception:
        # llama_cpp не установлен — это штатно, Оракул просто не будет
        # использовать локальную модель.
        pass

    for directory in search_dirs:
        try:
            if any(directory.glob("cudart64_*.dll")):
                return True
        except OSError:
            # Каталог недоступен — пропускаем его и ищем дальше.
            continue

    return False


if ORACULUM_GGUF_N_GPU_LAYERS is None:
    ORACULUM_GGUF_N_GPU_LAYERS = -1 if _detect_cuda() else 0

CUDA_AVAILABLE = _detect_cuda()