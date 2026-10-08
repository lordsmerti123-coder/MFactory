"""
core/curator.py
===============
ИИ-Куратор OracleAI Studio — фоновый аналитик с личностью.

Ответственность модуля:
  • Генерация весёлых имён для моделей.
  • Эмоциональные комментарии к процессу обучения в реальном времени.
  • Реакция на предсказания модели («45-68=67? Тебе учиться и учиться!»).
  • Интересные факты для пауз и ожидания.
  • Рекомендации следующего шага на основе истории сеансов пользователя.
  • Заглушки для будущего локального ИИ-бэкенда (GGUF / LLM).

Взаимодействие с другими модулями:
  • Импортирует типы из ``core.contracts`` (SessionState, TrainingHistory, ModelConfig).
  • Может дополнять технический анализ ``core.explainer`` эмоциональной обёрткой.
  • Не блокирует поток обучения: все методы синхронные и быстрые.

Режимы подсказок (hint_mode):
  • "static"  — только правило-ориентированные комментарии.
  • "ai"      — только ИИ-комментарии (если бэкенд доступен).
  • "hybrid"  — ИИ, если доступен; иначе правила. Режим по умолчанию.
  • "none"    — куратор молчит.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# ИМПОРТ КОНТРАКТОВ
# ---------------------------------------------------------------------------
# Согласно ТЗ, все типы берутся из core.contracts.
# Однако, если файл собирается до того, как contracts.py готов,
# используем локальные заглушки, чтобы модуль не падал.
try:
    from core.contracts import SessionState, TrainingHistory, ModelConfig
except ImportError:  # pragma: no cover - защита на время фазы 0
    logger.warning(
        "core.contracts не найден. Используются локальные заглушки типов."
    )

    @dataclass
    class SessionState:
        session_id: str = ""
        total_sessions: int = 0
        current_tab: str = ""
        completed_steps: List[str] = field(default_factory=list)
        user_actions_log: List[Dict] = field(default_factory=list)
        mistakes_count: int = 0
        successful_models: int = 0
        last_model_name: str = ""
        hint_mode: str = "hybrid"

    @dataclass
    class TrainingHistory:
        train_loss: List[float] = field(default_factory=list)
        val_loss: List[float] = field(default_factory=list)
        train_metric: List[float] = field(default_factory=list)
        val_metric: List[float] = field(default_factory=list)
        epochs_completed: int = 0
        best_epoch: int = 0
        best_val_loss: float = float("inf")
        total_time_seconds: float = 0.0
        sample_predictions: List[Dict] = field(default_factory=list)

    @dataclass
    class ModelConfig:
        format_version: str = "2.0"
        model_name: str = "Модель"
        architecture: str = "mlp"
        data_type: str = "numeric"
        task_type: str = "regression"


# ---------------------------------------------------------------------------
# ЗАГЛУШКА ИИ-БЭКЕНДА
# ---------------------------------------------------------------------------
# В будущем сюда подключится локальная GGUF-модель или другая LLM.
# Сейчас все методы возвращают None / кидают NotImplementedError,
# но структура полностью готова к интеграции.
class LLBackendStub:
    """Заглушка локального ИИ-бэкенда (например, GGUF).

    Когда появится реальная модель, достаточно заменить этот класс
    на настоящую реализацию, сохранив сигнатуры методов.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.available = False  # Пока ИИ не подключён

    def is_available(self) -> bool:
        """Доступен ли ИИ-бэкенд."""
        return self.available

    def load(self) -> bool:
        """Загрузить модель в память. Возвращает успех."""
        # Заглушка: модель ещё не выбрана
        logger.debug("LLBackendStub.load() вызван, но модель не подключена.")
        return False

    def generate(self, prompt: str) -> Optional[str]:
        """Сгенерировать текст по промпту."""
        raise NotImplementedError("ИИ-бэкенд не подключён.")

    def analyze_history(self, history: TrainingHistory) -> Optional[str]:
        """Проанализировать историю обучения и вернуть комментарий."""
        raise NotImplementedError("ИИ-бэкенд не подключён.")

    def comment_prediction(
        self, input_data: Any, expected: Any, predicted: Any
    ) -> Optional[str]:
        """Прокомментировать результат предсказания."""
        raise NotImplementedError("ИИ-бэкенд не подключён.")

    def suggest_next_step(self, session: SessionState) -> Optional[str]:
        """Предложить следующий шаг на основе истории сеанса."""
        raise NotImplementedError("ИИ-бэкенд не подключён.")


# ---------------------------------------------------------------------------
# ОСНОВНОЙ КЛАСС КУРАТОРА
# ---------------------------------------------------------------------------
class Curator:
    """Фоновый аналитик с личностью.

    Используется:
      • ``gui/panels/monitoring_panel.py`` — комментарии к графикам.
      • ``gui/panels/training_panel.py``   — имена моделей.
      • ``gui/panels/analysis_panel.py``   — эмоциональная обёртка отчёта.
      • ``gui/panels/project_panel.py``    — генерация имён.
    """

    # ------------------------------------------------------------------
    # ГЕНЕРАТОР ВЕСЁЛЫХ ИМЁН
    # ------------------------------------------------------------------
    FUNNY_NAMES: List[str] = [
        "Мозгожуй-3000",
        "Умник-2000",
        "НейроВася",
        "Гендальф Серый",
        "Скайнет-младший",
        "Пиксель",
        "Байт",
        "Тензорчик",
        "Градиентик",
        "Эпоха",
        "Лоссик",
        "Свёрточка",
        "Рекуррентик",
        "Трансформерчик",
        "Оракул",
        "Провидец",
        "Архитектор",
        "Демиург",
        "Отличник",
        "Ботаник",
        "Вундеркинд",
        "Профессор",
        "Шустрик",
        "Мечтатель",
        "Экспериментатор",
        "Мыслитель",
        "Знайка",
        "Почемучка",
        "Мудрец",
        "Искра",
    ]

    NAME_ADJECTIVES: List[str] = [
        "Быстрый",
        "Мудрый",
        "Смелый",
        "Тихий",
        "Громкий",
        "Хитрый",
        "Добрый",
        "Строгий",
        "Весёлый",
        "Задумчивый",
        "Любопытный",
        "Отважный",
        "Скромный",
        "Гениальный",
        "Пушистый",
        "Блестящий",
    ]

    NAME_NOUNS: List[str] = [
        "Нейрон",
        "Тензор",
        "Градиент",
        "Эмбеддинг",
        "Слой",
        "Батч",
        "Эпох",
        "Лосс",
        "Вектор",
        "Матрица",
        "Фильтр",
        "Коррелятор",
        "Активатор",
        "Регуляризатор",
        "Оптимизатор",
        "Классификатор",
    ]

    # ------------------------------------------------------------------
    # ИНТЕРЕСНЫЕ ФАКТЫ
    # ------------------------------------------------------------------
    FUN_FACTS: List[str] = [
        "Знаете ли вы? Первый перцептрон Фрэнк Розенблатт создал в 1957 году. "
        "Он умел только различать простые фигуры.",
        "В 2017 году статья «Attention Is All You Need» произвела революцию. "
        "Из неё выросли GPT и все современные чат-боты.",
        "Один трансформер на 175 миллиардов параметров потребляет за год обучения "
        "примерно как 5 автомобилей.",
        "Нейросети не понимают числа так, как люди. Для них это просто векторы "
        "в многомерном пространстве.",
        "Слово «эпоха» означает один полный проход нейросети по всем обучающим данным.",
        "Dropout случайно «выключает» нейроны во время обучения. Это как заставить "
        "студента сдавать экзамен с завязанными глазами — так знания становятся крепче.",
        "Свёрточные сети (CNN) изначально создавались для распознавания кошек и собак.",
        "Рекуррентные сети «читают» текст слева направо, как человек, но могут "
        "забыть начало длинного предложения.",
        "Механизм внимания позволяет Трансформеру смотреть на всё предложение сразу, "
        "а не по одному слову.",
        "Автоэнкодеры учатся сжимать данные, а потом восстанавливать их. "
        "Как архиватор, только для признаков.",
        "Самые большие нейросети содержат больше параметров, чем нейронов в мозге человека.",
        "Обучение с подкреплением — это как дрессировка: сеть получает награду "
        "за правильные действия.",
        "Нейросети могут ошибаться в простых задачах, но блестяще решать сложные. "
        "Всё зависит от данных.",
        "Термин «переобучение» означает, что сеть выучила примеры наизусть, "
        "но не поняла закономерность.",
        "Градиентный спуск — это как спуск с горы вслепую: на каждом шаге сеть "
        "ищет, где склон круче вниз.",
        "Скорость обучения (LR) — это размер шага при спуске с горы. "
        "Слишком большой шаг — и проскочишь долину.",
        "Функция активации решает, «загорится» ли нейрон. Без неё сеть была бы "
        "просто линейной регрессией.",
        "Batch size — это сколько примеров сеть видит за один присест. "
        "Как размер порции еды.",
        "Валидационная выборка — это как пробный экзамен перед настоящим. "
        "На ней сеть не учится, а только проверяется.",
        "Нейросети не умеют «думать» в человеческом смысле. Они находят "
        "статистические закономерности в данных.",
        "Ранние нейросети обучались неделями на одном компьютере. "
        "Сейчас тренировка маленькой модели занимает минуты.",
        "Токенизация — это разбиение текста на кусочки. Для символьной модели "
        "каждая буква — отдельный токен.",
        "Модель может знать миллионы параметров, но всё равно ошибиться в 2+2, "
        "если её учили не на том.",
        "Нейросети любят данные. Много данных. Чем больше, тем лучше.",
        "Шум в данных — как помехи в радио. Иногда он мешает, а иногда помогает "
        "не переобучиться.",
    ]

    # ------------------------------------------------------------------
    # ЭМОЦИОНАЛЬНЫЕ ШАБЛОНЫ ДЛЯ АНАЛИЗА В РЕАЛЬНОМ ВРЕМЕНИ
    # ------------------------------------------------------------------
    # Каждая ситуация имеет несколько вариантов, чтобы комментарии не повторялись.
    REALTIME_TEMPLATES: Dict[str, List[str]] = {
        "start": [
            "👀 Сеть делает первые шаги. Пока рано судить, подождём 2–3 эпохи.",
            "🌱 Начинаем! Нейросеть только проснулась и смотрит на данные.",
            "🚀 Поехали! Первые эпохи — как разминка перед марафоном.",
        ],
        "crash": [
            "🚨 Авария! Ошибка улетела в бесконечность! Жмите Стоп! "
            "Скорее всего, скорость обучения слишком высокая.",
            "💥 Бум! Графики ушли в космос. Это перебор. "
            "Попробуйте уменьшить «Скорость обучения (LR)» в 10 раз.",
            "🆘 SOS! Loss стал бесконечным. Сеть переполнена эмоциями. "
            "Нужен перерыв и LR поменьше.",
        ],
        "overfit": [
            "⚠️ Сеть начала зубрить! Оранжевый график пополз вверх. "
            "Может, остановимся и посмотрим, что она выучила?",
            "📚 Хм, модель заучивает примеры наизусть. "
            "Это как читать ответы в конце учебника. Попробуйте добавить Dropout.",
            "🔍 Вал-ошибка растёт — сеть переобучается. "
            "Ей нужны более разнообразные данные или регуляризация.",
        ],
        "plateau": [
            "🐢 Хм, графики стали горизонтальными. Сеть задумалась... или застряла. "
            "Подождём ещё 5 эпох или увеличим модель?",
            "⏳ Затишье. Ошибка почти не меняется. "
            "Иногда сети нужно больше времени, а иногда больше нейронов.",
            "🤔 Плато. Сеть достигла предела своей текущей архитектуры? "
            "Попробуйте добавить слой или увеличить эмбеддинг.",
        ],
        "progress": [
            "✅ Отлично! Ошибка плавно снижается. Сеть понимает закономерности.",
            "📉 Красота! Графики ползут вниз. Модель на верном пути.",
            "🎉 Прогресс! Нейросеть уже улавливает суть задачи.",
        ],
        "good": [
            "🌟 Блестяще! Вал-ошибка ниже тренировочной. Модель не зубрит, а понимает!",
            "🏆 Отличный результат! Сеть обобщает, а не заучивает.",
            "💚 Всё идёт по плану. Модель в прекрасной форме.",
        ],
        "waiting": [
            "⏳ Ждём... Нейросети нужно время, чтобы «переварить» данные.",
            "🍵 Самое время выпить чаю. Обучение — это надолго.",
            "🧘 Терпение. Хорошие модели не рождаются за секунду.",
        ],
    }

    # ------------------------------------------------------------------
    # ШАБЛОНЫ КОММЕНТАРИЕВ К ПРЕДСКАЗАНИЯМ
    # ------------------------------------------------------------------
    PREDICTION_PERFECT: List[str] = [
        "🎯 В яблочко! Модель справилась идеально.",
        "✨ Безупречно! Сеть щёлкает такие задачи как орешки.",
        "🏅 Идеальное предсказание! Можно гордиться.",
        "🌟 Точно в цель! Модель выучила закономерность.",
    ]

    PREDICTION_CLOSE: List[str] = [
        "🤏 Почти! Модель близка к истине, но чуть-чуть промахнулась.",
        "🎲 Хорошая попытка! Ошибка совсем небольшая.",
        "📏 Модель на верном пути, но точность можно подтянуть.",
        "🔍 Близко. Ещё пара эпох — и будет идеально.",
    ]

    PREDICTION_WRONG: List[str] = [
        "🤷 Что-то пошло не так. Модель пока путается.",
        "🌀 Хм, ответ мимо. Сеть ещё учится.",
        "📉 Неверно. Но это нормально — ошибки учат лучше всего.",
        "🧩 Модель пока не сложила пазл. Попробуем ещё.",
    ]

    PREDICTION_FUNNY: List[str] = [
        "{inp}={pred}? Что? Тебе учиться и учиться!",
        "Ожидали {exp}, а получили {pred}. Смело, но неверно.",
        "Модель ответила {pred}. Это... интересный взгляд на математику.",
        "{pred} вместо {exp}? Нейросеть, ты что, издеваешься?",
        "Ну, {pred} — это почти как {exp}, если прищуриться. Но нет.",
    ]

    # ------------------------------------------------------------------
    # ИНИЦИАЛИЗАЦИЯ
    # ------------------------------------------------------------------
    def __init__(self, session: Optional[SessionState] = None):
        self.session = session or SessionState()
        self.llm_backend: Optional[LLBackendStub] = LLBackendStub()
        self._last_fact_index = -1
        self._last_realtime_template = ""
        self._start_time = time.time()
        logger.info("Куратор инициализирован. Режим подсказок: %s", self.session.hint_mode)

    # ------------------------------------------------------------------
    # ПРОВЕРКА РЕЖИМА ПОДСКАЗОК
    # ------------------------------------------------------------------
    def _should_speak(self) -> bool:
        """Возвращает False, если куратор должен молчать."""
        return self.session.hint_mode != "none"

    def _try_ai(self, func_name: str, *args, **kwargs) -> Optional[str]:
        """Попытка получить ответ от ИИ-бэкенда."""
        if not self.llm_backend or not self.llm_backend.is_available():
            return None
        if self.session.hint_mode not in ("ai", "hybrid"):
            return None
        try:
            method = getattr(self.llm_backend, func_name)
            return method(*args, **kwargs)
        except NotImplementedError:
            return None
        except Exception as e:  # pragma: no cover
            logger.warning("ИИ-бэкенд упал: %s", e)
            return None

    # ------------------------------------------------------------------
    # ГЕНЕРАЦИЯ ИМЕНИ МОДЕЛИ
    # ------------------------------------------------------------------
    def generate_model_name(self) -> str:
        """Генерирует весёлое имя для новой модели.

        Сначала пытается использовать ИИ, потом правила.
        """
        if not self._should_speak():
            return "Модель"

        ai_name = self._try_ai("generate_model_name")
        if ai_name:
            return ai_name

        # 30% шанс — полностью случайная комбинация
        if random.random() < 0.3:
            adj = random.choice(self.NAME_ADJECTIVES)
            noun = random.choice(self.NAME_NOUNS)
            num = random.randint(1, 999)
            return f"{adj} {noun}-{num}"

        return random.choice(self.FUNNY_NAMES)

    # ------------------------------------------------------------------
    # АНАЛИЗ В РЕАЛЬНОМ ВРЕМЕНИ
    # ------------------------------------------------------------------
    def analyze_realtime(self, history: TrainingHistory) -> str:
        """Вызывается каждую эпоху. Возвращает живой комментарий к графикам."""
        if not self._should_speak():
            return ""

        # Попытка ИИ
        ai_comment = self._try_ai("analyze_history", history)
        if ai_comment:
            prefix = "🤖 " if self.session.hint_mode == "hybrid" else ""
            return f"{prefix}{ai_comment}"

        train_loss = history.train_loss
        val_loss = history.val_loss
        epochs = len(train_loss)

        # 0. Начало
        if epochs < 2:
            return self._pick_template("start")

        # 1. Авария (NaN / бесконечность)
        last_val = val_loss[-1] if val_loss else train_loss[-1]
        if last_val > 1000 or str(last_val).lower() == "nan":
            return self._pick_template("crash")

        # 2. Переобучение (вал растёт 3 эпохи подряд)
        if len(val_loss) >= 3:
            if val_loss[-1] > val_loss[-2] > val_loss[-3]:
                return self._pick_template("overfit")

        # 3. Плато
        if epochs > 5:
            progress = train_loss[-5] - train_loss[-1]
            if 0 < progress < 0.0001:
                return self._pick_template("plateau")

        # 4. Отличное обобщение
        if val_loss and train_loss and val_loss[-1] < train_loss[-1]:
            return self._pick_template("good")

        # 5. Нормальный прогресс
        if epochs > 1 and train_loss[-1] < train_loss[-2]:
            return self._pick_template("progress")

        # 6. Ожидание
        return self._pick_template("waiting")

    def _pick_template(self, category: str) -> str:
        """Выбирает случайный шаблон, избегая повтора предыдущего."""
        templates = self.REALTIME_TEMPLATES.get(category, [])
        if not templates:
            return ""
        # Если только один шаблон — возвращаем его
        if len(templates) == 1:
            return templates[0]
        choice = random.choice(templates)
        attempts = 0
        while choice == self._last_realtime_template and attempts < 5:
            choice = random.choice(templates)
            attempts += 1
        self._last_realtime_template = choice
        return choice

    # ------------------------------------------------------------------
    # КОММЕНТАРИЙ К ПРЕДСКАЗАНИЮ
    # ------------------------------------------------------------------
    def comment_prediction(
        self,
        input_data: Any,
        expected: Any,
        predicted: Any,
        is_correct: Optional[bool] = None,
    ) -> str:
        """Комментирует результат предсказания модели.

        Аргументы:
            input_data: вход (строка или список чисел).
            expected: ожидаемый ответ.
            predicted: ответ модели.
            is_correct: если известно, правильно ли (для числовых задач).
        """
        if not self._should_speak():
            return ""

        ai_comment = self._try_ai("comment_prediction", input_data, expected, predicted)
        if ai_comment:
            prefix = "🤖 " if self.session.hint_mode == "hybrid" else ""
            return f"{prefix}{ai_comment}"

        # Определяем тип комментария
        if is_correct is None:
            # Пытаемся сравнить как числа, иначе как строки
            try:
                exp_num = float(str(expected).strip())
                pred_num = float(str(predicted).strip())
                diff = abs(exp_num - pred_num)
                if diff < 1e-6:
                    is_correct = True
                elif diff <= max(abs(exp_num) * 0.1, 1.0):
                    is_correct = False  # близко
                else:
                    is_correct = False
            except (ValueError, TypeError):
                is_correct = str(expected).strip() == str(predicted).strip()

        if is_correct is True:
            return random.choice(self.PREDICTION_PERFECT)

        # Проверяем, насколько близко
        close = False
        try:
            exp_num = float(str(expected).strip())
            pred_num = float(str(predicted).strip())
            if abs(exp_num - pred_num) <= max(abs(exp_num) * 0.2, 2.0):
                close = True
        except (ValueError, TypeError):
            pass

        if close:
            return random.choice(self.PREDICTION_CLOSE)

        # Случайный смешной комментарий с подстановкой
        template = random.choice(self.PREDICTION_FUNNY)
        try:
            return template.format(inp=input_data, exp=expected, pred=predicted)
        except (KeyError, IndexError):
            return random.choice(self.PREDICTION_WRONG)

    # ------------------------------------------------------------------
    # ИНТЕРЕСНЫЙ ФАКТ
    # ------------------------------------------------------------------
    def get_random_fact(self) -> str:
        """Возвращает интересный факт, избегая повторений подряд."""
        if not self._should_speak():
            return ""
        idx = random.randrange(len(self.FUN_FACTS))
        if idx == self._last_fact_index:
            idx = (idx + 1) % len(self.FUN_FACTS)
        self._last_fact_index = idx
        return self.FUN_FACTS[idx]

    # ------------------------------------------------------------------
    # ПОДДЕРЖКА ПРИ ПЛАТО / ДОЛГОМ ОЖИДАНИИ
    # ------------------------------------------------------------------
    def get_encouragement(self, epochs_waiting: int = 0) -> str:
        """Подбадривает пользователя, когда графики не меняются."""
        if not self._should_speak():
            return ""
        phrases = [
            "Держитесь! Даже самые умные модели иногда задумываются.",
            "Плато — это не конец, а плато. Иногда нужно просто подождать.",
            "Нейросети, как и люди, иногда берут паузу, чтобы осмыслить.",
            "Если графики молчат — значит, сеть копит знания.",
            "Терпение и труд всё перетрут. Даже градиенты.",
        ]
        if epochs_waiting > 10:
            return "Уже долго ждём... Может, стоит увеличить модель или изменить скорость обучения?"
        return random.choice(phrases)

    # ------------------------------------------------------------------
    # ПРАЗДНОВАНИЕ УСПЕХА
    # ------------------------------------------------------------------
    def celebrate_success(self, metric_value: float) -> str:
        """Комментарий при завершении обучения с хорошим результатом."""
        if not self._should_speak():
            return ""
        if metric_value >= 0.9:
            return (
                "🎆 Это триумф! Модель показывает блестящий результат. "
                "Можно смело сохранять и хвастаться!"
            )
        if metric_value >= 0.7:
            return (
                "🎉 Хороший результат! Модель справляется с задачей. "
                "Есть куда расти, но уже можно тестировать."
            )
        return (
            "👍 Обучение завершено. Результат скромный, но это опыт. "
            "Попробуйте изменить параметры или добавить данных."
        )

    # ------------------------------------------------------------------
    # РЕКОМЕНДАЦИЯ СЛЕДУЮЩЕГО ШАГА
    # ------------------------------------------------------------------
    def suggest_next_step(self, session: Optional[SessionState] = None) -> str:
        """Советует, что делать дальше, на основе истории сеансов."""
        sess = session or self.session
        if not self._should_speak():
            return ""

        ai_suggestion = self._try_ai("suggest_next_step", sess)
        if ai_suggestion:
            return ai_suggestion

        steps = sess.completed_steps
        mistakes = sess.mistakes_count
        successes = sess.successful_models

        # Если пользователь новичок
        if sess.total_sessions <= 2:
            if "dataset_generated" not in steps:
                return "Начните с вкладки «Генератор» — создайте простой датасет."
            if "dataset_split" not in steps:
                return "Данные созданы. Перейдите во вкладку «Данные» и примените разбиение."
            if "model_created" not in steps:
                return "Теперь создайте модель во вкладке «Архитектура». Начните с Transformer."
            if "training_started" not in steps:
                return "Модель готова! Запустите обучение во вкладке «Обучение»."
            return "Следите за графиками в «Мониторинге». Не бойтесь останавливать и пробовать снова."

        # Если много ошибок
        if mistakes > 5:
            return (
                "Похоже, эксперименты идут полным ходом! Если что-то не получается, "
                "попробуйте вернуться к более простым настройкам."
            )

        # Если уже есть успехи
        if successes > 0:
            return (
                "Вы уже обучили несколько моделей. Попробуйте усложнить задачу: "
                "другая архитектура, больше данных, новые гиперпараметры."
            )

        # Общий совет
        return (
            "Продолжайте экспериментировать. Каждая попытка делает модель лучше, "
            "а вас — опытнее."
        )

    # ------------------------------------------------------------------
    # ФИНАЛЬНЫЙ АНАЛИЗ (ОБЁРТКА НАД EXPLAINER)
    # ------------------------------------------------------------------
    def analyze_training_results(self, history: TrainingHistory) -> List[str]:
        """Возвращает список эмоциональных рекомендаций после обучения.

        Техническую часть можно получить из ``core.explainer.Explainer``,
        а куратор добавляет живости и персонализации.
        """
        if not self._should_speak():
            return []

        comments: List[str] = []

        if not history.train_loss:
            comments.append("Обучение ещё не начиналось. Жду первых шагов!")
            return comments

        epochs = len(history.train_loss)
        final_train = history.train_loss[-1]
        final_val = history.val_loss[-1] if history.val_loss else final_train

        # Переобучение
        if final_val > final_train * 1.5:
            comments.append(
                "🚨 Сеть явно переобучилась. Она вызубрила тренировочные примеры, "
                "но растерялась на новых. Добавьте Dropout или уменьшите число эпох."
            )

        # Плато
        if epochs > 5 and (history.train_loss[-5] - final_train) < 0.0001:
            comments.append(
                "🐢 Обучение упёрлось в потолок. Возможно, модели не хватает ёмкости. "
                "Попробуйте добавить слой или увеличить эмбеддинг."
            )

        # Хороший результат
        if final_val < final_train:
            comments.append(
                "🌟 Отличный знак: ошибка на проверке ниже, чем на обучении. "
                "Модель умеет обобщать!"
            )

        # Если всё плохо
        if not comments:
            comments.append(
                "🧠 Процесс завершён. Посмотрите на графики и решите: "
                "нужно ли дообучить, упростить или оставить как есть."
            )

        return comments

    # ------------------------------------------------------------------
    # ЗАГЛУШКИ ДЛЯ БУДУЩИХ ФУНКЦИЙ
    # ------------------------------------------------------------------
    def generate_dataset_with_ai(self, task: str, num_samples: int) -> Optional[Any]:
        """Заглушка: генерация датасета через ИИ (например, пары переводов)."""
        logger.debug("generate_dataset_with_ai(%s) — заглушка.", task)
        return None

    def get_pretrained_model_path(self, name: str) -> Optional[str]:
        """Заглушка: путь к предзагруженной модели (речь, перевод и т.д.)."""
        logger.debug("get_pretrained_model_path(%s) — заглушка.", name)
        return None