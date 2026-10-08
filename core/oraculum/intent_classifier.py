"""
core/oraculum/intent_classifier.py
==================================
Классификатор намерений пользователя для Оракула.

ОТВЕТСТВЕННОСТЬ:
  • Определение намерения по текстовому запросу (правила и ключевые слова).
  • Учёт контекста: ошибки, отсутствие модели/данных усиливают намерения.
"""

from __future__ import annotations


class Intent:
    """Возможные намерения пользователя."""
    HELP = "help"
    EXPLAIN = "explain"
    TROUBLESHOOT = "troubleshoot"
    SUGGEST = "suggest"
    GREETING = "greeting"
    GENERAL = "general"
    ASK_MODEL = "ask_model"
    ASK_DATA = "ask_data"
    ASK_TRAINING = "ask_training"
    STATUS = "status"   # "что происходит?", "статус", "что сейчас происходит"

    ALL = (
        HELP, EXPLAIN, TROUBLESHOOT, SUGGEST, GREETING,
        GENERAL, ASK_MODEL, ASK_DATA, ASK_TRAINING, STATUS,
    )


class IntentClassifier:
    """Классифицирует намерение по ключевым словам и контексту."""

    def __init__(self):
        self.keywords = {
            Intent.HELP: ["помоги", "подскажи", "не знаю", "что делать", "help",
                          "помощ", "научи"],
            Intent.EXPLAIN: ["почему", "зачем", "объясни", "что значит", "как работает",
                             "что такое", "расскажи про", "объясн", "зачем нужен"],
            Intent.TROUBLESHOOT: ["ошибка", "не работает", "не получается", "сбой",
                                  "падение", "краш", "не запускается", "не грузит",
                                  "проблем", "не выходит", "поломал", "не работает"],
            Intent.SUGGEST: ["что дальше", "следующий шаг", "рекомендуй", "посоветуй",
                             "совет", "дальше"],
            Intent.GREETING: ["привет", "здравствуй", "добрый день", "hello", "hi",
                              "здорово", "добрый вечер", "доброе утро"],
            Intent.ASK_MODEL: ["модель", "нейросеть", "архитектура", "создать модель",
                               "сеть", "модел", "трансформер", "нейросет",
                               "создай модель", "сделай модель", "слоёв", "слой",
                               "attention", "внимание", "эмбеддинг", "mlp", "cnn",
                               "lstm", "rnn", "gru", "резнет", "resnet", "vit"],
            Intent.ASK_DATA: ["данные", "датасет", "загрузить", "сгенерировать",
                              "разбиение", "split", "данн", "датасет", "загруз",
                              "сгенерир", "генерац"],
            Intent.ASK_TRAINING: ["обучение", "тренировка", "эпоха", "loss", "точность",
                                  "train", "learn", "обуч", "тренир", "эпох",
                                  "научить", "lr", "гиперпараметр", "планировщик",
                                  "scheduler", "оптимизатор", "батч", "batch"],
            Intent.STATUS: ["что происходит", "что происходит сейчас", "что случилось",
                            "что сейчас", "статус", "в чём дело", "в чем дело",
                            "что вообще", "что происходит в программе", "где я",
                            "какое состояние", "обстановка"],
        }

    def classify(self, query: str, context=None) -> str:
        """
        Определяет намерение. context — AgentContext (может быть None).
        Правила:
        1. Точные ключевые слова имеют приоритет.
        2. Специфичные намерения (модель/данные/обучение) проверяются
           перед общими (explain/help).
        3. При многих ошибках — troubleshoot.
        4. Вопросы о текущем состоянии — status.
        """
        query_lower = (query or "").lower().strip()
        if not query_lower:
            return Intent.GENERAL

        # 1. Проблемы и объяснения — самые частые запросы
        for intent in (Intent.TROUBLESHOOT, Intent.EXPLAIN):
            if any(w in query_lower for w in self.keywords[intent]):
                return intent

        # 2. Статус (что происходит) — до специфичных тем, чтобы
        #    "что происходит с моделью?" не ушло в ask_model
        if any(w in query_lower for w in self.keywords[Intent.STATUS]):
            return Intent.STATUS

        # 3. Специфичные темы (модель / данные / обучение)
        for intent in (Intent.ASK_MODEL, Intent.ASK_DATA, Intent.ASK_TRAINING):
            if any(w in query_lower for w in self.keywords[intent]):
                return intent

        # 4. Остальные общие намерения
        for intent in (Intent.SUGGEST, Intent.GREETING, Intent.HELP):
            if any(w in query_lower for w in self.keywords[intent]):
                return intent

        # 5. Контекстные признаки
        if context is not None:
            errors = getattr(context, "errors_last_hour", 0)
            if errors > 2:
                return Intent.TROUBLESHOOT

        return Intent.GENERAL
