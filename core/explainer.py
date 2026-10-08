"""
core/explainer.py
=================
Интеллектуальный анализатор обучения: подсказки, советы, разбор полётов.
Версия формата: 2.0.

ОТВЕТСТВЕННОСТЬ:
- Анализ графиков обучения в реальном времени (каждую эпоху).
- Финальный разбор после обучения с конкретными рекомендациями.
- Комментарии к предсказаниям модели («45-68=67? Тебе учиться и учиться!»).
- Сравнение нескольких попыток обучения.
- Генерация интересных фактов для пауз.

ВХОДЫ:
- TrainingHistory / dict (старый формат)
- ModelConfig / dict
- Список предсказаний

ВЫХОДЫ:
- RealtimeAnalysis (для мониторинга)
- FinalReport (для вкладки анализа)
- Recommendation (конкретное действие)
- PredictionComment (комментарий к примеру)

ЗАВИСИМОСТИ:
- core.contracts (типы)
- core.logger

НЕ ЗАВИСИТ ОТ:
- GUI
- trainer.py
- model_factory.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from core.logger import get_logger

logger = get_logger(__name__)

# ============================================================
# КОНСТАНТЫ И ВСПОМОГАТЕЛЬНЫЕ ДАННЫЕ
# ============================================================

FORMAT_VERSION = "2.0"

# Интересные факты для пауз и ожидания
FUN_FACTS: List[str] = [
    "Первый перцептрон Фрэнк Розенблатт создал в 1957 году. Он умел только различать фигуры.",
    "В 2017 году статья «Attention Is All You Need» произвела революцию. Из неё выросли GPT и все современные чат-боты.",
    "Один трансформер на 175 миллиардов параметров потребляет как 5 автомобилей за год обучения.",
    "Слово «нейросеть» пришло из нейробиологии: искусственные нейроны моделируют реальные клетки мозга.",
    "Механизм внимания (attention) позволил сетям «смотреть» на весь текст сразу, а не читать по букве.",
    "Первая свёрточная сеть LeNet-5 в 1998 году распознавала рукописные цифры на чеках.",
    "Обучение с подкреплением (RL) научило AlphaGo побеждать чемпионов мира в го.",
    "Автоэнкодеры умеют «сжимать» изображения в крошечный код, а потом восстанавливать их.",
    "Дропаут (dropout) был вдохновлён идеей: если случайно «отключать» нейроны, сеть учится быть устойчивее.",
    "Функция активации ReLU появилась в 2010 году и ускорила обучение глубоких сетей в разы.",
    "Трансформеры не используют рекуррентность: они параллельно обрабатывают все символы сразу.",
    "В 2022 году модели диффузии научились генерировать фотореалистичные картинки из текста.",
    "Градиентный спуск — это как спуск с горы вслепую: сеть щупает склон и делает шаг вниз.",
    "Переобучение — как зубрёжка к экзамену: на знакомых задачах пятёрка, на новых — двойка.",
    "Пакетная нормализация (BatchNorm) позволяет обучать сети в 10 раз быстрее.",
    "Свёрточные сети видят изображения как набор «фильтров»: один ищет края, другой — углы.",
    "Рекуррентные сети читают текст как человек: слева направо, запоминая контекст.",
    "Слово «эпоха» означает один полный проход по всем данным обучения.",
    "Чем больше словарь модели, тем больше символов она умеет распознавать.",
    "Латентное пространство — это «воображение» нейросети: сжатое представление данных.",
]

# Забавные комментарии к ошибкам
ERROR_COMMENTS: List[str] = [
    "Тебе учиться и учиться!",
    "Хм, это неожиданно...",
    "Ну, почти!",
    "Сеть ещё путает цифры.",
    "Бывает. Главное — не сдаваться!",
    "Ой, а это что было?",
    "Модель явно не выспалась.",
    "Попробуем ещё раз!",
    "Это не баг, это фича... шутка.",
    "Ну, зато мы знаем, где проблема!",
]

# Забавные комментарии к успехам
SUCCESS_COMMENTS: List[str] = [
    "Отлично!",
    "Вот это да!",
    "Мозги работают!",
    "Прямо в яблочко!",
    "Сеть молодец!",
    "Идеально!",
    "Так держать!",
    "Вот это результат!",
]

# ============================================================
# СТРУКТУРЫ ДАННЫХ (результирующие)
# ============================================================

class Severity(str, Enum):
    """Серьёзность ситуации для мониторинга."""
    INFO = "info"         # Всё хорошо
    SUCCESS = "success"   # Отличный прогресс
    WARNING = "warning"   # Внимание
    DANGER = "danger"     # Авария
    NEUTRAL = "neutral"   # Нейтрально / жду данных


@dataclass
class RealtimeAnalysis:
    """Результат анализа в реальном времени (для панели мониторинга)."""
    severity: Severity = Severity.NEUTRAL
    message: str = "👀 Наблюдаю за первыми шагами нейросети..."
    icon: str = "👀"
    color_key: str = "neutral"  # для GUI: neutral, green, yellow, red
    detail: str = ""            # подробное объяснение (для тултипа)


@dataclass
class Recommendation:
    """Конкретная рекомендация с действием."""
    title: str
    description: str
    action_tab: str = ""         # куда перейти: "Гиперпараметры", "Архитектура", "Данные"
    action_field: str = ""       # какое поле изменить
    action_value: Any = None     # какое значение поставить
    severity: Severity = Severity.INFO
    auto_apply: bool = False     # можно ли применить автоматически


@dataclass
class PredictionComment:
    """Комментарий к одному предсказанию модели."""
    input_str: str
    expected_str: str
    predicted_str: str
    is_correct: bool
    comment: str                 # забавный или серьёзный комментарий
    detail: str = ""             # техническая деталь


@dataclass
class FinalReport:
    """Полный отчёт после обучения."""
    format_version: str = FORMAT_VERSION
    title: str = ""
    summary_html: str = ""
    recommendations: List[Recommendation] = field(default_factory=list)
    metrics_summary: Dict[str, Any] = field(default_factory=dict)
    comparison_with_previous: Optional[str] = None
    fun_fact: str = ""
    suggested_model_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_version": self.format_version,
            "title": self.title,
            "summary_html": self.summary_html,
            "recommendations": [
                {
                    "title": r.title,
                    "description": r.description,
                    "action_tab": r.action_tab,
                    "action_field": r.action_field,
                    "action_value": r.action_value,
                    "severity": r.severity.value,
                    "auto_apply": r.auto_apply,
                }
                for r in self.recommendations
            ],
            "metrics_summary": self.metrics_summary,
            "comparison_with_previous": self.comparison_with_previous,
            "fun_fact": self.fun_fact,
            "suggested_model_name": self.suggested_model_name,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "FinalReport":
        recs = []
        for r in d.get("recommendations", []):
            recs.append(Recommendation(
                title=r.get("title", ""),
                description=r.get("description", ""),
                action_tab=r.get("action_tab", ""),
                action_field=r.get("action_field", ""),
                action_value=r.get("action_value"),
                severity=Severity(r.get("severity", "info")),
                auto_apply=r.get("auto_apply", False),
            ))
        return FinalReport(
            format_version=d.get("format_version", FORMAT_VERSION),
            title=d.get("title", ""),
            summary_html=d.get("summary_html", ""),
            recommendations=recs,
            metrics_summary=d.get("metrics_summary", {}),
            comparison_with_previous=d.get("comparison_with_previous"),
            fun_fact=d.get("fun_fact", ""),
            suggested_model_name=d.get("suggested_model_name", ""),
        )


# ============================================================
# ОСНОВНОЙ КЛАСС
# ============================================================

class Explainer:
    """
    Интеллектуальный анализатор процесса обучения.
    
    Работает в двух режимах:
    1. Realtime: analyze_realtime() вызывается каждую эпоху.
    2. Final: analyze_training_results() вызывается после завершения.
    
    Также генерирует комментарии к предсказаниям и факты для пауз.
    """

    # ------------------------------------------------------------------
    # 1. АНАЛИЗ В РЕАЛЬНОМ ВРЕМЕНИ
    # ------------------------------------------------------------------

    @staticmethod
    def analyze_realtime(history: Union[Dict, Any], config: Optional[Dict] = None) -> RealtimeAnalysis:
        """
        Анализирует обучение каждую эпоху. Возвращает RealtimeAnalysis.
        
        Args:
            history: TrainingHistory или dict с ключами
                     train_loss, val_loss, train_metric, val_metric
            config: ModelConfig или dict с параметрами модели
        
        Returns:
            RealtimeAnalysis с severity, message, icon, color_key
        """
        # Нормализация входа: принимаем и dataclass, и dict
        h = Explainer._normalize_history(history)
        cfg = config or {}

        train_loss = h.get("train_loss", [])
        val_loss = h.get("val_loss", [])
        epochs = len(train_loss)

        # --- Нет данных ---
        if epochs == 0:
            return RealtimeAnalysis(
                severity=Severity.NEUTRAL,
                message="👀 Жду запуска обучения...",
                icon="👀",
                color_key="neutral",
            )

        # --- Первые эпохи ---
        if epochs < 2:
            return RealtimeAnalysis(
                severity=Severity.NEUTRAL,
                message="👀 Сеть только начала учиться. Ждём данных для анализа...",
                icon="👀",
                color_key="neutral",
                detail="Для выводов нужно хотя бы 2-3 эпохи.",
            )

        last_train = train_loss[-1]
        last_val = val_loss[-1] if val_loss else float('inf')

        # --- 1. АВАРИЯ: взрыв градиента / NaN ---
        if Explainer._is_nan_or_huge(last_val):
            return RealtimeAnalysis(
                severity=Severity.DANGER,
                message=(
                    "🚨 Авария! Ошибка улетела в бесконечность (NaN/Infinity). "
                    "Срочно жмите «Стоп» и уменьшайте «Скорость обучения (LR)»!"
                ),
                icon="🚨",
                color_key="red",
                detail=(
                    "Скорее всего LR слишком высокий. Попробуйте уменьшить в 10 раз. "
                    "Также проверьте «Защита от сбоев (Clip)» — поставьте 1.0."
                ),
            )

        # --- 2. ПЕРЕОБУЧЕНИЕ: Val растёт, Train падает ---
        if len(val_loss) >= 3:
            # Пересечение графиков
            if val_loss[-2] <= train_loss[-2] and val_loss[-1] > train_loss[-1]:
                return RealtimeAnalysis(
                    severity=Severity.WARNING,
                    message=(
                        "⚔️ Внимание! Оранжевый график (Val) пересёк синий (Train) "
                        "и пошёл вверх. Сеть начала зубрить ответы наизусть!"
                    ),
                    icon="⚔️",
                    color_key="yellow",
                    detail=(
                        "Это классическое переобучение. Увеличьте Dropout, "
                        "уменьшите модель или добавьте данных."
                    ),
                )
            # Вал растёт 3 эпохи подряд
            if val_loss[-1] > val_loss[-2] > val_loss[-3]:
                return RealtimeAnalysis(
                    severity=Severity.WARNING,
                    message=(
                        "📈 Ошибка на новых данных стабильно растёт. "
                        "Возможно, «Скорость обучения (LR)» слишком высокая."
                    ),
                    icon="📈",
                    color_key="yellow",
                    detail=(
                        "Сеть «перепрыгивает» правильные ответы. "
                        "Попробуйте уменьшить LR в 2-5 раз."
                    ),
                )

        # --- 3. ПЛАТО: застревание ---
        if epochs > 5:
            progress = train_loss[-5] - train_loss[-1]
            if 0 < progress < 0.0001:
                return RealtimeAnalysis(
                    severity=Severity.WARNING,
                    message=(
                        "🐢 Обучение застряло. Линии стали горизонтальными. "
                        "Сети не хватает «мозгов» или данных."
                    ),
                    icon="🐢",
                    color_key="yellow",
                    detail=(
                        "Попробуйте увеличить количество слоёв, размер эмбеддинга "
                        "или сгенерировать больше примеров."
                    ),
                )

        # --- 4. ОТЛИЧНЫЙ ПРОГРЕСС ---
        if last_val < last_train:
            return RealtimeAnalysis(
                severity=Severity.SUCCESS,
                message="✅ Полёт нормальный. Сеть отлично усваивает закономерности.",
                icon="✅",
                color_key="green",
                detail="Val Loss ниже Train Loss — это хорошо. Сеть не зубрит.",
            )

        # --- 5. НОРМАЛЬНЫЙ ПРОЦЕСС ---
        return RealtimeAnalysis(
            severity=Severity.INFO,
            message="🧠 Идёт процесс обучения. Графики плавно снижаются — хороший знак.",
            icon="🧠",
            color_key="green",
        )

    # ------------------------------------------------------------------
    # 2. ФИНАЛЬНЫЙ АНАЛИЗ
    # ------------------------------------------------------------------

    @staticmethod
    def analyze_training_results(
        history: Union[Dict, Any],
        config: Optional[Dict] = None,
        previous_histories: Optional[List[Union[Dict, Any]]] = None,
    ) -> FinalReport:
        """
        Финальный разбор полётов для вкладки «Анализ».
        
        Args:
            history: TrainingHistory или dict
            config: ModelConfig или dict
            previous_histories: список предыдущих попыток для сравнения
        
        Returns:
            FinalReport с рекомендациями, метриками, сравнением
        """
        h = Explainer._normalize_history(history)
        cfg = config or {}
        train_loss = h.get("train_loss", [])
        val_loss = h.get("val_loss", [])
        val_metric = h.get("val_metric", [])
        train_metric = h.get("train_metric", [])

        recommendations: List[Recommendation] = []
        metrics_summary: Dict[str, Any] = {}
        summary_parts: List[str] = []

        # --- Метрики ---
        if train_loss:
            metrics_summary["final_train_loss"] = train_loss[-1]
            metrics_summary["best_train_loss"] = min(train_loss)
        if val_loss:
            metrics_summary["final_val_loss"] = val_loss[-1]
            metrics_summary["best_val_loss"] = min(val_loss)
            metrics_summary["best_epoch"] = val_loss.index(min(val_loss)) + 1
        if val_metric:
            metrics_summary["final_val_metric"] = val_metric[-1]
            metrics_summary["best_val_metric"] = max(val_metric)
        metrics_summary["epochs_completed"] = len(train_loss)
        metrics_summary["data_type"] = cfg.get("data_type", "unknown")
        metrics_summary["architecture"] = cfg.get("type", "unknown")
        metrics_summary["model_name"] = cfg.get("model_name", "Модель")

        # --- Заголовок ---
        model_name = cfg.get("model_name", "Нейросеть")
        title = f"📊 Отчёт об обучении модели «{model_name}»"

        # --- Переобучение ---
        if val_loss and train_loss and len(val_loss) > 5:
            overfit_ratio = val_loss[-1] / (train_loss[-1] + 1e-9)
            if overfit_ratio > 1.5:
                recommendations.append(Recommendation(
                    title="🚨 Обнаружено сильное переобучение",
                    description=(
                        "Нейросеть вызубрила тренировочные примеры, но завалила "
                        "проверку на незнакомых данных."
                    ),
                    severity=Severity.DANGER,
                ))
                recommendations.append(Recommendation(
                    title="Увеличить штраф за зубрёжку (L2)",
                    description="Поставьте 0.01 или 0.1.",
                    action_tab="Гиперпараметры",
                    action_field="weight_decay",
                    action_value=0.01,
                    auto_apply=True,
                    severity=Severity.WARNING,
                ))
                if cfg.get("data_type") == "text":
                    recommendations.append(Recommendation(
                        title="Уменьшить подсказки учителя",
                        description="Снизьте Teacher Forcing до 0.3.",
                        action_tab="Гиперпараметры",
                        action_field="teacher_forcing_ratio",
                        action_value=0.3,
                        auto_apply=True,
                        severity=Severity.WARNING,
                    ))
                recommendations.append(Recommendation(
                    title="Уменьшить количество эпох",
                    description="Сеть слишком долго смотрела на одни и те же данные.",
                    action_tab="Гиперпараметры",
                    action_field="epochs",
                    action_value=max(10, len(train_loss) // 2),
                    severity=Severity.INFO,
                ))

        # --- Расходимость ---
        if val_loss and len(val_loss) >= 5:
            if val_loss[-1] > val_loss[-5]:
                recommendations.append(Recommendation(
                    title="⚠️ Ошибка нарастает (сеть расходится)",
                    description=(
                        "Вместо того чтобы учиться, сеть ошибается всё сильнее."
                    ),
                    severity=Severity.DANGER,
                ))
                recommendations.append(Recommendation(
                    title="Уменьшить скорость обучения",
                    description="Уменьшите LR в 10 раз.",
                    action_tab="Гиперпараметры",
                    action_field="learning_rate",
                    action_value=cfg.get("learning_rate", 0.001) / 10.0,
                    auto_apply=True,
                    severity=Severity.WARNING,
                ))
                recommendations.append(Recommendation(
                    title="Включить защиту от сбоев (Clip)",
                    description="Поставьте 0.5 или 1.0.",
                    action_tab="Гиперпараметры",
                    action_field="gradient_clipping",
                    action_value=1.0,
                    auto_apply=True,
                    severity=Severity.INFO,
                ))

        # --- Плато ---
        if len(train_loss) > 5 and (train_loss[-5] - train_loss[-1]) < 0.0001:
            recommendations.append(Recommendation(
                title="🐢 Обучение застряло (Плато)",
                description="Сеть перестала «умнеть» и остановилась на одном уровне.",
                severity=Severity.WARNING,
            ))
            if cfg.get("type") in ("transformer", "transformer_seq2seq"):
                recommendations.append(Recommendation(
                    title="Увеличить эмбеддинг",
                    description="Увеличьте d_model в 1.5-2 раза.",
                    action_tab="Архитектура",
                    action_field="embedding_dim",
                    action_value=int(cfg.get("embedding_dim", 128) * 1.5),
                    severity=Severity.INFO,
                ))
            elif cfg.get("type") == "mlp":
                recommendations.append(Recommendation(
                    title="Добавить скрытые слои",
                    description="Увеличьте количество слоёв на 1-2.",
                    action_tab="Архитектура",
                    action_field="hidden_layers",
                    severity=Severity.INFO,
                ))
            recommendations.append(Recommendation(
                title="Сгенерировать больше данных",
                description="Добавьте ещё 10 000–20 000 примеров.",
                action_tab="Генератор",
                severity=Severity.INFO,
            ))

        # --- Анализ точности для текста ---
        if val_metric:
            last_metric = val_metric[-1]
            if 0 <= last_metric <= 1:
                if last_metric < 0.3:
                    recommendations.append(Recommendation(
                        title="📝 Точность очень низкая",
                        description=(
                            f"Точность предсказания: {last_metric:.1%}. "
                            "Модель не может связать вопрос и ответ."
                        ),
                        severity=Severity.DANGER,
                    ))
                    recommendations.append(Recommendation(
                        title="Увеличить эмбеддинг и головы",
                        description="Во вкладке «Архитектура» увеличьте параметры.",
                        action_tab="Архитектура",
                        severity=Severity.INFO,
                    ))
                elif last_metric < 0.7:
                    recommendations.append(Recommendation(
                        title=f"📝 Точность: {last_metric:.1%}",
                        description=(
                            "Сеть уже понимает логику, но часто ошибается. "
                            "Дайте ей больше эпох."
                        ),
                        severity=Severity.INFO,
                    ))
                    recommendations.append(Recommendation(
                        title="Увеличить количество эпох",
                        description="Добавьте ещё 10-20 эпох.",
                        action_tab="Гиперпараметры",
                        action_field="epochs",
                        action_value=cfg.get("epochs", 50) + 20,
                        auto_apply=True,
                        severity=Severity.INFO,
                    ))
                elif last_metric >= 0.9:
                    recommendations.append(Recommendation(
                        title=f"🌟 Отличная точность: {last_metric:.1%}!",
                        description="Модель почти идеально читает и пишет символы.",
                        severity=Severity.SUCCESS,
                    ))

        # --- Если всё хорошо ---
        if not recommendations:
            recommendations.append(Recommendation(
                title="🌟 Блестящий результат!",
                description=(
                    "Графики стабильно идут вниз, оранжевая линия близка к синей. "
                    "Нейросеть успешно нашла закономерности. Можете переходить к тестированию!"
                ),
                severity=Severity.SUCCESS,
            ))

        # --- Сравнение с предыдущими попытками ---
        comparison = None
        if previous_histories:
            comparison = Explainer._compare_attempts(
                [Explainer._normalize_history(hh) for hh in previous_histories] + [h]
            )

        # --- Сборка HTML ---
        summary_parts.append(f"<h3>{title}</h3>")
        if metrics_summary.get("epochs_completed"):
            summary_parts.append(
                f"<p>Эпох завершено: <b>{metrics_summary['epochs_completed']}</b></p>"
            )
        if metrics_summary.get("final_val_loss") is not None:
            summary_parts.append(
                f"<p>Финальная ошибка (Val Loss): <b>{metrics_summary['final_val_loss']:.4f}</b></p>"
            )
        if metrics_summary.get("final_val_metric") is not None:
            m = metrics_summary["final_val_metric"]
            if 0 <= m <= 1:
                summary_parts.append(
                    f"<p>Финальная точность: <b>{m:.1%}</b></p>"
                )
            else:
                summary_parts.append(
                    f"<p>Финальная метрика: <b>{m:.4f}</b></p>"
                )
        if comparison:
            summary_parts.append(f"<hr><p>{comparison}</p>")

        summary_html = "\n".join(summary_parts)

        return FinalReport(
            format_version=FORMAT_VERSION,
            title=title,
            summary_html=summary_html,
            recommendations=recommendations,
            metrics_summary=metrics_summary,
            comparison_with_previous=comparison,
            fun_fact=Explainer.get_random_fact(),
            suggested_model_name=cfg.get("model_name", ""),
        )

    # ------------------------------------------------------------------
    # 3. КОММЕНТАРИИ К ПРЕДСКАЗАНИЯМ
    # ------------------------------------------------------------------

    @staticmethod
    def comment_prediction(
        input_data: Any,
        expected: Any,
        predicted: Any,
        config: Optional[Dict] = None,
    ) -> PredictionComment:
        """
        Генерирует комментарий к одному предсказанию модели.
        
        Пример:
            input: "45-68", expected: "-23", predicted: "67"
            → "45-68=67? Тебе учиться и учиться!"
        """
        input_str = str(input_data)
        expected_str = str(expected)
        predicted_str = str(predicted)

        # Проверяем корректность
        is_correct = Explainer._check_prediction(expected_str, predicted_str, config)

        if is_correct:
            comment = random.choice(SUCCESS_COMMENTS)
            detail = f"Вход: {input_str} → Ожидалось: {expected_str} → Получено: {predicted_str} ✅"
        else:
            # Забавный комментарий
            comment = random.choice(ERROR_COMMENTS)
            # Формируем фразу типа "45-68=67?"
            if config and config.get("data_type") == "text":
                phrase = f"{input_str}={predicted_str}?"
            else:
                phrase = f"Вход: {input_str} → Ответ: {predicted_str}, а надо: {expected_str}."
            comment = f"{phrase} {comment}"
            detail = (
                f"Вход: {input_str} | Ожидалось: {expected_str} | "
                f"Модель ответила: {predicted_str} ❌"
            )

        return PredictionComment(
            input_str=input_str,
            expected_str=expected_str,
            predicted_str=predicted_str,
            is_correct=is_correct,
            comment=comment,
            detail=detail,
        )

    @staticmethod
    def batch_comment_predictions(
        predictions: List[Dict[str, Any]],
        config: Optional[Dict] = None,
    ) -> Tuple[List[PredictionComment], float]:
        """
        Анализирует список предсказаний. Возвращает комментарии и точность.
        
        Args:
            predictions: список {"input": ..., "expected": ..., "predicted": ...}
        
        Returns:
            (список PredictionComment, точность 0.0-1.0)
        """
        comments = []
        correct_count = 0
        for p in predictions:
            c = Explainer.comment_prediction(
                p.get("input", ""),
                p.get("expected", ""),
                p.get("predicted", ""),
                config,
            )
            comments.append(c)
            if c.is_correct:
                correct_count += 1
        accuracy = correct_count / max(len(predictions), 1)
        return comments, accuracy

    # ------------------------------------------------------------------
    # 4. ФАКТЫ И ИМЕНА
    # ------------------------------------------------------------------

    @staticmethod
    def get_random_fact() -> str:
        """Возвращает случайный интересный факт."""
        return random.choice(FUN_FACTS)

    @staticmethod
    def generate_model_name() -> str:
        """Генерирует весёлое имя для модели."""
        prefixes = [
            "Мозгожуй", "Умник", "Нейро", "Гендальф", "Скайнет",
            "Пиксель", "Байт", "Тензорчик", "Градиентик", "Эпоха",
            "Лоссик", "Свёрточка", "Рекуррентик", "Трансформерчик",
            "Вектор", "Кубик", "Шарик", "Матрица", "Формула",
        ]
        suffixes = ["-3000", "-2000", "-1000", "-9000", "", "", "-мини", "-макс", "-про", ""]
        return random.choice(prefixes) + random.choice(suffixes)

    # ------------------------------------------------------------------
    # 5. СРАВНЕНИЕ ПОПЫТОК
    # ------------------------------------------------------------------

    @staticmethod
    def _compare_attempts(histories: List[Dict]) -> str:
        """Сравнивает несколько попыток обучения."""
        if len(histories) < 2:
            return ""
        
        lines = []
        for i, h in enumerate(histories, 1):
            val_loss = h.get("val_loss", [])
            val_metric = h.get("val_metric", [])
            epochs = len(val_loss)
            best_loss = min(val_loss) if val_loss else float('inf')
            best_metric = max(val_metric) if val_metric else 0.0
            lines.append(
                f"Попытка {i}: эпох={epochs}, лучшая ошибка={best_loss:.4f}, "
                f"лучшая метрика={best_metric:.1%}"
            )
        
        # Определяем прогресс
        if len(histories) >= 2:
            last = histories[-1]
            prev = histories[-2]
            last_metric = max(last.get("val_metric", [0]))
            prev_metric = max(prev.get("val_metric", [0]))
            if last_metric > prev_metric:
                lines.append("📈 Прогресс есть! Метрика улучшилась.")
            elif last_metric < prev_metric:
                lines.append("📉 Метрика ухудшилась. Попробуйте другие параметры.")
            else:
                lines.append("➡️ Результат не изменился.")
        
        return "<br>".join(lines)

    # ------------------------------------------------------------------
    # 6. КОНТЕКСТНЫЕ ПОДСКАЗКИ ДЛЯ ПАРАМЕТРОВ
    # ------------------------------------------------------------------

    @staticmethod
    def get_param_hint(
        param_name: str,
        current_value: Any,
        config: Optional[Dict] = None,
        dataset_meta: Optional[Dict] = None,
    ) -> str:
        """
        Возвращает контекстную подсказку для конкретного параметра.
        Используется hint_engine для режима «статические подсказки».
        
        Args:
            param_name: имя параметра ("lr_spin", "wd_spin", ...)
            current_value: текущее значение
            config: ModelConfig
            dataset_meta: DatasetMeta
        
        Returns:
            Текст подсказки
        """
        cfg = config or {}
        meta = dataset_meta or {}
        arch = cfg.get("type", "")
        data_type = meta.get("type", cfg.get("data_type", ""))

        # --- Скорость обучения ---
        if param_name in ("lr_spin", "learning_rate"):
            lr = float(current_value) if current_value else 0.001
            if arch in ("transformer", "transformer_seq2seq") and lr > 0.01:
                return (
                    "⚠️ Слишком высоко для Трансформера! Сеть разойдётся. "
                    "Попробуйте 1e-4 – 1e-3."
                )
            if arch == "mlp" and lr < 0.0001:
                return "🐢 Очень медленно. Для MLP попробуйте 0.001–0.01."
            if lr > 0.1:
                return "🚨 Очень высокий LR. Сеть может не сойтись."
            return (
                "Скорость обучения. Как сильно сеть корректирует веса "
                "после каждой ошибки. Стандарт: 0.001."
            )

        # --- Штраф за зубрёжку ---
        if param_name in ("wd_spin", "weight_decay"):
            wd = float(current_value) if current_value else 0.0
            if wd == 0:
                return (
                    "Штраф за зубрёжку (L2). Сейчас 0 — сеть может переобучиться. "
                    "Для маленьких датасетов поставьте 0.01."
                )
            return (
                "Представь, что нейросеть — ученик. L2-штраф говорит: "
                "«Не заучивай наизусть, понимай логику»."
            )

        # --- Подсказки учителя ---
        if param_name in ("tf_ratio_spin", "teacher_forcing_ratio"):
            tf = float(current_value) if current_value else 0.5
            if tf >= 0.9:
                return (
                    "Модель почти всегда видит правильный ответ при обучении. "
                    "Легче учиться, но на экзамене может растеряться."
                )
            if tf <= 0.1:
                return (
                    "Модель почти всегда использует свои предсказания. "
                    "Сложнее, но надёжнее. Рекомендуется снижать постепенно."
                )
            return (
                "Подсказки учителя при генерации. 1.0 = всегда правильный ответ "
                "(легче). 0.0 = всегда свой ответ (сложнее, но надёжнее). "
                "Рекомендуемо: начать с 0.7, потом снижать."
            )

        # --- Dropout ---
        if param_name in ("dropout_spin", "num_dropout_spin", "dropout"):
            d = float(current_value) if current_value else 0.1
            if d == 0:
                return "Dropout выключен. Сеть может переобучиться на маленьких данных."
            if d > 0.5:
                return "⚠️ Очень высокий Dropout. Сеть будет учиться медленно."
            return (
                "Dropout случайно «отключает» нейроны во время обучения. "
                "Это защищает от зубрёжки. Стандарт: 0.1–0.3."
            )

        # --- Батч ---
        if param_name in ("batch_spin", "batch_size"):
            bs = int(current_value) if current_value else 64
            if bs < 8:
                return "Очень маленький батч. Обучение будет шумным и медленным."
            if bs > 256:
                return "Большой батч. Убедитесь, что хватает видеопамяти."
            return "Размер партии. Сколько примеров сеть видит за один шаг. Стандарт: 32–128."

        # --- Эпохи ---
        if param_name in ("epochs_spin", "epochs"):
            return (
                "Количество полных проходов по всем данным. "
                "Для простых задач: 20-50. Для сложных: 100+."
            )

        # --- Эмбеддинг ---
        if param_name in ("tf_embed", "embedding_dim"):
            return (
                "Размер вектора для каждого символа. "
                "64 — для простых задач, 256 — для сложных. "
                "Должен делиться на количество голов."
            )

        # --- Головы ---
        if param_name in ("tf_heads", "num_heads"):
            return (
                "Сколько независимых потоков внимания ищут закономерности. "
                "4 — минимум, 8 — стандарт."
            )

        # --- По умолчанию ---
        return "Наведите на элемент, чтобы узнать подробнее."

    # ------------------------------------------------------------------
    # ВНУТРЕННИЕ ХЕЛПЕРЫ
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_history(history: Union[Dict, Any]) -> Dict[str, List[float]]:
        """Принимает TrainingHistory dataclass или dict, возвращает dict."""
        if isinstance(history, dict):
            return {
                "train_loss": history.get("train_loss", []),
                "val_loss": history.get("val_loss", []),
                "train_metric": history.get("train_metric", []),
                "val_metric": history.get("val_metric", []),
            }
        # Если dataclass из contracts
        return {
            "train_loss": getattr(history, "train_loss", []),
            "val_loss": getattr(history, "val_loss", []),
            "train_metric": getattr(history, "train_metric", []),
            "val_metric": getattr(history, "val_metric", []),
        }

    @staticmethod
    def _is_nan_or_huge(value: float, threshold: float = 1000.0) -> bool:
        """Проверяет, является ли значение NaN, Inf или слишком большим."""
        import math
        if value is None:
            return True
        try:
            v = float(value)
        except (TypeError, ValueError):
            return True
        if math.isnan(v) or math.isinf(v):
            return True
        return abs(v) > threshold

    @staticmethod
    def _check_prediction(
        expected: str,
        predicted: str,
        config: Optional[Dict] = None,
    ) -> bool:
        """Проверяет, совпадает ли предсказание с ожидаемым."""
        if not expected or not predicted:
            return False
        # Точное совпадение
        if expected.strip() == predicted.strip():
            return True
        # Числовое сравнение (для регрессии)
        try:
            e = float(expected)
            p = float(predicted)
            return abs(e - p) < 1e-4
        except (ValueError, TypeError):
            pass
        return False


# ============================================================
# ЗАГЛУШКА ДЛЯ ИИ-БЭКЕНДА
# ============================================================

class AIExplainerStub:
    """
    Заглушка для будущего ИИ-анализатора (локальная модель / GGUF).
    
    Когда будет подключена реальная модель, этот класс будет:
    - Анализировать графики глубже
    - Генерировать персонализированные рекомендации
    - Предсказывать переобучение до его начала
    
    Сейчас все методы возвращают None или пустые строки.
    """
    
    available = False
    
    @staticmethod
    def analyze_realtime_ai(history: Dict, config: Dict) -> Optional[str]:
        """Заглушка: вернёт анализ от ИИ-модели."""
        return None
    
    @staticmethod
    def generate_recommendations_ai(report: FinalReport) -> List[Recommendation]:
        """Заглушка: вернёт рекомендации от ИИ."""
        return []
    
    @staticmethod
    def comment_prediction_ai(input_str: str, expected: str, predicted: str) -> Optional[str]:
        """Заглушка: вернёт забавный комментарий от ИИ."""
        return None