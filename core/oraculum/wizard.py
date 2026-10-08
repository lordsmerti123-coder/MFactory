"""
core/oraculum/wizard.py
=======================
Режим «Мастер»: планировщик шагов + плеер, исполняющий план в человеческом темпе.

ОТВЕТСТВЕННОСТЬ:
  • WizardStep / WizardPlan — декларативные шаги (комментарий + инструмент + args).
  • WizardPlanner — строит планы: демо-трансформер (сложение) и интерактивный.
  • WizardPlayer — исполняет план по QTimer: на каждый шаг показывает комментарий,
    вызывает инструмент Оракула, при необходимости ждёт условие (например,
    завершение обучения), и делает паузу между шагами (скорость настраивается).

ЗАВИСИМОСТИ:
  • PyQt5.QtCore (QObject, pyqtSignal, QTimer)
  • Стандартная библиотека (dataclasses, typing, time)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from PyQt5.QtCore import QObject, QTimer, pyqtSignal

DEFAULT_STEP_DELAY_MS = 2200
TRAIN_WAIT_TIMEOUT_S = 300
STORY_INTERVAL_MS = 5000


# Рассказы о нейросетях — Оракул чередует их во время обучения.
NN_STORIES = [
    "Трансформер — это та же архитектура, что у ChatGPT: он смотрит на все "
    "символы сразу и решает, какие из них важны для ответа.",
    "Механизм внимания (attention) — модель спрашивает себя: «плюс здесь важнее, "
    "чем цифра 3?». Так она учится фокусироваться на главном.",
    "Каждый символ превращается в вектор чисел (эмбеддинг). Для компьютера «2» "
    "и «+» — просто наборы чисел, которым нужно научиться пользоваться.",
    "Loss — это «расстояние» между ответом модели и правильным ответом. Когда "
    "оно уменьшается — модель действительно учится, а не просто запоминает.",
    "val_loss — это экзамен: модель проверяется на примерах, которые никогда "
    "не видела. По нему видно, научилась ли она обобщать.",
    "Нейросеть — это много умножений матриц. Волшебство в том, что эти "
    "умножения настраиваются так, чтобы выдавать правильный ответ.",
    "Складывать числа нейросети учили ещё в 1980-х. Теперь это делает "
    "трансформер с вниманием — как в современных чат-ботах.",
]


def build_training_comment(agent, n: int) -> str:
    """
    Живой комментарий Оракула во время обучения: анализ текущей ситуации
    (loss, переобучение, точность) + рассказ о нейросетях.
    """
    t = agent._training_state
    metrics = t.get("last_metrics", {}) or {}
    epoch = t.get("last_epoch", 0)
    train_loss = metrics.get("train_loss")
    val_loss = metrics.get("val_loss")
    val_metric = metrics.get("val_metric")

    parts = []
    if isinstance(train_loss, (int, float)) and isinstance(val_loss, (int, float)):
        if val_loss > train_loss * 1.6 and train_loss > 0.05:
            parts.append(
                f"⚠️ Эпоха {epoch}: модель начинает зубрить — на обучении "
                f"loss {train_loss:.3f}, а на новых данных {val_loss:.3f}. "
                "Разница растёт, это признак переобучения."
            )
        elif train_loss <= 0.25 and val_loss <= 0.25:
            parts.append(
                f"✅ Эпоха {epoch}: loss {train_loss:.3f} (обучение) и "
                f"{val_loss:.3f} (проверка) — модель уверенно считает!"
            )
        else:
            parts.append(
                f"📉 Эпоха {epoch}: loss падает — {train_loss:.3f} на обучении, "
                f"{val_loss:.3f} на проверке. Модель учится!"
            )
    elif epoch:
        parts.append(f"Эпоха {epoch} обрабатывается…")

    if isinstance(val_metric, (int, float)) and val_metric > 0:
        parts.append(f"Точность токенов на проверке: {val_metric:.1%}.")

    if NN_STORIES:
        parts.append(NN_STORIES[(n - 1) % len(NN_STORIES)])

    return " ".join(parts)


def make_story_provider(agent) -> Callable[[], str]:
    """Возвращает callable, который при каждом вызове даёт новый комментарий."""
    counter = {"n": 0}

    def _provider() -> str:
        counter["n"] += 1
        return build_training_comment(agent, counter["n"])

    return _provider


@dataclass
class WizardStep:
    """Один шаг плана мастера."""
    step_id: str
    title: str                     # короткое имя для чек-листа
    narration: str                 # комментарий, который «проговаривает» Оракул
    tool: Optional[str]            # имя инструмента (None — просто пауза/пояснение)
    args: Dict[str, Any] = field(default_factory=dict)
    delay_ms: int = DEFAULT_STEP_DELAY_MS
    wait: Optional[Callable[[], bool]] = None   # условие завершения шага (например, конец обучения)
    wait_timeout: float = TRAIN_WAIT_TIMEOUT_S
    highlight: Optional[str] = None  # вкладка, которую подсвечивать
    story_interval_ms: int = 0       # если > 0 — во время ожидания Оракул комментирует
    story_provider: Optional[Callable[[], str]] = None


@dataclass
class WizardPlan:
    """План: заголовок + последовательность шагов."""
    scenario: str
    title: str
    steps: List[WizardStep]


class WizardPlanner:
    """Строит планы для режима «Мастер»."""

    def __init__(self, agent):
        self.agent = agent

    def plan_demo_transformer(self) -> WizardPlan:
        """Сценарий A: показать, как создаётся Трансформер и учится складывать числа."""
        agent = self.agent

        def _training_done() -> bool:
            return agent._training_state.get("status") in ("finished", "failed")

        steps = [
            WizardStep(
                "project", "Проект",
                "Начнём с проекта. Назову его «Сложение-демо», а модель — «Трансформерчик».",
                "set_project",
                {"name": "Сложение-демо", "model_name": "Трансформерчик"},
                highlight="Проект",
            ),
            WizardStep(
                "generate", "Данные",
                "Трансформеру нужны примеры. Генерирую 5000 примеров сложения чисел 0–10 "
                "(вопрос «2+3?» → ответ «5»).",
                "generate_dataset",
                {"task": "addition", "num_samples": 5000, "min": 0, "max": 10},
                highlight="Генератор",
            ),
            WizardStep(
                "split", "Данные",
                "Разбиваю данные на 3 части: train (модель учится на них), "
                "val (экзамен во время обучения) и test (финальная проверка). "
                "Словарь строится только по обучающей выборке, чтобы модель "
                "не «подглядывала» ответы.",
                "split_dataset", {},
                highlight="Данные",
            ),
            WizardStep(
                "arch", "Архитектура",
                "Выбираю Transformer Seq2Seq — именно такие модели (как ChatGPT) учатся "
                "понимать последовательности «вопрос → ответ». Для демо беру маленькую: "
                "d_model=64, 4 головы, 2+2 слоя.",
                "set_architecture",
                {"arch": "transformer_seq2seq", "arch_params": {
                    "embedding_dim": 64, "num_heads": 4,
                    "num_encoder_layers": 2, "num_decoder_layers": 2,
                    "dim_feedforward": 256, "dropout": 0.1,
                }},
                highlight="Архитектура",
            ),
            WizardStep(
                "hyper", "Гиперпараметры",
                "Для демо ставлю 20 эпох и скорость обучения 0.001 — покажу, "
                "как loss падает на графике. Настоящие модели учатся ещё дольше.",
                "set_hyperparams",
                {"epochs": 20, "lr": 0.001, "batch_size": 32},
                highlight="Гиперпараметры",
            ),
            WizardStep(
                "train", "Обучение",
                "Запускаю обучение. Я слежу за процессом: смотрю на графики loss "
                "и рассказываю, что происходит.",
                "start_training", {},
                highlight="Обучение",
                wait=_training_done,
                wait_timeout=TRAIN_WAIT_TIMEOUT_S,
                story_interval_ms=STORY_INTERVAL_MS,
                story_provider=make_story_provider(agent),
            ),
            WizardStep(
                "monitor", "Мониторинг",
                "Обучение завершено. Переключаю на вкладку «Мониторинг», где видно, "
                "как падал loss — модель училась! Посмотрите на график пару секунд.",
                "switch_tab", {"tab": "Мониторинг"},
                highlight="Мониторинг",
                delay_ms=9000,
            ),
            WizardStep(
                "analyze", "Анализ",
                "Теперь разберём результат во вкладке «Анализ».",
                "analyze", {},
                highlight="Анализ",
            ),
            WizardStep(
                "done", "Готово",
                "Готово! Трансформер научился решать примеры на сложение. "
                "Попробуйте сами в «Песочнице» или скажите «что происходит» — "
                "я покажу состояние.",
                None, {},
                highlight=None,
            ),
        ]
        return WizardPlan("demo_transformer", "Показываю, как создаётся Трансформер", steps)

    def plan_interactive(self, answers: Dict[str, Any]) -> WizardPlan:
        """Сценарий B: интерактивный помощник (каркас; уточнения — в чате)."""
        task = str(answers.get("task", "addition"))
        arch = str(answers.get("arch", "transformer_seq2seq"))
        num_samples = int(answers.get("num_samples", 5000))
        name = str(answers.get("name", "Мой проект"))
        model_name = str(answers.get("model_name", "Моя модель"))

        agent = self.agent

        def _training_done() -> bool:
            return agent._training_state.get("status") in ("finished", "failed")

        steps = [
            WizardStep("project", "Проект", f"Создаю проект «{name}».", "set_project",
                       {"name": name, "model_name": model_name}, highlight="Проект"),
            WizardStep("generate", "Данные",
                       f"Генерирую {num_samples} примеров.",
                       "generate_dataset", {"task": task, "num_samples": num_samples},
                       highlight="Генератор"),
            WizardStep("split", "Данные", "Применяю разбиение на train/val/test.",
                       "split_dataset", {}, highlight="Данные"),
            WizardStep("arch", "Архитектура", "Создаю модель.",
                       "set_architecture", {"arch": arch, "arch_params": {}},
                       highlight="Архитектура"),
            WizardStep("hyper", "Гиперпараметры", "Настраиваю обучение (5 эпох для старта).",
                       "set_hyperparams", {"epochs": 5, "lr": 0.001, "batch_size": 32},
                       highlight="Гиперпараметры"),
            WizardStep("train", "Обучение", "Запускаю обучение и слежу за ним: "
                       "комментирую loss и графики.",
                       "start_training", {}, highlight="Обучение",
                       wait=_training_done, wait_timeout=TRAIN_WAIT_TIMEOUT_S,
                       story_interval_ms=STORY_INTERVAL_MS,
                       story_provider=make_story_provider(agent)),
            WizardStep("monitor", "Мониторинг", "Показываю графики обучения.",
                       "switch_tab", {"tab": "Мониторинг"}, highlight="Мониторинг"),
            WizardStep("analyze", "Анализ", "Разбираю результат.",
                       "analyze", {}, highlight="Анализ"),
            WizardStep("done", "Готово", "Готово! Модель обучена. Спрашивайте «что происходит».",
                       None, {}, highlight=None),
        ]
        return WizardPlan("interactive", "Помогаю создать вашу нейросеть", steps)


class WizardPlayer(QObject):
    """Исполняет план по шагам в человеческом темпе (в главном потоке)."""

    step_started = pyqtSignal(str, dict)   # (step_id, step_info)
    narration = pyqtSignal(str)
    step_finished = pyqtSignal(str, dict)
    finished = pyqtSignal()
    aborted = pyqtSignal()
    error = pyqtSignal(str, str)           # (step_id, error_text)

    def __init__(self, execute_tool: Callable[[str, Dict[str, Any]], str],
                 parent: Optional[QObject] = None):
        super().__init__(parent)
        self._execute_tool = execute_tool
        self._plan: Optional[WizardPlan] = None
        self._index = 0
        self._speed = 1.0
        self._paused = False
        self._stopped = False
        self._active_step: Optional[WizardStep] = None
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(400)
        self._poll_timer.timeout.connect(self._poll_wait)
        self._delay_timer = QTimer(self)
        self._delay_timer.setSingleShot(True)
        self._delay_timer.timeout.connect(self._run_next_step)
        self._wait_deadline = 0.0
        self._last_story_time = 0.0

    # ============================================================
    # УПРАВЛЕНИЕ
    # ============================================================
    def start(self, plan: WizardPlan):
        self._plan = plan
        self._index = 0
        self._paused = False
        self._stopped = False
        self._run_next_step()

    def pause(self):
        self._paused = True
        self._poll_timer.stop()
        self._delay_timer.stop()

    def resume(self):
        if not self._paused or self._plan is None:
            return
        self._paused = False
        if self._active_step is not None:
            self._begin_wait(self._active_step)
        else:
            self._run_next_step()

    def stop(self):
        self._stopped = True
        self._poll_timer.stop()
        self._delay_timer.stop()
        self._active_step = None
        if self._plan is not None:
            self.aborted.emit()
        self._plan = None

    def set_speed(self, factor: float):
        self._speed = max(0.25, min(factor, 3.0))

    def is_running(self) -> bool:
        return self._plan is not None and not self._stopped

    # ============================================================
    # ПРОИГРЫВАНИЕ
    # ============================================================
    def _run_next_step(self):
        if self._stopped or self._paused:
            return
        if self._plan is None:
            return
        if self._index >= len(self._plan.steps):
            self.finished.emit()
            self._plan = None
            return

        step = self._plan.steps[self._index]
        self._active_step = step
        info = {
            "id": step.step_id,
            "title": step.title,
            "narration": step.narration,
            "highlight": step.highlight,
        }
        self.step_started.emit(step.step_id, info)
        self.narration.emit(step.narration)

        if step.tool:
            try:
                result = self._execute_tool(step.tool, step.args or {})
            except Exception as e:
                self.error.emit(step.step_id, str(e))
                result = f'{{"error": "{e}"}}'
            # Результат инструмента показываем как продолжение мысли (если не пусто)
            try:
                import json as _json
                parsed = _json.loads(result) if isinstance(result, str) else {}
                if parsed.get("error"):
                    self.narration.emit(f"⚠️ {parsed['error']}")
                elif parsed.get("report") and step.step_id == "arch":
                    self.narration.emit(f"{parsed['report']}")
            except Exception:
                pass

        if step.wait is not None:
            self._begin_wait(step)
        else:
            self._delay_next_step()

    def _begin_wait(self, step: WizardStep):
        self._wait_deadline = time.time() + step.wait_timeout
        self._last_story_time = time.time()
        self._poll_timer.start()
        self._poll_wait()

    def _poll_wait(self):
        step = self._active_step
        if step is None or step.wait is None:
            self._poll_timer.stop()
            self._delay_next_step()
            return
        if self._paused or self._stopped:
            return

        # Рассказы Оракула во время ожидания (например, обучения)
        if (step.story_provider is not None and step.story_interval_ms > 0
                and time.time() - self._last_story_time >= step.story_interval_ms / 1000.0):
            self._last_story_time = time.time()
            try:
                self.narration.emit(step.story_provider())
            except Exception:
                pass

        try:
            done = bool(step.wait())
        except Exception:
            done = False
        if done:
            self._poll_timer.stop()
            self.narration.emit("✅ Шаг выполнен.")
            self._delay_next_step()
        elif time.time() > self._wait_deadline:
            self._poll_timer.stop()
            self.narration.emit("⏭️ Процесс занял много времени — иду дальше.")
            self._delay_next_step()

    def _delay_next_step(self):
        step = self._active_step
        delay = int((step.delay_ms if step else DEFAULT_STEP_DELAY_MS) / self._speed)
        self._finish_step()
        self._delay_timer.start(delay)

    def _finish_step(self):
        step = self._active_step
        if step is not None:
            self.step_finished.emit(step.step_id, {"id": step.step_id, "title": step.title})
        self._index += 1
        self._active_step = None
