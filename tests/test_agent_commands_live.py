# -*- coding: utf-8 -*-
"""
Тест агента Оракула: выполнение и правильность команд от пользователя.

Часть A — детерминированная: идеальный «мозг» (scripted backend) выдаёт
правильные tool-call, проверяем конвейер: команда -> инструмент -> аргументы
-> изменение состояния -> финальный ответ.

Часть B — живая: реальная локальная GGUF-модель, реальные команды из жизни,
проверяем, что агент сам выбирает и корректно выполняет инструменты.

Запуск:  venv\\Scripts\\python.exe tests\\test_agent_commands_live.py
"""
import sys
import json
import time
import traceback
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402,F401  (torch до PyQt — требование проекта)

from core.contracts import SessionState, ModelConfig, TrainingHistory
from core.oraculum.agent import OraculumAgent


# ============================================================
# ЭМУЛЯТОР GUI: повторяет логику MainWindow._action_* с валидацией
# ============================================================
class FakeGui:
    TABS = ["Проект", "Генератор", "Данные", "Архитектура", "Гиперпараметры",
            "Обучение", "Мониторинг", "Анализ", "Экспорт", "Песочница", "Логи"]
    ARCHS = {"transformer_seq2seq", "mlp", "cnn", "lstm", "rnn", "gan",
             "transformer"}
    TASKS = {"addition", "subtraction", "multiplication", "division",
             "mixed_math", "linear_equation", "quadratic",
             "cipher_caesar", "cipher_atbash"}
    TASK_ALIASES = {
        "caesar_cipher": "cipher_caesar", "cezar": "cipher_caesar",
        "цезарь": "cipher_caesar", "atbash": "cipher_atbash",
        "сложение": "addition", "сумма": "addition",
        "вычитание": "subtraction", "умножение": "multiplication",
        "деление": "division", "смешанная": "mixed_math",
        "уравнение": "linear_equation", "квадратное": "quadratic",
    }

    def __init__(self):
        self.tab = "Проект"
        self.scenario = "new"
        self.project_name = None
        self.model_name = None
        self.dataset = False
        self.split = False
        self.model = False
        self.arch = None
        self.trained = False
        self.hyper = {}
        self.calls = []
        self.rejections = []

    def reset_call_log(self):
        self.calls = []
        self.rejections = []

    def dispatch(self, name, args):
        args = dict(args or {})
        self.calls.append((name, args))
        return getattr(self, "_do_" + name)(args)

    def _ok(self, **kw):
        return json.dumps({"done": True, **kw}, ensure_ascii=False)

    def _err(self, msg):
        self.rejections.append((msg,))
        return json.dumps({"error": msg}, ensure_ascii=False)

    def _do_switch_tab(self, args):
        wanted = str(args.get("tab", "")).strip()
        for t in self.TABS:
            if wanted.lower() in t.lower() or t.lower().startswith(wanted.lower()):
                self.tab = t
                return self._ok(tab=t)
        return self._err(f"вкладка не найдена: {wanted}")

    def _do_set_scenario(self, args):
        s = str(args.get("scenario", "")).strip().lower()
        if s not in ("new", "finetune", "play"):
            return self._err("сценарий должен быть new/finetune/play")
        self.scenario = s
        return self._ok(scenario=s)

    def _do_set_project(self, args):
        self.project_name = str(args.get("name", "")).strip() or "Проект"
        self.model_name = str(args.get("model_name", "")).strip()
        return self._ok(project=self.project_name)

    def _do_generate_dataset(self, args):
        task = str(args.get("task", "addition")).strip()
        task = self.TASK_ALIASES.get(task.lower(), task)
        if task not in self.TASKS:
            return self._err(f"неизвестная задача: {task}")
        try:
            num = int(args.get("num_samples", 5000))
        except (TypeError, ValueError):
            return self._err("num_samples должно быть числом")
        if num <= 0:
            return self._err("num_samples должно быть > 0")
        if "min" in args or "max" in args:
            lo, hi = int(args.get("min", 0)), int(args.get("max", 10))
            if lo > hi:
                lo, hi = hi, lo
        self.dataset = True
        return self._ok(task=task, samples=num)

    def _do_split_dataset(self, args):
        if not self.dataset:
            return self._err("данные не загружены")
        self.split = True
        return self._ok()

    def _do_create_model(self, args):
        if not (self.dataset and self.split):
            return self._err("нужны данные с разбиением")
        self.model = True
        return self._ok()

    def _do_set_architecture(self, args):
        arch = str(args.get("arch", "transformer_seq2seq")).strip()
        if arch.lower() in ("transformer", "transformers"):
            arch = "transformer_seq2seq"
        if arch not in self.ARCHS:
            return self._err(f"неизвестная архитектура: {arch}")
        self.model = True
        self.arch = arch
        return self._ok(arch=arch)

    def _do_set_hyperparams(self, args):
        for k in ("epochs", "lr", "batch_size"):
            if k in args:
                self.hyper[k] = args[k]
        return self._ok()

    def _do_start_training(self, args):
        if not (self.split and self.model):
            return self._err("нужны разбитые данные и модель")
        self.trained = True
        return self._ok()

    def _do_stop_training(self, args):
        return self._ok()

    def _do_analyze(self, args):
        if not self.trained:
            return self._err("нет истории обучения")
        return self._ok()

    def _do_get_state(self, args):
        return self._ok(current_tab=self.tab, dataset=self.dataset,
                        split=self.split, model=self.model,
                        trained=self.trained, arch=self.arch)


def make_agent(gui):
    state = {
        "session": SessionState(),
        "project": {"name": "P", "scenario": "new", "model_name": "M"},
        "dataset": None,
        "model": None,
        "config": ModelConfig(),
        "history": TrainingHistory(),
        "hint_mode": "hybrid",
        "user": {"username": "Admin", "level": "advanced", "is_admin": True},
    }
    agent = OraculumAgent(state, resource_limits={
        "max_ram_mb": 20000,
        "max_model_ram_mb": 20000,
        "max_llm_model_ram_mb": 20000,
        "max_cpu_percent": 100,
    })
    agent.settings["agent_enabled"] = True
    agent.settings["agent_max_steps"] = 6
    agent.settings["agent_temperature"] = 0.2
    # Регистрируем GUI-инструменты, как это делает MainWindow
    for name, desc, params in [
        ("switch_tab", "Переключить вкладку", {"tab": "название"}),
        ("set_scenario", "Сценарий", {"scenario": "new/finetune/play"}),
        ("create_model", "Создать модель", {}),
        ("split_dataset", "Разбиение данных", {}),
        ("start_training", "Запустить обучение", {}),
        ("stop_training", "Остановить обучение", {}),
        ("analyze", "Анализ результатов", {}),
        ("set_project", "Создать проект", {"name": "имя", "model_name": "имя модели"}),
        ("generate_dataset", "Сгенерировать датасет",
         {"task": "тип задачи", "num_samples": "кол-во", "min": "мин", "max": "макс"}),
        ("set_architecture", "Создать модель архитектуры",
         {"arch": "архитектура", "arch_params": "параметры"}),
        ("set_hyperparams", "Гиперпараметры", {"epochs": "число", "lr": "число",
                                                "batch_size": "число"}),
        ("get_state", "Состояние программы (JSON)", {}),
    ]:
        agent.register_tool(name, desc, params)
    agent.set_action_dispatcher(gui.dispatch)
    return agent


# ============================================================
# ЧАСТЬ A — детерминированный конвейер (scripted «мозг»)
# ============================================================
def part_a():
    print("\n" + "=" * 70)
    print("ЧАСТЬ A. Детерминированная проверка конвейера команд")
    print("=" * 70)

    # Сценарий «полный цикл» — как поступает новичок. Мозг выдаёт tool-call,
    # идеально соответствующие командам (как обучена реальная LLM).
    scenario = [
        # (команда пользователя, ответы мозга [tool-calls..., reply])
        ("переключи вкладку на Обучение",
         ['{"action": "switch_tab", "args": {"tab": "Обучение"}}',
          '{"reply": "Переключил на вкладку «Обучение»."}'],
         [("switch_tab", "Обучение")]),
        ("создай проект ШифроБот",
         ['{"action": "set_project", "args": {"name": "ШифроБот", "model_name": "ШифроБот-1"}}',
          '{"reply": "Проект ШифроБот создан."}'],
         [("set_project", "ШифроБот")]),
        ("сгенерируй данные: сложение от 0 до 10, 300 примеров",
         ['{"action": "generate_dataset", "args": {"task": "addition", "num_samples": 300, "min": 0, "max": 10}}',
          '{"reply": "Данные сгенерированы: 300 примеров."}'],
         [("generate_dataset", None)]),
        ("разбей данные на обучение и проверку",
         ['{"action": "split_dataset", "args": {}}',
          '{"reply": "Разбиение применено."}'],
         [("split_dataset", None)]),
        ("создай модель трансформер",
         ['{"action": "set_architecture", "args": {"arch": "transformer_seq2seq"}}',
          '{"reply": "Модель-трансформер создана."}'],
         [("set_architecture", "transformer_seq2seq")]),
        ("задай гиперпараметры: эпохи 10, lr 0.001",
         ['{"action": "set_hyperparams", "args": {"epochs": 10, "lr": 0.001}}',
          '{"reply": "Гиперпараметры заданы."}'],
         [("set_hyperparams", None)]),
        ("запусти обучение",
         ['{"action": "start_training", "args": {}}',
          '{"reply": "Обучение запущено."}'],
         [("start_training", None)]),
        ("проанализируй результат",
         ['{"action": "analyze", "args": {}}',
          '{"reply": "Анализ готов: модель обучена."}'],
         [("analyze", None)]),
    ]

    gui = FakeGui()
    agent = make_agent(gui)

    failures = 0
    for q, brain, expected_calls in scenario:
        gui.reset_call_log()
        n = {"i": 0}

        def fake_chat(messages):
            n["i"] += 1
            return brain[min(n["i"] - 1, len(brain) - 1)]

        agent._agent_chat = fake_chat
        agent._agent_backend_available = lambda: True

        before = list(gui.calls)
        reply = agent.run_agent_loop(q)
        calls = gui.calls[len(before):]

        names = [c[0] for c in calls]
        ok = True
        why = ""
        if expected_calls:
            if names != [e[0] for e in expected_calls]:
                ok = False
                why = f"ожидал инструменты {[e[0] for e in expected_calls]}, получил {names}"
            else:
                exp_val = expected_calls[0][1]
                if exp_val is not None:
                    vals = set(v for v in (calls[0][1] or {}).values())
                    if exp_val not in vals:
                        ok = False
                        why = f"аргумент {calls[0][1]!r} не содержит {exp_val!r}"
            if gui.rejections:
                ok = False
                why = f"GUI отклонил команду: {gui.rejections[-1][0]}"
        if ok:
            print(f"  [OK] {q!r} -> {names} -> {reply[:60]!r}")
        else:
            failures += 1
            print(f"  [FAIL] {q!r} -> {names} | {why} | reply={reply[:80]!r}")

    # Состояние после всего сценария
    print("\n  Итоговое состояние GUI:",
          f"project={gui.project_name}, dataset={gui.dataset}, split={gui.split}, "
          f"model={gui.model}({gui.arch}), trained={gui.trained}")
    if gui.dataset and gui.split and gui.model and gui.trained:
        print("  [OK] Сценарий «новичок -> обучение» доведён до конца.")
    else:
        failures += 1
        print("  [FAIL] Сценарий не доведён до обучения.")
    return failures


# ============================================================
# ЧАСТЬ B — живой тест на реальной локальной модели
# ============================================================
def part_b():
    print("\n" + "=" * 70)
    print("ЧАСТЬ B. Живой тест на реальной GGUF-модели")
    print("=" * 70)

    gui = FakeGui()
    agent = make_agent(gui)

    if not agent.load_local_model():
        print("  [SKIP] Локальная модель не загрузилась:", agent.local_model_error())
        return 0
    print("  Локальная модель загружена.")

    commands = [
        # (команда, ожидаемая проверка состояния после выполнения)
        ("что происходит",
         lambda g: g.rejections == [] and len(g.calls) == 0),
        ("переключись на вкладку Обучение",
         lambda g: g.tab == "Обучение"),
        ("создай проект ШифроБот и модель ШифроБот-1",
         lambda g: g.project_name == "ШифроБот" and g.model_name == "ШифроБот-1"),
        ("сгенерируй данные для шифра цезаря, 200 примеров",
         lambda g: g.dataset),
        ("разбей данные",
         lambda g: g.split),
        ("создай модель трансформер",
         lambda g: g.model and g.arch == "transformer_seq2seq"),
        ("задай эпохи 5 и скорость обучения 0.001",
         lambda g: g.hyper.get("epochs") == 5),
        ("запусти обучение",
         lambda g: g.trained),
        ("проанализируй результат",
         lambda g: any(n == "analyze" for n, _ in g.calls)),
    ]

    failures = 0
    for q, check in commands:
        gui.reset_call_log()
        t0 = time.time()
        try:
            reply = agent.process_user_query(q)
        except Exception as e:
            reply = f"<ИСКЛЮЧЕНИЕ: {type(e).__name__}: {e}>"
            traceback.print_exc()
        dt = time.time() - t0
        calls = [(n, dict(a)) for n, a in gui.calls]
        ok = bool(check(gui))
        if not ok:
            failures += 1
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {dt:5.1f}s | «{q}»")
        print(f"         инструменты: {calls if calls else '(нет)'}")
        if gui.rejections:
            print(f"         отклонено GUI: {gui.rejections}")
        print(f"         ответ: {str(reply)[:150]!r}")

    print("\n  Итоговое состояние:",
          f"dataset={gui.dataset}, split={gui.split}, model={gui.model}({gui.arch}), "
          f"trained={gui.trained}")
    if gui.dataset and gui.split and gui.model:
        print("  [OK] Агент самостоятельно довёл сценарий до готовности к обучению.")
    else:
        print("  [FAIL] Сценарий не выполнен полностью.")
        failures += 1

    agent.unload_local_model()
    return failures


if __name__ == "__main__":
    total = 0
    total += part_a()
    total += part_b()
    print("\n" + "=" * 70)
    print(f"ИТОГ: {total} проблем(ы)")
    print("=" * 70)
    sys.exit(1 if total else 0)
