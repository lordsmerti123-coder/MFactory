"""
tests/test_user_scenarios.py
============================
Интеграционные тесты пользовательских сценариев (что делает человек, куда нажимает).

Эти тесты «встают на место пользователя» и последовательно нажимают на логику
кнопок (без модальных диалогов — их ядро вынесено в silent-методы):

  «Создать проект» → «Сгенерировать данные» → «Загрузить» → «Разбить» →
  «Создать модель» → «Применить гиперпараметры» → «Начать обучение» → «Анализ».

В конце каждого сценария проверяется состояние программы, как его увидел бы
пользователь (данные есть, модель создана, обучение завершено, история есть).

ВАЖНО: torch импортируется строго ДО PyQt5 (ограничение проекта на Windows).
"""

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: F401,E402  (до PyQt5!)

from PyQt5.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication(sys.argv)

TRANSFORMER_PARAMS = {
    "embedding_dim": 32, "num_heads": 4,
    "num_encoder_layers": 1, "num_decoder_layers": 1,
    "dim_feedforward": 128, "dropout": 0.1,
}


def _spin_until(cond, timeout_s=120):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.2)
        if cond():
            return True
    return False


class TestUserScenarioCreateAndTrain(unittest.TestCase):
    """Полный сценарий нового пользователя: создать Трансформер и обучить на сложении."""

    @classmethod
    def setUpClass(cls):
        import gui.login_dialog as ld
        ld.LoginDialog.exec_ = lambda self: True
        ld.LoginDialog.get_selected_username = lambda self: "Admin"
        from gui.main_window import MainWindow
        cls.w = MainWindow()
        cls.w.show()
        app.processEvents()
        cls._created_files = []

    @classmethod
    def tearDownClass(cls):
        for f in cls._created_files:
            try:
                Path(f).unlink(missing_ok=True)
            except Exception:
                pass
        # Файлы от инструмента generate_dataset (внутреннее имя wizard_*.oai)
        try:
            subdirs = cls.w.session_manager.get_user_subdirs()
            d = subdirs.get("datasets")
            if d:
                for f in Path(d).glob("wizard_*.oai"):
                    try:
                        f.unlink(missing_ok=True)
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            cls.w.close()
        except Exception:
            pass

    def test_full_user_flow(self):
        w = self.w
        # --- Шаг 1: пользователь создаёт проект ---
        res = w.project_panel.create_project_silent("Сценарный-тест", "Трансформер-1")
        self.assertIn("создан", res)
        self.assertEqual(w.shared_state["project"]["name"], "Сценарный-тест")

        # --- Шаг 2: генерирует данные (примеры сложения) ---
        subdirs = w.session_manager.get_user_subdirs()
        save_path = subdirs["datasets"] / f"scenario_addition_{int(time.time())}.oai"
        self._created_files.append(str(save_path))
        msg = w.generator_panel.generate_dataset_silent("addition", 300, save_path)
        self.assertIn("создан", msg)

        # --- Шаг 3: загружает данные ---
        load = w.dataset_panel.load_dataset_silent(save_path)
        self.assertIn("загружен", load)
        self.assertIsNotNone(w.shared_state["dataset"])

        # --- Шаг 4: применяет разбиение ---
        w.dataset_panel.split_dataset()
        dataset = w.shared_state["dataset"]
        self.assertTrue(getattr(dataset, "is_split", False), "разбиение не применено")

        # --- Шаг 5: создаёт модель (Трансформер) ---
        rep = w.arch_panel.create_model_preset("transformer_seq2seq",
                                               dict(TRANSFORMER_PARAMS))
        self.assertIn("создана", rep)
        self.assertIsNotNone(w.shared_state["model"])

        # --- Шаг 6: гиперпараметры ---
        w.params_panel.set_hyperparams_silent(epochs=1, lr=0.001, batch_size=16)

        # --- Шаг 7: запускает обучение (не должно быть ошибки валидации) ---
        err = w.training_panel.start_training(silent=True)
        self.assertIsNone(err, f"валидация перед обучением вернула ошибку: {err}")

        finished = _spin_until(
            lambda: w.oraculum_agent._training_state.get("status", "idle")
            in ("finished", "failed"),
            timeout_s=120,
        )
        self.assertTrue(finished, "обучение не завершилось за отведённое время")
        self.assertEqual(w.oraculum_agent._training_state.get("status"), "finished")

        # --- Шаг 8: история и диагностика (что увидит пользователь) ---
        hist = w.shared_state.get("history")
        self.assertIsNotNone(hist)
        self.assertGreater(len(getattr(hist, "val_loss", []) or []), 0,
                           "история обучения пуста")
        diag = " ".join(w.oraculum_agent.diagnose())
        self.assertIn("Обучение завершено", diag)

    def test_train_without_dataset_returns_error_not_crash(self):
        """Пользователь жмёт «Обучить» до загрузки данных — не должно быть падения."""
        # Имитируем состояние без датасета: сбрасываем dataset/model/history
        w = self.w
        saved = (w.shared_state.get("dataset"), w.shared_state.get("model"),
                 w.shared_state.get("history"))
        w.shared_state["dataset"] = None
        w.shared_state["model"] = None
        try:
            err = w.training_panel.start_training(silent=True)
            self.assertIsNotNone(err, "ожидалась ошибка валидации, а не старт обучения")
        finally:
            w.shared_state["dataset"], w.shared_state["model"], w.shared_state["history"] = saved

    def test_diagnose_what_is_happening(self):
        """Пользователь спрашивает «что происходит» — получает конкретный ответ."""
        w = self.w
        resp = w.oraculum_agent.process_user_query("что происходит")
        self.assertTrue(len(resp) > 10)
        self.assertNotIn("технологиях Яндекс", resp)

    def test_addition_0_10_generation(self):
        """Генератор выдаёт числа строго в диапазоне 0–10."""
        from core.generator import DatasetGenerator
        g = DatasetGenerator(seed=7)
        d = g.generate_task("addition", {"num_range": (0, 10)}, 1000)
        nums = set()
        for e in d.raw_inputs:
            for part in str(e).split("+"):
                nums.add(int(part))
        self.assertGreaterEqual(min(nums), 0)
        self.assertLessEqual(max(nums), 10)

    def test_addition_0_10_via_agent_tool(self):
        """ИИ-инструмент generate_dataset с min=0/max=10 создаёт датасет 0–10."""
        w = self.w
        r = w._execute_agent_action(
            "generate_dataset", {"task": "addition", "num_samples": 300, "min": 0, "max": 10}
        )
        self.assertIn("done", r)
        self.assertNotIn("error", r)
        self.assertIsNotNone(w.shared_state["dataset"])


    def test_hyperparams_preserved_on_model_create(self):
        """Смена архитектуры не должна сбрасывать выставленные гиперпараметры."""
        w = self.w
        # Выставляем гиперпараметры во вкладке «Гиперпараметры»
        w.params_panel.set_hyperparams_silent(epochs=7, lr=0.0003, batch_size=64)
        # Создаём модель (как при смене архитектуры)
        rep = w.arch_panel.create_model_preset("transformer_seq2seq",
                                               dict(TRANSFORMER_PARAMS))
        self.assertIn("создана", rep)
        cfg = w.shared_state.get("config")
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.get("epochs"), 7, "эпохи сбросились при создании модели")
        self.assertEqual(cfg.get("learning_rate"), 0.0003, "LR сбросился")
        self.assertEqual(cfg.get("batch_size"), 64, "batch сбросился")


if __name__ == "__main__":
    unittest.main()
