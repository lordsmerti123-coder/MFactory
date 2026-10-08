"""
tests/test_hint_engine.py
=========================
Регрессия движка подсказок: сопоставление «голых» id элементов с записями
базы, ИИ-подсказки через агента Оракула, отсутствующие записи панелей.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.hint_engine import HintEngine, HintContext


class _FakeAgent:
    def __init__(self, text="Совет: выберите Transformer для текста."):
        self._text = text

    def generate_free_text(self, prompt):
        return self._text


class TestHintResolution(unittest.TestCase):
    def setUp(self):
        self.engine = HintEngine()

    def test_prefixed_id_exact(self):
        h = self.engine.get_hint(
            HintContext(widget_id="dataset.load_btn", panel_name="dataset_panel"))
        self.assertIn("данными", h)

    def test_unique_bare_id_resolves(self):
        # "train_spin" есть только в dataset — должен найтись
        h = self.engine.get_hint(
            HintContext(widget_id="train_spin", panel_name="dataset_panel"))
        self.assertTrue(h)
        self.assertIn("обучени", h.lower())

    def test_ambiguous_bare_id_stays_empty(self):
        # "load_btn" есть и в project, и в dataset — неоднозначно, пусто
        h = self.engine.get_hint(
            HintContext(widget_id="load_btn", panel_name="dataset_panel"))
        self.assertEqual(h, "")

    def test_missing_entries_now_present(self):
        for wid in ("dataset.input_dim_label", "dataset.quality_label",
                    "dataset.split_info_label", "monitoring.stop_btn",
                    "monitoring.predictions", "monitoring.fact"):
            h = self.engine.get_hint(HintContext(widget_id=wid, panel_name="x"))
            self.assertTrue(h, f"нет подсказки для {wid}")


class TestAIHint(unittest.TestCase):
    def test_ai_hint_uses_agent(self):
        engine = HintEngine()
        HintEngine.set_default_agent(_FakeAgent())
        try:
            # В режиме "hybrid" ИИ-подсказка получает префикс 🤖
            h = engine.get_hint(HintContext(
                widget_id="dataset.load_btn", panel_name="dataset_panel"))
            self.assertIn("Transformer", h)
        finally:
            HintEngine.set_default_agent(None)

    def test_ai_hint_without_agent_returns_empty(self):
        engine = HintEngine()
        HintEngine.set_default_agent(None)
        # Режим "ai" без агента → сообщение о недоступности
        class _Sess:
            hint_mode = "ai"
        h = engine.get_hint(HintContext(
            widget_id="dataset.load_btn", panel_name="dataset_panel",
            session=_Sess()))
        self.assertTrue(h)  # вернёт сообщение «ИИ-подсказки недоступны»


if __name__ == "__main__":
    unittest.main()
