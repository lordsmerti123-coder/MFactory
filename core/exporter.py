"""
core/exporter.py
================
Модуль экспорта моделей и проектов.

Сценарии экспорта:
  1. export_project       — полный проект (.oai): датасет + модель + история
  2. export_model_only    — только модель (.pth): веса + конфиг
  3. export_flash_chat    — папка для флешки: автономный чат без установки
  4. export_html_sandbox  — HTML-песочница (заглушка)
  5. export_onnx          — экспорт в ONNX
  6. export_report        — отчёт в Markdown

Контракты данных:
  - Все сохраняемые файлы содержат "format_version".
  - Работает с объектами DatasetContainer, ModelConfig, TrainingHistory.
  - Обратная совместимость: from_dict принимает данные версии 1.x.

Зависимости:
  - core.model_factory (ModelFactory)
  - core.logger (get_logger)
  - core.contracts (DatasetContainer, ModelConfig, TrainingHistory)
  - torch, json, datetime, pathlib

ВНИМАНИЕ: модуль НЕ создаёт новых ключей в shared_state.
"""

import json
import torch
import torch.nn as nn
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from core.logger import get_logger
from core.contracts import (
    FORMAT_VERSION,
    DatasetContainer,
    ModelConfig,
    TrainingHistory,
    ModelCheckpoint,
)

logger = get_logger("Exporter")

# ============================================================
#  КОНСТАНТЫ
# ============================================================
PROJECT_EXTENSION = ".oai"
MODEL_EXTENSION = ".pth"


# ============================================================
#  ШАБЛОН: chat.py (автономный чат для флешки)
# ============================================================
CHAT_PY_TEMPLATE = '''#!/usr/bin/env python3
"""
Автономный чат с нейросетью "{model_name}".
Сгенерировано OracleAI Studio.
Зависимости: только torch.

Запуск: python chat.py
"""
import json
import sys
import math
import torch
import torch.nn as nn
from pathlib import Path

# ============================================================
#  МИНИМАЛЬНЫЙ ФАБРИК МОДЕЛЕЙ (автономный, без основного проекта)
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=8,
                 num_encoder_layers=2, num_decoder_layers=2,
                 dim_feedforward=512, activation="relu",
                 max_len=200, dropout=0.1,
                 pad_idx=0, bos_idx=2, eos_idx=3):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            activation=activation, batch_first=True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, trg, teacher_forcing_ratio=1.0):
        trg_input = trg[:, :-1]
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        trg_emb = self.pos_encoder(self.embedding(trg_input) * math.sqrt(self.d_model))
        src_pad_mask = (src == self.pad_idx)
        tgt_pad_mask = (trg_input == self.pad_idx)
        tgt_mask_float = self.transformer.generate_square_subsequent_mask(
            trg_input.size(1)
        ).to(src.device)
        tgt_mask = (tgt_mask_float == float("-inf"))
        out = self.transformer(
            src_emb, trg_emb,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_pad_mask
        )
        return self.fc_out(out)

    @torch.no_grad()
    def generate(self, src, max_len=50):
        self.eval()
        device = src.device
        batch_size = src.size(0)
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        src_pad_mask = (src == self.pad_idx)
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_pad_mask)
        ys = torch.full((batch_size, 1), self.bos_idx, dtype=torch.long, device=device)
        for _ in range(max_len):
            trg_emb = self.pos_encoder(self.embedding(ys) * math.sqrt(self.d_model))
            tgt_mask_float = self.transformer.generate_square_subsequent_mask(
                ys.size(1)
            ).to(device)
            tgt_mask = (tgt_mask_float == float("-inf"))
            out = self.transformer.decoder(
                trg_emb, memory, tgt_mask=tgt_mask,
                memory_key_padding_mask=src_pad_mask
            )
            prob = self.fc_out(out[:, -1, :])
            next_word = prob.argmax(dim=-1)
            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
            if (next_word == self.eos_idx).all():
                break
        return ys[:, 1:]


class NumericRNN(nn.Module):
    """Обёртка для RNN с числовым входом."""
    def __init__(self, rnn_layer, hidden_size, output_dim):
        super().__init__()
        self.rnn = rnn_layer
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class TextRNN(nn.Module):
    """Обёртка для RNN с текстовым входом (эмбеддинг)."""
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers,
                 output_dim, rnn_type="lstm", pad_idx=0, bos_idx=2, eos_idx=3):
        super().__init__()
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        rnn_classes = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
        rnn_cls = rnn_classes.get(rnn_type, nn.LSTM)
        self.rnn = rnn_cls(
            embedding_dim, hidden_size, num_layers,
            batch_first=True, dropout=0.0 if num_layers <= 1 else 0.1
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, src, trg=None, teacher_forcing_ratio=0.0):
        src_emb = self.embedding(src)
        out, _ = self.rnn(src_emb)
        return self.fc(out)

    @torch.no_grad()
    def generate(self, src, max_len=50):
        self.eval()
        device = src.device
        src_emb = self.embedding(src)
        out, hidden = self.rnn(src_emb)
        # Жадная генерация: подаём последний выход как вход для следующего шага
        last_output = self.fc(out[:, -1, :])
        next_char = last_output.argmax(dim=-1)
        generated = [next_char]
        current_input = next_char.unsqueeze(1)
        for _ in range(max_len - 1):
            emb = self.embedding(current_input)
            out, hidden = self.rnn(emb, hidden)
            logits = self.fc(out[:, -1, :])
            next_char = logits.argmax(dim=-1)
            generated.append(next_char)
            if next_char.item() == self.eos_idx:
                break
            current_input = next_char.unsqueeze(1)
        return torch.stack(generated, dim=1)


def create_model_from_config(config_dict):
    """Минимальный фабрик моделей для автономного запуска.
    
    Принимает словарь (совместим с ModelConfig.to_dict()).
    """
    model_type = config_dict.get("type", "mlp").lower()
    data_type = config_dict.get("data_type", "numeric")

    if model_type == "mlp":
        input_dim = config_dict.get("input_dim", 1)
        output_dim = config_dict.get("output_dim", 1)
        hidden_layers = config_dict.get("hidden_layers", [64, 128, 64])
        activation_name = config_dict.get("activation", "relu")
        dropout = config_dict.get("dropout", 0.0)
        batch_norm = config_dict.get("batch_norm", False)
        activations = {
            "relu": nn.ReLU, "gelu": nn.GELU,
            "tanh": nn.Tanh, "sigmoid": nn.Sigmoid
        }
        act_layer = activations.get(activation_name, nn.ReLU)
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_layer())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    elif model_type in ("rnn", "lstm", "gru"):
        if data_type == "text":
            vocab_size = config_dict.get("vocab_size", 100)
            embedding_dim = config_dict.get("embedding_dim", 64)
            hidden_size = config_dict.get("hidden_size", 128)
            num_layers = config_dict.get("num_layers", 2)
            return TextRNN(
                vocab_size, embedding_dim, hidden_size, num_layers,
                vocab_size, rnn_type=model_type,
                pad_idx=config_dict.get("pad_idx", 0),
                bos_idx=config_dict.get("bos_idx", 2),
                eos_idx=config_dict.get("eos_idx", 3)
            )
        else:
            input_dim = config_dict.get("input_dim", 1)
            output_dim = config_dict.get("output_dim", 1)
            hidden_size = config_dict.get("hidden_size", 128)
            num_layers = config_dict.get("num_layers", 2)
            rnn_classes = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
            rnn_layer = rnn_classes.get(model_type, nn.LSTM)(
                input_dim, hidden_size, num_layers,
                batch_first=True,
                dropout=0.0 if num_layers <= 1 else config_dict.get("dropout", 0.0)
            )
            return NumericRNN(rnn_layer, hidden_size, output_dim)

    elif model_type in ("transformer", "transformer_seq2seq"):
        return TransformerSeq2Seq(
            vocab_size=config_dict.get("vocab_size", 100),
            d_model=config_dict.get("embedding_dim", 128),
            num_heads=config_dict.get("num_heads", 8),
            num_encoder_layers=config_dict.get("num_encoder_layers", 2),
            num_decoder_layers=config_dict.get("num_decoder_layers", 2),
            dim_feedforward=config_dict.get("dim_feedforward", 512),
            activation=config_dict.get("activation", "relu"),
            max_len=config_dict.get("max_len", 100),
            dropout=config_dict.get("dropout", 0.1),
            pad_idx=config_dict.get("pad_idx", 0),
            bos_idx=config_dict.get("bos_idx", 2),
            eos_idx=config_dict.get("eos_idx", 3)
        )

    else:
        # Fallback: если архитектура не поддерживается, создаём MLP
        input_dim = config_dict.get("input_dim", 1)
        output_dim = config_dict.get("output_dim", 1)
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )


def load_everything(script_dir):
    """Загружает модель, конфиг и словарь из папки."""
    script_dir = Path(script_dir)

    # 1. Пробуем TorchScript модель
    jit_path = script_dir / "model_jit.pt"
    pth_path = script_dir / "model.pth"
    config_path = script_dir / "config.json"
    vocab_path = script_dir / "vocab.json"

    config_dict = {{}}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

    vocab = None
    if vocab_path.exists():
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

    model = None
    load_method = ""

    # Пытаемся загрузить TorchScript
    if jit_path.exists():
        try:
            model = torch.jit.load(str(jit_path), map_location="cpu")
            load_method = "TorchScript"
        except Exception:
            pass

    # Если не получилось — воссоздаём из state_dict
    if model is None and pth_path.exists():
        try:
            checkpoint = torch.load(str(pth_path), map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                ckpt_config = checkpoint.get("config", config_dict)
                if not config_dict:
                    config_dict = ckpt_config
                model = create_model_from_config(config_dict)
                model.load_state_dict(checkpoint["model_state_dict"])
                load_method = "state_dict"
            else:
                model = checkpoint
                load_method = "full_object"
        except Exception as e:
            print(f"Ошибка загрузки модели: {{e}}")
            sys.exit(1)

    if model is None:
        print("Файл модели не найден. Убедитесь, что рядом есть model.pth или model_jit.pt")
        sys.exit(1)

    model.eval()
    return model, config_dict, vocab, load_method


def predict_text(model, config_dict, vocab, input_text):
    """Предсказание для текстовой модели."""
    if vocab is None:
        return "(нет словаря — текстовое предсказание невозможно)"

    inv_vocab = {{v: k for k, v in vocab.items()}}
    pad_idx = vocab.get("<PAD>", 0)
    bos_idx = vocab.get("<BOS>", 2)
    eos_idx = vocab.get("<EOS>", 3)

    x_seq = [vocab.get(ch, vocab.get("<UNK>", 1)) for ch in input_text]
    x_tensor = torch.tensor([x_seq], dtype=torch.long)

    if hasattr(model, "generate"):
        max_len = config_dict.get("max_len", 50)
        output_indices = model.generate(x_tensor, max_len=max_len)
        generated = output_indices[0].cpu().tolist()
        chars = []
        for idx in generated:
            if idx == eos_idx:
                break
            if idx in (pad_idx, bos_idx):
                continue
            chars.append(inv_vocab.get(idx, "?"))
        return "".join(chars)
    else:
        return "(модель не поддерживает генерацию текста)"


def predict_numeric(model, config_dict, input_text):
    """Предсказание для числовой модели."""
    try:
        x_vals = [float(x) for x in input_text.replace(",", " ").split()]
    except ValueError:
        return "Ошибка: введите числа через пробел (например: 1.5 2.0)"

    expected_dim = config_dict.get("input_dim", 1)
    if len(x_vals) != expected_dim:
        return (f"Ошибка: модель ожидает {{expected_dim}} чисел на вход, "
                f"а вы ввели {{len(x_vals)}}")

    x_tensor = torch.tensor([x_vals], dtype=torch.float32)
    with torch.no_grad():
        output = model(x_tensor)
    result = output[0].cpu().numpy()
    return str([round(float(v), 4) for v in result])


def main():
    script_dir = Path(__file__).parent
    model, config_dict, vocab, load_method = load_everything(script_dir)

    data_type = config_dict.get("data_type", "numeric")
    model_name = config_dict.get("model_name", "Нейросеть")

    print("=" * 50)
    print(f"  Чат с нейросетью: {{model_name}}")
    print(f"  Архитектура: {{config_dict.get('type', 'unknown').upper()}}")
    print(f"  Тип данных: {{data_type}}")
    print(f"  Загрузка: {{load_method}}")
    print("  Введите 'выход' для завершения.")
    print("=" * 50)

    while True:
        try:
            user_input = input("\\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\\nДо свидания!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("выход", "quit", "exit"):
            print("До свидания!")
            break

        with torch.no_grad():
            if data_type == "text":
                result = predict_text(model, config_dict, vocab, user_input)
            else:
                result = predict_numeric(model, config_dict, user_input)

        print(f"Нейросеть: {{result}}")


if __name__ == "__main__":
    main()
'''

# ============================================================
#  ШАБЛОН: run_chat.bat (Windows)
# ============================================================
RUN_BAT_TEMPLATE = '''@echo off
chcp 65001 >nul
title Чат с нейросетью "{model_name}"
echo ========================================
echo   Запуск чата с нейросетью "{model_name}"
echo ========================================
echo.

REM Пытаемся найти Python
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    python "%~dp0chat.py"
) else (
    echo ОШИБКА: Python не найден в системе.
    echo Установите Python 3.9+ с сайта https://python.org
    echo При установке ОБЯЗАТЕЛЬНО поставьте галочку "Add Python to PATH".
    echo.
    pause
)
pause
'''

# ============================================================
#  ШАБЛОН: run_chat.sh (Linux/Mac)
# ============================================================
RUN_SH_TEMPLATE = '''#!/bin/bash
echo "========================================"
echo "  Запуск чата с нейросетью \\"{model_name}\\"
echo "========================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if command -v python3 &> /dev/null; then
    python3 "$SCRIPT_DIR/chat.py"
elif command -v python &> /dev/null; then
    python "$SCRIPT_DIR/chat.py"
else
    echo "ОШИБКА: Python не найден."
    echo "Установите Python 3.9+: sudo apt install python3 python3-pip"
fi
'''

# ============================================================
#  ШАБЛОН: README.txt
# ============================================================
README_TEMPLATE = '''
========================================
  НЕЙРОСЕТЬ "{model_name}" НА ФЛЕШКЕ
========================================

Что это?
--------
Это обученная нейросеть, которую ты создал(а) в OracleAI Studio.
Она умеет: {task_description}

Как запустить?
--------------
1. Дважды кликни на файл:
   - На Windows:  run_chat.bat
   - На Linux:    run_chat.sh (нужно сделать исполняемым: chmod +x run_chat.sh)

2. Откроется консоль. Введи свой вопрос или данные.

3. Напиши "выход" чтобы завершить.

Требования
----------
- Установленный Python 3.9 или новее
- Библиотека PyTorch: pip install torch

Что внутри папки?
-----------------
- model.pth      — веса нейросети (её "мозг")
- model_jit.pt   — оптимизированная версия (если создана)
- config.json    — настройки модели
- vocab.json     — словарь символов (для текстовых моделей)
- chat.py        — программа чата
- run_chat.bat   — запуск для Windows
- run_chat.sh    — запуск для Linux
- README.txt     — этот файл

Можно ли использовать в своём проекте?
---------------------------------------
Да! Файл model.pth можно загрузить обратно в OracleAI Studio
или использовать в любом Python-проекте с PyTorch.

Создано: {created_at}
Версия формата: {format_version}
'''


# ============================================================
#  КЛАСС ЭКСПОРТЁРА
# ============================================================
class Exporter:
    """
    Единая точка экспорта моделей и проектов.

    Все методы принимают и возвращают объекты (не словари).
    """

    # --------------------------------------------------------
    #  1. ПОЛНЫЙ ПРОЕКТ (.oai)
    # --------------------------------------------------------
    @staticmethod
    def export_project(
        dataset: Optional[DatasetContainer],
        model: Optional[nn.Module],
        config: ModelConfig,
        history: Optional[TrainingHistory],
        path: Path,
        project_meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Сохраняет весь проект в один файл .oai.

        Аргументы:
          dataset: DatasetContainer или None
          model: nn.Module или None
          config: ModelConfig (обязательно)
          history: TrainingHistory или None
          path: путь к файлу .oai
          project_meta: метаданные проекта (имя, описание и т.д.)

        Возвращает:
          Path к сохранённому файлу

        Бросает:
          IOError при ошибке записи
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Проверяем, что конфиг — объект ModelConfig
        if not isinstance(config, ModelConfig):
            raise TypeError(
                f"Ожидался ModelConfig, получен {type(config).__name__}"
            )

        # Проверяем, что история — объект TrainingHistory
        if history is not None and not isinstance(history, TrainingHistory):
            raise TypeError(
                f"Ожидался TrainingHistory, получен {type(history).__name__}"
            )

        # Формируем данные проекта
        project_data: Dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "export_type": "full_project",
            "metadata": {
                "created_at": Exporter._get_timestamp(),
                "app_version": "2.0",
                "torch_version": torch.__version__,
            },
            "project_meta": project_meta or {
                "name": config.model_name or "Без названия",
                "model_name": config.model_name,
                "description": "",
                "scenario": "new",
            },
            "config": config.to_dict(),
        }

        # --- Датасет ---
        if dataset is not None:
            if not isinstance(dataset, DatasetContainer):
                raise TypeError(
                    f"Ожидался DatasetContainer, получен {type(dataset).__name__}"
                )
            project_data["dataset"] = dataset.to_dict()
        else:
            project_data["dataset"] = None

        # --- История ---
        if history is not None:
            project_data["history"] = history.to_dict()
        else:
            project_data["history"] = None

        # --- Модель ---
        if model is not None:
            # Сохраняем state_dict отдельно
            project_data["model_state_dict"] = {
                k: v.cpu() for k, v in model.state_dict().items()
            }
            project_data["model_params_count"] = sum(
                p.numel() for p in model.parameters()
            )
        else:
            project_data["model_state_dict"] = None
            project_data["model_params_count"] = 0

        # --- Сохранение через torch.save (поддерживает тензоры) ---
        try:
            torch.save(project_data, path)
            logger.info(
                f"Проект сохранён: {path} "
                f"(модель: {'есть' if model else 'нет'}, "
                f"датасет: {'есть' if dataset else 'нет'})"
            )
        except Exception as e:
            logger.error(f"Ошибка сохранения проекта: {e}")
            raise IOError(f"Не удалось сохранить проект: {e}") from e

        return path

    # --------------------------------------------------------
    #  2. ТОЛЬКО МОДЕЛЬ (.pth)
    # --------------------------------------------------------
    @staticmethod
    def export_model_only(
        model: nn.Module,
        config: ModelConfig,
        path: Path,
        vocab: Optional[Dict[str, int]] = None,
        normalization: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Сохраняет только модель в .pth файл.

        Аргументы:
          model: обученная модель
          config: ModelConfig
          path: путь к .pth файлу
          vocab: словарь токенов (для текстовых моделей)
          normalization: параметры нормализации (для числовых)

        Возвращает:
          Path к сохранённому файлу
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not isinstance(config, ModelConfig):
            raise TypeError(
                f"Ожидался ModelConfig, получен {type(config).__name__}"
            )

        checkpoint = {
            "format_version": FORMAT_VERSION,
            "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "config": config.to_dict(),
            "vocab": vocab or {},
            "normalization": normalization,
            "metadata": {
                "created_at": Exporter._get_timestamp(),
                "torch_version": torch.__version__,
                "params_count": sum(p.numel() for p in model.parameters()),
                "model_name": config.model_name,
            },
        }

        try:
            torch.save(checkpoint, path)
            logger.info(f"Модель сохранена: {path}")
        except Exception as e:
            logger.error(f"Ошибка сохранения модели: {e}")
            raise IOError(f"Не удалось сохранить модель: {e}") from e

        return path

    # --------------------------------------------------------
    #  3. ФЛЕШКА-ЧАТ (папка с файлами)
    # --------------------------------------------------------
    @staticmethod
    def export_flash_chat(
        model: nn.Module,
        config: ModelConfig,
        vocab: Optional[Dict[str, int]],
        path_dir: Path,
        project_meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Создаёт папку для флешки с автономным чатом.

        Аргументы:
          model: обученная модель
          config: ModelConfig
          vocab: словарь токенов
          path_dir: путь к папке назначения
          project_meta: метаданные проекта

        Возвращает:
          Path к созданной папке
        """
        path_dir = Path(path_dir)
        path_dir.mkdir(parents=True, exist_ok=True)

        if not isinstance(config, ModelConfig):
            raise TypeError(
                f"Ожидался ModelConfig, получен {type(config).__name__}"
            )

        model_name = config.model_name or "Нейросеть"
        data_type = config.data_type.value if hasattr(config.data_type, 'value') else str(config.data_type)
        meta = project_meta or {}
        task_description = meta.get("description", "предсказание")

        # --- 1. model.pth (state_dict + config) ---
        pth_path = path_dir / "model.pth"
        checkpoint = {
            "format_version": FORMAT_VERSION,
            "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "config": config.to_dict(),
            "metadata": {
                "created_at": Exporter._get_timestamp(),
                "model_name": model_name,
            },
        }
        torch.save(checkpoint, pth_path)

        # --- 2. model_jit.pt (TorchScript, если получится) ---
        jit_path = path_dir / "model_jit.pt"
        Exporter._try_export_torchscript(model, config, jit_path)

        # --- 3. config.json ---
        config_path = path_dir / "config.json"
        config_dict = config.to_dict()
        # Убираем несериализуемые поля для JSON
        config_json = {}
        for k, v in config_dict.items():
            try:
                json.dumps(v)
                config_json[k] = v
            except (TypeError, ValueError):
                config_json[k] = str(v)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_json, f, ensure_ascii=False, indent=2)

        # --- 4. vocab.json ---
        if vocab and data_type == "text":
            vocab_path = path_dir / "vocab.json"
            with open(vocab_path, "w", encoding="utf-8") as f:
                json.dump(vocab, f, ensure_ascii=False, indent=2)

        # --- 5. chat.py ---
        chat_path = path_dir / "chat.py"
        chat_code = CHAT_PY_TEMPLATE.replace("{model_name}", model_name)
        with open(chat_path, "w", encoding="utf-8") as f:
            f.write(chat_code)

        # --- 6. run_chat.bat ---
        bat_path = path_dir / "run_chat.bat"
        bat_code = RUN_BAT_TEMPLATE.replace("{model_name}", model_name)
        with open(bat_path, "w", encoding="utf-8") as f:
            f.write(bat_code)

        # --- 7. run_chat.sh ---
        sh_path = path_dir / "run_chat.sh"
        sh_code = RUN_SH_TEMPLATE.replace("{model_name}", model_name)
        with open(sh_path, "w", encoding="utf-8") as f:
            f.write(sh_code)
        try:
            sh_path.chmod(0o755)
        except OSError:
            pass  # На Windows может не сработать

        # --- 8. README.txt ---
        readme_path = path_dir / "README.txt"
        readme_code = README_TEMPLATE.format(
            model_name=model_name,
            task_description=task_description,
            created_at=Exporter._get_timestamp(),
            format_version=FORMAT_VERSION,
        )
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_code)

        logger.info(f"Флешка-чат создана: {path_dir}")
        return path_dir

    # --------------------------------------------------------
    #  4. HTML-ПЕСОЧНИЦА (заглушка)
    # --------------------------------------------------------
    @staticmethod
    def export_html_sandbox(
        model: nn.Module,
        config: ModelConfig,
        vocab: Optional[Dict[str, int]],
        path: Path,
    ) -> Path:
        """
        Экспорт в HTML-песочницу.

        ТЕКУЩИЙ СТАТУС: ЗАГЛУШКА.
        Полноценная реализация требует ONNX Runtime Web.
        Сейчас генерирует HTML-страницу с инструкцией.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not isinstance(config, ModelConfig):
            raise TypeError(
                f"Ожидался ModelConfig, получен {type(config).__name__}"
            )

        model_name = config.model_name or "Нейросеть"
        model_type = config.architecture.value if hasattr(config.architecture, 'value') else str(config.architecture)
        data_type = config.data_type.value if hasattr(config.data_type, 'value') else str(config.data_type)
        params_count = sum(p.numel() for p in model.parameters())

        html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{model_name} — Песочница</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            background-color: #1e1e1e;
            color: #a9b7c6;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
        }}
        .card {{
            background-color: #2b2b2b;
            border: 1px solid #555;
            border-radius: 8px;
            padding: 40px;
            max-width: 600px;
            text-align: center;
        }}
        h1 {{ color: #cc7832; }}
        .info {{ color: #888; margin: 10px 0; }}
        .badge {{
            display: inline-block;
            background-color: #36414f;
            border: 1px solid #4c5052;
            border-radius: 4px;
            padding: 4px 12px;
            margin: 4px;
            font-size: 14px;
        }}
        .warning {{
            background-color: #3a2a1a;
            border: 1px solid #cc7832;
            border-radius: 4px;
            padding: 15px;
            margin-top: 20px;
            text-align: left;
        }}
    </style>
</head>
<body>
    <div class="card">
        <h1>🧠 {model_name}</h1>
        <p class="info">Модель сгенерирована в OracleAI Studio</p>
        <div>
            <span class="badge">Архитектура: {model_type.upper()}</span>
            <span class="badge">Данные: {data_type}</span>
            <span class="badge">Параметры: {params_count:,}</span>
        </div>
        <div class="warning">
            <b>⚠️ Веб-инференс пока не реализован.</b><br><br>
            Для запуска этой модели в браузере требуется
            интеграция с ONNX Runtime Web.<br><br>
            Сейчас вы можете:<br>
            • Запустить чат через <b>флешку</b> (Экспорт → Флешка-чат)<br>
            • Загрузить модель обратно в <b>OracleAI Studio</b><br>
            • Использовать файл <b>.pth</b> в своём Python-проекте
        </div>
        <p class="info" style="margin-top: 20px; font-size: 12px;">
            Версия формата: {FORMAT_VERSION} |
            Создано: {Exporter._get_timestamp()}
        </p>
    </div>
</body>
</html>"""

        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(
            f"HTML-песочница создана (заглушка): {path}. "
            f"Полноценный веб-инференс требует интеграции с ONNX Runtime Web."
        )
        return path

    # --------------------------------------------------------
    #  5. ЭКСПОРТ В ONNX
    # --------------------------------------------------------
    @staticmethod
    def export_onnx(
        model: nn.Module,
        config: ModelConfig,
        path: Path,
    ) -> Path:
        """
        Экспорт модели в формат ONNX.

        Поддерживает:
          - MLP (числовые данные)
          - Простые архитектуры без динамических циклов

        НЕ поддерживает (бросает предупреждение):
          - Трансформеры с автокорреляционной генерацией
          - Модели с условной логикой в forward
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not isinstance(config, ModelConfig):
            raise TypeError(
                f"Ожидался ModelConfig, получен {type(config).__name__}"
            )

        model_type = config.architecture.value if hasattr(config.architecture, 'value') else str(config.architecture)
        model_type = model_type.lower()
        data_type = config.data_type.value if hasattr(config.data_type, 'value') else str(config.data_type)

        # --- Определяем dummy input ---
        try:
            if data_type == "numeric":
                input_dim = config.input_dim or 1
                if isinstance(input_dim, (tuple, list)):
                    input_dim = input_dim[0] if input_dim else 1
                dummy_input = torch.randn(1, int(input_dim))
            elif data_type == "text":
                # Для текстовых моделей нужен 2D вход (батч, последовательность)
                max_len = config.arch_params.get("max_len", 100)
                dummy_input = torch.randint(0, 10, (1, int(max_len)))
                # Для seq2seq нужен и вход, и цель
                if "transformer" in model_type:
                    dummy_target = torch.randint(0, 10, (1, int(max_len)))
                    torch.onnx.export(
                        model,
                        (dummy_input, dummy_target),
                        str(path),
                        input_names=["src", "trg"],
                        output_names=["output"],
                        dynamic_axes={
                            "src": {0: "batch_size", 1: "seq_len"},
                            "trg": {0: "batch_size", 1: "seq_len"},
                            "output": {0: "batch_size", 1: "seq_len"},
                        },
                        opset_version=14,
                    )
                    logger.info(f"ONNX-экспорт успешен (seq2seq): {path}")
                    return path
            else:
                raise RuntimeError(
                    f"ONNX-экспорт не поддерживает тип данных: {data_type}"
                )

            torch.onnx.export(
                model,
                dummy_input,
                str(path),
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                opset_version=14,
            )

            logger.info(f"ONNX-экспорт успешен: {path}")
            return path

        except Exception as e:
            error_msg = (
                f"ONNX-экспорт не удался для архитектуры '{model_type}': {e}. "
                f"Некоторые архитектуры (трансформеры с генерацией, GAN) "
                f"не поддерживают статический экспорт. "
                f"Используйте сохранение в .pth вместо ONNX."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    # --------------------------------------------------------
    #  6. ОТЧЁТ (Markdown)
    # --------------------------------------------------------
    @staticmethod
    def export_report(
        config: ModelConfig,
        history: Optional[TrainingHistory],
        path: Path,
        project_meta: Optional[Dict[str, Any]] = None,
        dataset_meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Генерирует отчёт об обучении в формате Markdown.

        Аргументы:
          config: ModelConfig
          history: TrainingHistory
          path: путь к .md файлу
          project_meta: метаданные проекта
          dataset_meta: метаданные датасета (словарь)

        Возвращает:
          Path к созданному файлу
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not isinstance(config, ModelConfig):
            raise TypeError(
                f"Ожидался ModelConfig, получен {type(config).__name__}"
            )

        meta = project_meta or {}
        ds_meta = dataset_meta or {}
        model_name = config.model_name or "Модель"
        model_type = config.architecture.value if hasattr(config.architecture, 'value') else str(config.architecture)
        data_type = config.data_type.value if hasattr(config.data_type, 'value') else str(config.data_type)

        # --- Собираем статистику из истории ---
        if history is not None and isinstance(history, TrainingHistory):
            train_loss = history.train_loss or []
            val_loss = history.val_loss or []
            val_metric = history.val_metric or []

            epochs_done = len(train_loss)
            final_train_loss = train_loss[-1] if train_loss else "N/A"
            final_val_loss = val_loss[-1] if val_loss else "N/A"
            best_val_loss = min(val_loss) if val_loss else "N/A"
            best_epoch = val_loss.index(min(val_loss)) + 1 if val_loss else "N/A"

            final_metric_str = "N/A"
            if val_metric:
                last_m = val_metric[-1]
                if 0 <= last_m <= 1:
                    final_metric_str = f"{last_m:.1%}"
                else:
                    final_metric_str = f"{last_m:.4f}"
        else:
            epochs_done = 0
            final_train_loss = "N/A"
            final_val_loss = "N/A"
            best_val_loss = "N/A"
            best_epoch = "N/A"
            final_metric_str = "N/A"

        # --- Формируем Markdown ---
        lines = [
            f"# 📊 Отчёт об обучении модели «{model_name}»",
            "",
            f"**Дата:** {Exporter._get_timestamp()}",
            f"**Версия формата:** {FORMAT_VERSION}",
            "",
            "---",
            "",
            "## 🧠 Модель",
            "",
            f"| Параметр | Значение |",
            f"|---|---|",
            f"| Имя | {model_name} |",
            f"| Архитектура | {model_type.upper()} |",
            f"| Тип данных | {data_type} |",
            f"| Оптимизатор | {config.optimizer} |",
            f"| Скорость обучения | {config.learning_rate} |",
            f"| Эпох запланировано | {config.epochs} |",
            f"| Эпох завершено | {epochs_done} |",
            f"| Batch size | {config.batch_size} |",
            "",
        ]

        if ds_meta:
            lines.extend([
                "## 📁 Датасет",
                "",
                f"| Параметр | Значение |",
                f"|---|---|",
                f"| Тип | {ds_meta.get('type', 'unknown')} |",
                f"| Примеров | {ds_meta.get('num_samples', 'N/A')} |",
                f"| Задача | {ds_meta.get('task_type', ds_meta.get('dataset_id', 'N/A'))} |",
                "",
            ])

        lines.extend([
            "## 📈 Результаты обучения",
            "",
            f"| Метрика | Значение |",
            f"|---|---|",
            f"| Финальный Train Loss | {final_train_loss} |",
            f"| Финальный Val Loss | {final_val_loss} |",
            f"| Лучший Val Loss | {best_val_loss} |",
            f"| Лучшая эпоха | {best_epoch} |",
            f"| Финальная метрика (Val) | {final_metric_str} |",
            "",
        ])

        # --- Рекомендации ---
        lines.extend([
            "## 💡 Наблюдения",
            "",
        ])

        if history and isinstance(history, TrainingHistory) and len(history.val_loss) > 3:
            val_loss = history.val_loss
            train_loss = history.train_loss
            if val_loss and train_loss:
                if val_loss[-1] > train_loss[-1] * 1.5:
                    lines.append("- ⚠️ **Переобучение:** Val Loss значительно выше Train Loss. "
                                 "Рекомендуется увеличить L2-штраф или уменьшить число эпох.")
                elif val_loss[-1] > val_loss[-4] if len(val_loss) >= 4 else False:
                    lines.append("- ⚠️ **Расходимость:** Val Loss растёт. "
                                 "Рекомендуется уменьшить скорость обучения.")
                elif len(train_loss) > 5 and (train_loss[-5] - train_loss[-1]) < 0.0001:
                    lines.append("- 🐢 **Плато:** Обучение застряло. "
                                 "Рекомендуется увеличить размер модели.")
                else:
                    lines.append("- ✅ Обучение прошло стабильно. Графики сходятся.")
        else:
            lines.append("- Недостаточно данных для анализа.")

        lines.extend([
            "",
            "---",
            "",
            f"*Отчёт сгенерирован автоматически. "
            f"OracleAI Studio, {Exporter._get_timestamp()}*",
        ])

        report_text = "\n".join(lines)

        with open(path, "w", encoding="utf-8") as f:
            f.write(report_text)

        logger.info(f"Отчёт сохранён: {path}")
        return path

    # --------------------------------------------------------
    #  7. ЗАГРУЗКА ПРОЕКТА (.oai)
    # --------------------------------------------------------
    @staticmethod
    def load_project(path: Path) -> Dict[str, Any]:
        """
        Загружает полный проект из .oai файла.
        Возвращает словарь с объектами:
            - format_version: str
            - project_meta: dict
            - config: ModelConfig
            - history: TrainingHistory | None
            - dataset: DatasetContainer | None
            - model_state_dict: dict | None

        Бросает:
          FileNotFoundError если файл не существует
          ValueError если формат не поддерживается
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Файл проекта не найден: {path}")

        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise ValueError(f"Файл проекта повреждён: {e}") from e

        if not isinstance(data, dict):
            raise ValueError("Файл проекта имеет неверный формат (ожидался словарь).")

        # --- Миграция версии ---
        file_version = data.get("format_version", "1.0")
        if file_version < "2.0":
            logger.warning(
                f"Файл проекта версии {file_version}. "
                f"Выполняется миграция на {FORMAT_VERSION}."
            )
            data = Exporter._migrate_v1_to_v2(data)

        # --- Восстанавливаем объекты ---
        result = {
            "format_version": data.get("format_version", FORMAT_VERSION),
            "project_meta": data.get("project_meta", {}),
        }

        # Конфиг
        config_data = data.get("config", {})
        if config_data:
            result["config"] = ModelConfig.from_dict(config_data)
        else:
            result["config"] = ModelConfig()

        # История
        history_data = data.get("history")
        if history_data:
            result["history"] = TrainingHistory.from_dict(history_data)
        else:
            result["history"] = None

        # Датасет
        dataset_data = data.get("dataset")
        if dataset_data:
            result["dataset"] = DatasetContainer.from_dict(dataset_data)
        else:
            result["dataset"] = None

        # State_dict
        result["model_state_dict"] = data.get("model_state_dict")

        logger.info(f"Проект загружен: {path} (версия {file_version})")
        return result

    # --------------------------------------------------------
    #  8. ЗАГРУЗКА МОДЕЛИ (.pth) И ПОЛУЧЕНИЕ ОБЪЕКТОВ
    # --------------------------------------------------------
    @staticmethod
    def load_model_checkpoint(path: Path) -> Dict[str, Any]:
        """
        Загружает модель из .pth и возвращает объекты.
        Возвращает словарь:
            - model: nn.Module (созданная модель)
            - config: ModelConfig
            - vocab: dict
            - normalization: dict | None
            - history: TrainingHistory | None (если есть)
            - metadata: dict
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {path}")

        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise ValueError(f"Файл модели повреждён: {e}") from e

        if not isinstance(data, dict):
            raise ValueError("Файл модели имеет неверный формат (ожидался словарь).")

        # Восстанавливаем конфиг
        config_data = data.get("config", {})
        if config_data:
            config = ModelConfig.from_dict(config_data)
        else:
            config = ModelConfig()

        # Создаём модель
        from core.model_factory import ModelFactory
        model = ModelFactory.create_model(config)

        # Загружаем веса
        if "model_state_dict" in data:
            model.load_state_dict(data["model_state_dict"])

        # Восстанавливаем историю если есть
        history = None
        if "history" in data:
            history = TrainingHistory.from_dict(data["history"])

        return {
            "model": model,
            "config": config,
            "vocab": data.get("vocab", {}),
            "normalization": data.get("normalization"),
            "history": history,
            "metadata": data.get("metadata", {}),
        }

    # ============================================================
    #  ВНУТРЕННИЕ ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ============================================================

    @staticmethod
    def _get_timestamp() -> str:
        """Возвращает текущую дату и время в формате ISO."""
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _try_export_torchscript(
        model: nn.Module,
        config: ModelConfig,
        path: Path,
    ) -> bool:
        """
        Пытается экспортировать модель в TorchScript.
        Возвращает True если удалось, False если нет.
        Не бросает исключений.
        """
        try:
            data_type = config.data_type.value if hasattr(config.data_type, 'value') else str(config.data_type)
            model_type = config.architecture.value if hasattr(config.architecture, 'value') else str(config.architecture)
            model_type = model_type.lower()

            if data_type == "numeric":
                input_dim = config.input_dim or 1
                if isinstance(input_dim, (tuple, list)):
                    input_dim = input_dim[0] if input_dim else 1
                example_input = torch.randn(1, int(input_dim))
            elif data_type == "text":
                max_len = config.arch_params.get("max_len", 100)
                example_input = torch.randint(0, 10, (1, int(max_len)))
            else:
                logger.warning(
                    f"TorchScript: неподдерживаемый тип данных {data_type}"
                )
                return False

            # Для seq2seq нужен второй аргумент (целевая последовательность)
            if "transformer" in model_type:
                max_len = config.arch_params.get("max_len", 100)
                example_target = torch.randint(0, 10, (1, int(max_len)))
                scripted = torch.jit.trace(
                    model, (example_input, example_target),
                    check_trace=False,
                )
            else:
                scripted = torch.jit.trace(model, example_input, check_trace=False)

            scripted.save(str(path))
            logger.info(f"TorchScript-экспорт успешен: {path}")
            return True

        except Exception as e:
            logger.warning(
                f"TorchScript-экспорт не удался ({e}). "
                f"chat.py будет использовать state_dict."
            )
            return False

    @staticmethod
    def _migrate_v1_to_v2(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Миграция данных формата 1.x на формат 2.0.
        """
        migrated = dict(data)

        # Добавляем версию
        migrated["format_version"] = FORMAT_VERSION

        # Добавляем project_meta если нет
        if "project_meta" not in migrated:
            config = migrated.get("config", {})
            migrated["project_meta"] = {
                "name": "Мигрированный проект",
                "model_name": config.get("model_name", "Модель"),
                "description": "Проект мигрирован из формата 1.x",
                "scenario": "new",
            }

        # Добавляем metadata если нет
        if "metadata" not in migrated:
            migrated["metadata"] = {
                "created_at": Exporter._get_timestamp(),
                "migrated_from": data.get("format_version", "1.0"),
                "migrated_at": Exporter._get_timestamp(),
            }

        # Убеждаемся что в config есть model_name и другие поля
        if "config" in migrated and isinstance(migrated["config"], dict):
            if "model_name" not in migrated["config"]:
                migrated["config"]["model_name"] = "Модель"

        # Если есть dataset в старом формате — преобразуем
        if "dataset" in migrated and isinstance(migrated["dataset"], dict):
            # Старый формат: просто словарь с данными
            if "meta" not in migrated["dataset"]:
                migrated["dataset"] = {
                    "meta": {
                        "type": migrated["dataset"].get("type", "text"),
                        "num_samples": len(migrated["dataset"].get("raw_inputs", [])),
                    },
                    "raw_inputs": migrated["dataset"].get("raw_inputs", []),
                    "raw_outputs": migrated["dataset"].get("raw_outputs", []),
                    "vocab": migrated["dataset"].get("vocab", {}),
                }

        logger.info("Миграция проекта с 1.x на 2.0 завершена.")
        return migrated