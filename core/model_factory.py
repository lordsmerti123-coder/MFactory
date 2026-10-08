"""
core/model_factory.py
=====================
Единая точка создания всех нейросетевых архитектур проекта (зоопарк).

Отвечает за:
1. Валидацию совместимости architecture × data_type × task_type
2. Создание экземпляра nn.Module по ModelConfig
3. Подсчёт параметров и оценку памяти
4. Сохранение и загрузку моделей в формате ModelCheckpoint

Правила:
- Импортируем типы ТОЛЬКО из core.contracts.
- Каждая архитектура — отдельный приватный метод _create_<name>.
- Архитектуры со статусом "Заглушка" возвращают упрощённую модель с атрибутом
  `_is_stub = True` и понятным описанием в `_stub_message`.
- Поддерживаем старые конфиги (Dict[str, Any]) для обратной совместимости.
- Все методы возвращают объекты, а не словари.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.contracts import (
    ArchitectureType,
    DataType,
    FORMAT_VERSION,
    ModelCheckpoint,
    ModelConfig,
    TaskType,
)
from core.logger import get_logger

logger = get_logger()


# =====================================================================
#  БАЗОВЫЕ СТРОИТЕЛЬНЫЕ БЛОКИ
# =====================================================================


class PositionalEncoding(nn.Module):
    """Синусоидальное позиционное кодирование для Transformer."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


def _get_activation(name: str) -> nn.Module:
    """Возвращает слой активации по имени."""
    name = (name or "relu").lower()
    mapping = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
    }
    cls = mapping.get(name, nn.ReLU)
    return cls()


# =====================================================================
#  TRANSFORMER SEQ2SEQ (полная реализация)
# =====================================================================


class TransformerSeq2Seq(nn.Module):
    """Transformer Encoder-Decoder для задач seq2seq (математика, перевод, шифры)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 512,
        activation: str = "relu",
        max_len: int = 200,
        dropout: float = 0.1,
        pad_idx: int = 0,
        bos_idx: int = 2,
        eos_idx: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        trg_input = trg[:, :-1]
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        trg_emb = self.pos_encoder(self.embedding(trg_input) * math.sqrt(self.d_model))

        src_pad_mask = src == self.pad_idx
        tgt_pad_mask = trg_input == self.pad_idx
        tgt_mask_float = self.transformer.generate_square_subsequent_mask(
            trg_input.size(1)
        ).to(src.device)
        tgt_mask = tgt_mask_float == float("-inf")

        out = self.transformer(
            src_emb,
            trg_emb,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        return self.fc_out(out)

    @torch.no_grad()
    def generate(self, src: torch.Tensor, max_len: int = 50) -> torch.Tensor:
        self.eval()
        device = src.device
        batch_size = src.size(0)

        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        src_pad_mask = src == self.pad_idx
        memory = self.transformer.encoder(
            src_emb, src_key_padding_mask=src_pad_mask
        )

        ys = torch.full((batch_size, 1), self.bos_idx, dtype=torch.long, device=device)
        for _ in range(max_len):
            trg_emb = self.pos_encoder(self.embedding(ys) * math.sqrt(self.d_model))
            tgt_mask_float = self.transformer.generate_square_subsequent_mask(
                ys.size(1)
            ).to(device)
            tgt_mask = tgt_mask_float == float("-inf")
            out = self.transformer.decoder(
                trg_emb,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_pad_mask,
            )
            prob = self.fc_out(out[:, -1, :])
            next_word = prob.argmax(dim=-1)
            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
            if (next_word == self.eos_idx).all():
                break
        return ys[:, 1:]


# =====================================================================
#  TRANSFORMER ENCODER (классификация текста)
# =====================================================================


class TransformerEncoderClassifier(nn.Module):
    """Transformer-энкодер + пулинг + классификационная голова."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        max_len: int = 200,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )
        self.pad_idx = pad_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoder(self.embedding(x) * math.sqrt(self.embedding.embedding_dim))
        # Mean-pooling по не-pad позициям
        pad_mask = (x != self.pad_idx).unsqueeze(-1).float()
        out = self.encoder(x, src_key_padding_mask=(x == self.pad_idx))
        pooled = (out * pad_mask).sum(dim=1) / pad_mask.sum(dim=1).clamp(min=1)
        return self.classifier(pooled)


# =====================================================================
#  CNN (для изображений и спектрограмм)
# =====================================================================


class SimpleCNN(nn.Module):
    """Универсальная свёрточная сеть для классификации изображений."""

    def __init__(
        self,
        image_shape: Tuple[int, int, int],
        num_classes: int,
        num_conv_layers: int = 3,
        base_filters: int = 32,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        H, W, C = image_shape

        layers: List[nn.Module] = []
        in_ch = C
        cur_h, cur_w = H, W
        for i in range(num_conv_layers):
            out_ch = base_filters * (2 ** i)
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout),
            ]
            cur_h //= 2
            cur_w //= 2
            in_ch = out_ch

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C) → (B, C, H, W)
        if x.dim() == 4 and x.shape[-1] in (1, 3, 4) and x.shape[1] != x.shape[-1]:
            x = x.permute(0, 3, 1, 2).contiguous()
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class Conv1DClassifier(nn.Module):
    """1D-CNN для звука/спектрограмм/временных рядов."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_layers: int = 3,
        base_filters: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        in_ch = input_dim
        for i in range(num_layers):
            out_ch = base_filters * (2 ** i)
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C) → (B, C, L)
        if x.dim() == 3:
            x = x.transpose(1, 2)
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# =====================================================================
#  RNN / LSTM / GRU (числовые последовательности и time_series)
# =====================================================================


class NumericRNN(nn.Module):
    """RNN/LSTM/GRU для числовых последовательностей."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        rnn_type: str = "lstm",
        dropout: float = 0.0,
        bidirectional: bool = False,
        return_sequence: bool = False,
    ):
        super().__init__()
        rnn_classes = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
        rnn_cls = rnn_classes.get(rnn_type.lower(), nn.LSTM)
        self.rnn = rnn_cls(
            input_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.return_sequence = return_sequence
        self.is_lstm = rnn_type.lower() == "lstm"
        mult = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * mult, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        out, _ = self.rnn(x)
        if self.return_sequence:
            return self.fc(out)
        return self.fc(out[:, -1, :])


# =====================================================================
#  AUTOENCODER (числовой)
# =====================================================================


class Autoencoder(nn.Module):
    """MLP-автоэнкодер для числовых данных."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]
        enc_layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: List[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


# =====================================================================
#  CONVOLUTIONAL AUTOENCODER (для изображений)
# =====================================================================


class ConvAutoencoder(nn.Module):
    """Свёрточный автоэнкодер для очистки изображений от шума."""

    def __init__(
        self,
        image_shape: Tuple[int, int, int],
        latent_dim: int = 64,
        base_filters: int = 16,
    ):
        super().__init__()
        H, W, C = image_shape

        self.encoder = nn.Sequential(
            nn.Conv2d(C, base_filters, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 2, base_filters * 4, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(base_filters * 4 * 4, latent_dim),
        )
        self.latent_dim = latent_dim
        self.base_filters = base_filters
        self.feature_shape = (base_filters * 4, 2, 2)

        # Декодер через ConvTranspose2d.
        # Чтобы гарантировать выходной размер — используем interpolation.
        self.fc_decode = nn.Linear(latent_dim, base_filters * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_filters * 2, base_filters, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_filters, C, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        self.out_h = H
        self.out_w = W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[-1] in (1, 3, 4) and x.shape[1] != x.shape[-1]:
            x = x.permute(0, 3, 1, 2).contiguous()
        z = self.encoder(x)
        h = self.fc_decode(z).view(-1, *self.feature_shape)
        out = self.decoder(h)
        # Приводим к исходному размеру (может отличаться из-за stride)
        if out.shape[-2] != self.out_h or out.shape[-1] != self.out_w:
            out = F.interpolate(out, size=(self.out_h, self.out_w), mode="bilinear", align_corners=False)
        return out


# =====================================================================
#  VAE (Variational Autoencoder) — числовой
# =====================================================================


class VAE(nn.Module):
    """Вариационный автоэнкодер для генерации числовых данных."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]
        enc_layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        dec_layers: List[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(n, self.fc_mu.out_features, device=device)
        return self.decoder(z)


# =====================================================================
#  GAN (простейший MLP-GAN для числовых данных)
# =====================================================================


class Generator(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, output_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GAN(nn.Module):
    """
    GAN-обёртка: хранит генератор и дискриминатор.
    forward() по умолчанию пропускает вход через генератор.
    """

    def __init__(
        self,
        data_dim: int,
        latent_dim: int = 64,
        hidden: int = 128,
    ):
        super().__init__()
        self.generator = Generator(latent_dim, data_dim, hidden)
        self.discriminator = Discriminator(data_dim, hidden)
        self.latent_dim = latent_dim

    def forward(self, z: Optional[torch.Tensor] = None, batch_size: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
        if z is None:
            dev = device or next(self.parameters()).device
            z = torch.randn(batch_size, self.latent_dim, device=dev)
        return self.generator(z)

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(n, self.latent_dim, device=device)
        return self.generator(z)


# =====================================================================
#  DCGAN (свёрточный GAN для изображений)
# =====================================================================


class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim: int, image_shape: Tuple[int, int, int], base_filters: int = 64):
        super().__init__()
        H, W, C = image_shape
        self.init_size = max(4, H // 8, W // 8)
        self.base_filters = base_filters
        self.fc = nn.Linear(latent_dim, base_filters * 4 * self.init_size * self.init_size)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_filters * 2, base_filters, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_filters, C, 4, stride=2, padding=1),
            nn.Tanh(),
        )
        self.out_h = H
        self.out_w = W

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, self.base_filters * 4, self.init_size, self.init_size)
        out = self.net(x)
        if out.shape[-2] != self.out_h or out.shape[-1] != self.out_w:
            out = F.interpolate(out, size=(self.out_h, self.out_w), mode="bilinear", align_corners=False)
        return out


class DCGANDiscriminator(nn.Module):
    def __init__(self, image_shape: Tuple[int, int, int], base_filters: int = 64):
        super().__init__()
        H, W, C = image_shape
        self.net = nn.Sequential(
            nn.Conv2d(C, base_filters, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_filters, base_filters * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_filters * 2, base_filters * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(base_filters * 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[-1] in (1, 3, 4) and x.shape[1] != x.shape[-1]:
            x = x.permute(0, 3, 1, 2).contiguous()
        return self.net(x)


class DCGAN(nn.Module):
    def __init__(self, image_shape: Tuple[int, int, int], latent_dim: int = 100, base_filters: int = 64):
        super().__init__()
        self.generator = DCGANGenerator(latent_dim, image_shape, base_filters)
        self.discriminator = DCGANDiscriminator(image_shape, base_filters)
        self.latent_dim = latent_dim

    def forward(self, z: Optional[torch.Tensor] = None, batch_size: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
        if z is None:
            dev = device or next(self.parameters()).device
            z = torch.randn(batch_size, self.latent_dim, device=dev)
        return self.generator(z)

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(n, self.latent_dim, device=device)
        return self.generator(z)


# =====================================================================
#  RESNET (упрощённый, для изображений)
# =====================================================================


class _ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    """Упрощённый ResNet для учебных задач на изображениях."""

    def __init__(
        self,
        image_shape: Tuple[int, int, int],
        num_classes: int,
        blocks_per_stage: Tuple[int, ...] = (2, 2, 2),
        base_filters: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        H, W, C = image_shape
        self.conv1 = nn.Conv2d(C, base_filters, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_filters)

        layers: List[nn.Module] = []
        in_ch = base_filters
        for i, n_blocks in enumerate(blocks_per_stage):
            out_ch = base_filters * (2 ** i)
            stride = 1 if i == 0 else 2
            layers.append(_ResidualBlock(in_ch, out_ch, stride=stride))
            in_ch = out_ch
            for _ in range(n_blocks - 1):
                layers.append(_ResidualBlock(out_ch, out_ch, stride=1))
        self.res_layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[-1] in (1, 3, 4) and x.shape[1] != x.shape[-1]:
            x = x.permute(0, 3, 1, 2).contiguous()
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res_layers(x)
        x = self.pool(x)
        return self.classifier(x)


# =====================================================================
#  VISION TRANSFORMER (упрощённый)
# =====================================================================


class _PatchEmbedding(nn.Module):
    def __init__(self, image_shape: Tuple[int, int, int], patch_size: int, d_model: int):
        super().__init__()
        H, W, C = image_shape
        assert H % patch_size == 0 and W % patch_size == 0, (
            f"Размеры изображения ({H}x{W}) должны делиться на patch_size={patch_size}"
        )
        self.num_patches = (H // patch_size) * (W // patch_size)
        self.proj = nn.Conv2d(C, d_model, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[-1] in (1, 3, 4) and x.shape[1] != x.shape[-1]:
            x = x.permute(0, 3, 1, 2).contiguous()
        B = x.size(0)
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, N, d_model)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        return x


class VisionTransformer(nn.Module):
    """ViT для классификации изображений (упрощённый)."""

    def __init__(
        self,
        image_shape: Tuple[int, int, int],
        num_classes: int,
        patch_size: int = 8,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = _PatchEmbedding(image_shape, patch_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.encoder(x)
        cls_token = x[:, 0]
        return self.head(self.norm(cls_token))


# =====================================================================
#  RBFN (сеть радиально-базисных функций)
# =====================================================================


class RBFN(nn.Module):
    """Сеть радиально-базисных функций. Обучается в два этапа,
    но здесь — сквозное обучение через backprop."""

    def __init__(self, input_dim: int, output_dim: int, num_centers: int = 32):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_centers, input_dim))
        self.log_sigma = nn.Parameter(torch.zeros(num_centers))
        self.linear = nn.Linear(num_centers, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D), centers: (K, D)
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)
        dist2 = (diff ** 2).sum(dim=-1)
        sigma = torch.exp(self.log_sigma).clamp(min=1e-3)
        phi = torch.exp(-dist2 / (2 * sigma.unsqueeze(0) ** 2))
        return self.linear(phi)


# =====================================================================
#  SOM (самоорганизующаяся карта Кохонена) — дифференцируемая версия
# =====================================================================


class SOM(nn.Module):
    """
    Мягкая (soft) SOM: веса — обучаемый параметр, а ближайший нейрон
    определяется по расстоянию. Для учебных целей достаточно.
    """

    def __init__(self, input_dim: int, map_size: int = 8, num_classes: int = 0):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(map_size * map_size, input_dim))
        self.map_size = map_size
        # Если нужно классифицировать — добавляем голову
        self.head = nn.Linear(map_size * map_size, max(num_classes, 1)) if num_classes > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(1) - self.weights.unsqueeze(0)
        dist2 = (diff ** 2).sum(dim=-1)
        # мягкое распределение по нейронам (softmax по отрицательному расстоянию)
        activations = F.softmax(-dist2, dim=-1)
        if self.head is not None:
            return self.head(activations)
        return activations


# =====================================================================
#  ЗАГЛУШКИ (экспериментальные архитектуры)
# =====================================================================


class _StubModel(nn.Module):
    """
    Базовый класс-заглушка. Возвращает простой MLP-подобный результат,
    помечен флагом _is_stub и содержит _stub_message.
    Интерфейс обязан показать предупреждение о заглушке.
    """

    _is_stub: bool = True
    _stub_message: str = "Архитектура находится в режиме заглушки."

    def __init__(self, input_dim: int, output_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        return self.net(x)


class DiffusionStub(_StubModel):
    _stub_message = (
        "Диффузионная модель (DDPM) пока работает в упрощённом режиме. "
        "Реальный итеративный процесс денойзинга требует отдельного sampler-а. "
        "В обучающих целях модель имитирует поведение обычного автоэнкодера."
    )


class LatentDiffusionStub(_StubModel):
    _stub_message = (
        "Latent Diffusion — экспериментальная заглушка. "
        "В полной версии здесь должен быть автоэнкодер + диффузия в латентном пространстве."
    )


class GNNStub(_StubModel):
    _stub_message = (
        "Графовая нейросеть требует библиотеки PyTorch Geometric (torch_geometric), "
        "которая сейчас не подключена. Возвращаем упрощённый MLP."
    )


class GraphTransformerStub(_StubModel):
    _stub_message = "Graph Transformer — экспериментальная заглушка. Требуется torch_geometric."


class LiquidNNStub(_StubModel):
    _stub_message = (
        "Liquid Neural Network — экспериментальная архитектура. "
        "Сейчас работает как обычный RNN-подобный MLP."
    )


class SpikingNNStub(_StubModel):
    _stub_message = (
        "Spiking Neural Network — требует библиотеки snnTorch или Norse. "
        "Сейчас работает как обычный MLP."
    )


class PINNStub(_StubModel):
    _stub_message = (
        "Physics-Informed NN — требует специальных loss-функций с производными. "
        "Сейчас работает как обычная регрессия."
    )


class NeuralODEStub(_StubModel):
    _stub_message = (
        "Neural ODE требует библиотеки torchdiffeq. "
        "Сейчас работает как одношаговый MLP."
    )


class CapsNetStub(_StubModel):
    _stub_message = (
        "Capsule Network — экспериментальная реализация. "
        "Сейчас работает как упрощённый CNN."
    )


class CLIPStub(_StubModel):
    _stub_message = (
        "CLIP-подобная мультимодальная модель — заглушка. "
        "Для работы нужны пары (изображение, текст) и двойная архитектура."
    )


class CNNRNNStub(_StubModel):
    _stub_message = (
        "CNN+RNN (image-to-text) — экспериментальная заглушка. "
        "Полная версия требует двухмодального пайплайна."
    )


# =====================================================================
#  MODEL FACTORY
# =====================================================================


class ModelFactory:
    """
    Единая точка создания моделей.

    Правила:
    1. На вход принимает ModelConfig ИЛИ Dict[str, Any] (для обратной совместимости).
    2. Проверяет совместимость architecture × data_type.
    3. Возвращает nn.Module.
    """

    # Матрица совместимости: архитектура → множество допустимых DataType
    COMPATIBILITY: Dict[ArchitectureType, set] = {
        ArchitectureType.MLP: {DataType.NUMERIC},
        ArchitectureType.CNN: {DataType.IMAGE, DataType.AUDIO, DataType.TIME_SERIES},
        ArchitectureType.RNN: {DataType.TEXT, DataType.NUMERIC, DataType.TIME_SERIES, DataType.AUDIO},
        ArchitectureType.LSTM: {DataType.TEXT, DataType.NUMERIC, DataType.TIME_SERIES, DataType.AUDIO},
        ArchitectureType.GRU: {DataType.TEXT, DataType.NUMERIC, DataType.TIME_SERIES, DataType.AUDIO},
        ArchitectureType.TRANSFORMER_SEQ2SEQ: {DataType.TEXT},
        ArchitectureType.TRANSFORMER_ENCODER: {DataType.TEXT, DataType.TIME_SERIES},
        ArchitectureType.TRANSFORMER_DECODER: {DataType.TEXT},
        ArchitectureType.AUTOENCODER: {DataType.NUMERIC, DataType.IMAGE},
        ArchitectureType.VAE: {DataType.NUMERIC, DataType.IMAGE},
        ArchitectureType.CVAE: {DataType.NUMERIC, DataType.IMAGE},
        ArchitectureType.GAN: {DataType.NUMERIC},
        ArchitectureType.DCGAN: {DataType.IMAGE},
        ArchitectureType.DIFFUSION: {DataType.IMAGE, DataType.NUMERIC},
        ArchitectureType.LATENT_DIFFUSION: {DataType.IMAGE},
        ArchitectureType.VIT: {DataType.IMAGE},
        ArchitectureType.CAPSNET: {DataType.IMAGE},
        ArchitectureType.RESNET: {DataType.IMAGE},
        ArchitectureType.GNN: {DataType.GRAPH},
        ArchitectureType.GCN: {DataType.GRAPH},
        ArchitectureType.GAT: {DataType.GRAPH},
        ArchitectureType.GRAPH_TRANSFORMER: {DataType.GRAPH},
        ArchitectureType.RBFN: {DataType.NUMERIC},
        ArchitectureType.SOM: {DataType.NUMERIC},
        ArchitectureType.LIQUID_NN: {DataType.TIME_SERIES, DataType.NUMERIC},
        ArchitectureType.SPIKING_NN: {DataType.NUMERIC, DataType.TIME_SERIES},
        ArchitectureType.PINN: {DataType.NUMERIC, DataType.TIME_SERIES},
        ArchitectureType.NEURAL_ODE: {DataType.TIME_SERIES, DataType.NUMERIC},
        ArchitectureType.CONV_AE: {DataType.IMAGE},
        ArchitectureType.CNN_RNN: {DataType.MULTIMODAL, DataType.IMAGE},
        ArchitectureType.CLIP_LIKE: {DataType.MULTIMODAL},
        ArchitectureType.CUSTOM: set(DataType),  # разрешено всё
    }

    # ------------------------------------------------------------------
    #  Публичный API
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_config(config: Union[ModelConfig, Dict[str, Any]]) -> ModelConfig:
        """Принимает ModelConfig или старый Dict и возвращает ModelConfig."""
        if isinstance(config, ModelConfig):
            return config
        if isinstance(config, dict):
            return ModelConfig.from_dict(config)
        raise TypeError(f"Ожидался ModelConfig или dict, получено: {type(config).__name__}")

    @staticmethod
    def validate_compatibility(config: Union[ModelConfig, Dict[str, Any]]) -> List[str]:
        """
        Возвращает список ошибок совместимости.
        Пустой список = всё ок.
        """
        cfg = ModelFactory.normalize_config(config)
        errors: List[str] = []

        allowed = ModelFactory.COMPATIBILITY.get(cfg.architecture, set())
        if allowed and cfg.data_type not in allowed:
            allowed_str = ", ".join(t.value for t in allowed)
            errors.append(
                f"Архитектура '{cfg.architecture.value}' не поддерживает "
                f"тип данных '{cfg.data_type.value}'. Допустимо: {allowed_str}."
            )

        # Специфические проверки
        if cfg.architecture in (
            ArchitectureType.TRANSFORMER_SEQ2SEQ,
            ArchitectureType.TRANSFORMER_ENCODER,
            ArchitectureType.TRANSFORMER_DECODER,
            ArchitectureType.VIT,
        ):
            d_model = cfg.arch_params.get("embedding_dim") or cfg.arch_params.get("d_model", 128)
            num_heads = cfg.arch_params.get("num_heads", 8)
            if num_heads <= 0:
                errors.append("Количество голов внимания должно быть > 0.")
            elif d_model % num_heads != 0:
                suggested = ((d_model // num_heads) + 1) * num_heads
                errors.append(
                    f"Размер эмбеддинга ({d_model}) должен делиться на "
                    f"количество голов ({num_heads}) без остатка. "
                    f"Попробуйте {suggested}."
                )

        if cfg.data_type == DataType.TEXT and cfg.vocab_size < 4:
            errors.append(
                "Для текстовых моделей словарь должен содержать минимум 4 токена "
                "(PAD, UNK, BOS, EOS). Примените разбиение данных."
            )

        if cfg.data_type == DataType.IMAGE and cfg.image_shape is None:
            errors.append("Для работы с изображениями не указан image_shape в конфиге.")

        return errors

    @staticmethod
    def create_model(config: Union[ModelConfig, Dict[str, Any]]) -> nn.Module:
        """Создаёт модель по конфигу. Бросает ValueError при несовместимости."""
        cfg = ModelFactory.normalize_config(config)
        errors = ModelFactory.validate_compatibility(cfg)
        if errors:
            raise ValueError("\n".join(errors))

        dispatch = {
            ArchitectureType.MLP: ModelFactory._create_mlp,
            ArchitectureType.CNN: ModelFactory._create_cnn,
            ArchitectureType.RNN: ModelFactory._create_rnn,
            ArchitectureType.LSTM: ModelFactory._create_rnn,
            ArchitectureType.GRU: ModelFactory._create_rnn,
            ArchitectureType.TRANSFORMER_SEQ2SEQ: ModelFactory._create_transformer_seq2seq,
            ArchitectureType.TRANSFORMER_ENCODER: ModelFactory._create_transformer_encoder,
            ArchitectureType.TRANSFORMER_DECODER: ModelFactory._create_transformer_decoder,
            ArchitectureType.AUTOENCODER: ModelFactory._create_autoencoder,
            ArchitectureType.VAE: ModelFactory._create_vae,
            ArchitectureType.CVAE: ModelFactory._create_vae,
            ArchitectureType.GAN: ModelFactory._create_gan,
            ArchitectureType.DCGAN: ModelFactory._create_dcgan,
            ArchitectureType.DIFFUSION: ModelFactory._create_diffusion,
            ArchitectureType.LATENT_DIFFUSION: ModelFactory._create_latent_diffusion,
            ArchitectureType.VIT: ModelFactory._create_vit,
            ArchitectureType.CAPSNET: ModelFactory._create_capsnet,
            ArchitectureType.RESNET: ModelFactory._create_resnet,
            ArchitectureType.GNN: ModelFactory._create_gnn,
            ArchitectureType.GCN: ModelFactory._create_gnn,
            ArchitectureType.GAT: ModelFactory._create_gnn,
            ArchitectureType.GRAPH_TRANSFORMER: ModelFactory._create_graph_transformer,
            ArchitectureType.RBFN: ModelFactory._create_rbfn,
            ArchitectureType.SOM: ModelFactory._create_som,
            ArchitectureType.LIQUID_NN: ModelFactory._create_liquid_nn,
            ArchitectureType.SPIKING_NN: ModelFactory._create_spiking_nn,
            ArchitectureType.PINN: ModelFactory._create_pinn,
            ArchitectureType.NEURAL_ODE: ModelFactory._create_neural_ode,
            ArchitectureType.CONV_AE: ModelFactory._create_conv_ae,
            ArchitectureType.CNN_RNN: ModelFactory._create_cnn_rnn,
            ArchitectureType.CLIP_LIKE: ModelFactory._create_clip_like,
            ArchitectureType.CUSTOM: ModelFactory._create_custom,
        }

        creator = dispatch.get(cfg.architecture)
        if creator is None:
            raise ValueError(f"Неизвестная архитектура: {cfg.architecture}")

        model = creator(cfg)
        # Проставляем метаданные (удобно для GUI / экспорта)
        model._architecture = cfg.architecture.value  # type: ignore[attr-defined]
        model._data_type = cfg.data_type.value  # type: ignore[attr-defined]
        model._is_stub = getattr(model, "_is_stub", False)
        return model

    @staticmethod
    def get_param_count(model: nn.Module) -> int:
        """Общее число параметров."""
        return sum(p.numel() for p in model.parameters())

    @staticmethod
    def get_trainable_param_count(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def estimate_memory(model: nn.Module) -> float:
        """Оценка памяти модели в МБ (только веса и буферы, без градиентов)."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 ** 2)

    @staticmethod
    def estimate_training_memory(model: nn.Module, batch_size: int = 64) -> float:
        """Грубая оценка общей VRAM при обучении: веса + градиенты + оптим. ≈ 3× веса."""
        base = ModelFactory.estimate_memory(model)
        return base * 3.5 + batch_size * 0.05  # + небольшой оверхед на батч

    @staticmethod
    def complexity_label(model: nn.Module) -> Tuple[str, str]:
        """Возвращает (emoji, текст) для GUI-светофора."""
        n = ModelFactory.get_param_count(model)
        if n < 1_000_000:
            return "🟢", "Лёгкая — обучится за секунды на ноутбуке"
        if n < 10_000_000:
            return "🟡", "Средняя — нужна видеокарта, 5–15 минут"
        if n < 100_000_000:
            return "🔴", "Тяжёлая — нужен GPU с 4+ ГБ"
        return "💀", "Экстремальная — может не влезть в память"

    @staticmethod
    def save_model(
        model: nn.Module,
        path: Union[str, Path],
        config: Union[ModelConfig, Dict[str, Any]],
        history: Optional[Dict[str, List[float]]] = None,
        vocab: Optional[Dict[str, int]] = None,
        normalization: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Сохраняет модель в едином формате ModelCheckpoint.
        Всегда содержит format_version.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        cfg = ModelFactory.normalize_config(config)
        checkpoint = ModelCheckpoint(
            format_version=FORMAT_VERSION,
            model_state_dict={k: v.cpu() for k, v in model.state_dict().items()},
            config=cfg.to_dict(),
            history=history or {},
            vocab=vocab or {},
            normalization=normalization,
            metadata=metadata or {},
        )
        torch.save(checkpoint.to_dict(), path)
        logger.info(f"Модель сохранена: {path} (версия формата {FORMAT_VERSION})")
        return path

    @staticmethod
    def load_model(path: Union[str, Path]) -> Tuple[nn.Module, ModelConfig, Dict[str, Any]]:
        """
        Загружает модель. Возвращает (model, config, extra).
        extra содержит history, vocab, normalization, metadata.
        Поддерживает старые чекпоинты (без format_version).
        """
        path = Path(path)
        raw = torch.load(path, map_location="cpu", weights_only=False)

        # Старый формат: {"model_state_dict": ..., "config": Dict}
        if "format_version" not in raw:
            warnings.warn(
                f"Загружается чекпоинт старого формата ({path}). "
                "Будет выполнена миграция.",
                UserWarning,
            )
            config_dict = raw.get("config", {})
            model = ModelFactory.create_model(config_dict)
            model.load_state_dict(raw["model_state_dict"])
            cfg = ModelFactory.normalize_config(config_dict)
            extra = {
                "history": {},
                "vocab": {},
                "normalization": None,
                "metadata": {"migrated_from": "v1"},
            }
            return model, cfg, extra

        # Новый формат: ModelCheckpoint.to_dict()
        checkpoint = ModelCheckpoint.from_dict(raw)
        model = ModelFactory.create_model(checkpoint.config)
        model.load_state_dict(checkpoint.model_state_dict)
        cfg = ModelFactory.normalize_config(checkpoint.config)
        extra = {
            "history": checkpoint.history,
            "vocab": checkpoint.vocab,
            "normalization": checkpoint.normalization,
            "metadata": checkpoint.metadata,
        }
        logger.info(
            f"Модель загружена: {path} (версия формата {checkpoint.format_version})"
        )
        return model, cfg, extra

    # ------------------------------------------------------------------
    #  Приватные создатели архитектур
    # ------------------------------------------------------------------

    @staticmethod
    def _create_mlp(cfg: ModelConfig) -> nn.Module:
        input_dim = _resolve_input_dim(cfg)
        output_dim = _resolve_output_dim(cfg)
        hidden = cfg.arch_params.get("hidden_layers", [64, 128, 64])
        activation = cfg.arch_params.get("activation", "relu")
        dropout = float(cfg.arch_params.get("dropout", 0.0))
        batch_norm = bool(cfg.arch_params.get("batch_norm", False))

        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(_get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        return nn.Sequential(*layers)

    @staticmethod
    def _create_cnn(cfg: ModelConfig) -> nn.Module:
        if cfg.data_type == DataType.IMAGE:
            image_shape = cfg.image_shape
            if image_shape is None:
                raise ValueError("CNN: не указан image_shape для изображений.")
            return SimpleCNN(
                image_shape=image_shape,
                num_classes=_resolve_output_dim(cfg),
                num_conv_layers=cfg.arch_params.get("num_conv_layers", 3),
                base_filters=cfg.arch_params.get("base_filters", 32),
                kernel_size=cfg.arch_params.get("kernel_size", 3),
                dropout=float(cfg.arch_params.get("dropout", 0.2)),
            )
        # AUDIO / TIME_SERIES → Conv1D
        return Conv1DClassifier(
            input_dim=_resolve_input_dim(cfg),
            num_classes=_resolve_output_dim(cfg),
            num_layers=cfg.arch_params.get("num_conv_layers", 3),
            base_filters=cfg.arch_params.get("base_filters", 32),
            dropout=float(cfg.arch_params.get("dropout", 0.2)),
        )

    @staticmethod
    def _create_rnn(cfg: ModelConfig) -> nn.Module:
        rnn_type = cfg.architecture.value  # "rnn" | "lstm" | "gru"
        return NumericRNN(
            input_dim=_resolve_input_dim(cfg, default=1),
            output_dim=_resolve_output_dim(cfg),
            hidden_size=cfg.arch_params.get("hidden_size", 128),
            num_layers=cfg.arch_params.get("num_layers", 2),
            rnn_type=rnn_type,
            dropout=float(cfg.arch_params.get("dropout", 0.0)),
            bidirectional=bool(cfg.arch_params.get("bidirectional", False)),
            return_sequence=bool(cfg.arch_params.get("return_sequence", False)),
        )

    @staticmethod
    def _create_transformer_seq2seq(cfg: ModelConfig) -> nn.Module:
        d_model = cfg.arch_params.get("embedding_dim", cfg.arch_params.get("d_model", 128))
        num_layers = cfg.arch_params.get("num_layers", 2)
        return TransformerSeq2Seq(
            vocab_size=cfg.vocab_size,
            d_model=d_model,
            num_heads=cfg.arch_params.get("num_heads", 8),
            num_encoder_layers=cfg.arch_params.get("num_encoder_layers", num_layers),
            num_decoder_layers=cfg.arch_params.get("num_decoder_layers", num_layers),
            dim_feedforward=cfg.arch_params.get("dim_feedforward", d_model * 4),
            activation=cfg.arch_params.get("activation", "relu"),
            max_len=cfg.arch_params.get("max_len", 200),
            dropout=float(cfg.arch_params.get("dropout", 0.1)),
            pad_idx=cfg.arch_params.get("pad_idx", 0),
            bos_idx=cfg.arch_params.get("bos_idx", 2),
            eos_idx=cfg.arch_params.get("eos_idx", 3),
        )

    @staticmethod
    def _create_transformer_encoder(cfg: ModelConfig) -> nn.Module:
        d_model = cfg.arch_params.get("embedding_dim", cfg.arch_params.get("d_model", 128))
        num_layers = cfg.arch_params.get("num_layers", 2)
        return TransformerEncoderClassifier(
            vocab_size=cfg.vocab_size,
            num_classes=_resolve_output_dim(cfg),
            d_model=d_model,
            num_heads=cfg.arch_params.get("num_heads", 8),
            num_layers=cfg.arch_params.get("num_encoder_layers", num_layers),
            dim_feedforward=cfg.arch_params.get("dim_feedforward", d_model * 4),
            max_len=cfg.arch_params.get("max_len", 200),
            dropout=float(cfg.arch_params.get("dropout", 0.1)),
            pad_idx=cfg.arch_params.get("pad_idx", 0),
        )

    @staticmethod
    def _create_transformer_decoder(cfg: ModelConfig) -> nn.Module:
        # Упрощённо: используем Seq2Seq без энкодера (заглушка)
        warnings.warn("Transformer Decoder пока реализован как упрощённый Seq2Seq.")
        return ModelFactory._create_transformer_seq2seq(cfg)

    @staticmethod
    def _create_autoencoder(cfg: ModelConfig) -> nn.Module:
        if cfg.data_type == DataType.IMAGE:
            return ModelFactory._create_conv_ae(cfg)
        return Autoencoder(
            input_dim=_resolve_input_dim(cfg),
            latent_dim=cfg.arch_params.get("latent_dim", 32),
            hidden_dims=cfg.arch_params.get("hidden_dims", [128, 64]),
            dropout=float(cfg.arch_params.get("dropout", 0.1)),
        )

    @staticmethod
    def _create_vae(cfg: ModelConfig) -> nn.Module:
        return VAE(
            input_dim=_resolve_input_dim(cfg),
            latent_dim=cfg.arch_params.get("latent_dim", 32),
            hidden_dims=cfg.arch_params.get("hidden_dims", [128, 64]),
        )

    @staticmethod
    def _create_gan(cfg: ModelConfig) -> nn.Module:
        return GAN(
            data_dim=_resolve_input_dim(cfg),
            latent_dim=cfg.arch_params.get("latent_dim", 64),
            hidden=cfg.arch_params.get("hidden", 128),
        )

    @staticmethod
    def _create_dcgan(cfg: ModelConfig) -> nn.Module:
        if cfg.image_shape is None:
            raise ValueError("DCGAN: требуется image_shape.")
        return DCGAN(
            image_shape=cfg.image_shape,
            latent_dim=cfg.arch_params.get("latent_dim", 100),
            base_filters=cfg.arch_params.get("base_filters", 64),
        )

    @staticmethod
    def _create_diffusion(cfg: ModelConfig) -> nn.Module:
        in_dim = _resolve_input_dim(cfg)
        out_dim = _resolve_output_dim(cfg, default=in_dim)
        return DiffusionStub(in_dim, out_dim)

    @staticmethod
    def _create_latent_diffusion(cfg: ModelConfig) -> nn.Module:
        in_dim = _resolve_input_dim(cfg)
        out_dim = _resolve_output_dim(cfg, default=in_dim)
        return LatentDiffusionStub(in_dim, out_dim)

    @staticmethod
    def _create_vit(cfg: ModelConfig) -> nn.Module:
        if cfg.image_shape is None:
            raise ValueError("ViT: требуется image_shape.")
        return VisionTransformer(
            image_shape=cfg.image_shape,
            num_classes=_resolve_output_dim(cfg),
            patch_size=cfg.arch_params.get("patch_size", 8),
            d_model=cfg.arch_params.get("embedding_dim", cfg.arch_params.get("d_model", 128)),
            num_heads=cfg.arch_params.get("num_heads", 8),
            num_layers=cfg.arch_params.get("num_layers", 4),
            dim_feedforward=cfg.arch_params.get("dim_feedforward", 512),
            dropout=float(cfg.arch_params.get("dropout", 0.1)),
        )

    @staticmethod
    def _create_capsnet(cfg: ModelConfig) -> nn.Module:
        in_dim = _resolve_input_dim(cfg)
        out_dim = _resolve_output_dim(cfg, default=10)
        return CapsNetStub(in_dim, out_dim)

    @staticmethod
    def _create_resnet(cfg: ModelConfig) -> nn.Module:
        if cfg.image_shape is None:
            raise ValueError("ResNet: требуется image_shape.")
        return ResNet(
            image_shape=cfg.image_shape,
            num_classes=_resolve_output_dim(cfg),
            blocks_per_stage=tuple(cfg.arch_params.get("blocks_per_stage", [2, 2, 2])),
            base_filters=cfg.arch_params.get("base_filters", 32),
            dropout=float(cfg.arch_params.get("dropout", 0.2)),
        )

    @staticmethod
    def _create_gnn(cfg: ModelConfig) -> nn.Module:
        in_dim = _resolve_input_dim(cfg, default=16)
        out_dim = _resolve_output_dim(cfg, default=2)
        return GNNStub(in_dim, out_dim)

    @staticmethod
    def _create_graph_transformer(cfg: ModelConfig) -> nn.Module:
        in_dim = _resolve_input_dim(cfg, default=16)
        out_dim = _resolve_output_dim(cfg, default=2)
        return GraphTransformerStub(in_dim, out_dim)

    @staticmethod
    def _create_rbfn(cfg: ModelConfig) -> nn.Module:
        return RBFN(
            input_dim=_resolve_input_dim(cfg),
            output_dim=_resolve_output_dim(cfg),
            num_centers=cfg.arch_params.get("num_centers", 32),
        )

    @staticmethod
    def _create_som(cfg: ModelConfig) -> nn.Module:
        return SOM(
            input_dim=_resolve_input_dim(cfg),
            map_size=cfg.arch_params.get("map_size", 8),
            num_classes=_resolve_output_dim(cfg, default=0),
        )

    @staticmethod
    def _create_liquid_nn(cfg: ModelConfig) -> nn.Module:
        in_dim = _resolve_input_dim(cfg)
        out_dim = _resolve_output_dim(cfg)
        return LiquidNNStub(in_dim, out_dim)

    @staticmethod
    def _create_spiking_nn(cfg: ModelConfig) -> nn.Module:
        in_dim = _resolve_input_dim(cfg)
        out_dim = _resolve_output_dim(cfg)
        return SpikingNNStub(in_dim, out_dim)

    @staticmethod
    def _create_pinn(cfg: ModelConfig) -> nn.Module:
        in_dim = _resolve_input_dim(cfg)
        out_dim = _resolve_output_dim(cfg)
        return PINNStub(in_dim, out_dim)

    @staticmethod
    def _create_neural_ode(cfg: ModelConfig) -> nn.Module:
        in_dim = _resolve_input_dim(cfg)
        out_dim = _resolve_output_dim(cfg)
        return NeuralODEStub(in_dim, out_dim)

    @staticmethod
    def _create_conv_ae(cfg: ModelConfig) -> nn.Module:
        if cfg.image_shape is None:
            raise ValueError("ConvAE: требуется image_shape.")
        return ConvAutoencoder(
            image_shape=cfg.image_shape,
            latent_dim=cfg.arch_params.get("latent_dim", 64),
            base_filters=cfg.arch_params.get("base_filters", 16),
        )

    @staticmethod
    def _create_cnn_rnn(cfg: ModelConfig) -> nn.Module:
        in_dim = _resolve_input_dim(cfg, default=128)
        out_dim = _resolve_output_dim(cfg, default=10)
        return CNNRNNStub(in_dim, out_dim)

    @staticmethod
    def _create_clip_like(cfg: ModelConfig) -> nn.Module:
        in_dim = _resolve_input_dim(cfg, default=128)
        out_dim = _resolve_output_dim(cfg, default=128)
        return CLIPStub(in_dim, out_dim)

    @staticmethod
    def _create_custom(cfg: ModelConfig) -> nn.Module:
        """
        CUSTOM — это обобщённый MLP для произвольных размерностей.
        Продвинутый пользователь может subclass-ить в своём коде.
        """
        return ModelFactory._create_mlp(cfg)


# =====================================================================
#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =====================================================================


def _resolve_input_dim(cfg: ModelConfig, default: int = 1) -> int:
    """Извлекает input_dim из конфига, учитывая разные типы данных."""
    if cfg.input_dim is not None:
        if isinstance(cfg.input_dim, tuple):
            # Для изображений: (H, W, C) → плоский размер (если это не CNN)
            n = 1
            for v in cfg.input_dim:
                n *= int(v)
            return n
        return int(cfg.input_dim)
    if cfg.image_shape is not None:
        H, W, C = cfg.image_shape
        return int(H) * int(W) * int(C)
    if cfg.vocab_size > 0:
        # Для текстовых моделей input_dim часто не нужен (используется embedding)
        return cfg.vocab_size
    return default


def _resolve_output_dim(cfg: ModelConfig, default: int = 1) -> int:
    if cfg.output_dim is not None:
        if isinstance(cfg.output_dim, tuple):
            n = 1
            for v in cfg.output_dim:
                n *= int(v)
            return n
        return int(cfg.output_dim)
    if cfg.num_classes > 0:
        return int(cfg.num_classes)
    if cfg.task_type == TaskType.SEQ2SEQ and cfg.vocab_size > 0:
        return cfg.vocab_size
    return default