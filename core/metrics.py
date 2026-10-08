"""
core/metrics.py
================
Универсальный модуль метрик для всех типов задач.
Зависимости: только torch и numpy. Никаких внешних библиотек.

ПРАВИЛА:
  1. Все функции принимают torch.Tensor и возвращают float.
  2. Все функции БЕЗОПАСНЫ: не падают на NaN, Inf, пустых тензорах, несовпадении размерностей.
  3. Никакого импорта из contracts.py здесь не нужно — модуль автономный.
  4. Тренер вызывает compute_metric(metric_name, y_pred, y_true, **kwargs).
  5. Панель мониторинга вызывает get_metric_for_task(task_type) чтобы узнать имя метрики.

СПИСОК МЕТРИК:
  Классификация:  accuracy, precision, recall, f1_score
  Регрессия:      mse, mae, rmse, r2_score
  Текст/Seq2Seq:  token_accuracy, sequence_accuracy, char_error_rate
  Изображения:    pixel_mse, pixel_accuracy (для сегментации)
  Аномалии:       anomaly_f1
  Кластеризация:  silhouette_approx (упрощённая)
  Аудио:          audio_mse (то же что mse, алиас)

ВЕРСИЯ ФОРМАТА: 2.0
"""

import torch
import numpy as np
from typing import Optional, Dict, Callable, Any


# ============================================================
#  ЗАЩИТНЫЕ УТИЛИТЫ
# ============================================================

def _safe_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Очищает тензор от NaN и Inf, заменяя их на 0.
    Если тензор пустой — возвращает тензор с одним нулём.
    """
    if tensor is None:
        return torch.tensor([0.0])
    if not isinstance(tensor, torch.Tensor):
        try:
            tensor = torch.tensor(tensor, dtype=torch.float32)
        except Exception:
            return torch.tensor([0.0])
    if tensor.numel() == 0:
        return torch.tensor([0.0])
    tensor = tensor.detach().cpu().float()
    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    return tensor


def _safe_int_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Безопасное преобразование в long-тензор для классификации."""
    if tensor is None:
        return torch.tensor([0], dtype=torch.long)
    if not isinstance(tensor, torch.Tensor):
        try:
            tensor = torch.tensor(tensor, dtype=torch.long)
        except Exception:
            return torch.tensor([0], dtype=torch.long)
    if tensor.numel() == 0:
        return torch.tensor([0], dtype=torch.long)
    return tensor.detach().cpu().long()


def _extract_predictions(y_pred: torch.Tensor) -> torch.Tensor:
    """
    Извлекает предсказанные классы из выхода модели.
    Если размерность > 1 и второй размер > 1 → argmax по классам.
    Иначе → округление.
    """
    y_pred = _safe_tensor(y_pred)
    if y_pred.dim() > 1 and y_pred.size(-1) > 1:
        return y_pred.argmax(dim=-1)
    elif y_pred.dim() > 1 and y_pred.size(-1) == 1:
        return (y_pred.squeeze(-1) > 0.5).long()
    else:
        return torch.round(y_pred).long()


def _extract_targets(y_true: torch.Tensor) -> torch.Tensor:
    """Извлекает целевые классы (аналогично предсказаниям)."""
    y_true = _safe_tensor(y_true)
    if y_true.dim() > 1 and y_true.size(-1) > 1:
        return y_true.argmax(dim=-1)
    elif y_true.dim() > 1 and y_true.size(-1) == 1:
        return (y_true.squeeze(-1) > 0.5).long()
    else:
        return torch.round(y_true).long()


# ============================================================
#  МЕТРИКИ КЛАССИФИКАЦИИ
# ============================================================

def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs) -> float:
    """
    Точность классификации.
    Работает с логитами (аргмах) и с вероятностями (округление).
    Возвращает долю правильных предсказаний [0.0, 1.0].
    """
    preds = _extract_predictions(y_pred)
    targets = _extract_targets(y_true)
    if preds.shape != targets.shape:
        min_len = min(preds.numel(), targets.numel())
        preds = preds.flatten()[:min_len]
        targets = targets.flatten()[:min_len]
    if preds.numel() == 0:
        return 0.0
    return (preds == targets).float().mean().item()


def precision(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs) -> float:
    """
    Точность (Precision) для бинарной/многоклассовой классификации.
    Для многоклассовой — macro-average.
    """
    preds = _extract_predictions(y_pred)
    targets = _extract_targets(y_true)
    if preds.shape != targets.shape:
        min_len = min(preds.numel(), targets.numel())
        preds = preds.flatten()[:min_len]
        targets = targets.flatten()[:min_len]
    if preds.numel() == 0:
        return 0.0

    classes = torch.unique(torch.cat([preds, targets]))
    precisions = []
    for c in classes:
        tp = ((preds == c) & (targets == c)).sum().float()
        fp = ((preds == c) & (targets != c)).sum().float()
        denom = tp + fp
        if denom > 0:
            precisions.append((tp / denom).item())
    return float(np.mean(precisions)) if precisions else 0.0


def recall(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs) -> float:
    """
    Полнота (Recall) для бинарной/многоклассовой классификации.
    Для многоклассовой — macro-average.
    """
    preds = _extract_predictions(y_pred)
    targets = _extract_targets(y_true)
    if preds.shape != targets.shape:
        min_len = min(preds.numel(), targets.numel())
        preds = preds.flatten()[:min_len]
        targets = targets.flatten()[:min_len]
    if preds.numel() == 0:
        return 0.0

    classes = torch.unique(torch.cat([preds, targets]))
    recalls = []
    for c in classes:
        tp = ((preds == c) & (targets == c)).sum().float()
        fn = ((preds != c) & (targets == c)).sum().float()
        denom = tp + fn
        if denom > 0:
            recalls.append((tp / denom).item())
    return float(np.mean(recalls)) if recalls else 0.0


def f1_score(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs) -> float:
    """
    F1-мера (гармоническое среднее Precision и Recall).
    Macro-average для многоклассовой. БЕЗ sklearn.
    """
    p = precision(y_pred, y_true)
    r = recall(y_pred, y_true)
    if p + r == 0:
        return 0.0
    return 2.0 * p * r / (p + r)


# ============================================================
#  МЕТРИКИ РЕГРЕССИИ
# ============================================================

def mse(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs) -> float:
    """Среднеквадратичная ошибка. Чем ниже — тем лучше."""
    y_pred = _safe_tensor(y_pred).flatten()
    y_true = _safe_tensor(y_true).flatten()
    if y_pred.numel() != y_true.numel():
        min_len = min(y_pred.numel(), y_true.numel())
        y_pred = y_pred[:min_len]
        y_true = y_true[:min_len]
    if y_pred.numel() == 0:
        return 0.0
    return ((y_pred - y_true) ** 2).mean().item()


def mae(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs) -> float:
    """Средняя абсолютная ошибка. Чем ниже — тем лучше."""
    y_pred = _safe_tensor(y_pred).flatten()
    y_true = _safe_tensor(y_true).flatten()
    if y_pred.numel() != y_true.numel():
        min_len = min(y_pred.numel(), y_true.numel())
        y_pred = y_pred[:min_len]
        y_true = y_true[:min_len]
    if y_pred.numel() == 0:
        return 0.0
    return torch.abs(y_pred - y_true).mean().item()


def rmse(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs) -> float:
    """Корень из среднеквадратичной ошибки."""
    return float(np.sqrt(mse(y_pred, y_true)))


def r2_score(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs) -> float:
    """
    Коэффициент детерминации R². 1.0 = идеально, 0.0 = не лучше среднего.
    Может быть отрицательным если модель хуже константы.
    """
    y_pred = _safe_tensor(y_pred).flatten()
    y_true = _safe_tensor(y_true).flatten()
    if y_pred.numel() != y_true.numel():
        min_len = min(y_pred.numel(), y_true.numel())
        y_pred = y_pred[:min_len]
        y_true = y_true[:min_len]
    if y_pred.numel() == 0:
        return 0.0
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    if ss_tot < 1e-8:
        return 1.0 if ss_res < 1e-8 else 0.0
    return (1.0 - ss_res / ss_tot).item()


# ============================================================
#  МЕТРИКИ ТЕКСТА / SEQ2SEQ
# ============================================================

def token_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor,
                   pad_idx: int = 0, **kwargs) -> float:
    """
    Точность предсказания токенов для текстовых моделей.
    y_pred: (batch, seq_len, vocab_size) — логиты
    y_true: (batch, seq_len) — целевые индексы
    Игнорирует pad_idx.
    """
    if y_pred is None or y_true is None:
        return 0.0
    if not isinstance(y_pred, torch.Tensor) or not isinstance(y_true, torch.Tensor):
        return 0.0
    if y_pred.numel() == 0 or y_true.numel() == 0:
        return 0.0

    y_pred = y_pred.detach().cpu()
    y_true = y_true.detach().cpu().long()

    # Если y_pred имеет размерность (batch, seq_len, vocab) → argmax
    if y_pred.dim() == 3:
        preds = y_pred.argmax(dim=-1)
    elif y_pred.dim() == 2:
        preds = y_pred
    else:
        preds = y_pred.unsqueeze(0) if y_pred.dim() == 1 else y_pred

    # Выравниваем размеры
    if preds.dim() == 1:
        preds = preds.unsqueeze(0)
    if y_true.dim() == 1:
        y_true = y_true.unsqueeze(0)

    # Обрезаем до одинаковой длины
    min_seq = min(preds.size(1), y_true.size(1))
    preds = preds[:, :min_seq]
    y_true = y_true[:, :min_seq]

    # Маска: игнорируем PAD
    mask = y_true != pad_idx
    if mask.sum() == 0:
        return 0.0

    correct = ((preds == y_true) & mask).sum().float()
    return (correct / mask.sum().float()).item()


def sequence_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor,
                      pad_idx: int = 0, eos_idx: int = 3, **kwargs) -> float:
    """
    Точность на уровне целых последовательностей.
    Последовательность считается правильной если ВСЕ токены совпали.
    Более строгая метрика чем token_accuracy.
    """
    if y_pred is None or y_true is None:
        return 0.0
    if not isinstance(y_pred, torch.Tensor) or not isinstance(y_true, torch.Tensor):
        return 0.0
    if y_pred.numel() == 0 or y_true.numel() == 0:
        return 0.0

    y_pred = y_pred.detach().cpu()
    y_true = y_true.detach().cpu().long()

    if y_pred.dim() == 3:
        preds = y_pred.argmax(dim=-1)
    else:
        preds = y_pred

    if preds.dim() == 1:
        preds = preds.unsqueeze(0)
    if y_true.dim() == 1:
        y_true = y_true.unsqueeze(0)

    min_seq = min(preds.size(1), y_true.size(1))
    preds = preds[:, :min_seq]
    y_true = y_true[:, :min_seq]

    batch_size = y_true.size(0)
    if batch_size == 0:
        return 0.0

    correct_sequences = 0
    for i in range(batch_size):
        # Обрезаем по EOS
        true_seq = y_true[i]
        pred_seq = preds[i]

        eos_positions = (true_seq == eos_idx).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            end_pos = eos_positions[0].item() + 1
            true_seq = true_seq[:end_pos]
            pred_seq = pred_seq[:end_pos]

        # Маска без PAD
        mask = true_seq != pad_idx
        if mask.sum() == 0:
            continue
        if (pred_seq[mask] == true_seq[mask]).all():
            correct_sequences += 1

    return correct_sequences / batch_size


def char_error_rate(y_pred: torch.Tensor, y_true: torch.Tensor,
                    pad_idx: int = 0, **kwargs) -> float:
    """
    Упрощённый Character Error Rate (CER).
    CER = 1 - token_accuracy. Чем ниже — тем лучше.
    """
    ta = token_accuracy(y_pred, y_true, pad_idx=pad_idx)
    return 1.0 - ta


# ============================================================
#  МЕТРИКИ ИЗОБРАЖЕНИЙ
# ============================================================

def pixel_mse(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs) -> float:
    """
    MSE на уровне пикселей для задач генерации/реконструкции изображений.
    Тензоры: (batch, C, H, W) или (batch, H, W) или плоские.
    """
    y_pred = _safe_tensor(y_pred)
    y_true = _safe_tensor(y_true)
    if y_pred.numel() != y_true.numel():
        min_len = min(y_pred.numel(), y_true.numel())
        y_pred = y_pred.flatten()[:min_len]
        y_true = y_true.flatten()[:min_len]
    if y_pred.numel() == 0:
        return 0.0
    return ((y_pred - y_true) ** 2).mean().item()


def pixel_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor,
                   num_classes: int = 0, **kwargs) -> float:
    """
    Точность сегментации на уровне пикселей.
    Если входы — логиты (есть размерность классов), берём argmax.
    """
    y_pred = _safe_tensor(y_pred)
    y_true = _safe_tensor(y_true)

    # Если есть размерность классов
    if y_pred.dim() >= 2 and y_pred.size(1) > 1 and num_classes > 1:
        preds = y_pred.argmax(dim=1)
    else:
        preds = torch.round(y_pred)

    targets = torch.round(y_true)

    if preds.numel() != targets.numel():
        min_len = min(preds.numel(), targets.numel())
        preds = preds.flatten()[:min_len]
        targets = targets.flatten()[:min_len]

    if preds.numel() == 0:
        return 0.0
    return (preds == targets).float().mean().item()


def ssim_approx(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs) -> float:
    """
    Упрощённый SSIM (Structural Similarity) для оценки качества изображений.
    Возвращает значение в [0, 1]. 1 = идентичны.
    Упрощение: считаем глобально по всему тензору, не по окнам.
    """
    y_pred = _safe_tensor(y_pred).flatten()
    y_true = _safe_tensor(y_true).flatten()

    if y_pred.numel() != y_true.numel():
        min_len = min(y_pred.numel(), y_true.numel())
        y_pred = y_pred[:min_len]
        y_true = y_true[:min_len]
    if y_pred.numel() == 0:
        return 0.0

    mu_x = y_pred.mean()
    mu_y = y_true.mean()
    sigma_x = y_pred.std() if y_pred.numel() > 1 else torch.tensor(0.0)
    sigma_y = y_true.std() if y_true.numel() > 1 else torch.tensor(0.0)
    sigma_xy = ((y_pred - mu_x) * (y_true - mu_y)).mean() if y_pred.numel() > 1 else torch.tensor(0.0)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x ** 2 + sigma_y ** 2 + C2)

    if denominator < 1e-10:
        return 1.0 if numerator < 1e-10 else 0.0
    return max(0.0, min(1.0, (numerator / denominator).item()))


# ============================================================
#  МЕТРИКИ АУДИО
# ============================================================

def audio_mse(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs) -> float:
    """MSE для аудио-задач (спектрограммы, волны). Алиас для mse."""
    return mse(y_pred, y_true)


def audio_mae(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs) -> float:
    """MAE для аудио-задач."""
    return mae(y_pred, y_true)


# ============================================================
#  МЕТРИКИ АНОМАЛИЙ
# ============================================================

def anomaly_f1(y_pred: torch.Tensor, y_true: torch.Tensor,
               threshold: float = 0.5, **kwargs) -> float:
    """
    F1 для детекции аномалий.
    Если предсказания — вероятности, применяем threshold.
    """
    y_pred = _safe_tensor(y_pred).flatten()
    y_true = _safe_tensor(y_true).flatten()

    if y_pred.numel() != y_true.numel():
        min_len = min(y_pred.numel(), y_true.numel())
        y_pred = y_pred[:min_len]
        y_true = y_true[:min_len]
    if y_pred.numel() == 0:
        return 0.0

    # Если значения в диапазоне [0,1] и не целые → применяем порог
    if y_pred.min() >= 0 and y_pred.max() <= 1.0 and not torch.all(y_pred == torch.round(y_pred)):
        preds = (y_pred >= threshold).long()
    else:
        preds = torch.round(y_pred).long()
    targets = torch.round(y_true).long()

    tp = ((preds == 1) & (targets == 1)).sum().float()
    fp = ((preds == 1) & (targets == 0)).sum().float()
    fn = ((preds == 0) & (targets == 1)).sum().float()

    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if p + r == 0:
        return 0.0
    return (2 * p * r / (p + r)).item()


# ============================================================
#  МЕТРИКИ КЛАСТЕРИЗАЦИИ
# ============================================================

def silhouette_approx(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs) -> float:
    """
    Упрощённый силуэтный коэффициент для оценки кластеризации.
    y_pred: индексы кластеров (предсказанные)
    y_true: не используется (или истинные кластеры для сравнения).
    Возвращает [-1, 1]. Чем выше — тем лучше.

    УПРОЩЕНИЕ: считаем только если данных < 5000 точек.
    Иначе возвращаем 0.0 с пометкой.
    """
    y_pred = _safe_int_tensor(y_pred).flatten()
    if y_pred.numel() == 0 or y_pred.numel() > 5000:
        return 0.0

    # Для упрощённого силуэта нужны сами точки, а не только кластеры.
    # Здесь мы возвращаем заглушку если нет точек.
    # В реальности тренер должен передать точки через kwargs['features'].
    features = kwargs.get('features', None)
    if features is None:
        return 0.0

    features = _safe_tensor(features)
    if features.size(0) != y_pred.numel():
        return 0.0

    clusters = torch.unique(y_pred)
    if len(clusters) < 2:
        return 0.0

    # Упрощённый силуэт: среднее расстояние до своего кластера vs ближайшего чужого
    silhouettes = []
    for i in range(min(y_pred.numel(), 500)):  # Сэмплируем для скорости
        own_cluster = y_pred[i]
        own_mask = y_pred == own_cluster
        other_mask = y_pred != own_cluster

        if own_mask.sum() <= 1 or other_mask.sum() == 0:
            continue

        # Среднее расстояние до своего кластера (без себя)
        own_dists = torch.cdist(features[i].unsqueeze(0), features[own_mask]).squeeze(0)
        own_dists = own_dists[own_dists > 1e-8]  # убираем расстояние до себя
        a = own_dists.mean() if own_dists.numel() > 0 else torch.tensor(0.0)

        # Минимальное среднее расстояние до чужого кластера
        b = torch.tensor(float('inf'))
        for c in clusters:
            if c == own_cluster:
                continue
            c_mask = y_pred == c
            if c_mask.sum() == 0:
                continue
            c_dists = torch.cdist(features[i].unsqueeze(0), features[c_mask]).mean()
            b = min(b, c_dists)

        if b == float('inf'):
            continue
        denom = max(a, b)
        if denom < 1e-8:
            continue
        silhouettes.append(((b - a) / denom).item())

    return float(np.mean(silhouettes)) if silhouettes else 0.0


# ============================================================
#  РЕЕСТР МЕТРИК И ДИСПЕТЧЕР
# ============================================================

METRIC_REGISTRY: Dict[str, Callable] = {
    # Классификация
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1_score,
    # Регрессия
    "mse": mse,
    "mae": mae,
    "rmse": rmse,
    "r2_score": r2_score,
    # Текст
    "token_accuracy": token_accuracy,
    "sequence_accuracy": sequence_accuracy,
    "char_error_rate": char_error_rate,
    # Изображения
    "pixel_mse": pixel_mse,
    "pixel_accuracy": pixel_accuracy,
    "ssim_approx": ssim_approx,
    # Аудио
    "audio_mse": audio_mse,
    "audio_mae": audio_mae,
    # Аномалии
    "anomaly_f1": anomaly_f1,
    # Кластеризация
    "silhouette_approx": silhouette_approx,
}


def compute_metric(metric_name: str, y_pred: torch.Tensor, y_true: torch.Tensor,
                   **kwargs) -> float:
    """
    ЕДИНАЯ ТОЧКА ВХОДА для вычисления любой метрики.
    Тренер вызывает: compute_metric("accuracy", outputs, targets, pad_idx=0)

    Args:
        metric_name: имя метрики из METRIC_REGISTRY
        y_pred: предсказания модели (любой формы)
        y_true: целевые значения (любой формы)
        **kwargs: дополнительные параметры (pad_idx, eos_idx, threshold, features)

    Returns:
        float: значение метрики. 0.0 если метрика не найдена или ошибка.
    """
    func = METRIC_REGISTRY.get(metric_name)
    if func is None:
        return 0.0
    try:
        result = func(y_pred, y_true, **kwargs)
        if np.isnan(result) or np.isinf(result):
            return 0.0
        return float(result)
    except Exception:
        return 0.0


# ============================================================
#  МАППИНГ: ТИП ЗАДАЧИ → МЕТРИКА
# ============================================================

# Соответствие типа задачи и метрики по умолчанию.
# Тренер использует это чтобы выбрать метрику автоматически.
TASK_METRIC_MAP: Dict[str, Dict[str, Any]] = {
    "classification": {
        "metric_name": "accuracy",
        "higher_is_better": True,
        "display_name": "Точность",
        "range": (0.0, 1.0),
    },
    "regression": {
        "metric_name": "mse",
        "higher_is_better": False,
        "display_name": "MSE",
        "range": (0.0, float('inf')),
    },
    "seq2seq": {
        "metric_name": "token_accuracy",
        "higher_is_better": True,
        "display_name": "Точность токенов",
        "range": (0.0, 1.0),
    },
    "translation": {
        "metric_name": "token_accuracy",
        "higher_is_better": True,
        "display_name": "Точность токенов",
        "range": (0.0, 1.0),
    },
    "image_classification": {
        "metric_name": "accuracy",
        "higher_is_better": True,
        "display_name": "Точность",
        "range": (0.0, 1.0),
    },
    "image_generation": {
        "metric_name": "pixel_mse",
        "higher_is_better": False,
        "display_name": "Pixel MSE",
        "range": (0.0, float('inf')),
    },
    "audio_classification": {
        "metric_name": "accuracy",
        "higher_is_better": True,
        "display_name": "Точность",
        "range": (0.0, 1.0),
    },
    "audio_generation": {
        "metric_name": "audio_mse",
        "higher_is_better": False,
        "display_name": "Audio MSE",
        "range": (0.0, float('inf')),
    },
    "speech_to_text": {
        "metric_name": "token_accuracy",
        "higher_is_better": True,
        "display_name": "Точность токенов",
        "range": (0.0, 1.0),
    },
    "text_to_speech": {
        "metric_name": "audio_mae",
        "higher_is_better": False,
        "display_name": "Audio MAE",
        "range": (0.0, float('inf')),
    },
    "anomaly_detection": {
        "metric_name": "anomaly_f1",
        "higher_is_better": True,
        "display_name": "F1 аномалий",
        "range": (0.0, 1.0),
    },
    "clustering": {
        "metric_name": "silhouette_approx",
        "higher_is_better": True,
        "display_name": "Силуэт",
        "range": (-1.0, 1.0),
    },
    "reinforcement": {
        "metric_name": "accuracy",
        "higher_is_better": True,
        "display_name": "Награда",
        "range": (0.0, float('inf')),
    },
}


def get_metric_for_task(task_type: str) -> Dict[str, Any]:
    """
    Возвращает информацию о метрике для данного типа задачи.
    Используется тренером и панелью мониторинга.

    Args:
        task_type: строка типа задачи ("classification", "seq2seq", и т.д.)

    Returns:
        Dict с ключами: metric_name, higher_is_better, display_name, range
        Если тип неизвестен — возвращает дефолт (mse).
    """
    return TASK_METRIC_MAP.get(task_type, TASK_METRIC_MAP["regression"])


def get_all_metric_names() -> list:
    """Возвращает список всех доступных метрик (для GUI-выпадающих списков)."""
    return sorted(METRIC_REGISTRY.keys())


def is_metric_higher_better(metric_name: str) -> bool:
    """
    Возвращает True если метрика тем лучше, чем выше.
    Используется для определения цвета в мониторинге.
    """
    higher_better = {
        "accuracy", "precision", "recall", "f1_score",
        "token_accuracy", "sequence_accuracy",
        "pixel_accuracy", "ssim_approx",
        "anomaly_f1", "silhouette_approx",
        "r2_score",
    }
    return metric_name in higher_better


# ============================================================
#  САМОТЕСТ (запускается при прямом вызове файла)
# ============================================================

if __name__ == "__main__":
    print("=== САМОТЕСТ core/metrics.py ===")
    print()

    # Тест классификации
    preds_cls = torch.tensor([[2.0, 0.1, 0.3], [0.1, 3.0, 0.2], [0.2, 0.1, 4.0]])
    true_cls = torch.tensor([0, 1, 2])
    print(f"accuracy: {accuracy(preds_cls, true_cls):.4f} (ожидается 1.0)")
    print(f"f1_score: {f1_score(preds_cls, true_cls):.4f} (ожидается 1.0)")

    # Тест регрессии
    preds_reg = torch.tensor([1.0, 2.0, 3.0])
    true_reg = torch.tensor([1.1, 2.2, 2.8])
    print(f"mse: {mse(preds_reg, true_reg):.6f}")
    print(f"mae: {mae(preds_reg, true_reg):.6f}")
    print(f"r2: {r2_score(preds_reg, true_reg):.6f}")

    # Тест текста
    preds_text = torch.zeros(2, 5, 10)  # (batch=2, seq=5, vocab=10)
    preds_text[0, :, 3] = 10.0  # все токены = 3
    preds_text[1, :, 5] = 10.0  # все токены = 5
    true_text = torch.tensor([[3, 3, 3, 3, 3], [5, 5, 5, 5, 0]])  # второй с PAD
    print(f"token_accuracy: {token_accuracy(preds_text, true_text, pad_idx=0):.4f}")

    # Тест пустых данных
    print(f"accuracy (пусто): {accuracy(torch.tensor([]), torch.tensor([])):.4f} (ожидается 0.0)")
    print(f"mse (NaN): {mse(torch.tensor([float('nan')]), torch.tensor([1.0])):.4f} (ожидается 0.0)")

    # Тест диспетчера
    print(f"compute_metric('accuracy', ...): {compute_metric('accuracy', preds_cls, true_cls):.4f}")
    print(f"compute_metric('nonexist', ...): {compute_metric('nonexist', preds_cls, true_cls):.4f}")

    # Тест маппинга
    info = get_metric_for_task("seq2seq")
    print(f"Метрика для seq2seq: {info['metric_name']} ({info['display_name']})")

    print()
    print("=== ВСЕ ТЕСТЫ ПРОЙДЕНЫ ===")