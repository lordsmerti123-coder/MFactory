"""
core/text_tokenizer.py
======================
Универсальный токенизатор для текстовых данных.

Поддерживает:
  • Символьную токенизацию (char-level) — основной режим
  • Заготовку под BPE / WordPiece / субтокены (заглушка, режим "bpe")
  • Версионирование формата (FORMAT_VERSION из contracts)
  • Обратную совместимость со старыми токенизаторами (версия 1.x)
  • Ограничение словаря (Top-K) для симуляции OOV
  • Пакетное кодирование/декодирование (encode_batch / decode_batch)

Зависимости:
  • core/contracts.py — FORMAT_VERSION, DataType (обязателен)
  • core/logger.py

Контракт:
  • Все сохраняемые файлы содержат "format_version"
  • from_dict / load принимают данные версии 1.x и мигрируют их
  • Неизвестные ключи при десериализации сохраняются в _extra
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from collections import Counter

from core.logger import get_logger

# ============================================================
#  ИМПОРТ КОНТРАКТОВ (обязателен, но защищён для отладки)
# ============================================================
try:
    from core.contracts import FORMAT_VERSION, DataType
except ImportError:
    # Fallback на случай, если contracts.py ещё не собран.
    # В боевом режиме этого быть не должно.
    FORMAT_VERSION = "2.0"
    class DataType:
        TEXT = "text"

logger = get_logger(__name__)


# ============================================================
#  КОНСТАНТЫ СПЕЦИАЛЬНЫХ ТОКЕНОВ
# ============================================================
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
MASK_TOKEN = "<MASK>"   # Заготовка для masked language modeling (BERT-стиль)

PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3
MASK_IDX = 4            # Зарезервирован, но не используется в базовом режиме


# ============================================================
#  ОСНОВНОЙ КЛАСС ТОКЕНИЗАТОРА
# ============================================================
class TextTokenizer:
    """
    Универсальный токенизатор для текстовых задач.

    Режимы работы:
      • "char"  — каждый символ текста = отдельный токен (основной, полностью реализован)
      • "bpe"   — заглушка под Byte-Pair Encoding (методы бросают NotImplementedError)
      • "word"  — заглушка под словесную токенизацию (методы бросают NotImplementedError)

    Пример использования:
        tok = TextTokenizer(tokenizer_type="char")
        tok.build_vocab(["2+2", "3-1"], max_vocab_size=50)
        ids = tok.encode("2+2", add_bos=True, add_eos=True)
        text = tok.decode(ids)
    """

    # --- Специальные токены (классовые константы для внешнего доступа) ---
    PAD_IDX = PAD_IDX
    UNK_IDX = UNK_IDX
    BOS_IDX = BOS_IDX
    EOS_IDX = EOS_IDX
    MASK_IDX = MASK_IDX

    def __init__(self, tokenizer_type: str = "char"):
        """
        Args:
            tokenizer_type: "char" | "bpe" | "word"
                • "char" — полностью реализован
                • "bpe", "word" — заглушки, вызовут ошибку при попытке обучения
        """
        self.tokenizer_type = tokenizer_type.lower().strip()
        self.format_version = FORMAT_VERSION

        # Основной словарь: символ -> индекс
        self.vocab: Dict[str, int] = {
            PAD_TOKEN: PAD_IDX,
            UNK_TOKEN: UNK_IDX,
            BOS_TOKEN: BOS_IDX,
            EOS_TOKEN: EOS_IDX,
            MASK_TOKEN: MASK_IDX,
        }
        # Обратный словарь: индекс -> символ
        self.inv_vocab: Dict[int, str] = {
            PAD_IDX: PAD_TOKEN,
            UNK_IDX: UNK_TOKEN,
            BOS_IDX: BOS_TOKEN,
            EOS_IDX: EOS_TOKEN,
            MASK_IDX: MASK_TOKEN,
        }

        # Заготовка под BPE: словарь мерджей (пока не реализован)
        self.bpe_merges: Dict[Tuple[str, str], int] = {}

        # Дополнительные поля для обратной совместимости и расширяемости
        self._extra: Dict[str, Any] = {}

        # Флаг: построен ли словарь
        self._vocab_built = False

        if self.tokenizer_type not in ("char", "bpe", "word"):
            logger.warning(
                f"Неизвестный тип токенизатора: '{tokenizer_type}'. "
                f"Использую 'char' по умолчанию."
            )
            self.tokenizer_type = "char"

    # ============================================================
    #  СВОЙСТВА
    # ============================================================
    @property
    def vocab_size(self) -> int:
        """Текущий размер словаря (включая спецтокены)."""
        return len(self.vocab)

    @property
    def is_built(self) -> bool:
        """Построен ли словарь."""
        return self._vocab_built

    @property
    def special_tokens(self) -> Dict[str, int]:
        """Словарь специальных токенов."""
        return {
            PAD_TOKEN: PAD_IDX,
            UNK_TOKEN: UNK_IDX,
            BOS_TOKEN: BOS_IDX,
            EOS_TOKEN: EOS_IDX,
            MASK_TOKEN: MASK_IDX,
        }

    # ============================================================
    #  ПОСТРОЕНИЕ СЛОВАРЯ
    # ============================================================
    def build_vocab(self, texts: List[str], max_vocab_size: Optional[int] = None) -> None:
        """
        Строит частотный словарь из списка текстов.

        Для режима "char":
            Каждый уникальный символ становится токеном.
            Если задан max_vocab_size, остаются только самые частые символы.

        Для режима "bpe" / "word":
            Заглушка — бросает NotImplementedError.

        Args:
            texts: Список строк для анализа.
            max_vocab_size: Максимальный размер словаря (включая 5 спецтокенов).
                            Если задан, остаются только Top-K частых символов.
                            Остальные будут заменены на <UNK>.
        """
        if self.tokenizer_type != "char":
            raise NotImplementedError(
                f"Токенизатор '{self.tokenizer_type}' пока не реализован. "
                f"Используйте 'char' или дождитесь обновления."
            )

        if not texts:
            logger.warning("build_vocab вызван с пустым списком текстов. Словарь не изменён.")
            return

        # --- Подсчёт частоты символов ---
        char_counts: Counter = Counter()
        total_chars = 0
        for text in texts:
            text_str = str(text)
            char_counts.update(text_str)
            total_chars += len(text_str)

        if total_chars == 0:
            logger.warning("Все тексты пусты. Словарь не изменён.")
            return

        # --- Применение лимита (если задан) ---
        # Учитываем, что 5 позиций заняты спецтокенами
        num_special = len(self.special_tokens)
        if max_vocab_size is not None and max_vocab_size > num_special:
            limit = max_vocab_size - num_special
            most_common = char_counts.most_common(limit)
            allowed_chars = [char for char, _ in most_common]
            excluded_count = len(char_counts) - len(allowed_chars)
        else:
            allowed_chars = list(char_counts.keys())
            excluded_count = 0
            # Если лимит задан, но меньше спецтокенов — предупреждаем
            if max_vocab_size is not None and max_vocab_size <= num_special:
                logger.warning(
                    f"max_vocab_size={max_vocab_size} слишком мал "
                    f"(спецтокенов уже {num_special}). Лимит игнорируется."
                )

        # --- Добавление новых символов в словарь ---
        idx = len(self.vocab)
        added_count = 0
        for char in allowed_chars:
            if char not in self.vocab:
                self.vocab[char] = idx
                self.inv_vocab[idx] = char
                idx += 1
                added_count += 1

        # --- Логирование ---
        if excluded_count > 0:
            logger.warning(
                f"Токенизатор отсек {excluded_count} редких символов в <UNK> "
                f"из-за лимита max_vocab_size={max_vocab_size}."
            )
        logger.info(
            f"Словарь построен: {self.vocab_size} токенов "
            f"(включая {num_special} специальных, добавлено {added_count} новых)."
        )

        self._vocab_built = True

    # ============================================================
    #  КОДИРОВАНИЕ (encode)
    # ============================================================
    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        add_special_tokens: bool = False,
    ) -> List[int]:
        """
        Преобразует текст в список индексов токенов.

        Для режима "char":
            Каждый символ -> индекс в словаре. Неизвестные -> <UNK>.

        Args:
            text: Входной текст.
            add_bos: Добавить <BOS> в начало.
            add_eos: Добавить <EOS> в конец.
            add_special_tokens: Добавить <BOS> и <EOS> одновременно
                                (перекрывает add_bos/add_eos).

        Returns:
            Список целых чисел (индексов токенов).
        """
        if self.tokenizer_type != "char":
            raise NotImplementedError(
                f"encode() для '{self.tokenizer_type}' пока не реализован."
            )

        if add_special_tokens:
            add_bos = True
            add_eos = True

        seq = [self.vocab.get(char, UNK_IDX) for char in str(text)]

        if add_bos:
            seq = [BOS_IDX] + seq
        if add_eos:
            seq = seq + [EOS_IDX]

        return seq

    # ============================================================
    #  ДЕКОДИРОВАНИЕ (decode)
    # ============================================================
    def decode(
        self,
        indices: List[int],
        stop_at_eos: bool = True,
        skip_special: bool = True,
    ) -> str:
        """
        Преобразует список индексов обратно в текст.

        Для режима "char":
            Каждый индекс -> символ. Неизвестные -> "?".

        Args:
            indices: Список индексов токенов.
            stop_at_eos: Остановить декодирование при встрече <EOS>.
            skip_special: Пропускать спецтокены (<PAD>, <BOS>, <EOS>, <MASK>).
                          Если False, они будут выведены как есть.

        Returns:
            Декодированная строка.
        """
        if self.tokenizer_type != "char":
            raise NotImplementedError(
                f"decode() для '{self.tokenizer_type}' пока не реализован."
            )

        special_indices = set(self.special_tokens.values())
        chars: List[str] = []

        for idx in indices:
            if stop_at_eos and idx == EOS_IDX:
                break
            if skip_special and idx in special_indices:
                continue
            if idx == UNK_IDX:
                chars.append("?")  # Визуализация OOV символа
            else:
                chars.append(self.inv_vocab.get(idx, "?"))

        return "".join(chars)

    # ============================================================
    #  ПАКЕТНЫЕ ОПЕРАЦИИ (для удобства и будущих расширений)
    # ============================================================
    def encode_batch(
        self,
        texts: List[str],
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[List[int]]:
        """
        Кодирует список текстов.

        ВНИМАНИЕ: не выполняет паддинг. Для паддинга используйте
        torch.nn.utils.rnn.pad_sequence в dataset.py.

        Args:
            texts: Список строк.
            add_bos: Добавить <BOS> в начало каждого.
            add_eos: Добавить <EOS> в конец каждого.

        Returns:
            Список списков индексов.
        """
        return [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]

    def decode_batch(
        self,
        batch: List[List[int]],
        stop_at_eos: bool = True,
        skip_special: bool = True,
    ) -> List[str]:
        """
        Декодирует список списков индексов.

        Args:
            batch: Список списков индексов.
            stop_at_eos: Останавливать при <EOS>.
            skip_special: Пропускать спецтокены.

        Returns:
            Список декодированных строк.
        """
        return [self.decode(seq, stop_at_eos, skip_special) for seq in batch]

    # ============================================================
    #  СЛУЖЕБНЫЕ МЕТОДЫ
    # ============================================================
    def get_special_tokens(self) -> Dict[str, int]:
        """Возвращает словарь специальных токенов. (Обратная совместимость)"""
        return self.special_tokens

    def __contains__(self, token: str) -> bool:
        """Проверка наличия токена в словаре: 'а' in tokenizer."""
        return token in self.vocab

    def __len__(self) -> int:
        """Размер словаря: len(tokenizer)."""
        return self.vocab_size

    def __repr__(self) -> str:
        return (
            f"TextTokenizer(type='{self.tokenizer_type}', "
            f"vocab_size={self.vocab_size}, "
            f"built={self._vocab_built}, "
            f"format_version='{self.format_version}')"
        )

    # ============================================================
    #  СЕРИАЛИЗАЦИЯ / ДЕСЕРИАЛИЗАЦИЯ (с версионированием)
    # ============================================================
    def to_dict(self) -> Dict[str, Any]:
        """
        Сериализует токенизатор в словарь (для сохранения в JSON / .oai).

        Формат 2.0:
        {
            "format_version": "2.0",
            "tokenizer_type": "char",
            "vocab": {"<PAD>": 0, "а": 5, ...},
            "vocab_built": true,
            "bpe_merges": {},   // заглушка
            "extra": {}         // расширяемость
        }
        """
        return {
            "format_version": self.format_version,
            "tokenizer_type": self.tokenizer_type,
            "vocab": dict(self.vocab),
            "vocab_built": self._vocab_built,
            "bpe_merges": {},  # заглушка для будущего
            "extra": dict(self._extra),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TextTokenizer":
        """
        Десериализует токенизатор из словаря.

        Поддерживает:
          • Формат 2.0 (полный)
          • Формат 1.x (старый: просто словарь {"<PAD>": 0, ...})
          • Неизвестные ключи сохраняются в _extra

        Args:
            data: Словарь с данными токенизатора.

        Returns:
            Восстановленный TextTokenizer.
        """
        # --- Определение версии формата ---
        version = data.get("format_version", "1.0")

        if version.startswith("2"):
            # ---- ФОРМАТ 2.0 ----
            tok = TextTokenizer(tokenizer_type=data.get("tokenizer_type", "char"))
            tok.format_version = version

            # Восстанавливаем словарь
            loaded_vocab = data.get("vocab", {})
            tok.vocab = dict(loaded_vocab)
            tok.inv_vocab = {int(v): k for k, v in loaded_vocab.items()}

            # Флаг построенного словаря
            tok._vocab_built = data.get("vocab_built", len(loaded_vocab) > 5)

            # BPE-мерджи (заглушка)
            tok.bpe_merges = data.get("bpe_merges", {})

            # Расширения
            tok._extra = data.get("extra", {})

            logger.info(
                f"Токенизатор загружен из формата {version}: "
                f"{tok.vocab_size} токенов, тип='{tok.tokenizer_type}'."
            )
            return tok

        else:
            # ---- ФОРМАТ 1.x (обратная совместимость) ----
            # Старый формат: data — это просто словарь {"<PAD>": 0, "а": 5, ...}
            # Или содержит {"vocab": {...}} без format_version
            logger.info(
                f"Миграция токенизатора из формата {version} → {FORMAT_VERSION}."
            )

            # Пытаемся извлечь словарь из разных форматов
            if "vocab" in data and isinstance(data["vocab"], dict):
                raw_vocab = data["vocab"]
            elif all(isinstance(v, int) for v in data.values()):
                # Старый формат: сам data является словарём токенов
                raw_vocab = data
            else:
                raise ValueError(
                    f"Не удалось распознать формат токенизатора версии {version}."
                )

            tok = TextTokenizer(tokenizer_type="char")
            tok.format_version = FORMAT_VERSION  # обновляем версию после миграции

            # Миграция: добавляем отсутствующие спецтокены
            special_map = {
                PAD_TOKEN: PAD_IDX,
                UNK_TOKEN: UNK_IDX,
                BOS_TOKEN: BOS_IDX,
                EOS_TOKEN: EOS_IDX,
                MASK_TOKEN: MASK_IDX,
            }
            # Сначала спецтокены
            for tok_name, tok_idx in special_map.items():
                if tok_name not in raw_vocab:
                    raw_vocab[tok_name] = tok_idx

            tok.vocab = dict(raw_vocab)
            tok.inv_vocab = {int(v): k for k, v in raw_vocab.items()}
            tok._vocab_built = True

            # Сохраняем неизвестные поля в _extra
            known_keys = {"format_version", "vocab", "tokenizer_type"}
            for key, value in data.items():
                if key not in known_keys:
                    tok._extra[key] = value

            logger.info(
                f"Токенизатор мигрирован: {tok.vocab_size} токенов."
            )
            return tok

    # ============================================================
    #  СОХРАНЕНИЕ / ЗАГРУЗКА В ФАЙЛ
    # ============================================================
    def save(self, path: Path) -> None:
        """
        Сохраняет токенизатор в JSON-файл.

        Файл содержит format_version для будущей миграции.

        Args:
            path: Путь к файлу (.json или любой другой).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Токенизатор сохранён: {path} ({self.vocab_size} токенов)")

    @staticmethod
    def load(path: Path) -> "TextTokenizer":
        """
        Загружает токенизатор из файла.

        Поддерживает:
          • Новый формат 2.0 (JSON с format_version)
          • Старый формат 1.x (просто словарь в JSON)

        Args:
            path: Путь к файлу.

        Returns:
            Восстановленный TextTokenizer.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Файл токенизатора не найден: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tok = TextTokenizer.from_dict(data)
        logger.info(f"Токенизатор загружен: {path} ({tok.vocab_size} токенов)")
        return tok

    # ============================================================
    #  ЗАГОТОВКИ ПОД BPE / WORDPIECE (будущие расширения)
    # ============================================================
    def train_bpe(self, texts: List[str], num_merges: int = 1000) -> None:
        """
        Заглушка: обучение BPE-мерджей.
        Будет реализовано в будущих версиях.

        Сейчас бросает NotImplementedError с понятным сообщением.
        """
        raise NotImplementedError(
            "BPE-токенизация пока не реализована. "
            "Используйте 'char' режим или дождитесь обновления. "
            "Заглушка оставлена для будущей расширяемости."
        )

    def apply_bpe(self, text: str) -> List[str]:
        """
        Заглушка: применение BPE к тексту.
        """
        raise NotImplementedError(
            "BPE-токенизация пока не реализована."
        )

    # ============================================================
    #  ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ДЛЯ АНАЛИЗА
    # ============================================================
    def get_oov_stats(self, texts: List[str]) -> Dict[str, Any]:
        """
        Анализирует, сколько символов в текстах выйдут за пределы словаря (OOV).

        Полезно для отладки: перед обучением можно проверить,
        насколько хорошо словарь покрывает данные.

        Args:
            texts: Список текстов для анализа.

        Returns:
            Словарь со статистикой:
            {
                "total_chars": int,
                "oov_chars": int,
                "oov_ratio": float,  # доля неизвестных символов (0.0–1.0)
                "unique_oov": List[str],  # уникальные неизвестные символы
            }
        """
        total = 0
        oov = 0
        unique_oov = set()

        for text in texts:
            for char in str(text):
                total += 1
                if char not in self.vocab:
                    oov += 1
                    unique_oov.add(char)

        return {
            "total_chars": total,
            "oov_chars": oov,
            "oov_ratio": oov / total if total > 0 else 0.0,
            "unique_oov": sorted(unique_oov),
        }

    def get_vocab_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику словаря.

        Returns:
            {
                "vocab_size": int,
                "num_special": int,
                "num_regular": int,
                "special_tokens": Dict[str, int],
            }
        """
        num_special = len(self.special_tokens)
        return {
            "vocab_size": self.vocab_size,
            "num_special": num_special,
            "num_regular": self.vocab_size - num_special,
            "special_tokens": self.special_tokens,
        }