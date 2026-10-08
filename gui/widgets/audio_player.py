"""
gui/widgets/audio_player.py
=============================
Поддерживает numpy-массивы и файлы .wav.
Не создаёт ключей в shared_state.
"""

import numpy as np
from typing import Optional, List
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QSlider, QComboBox, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QPainter, QPen, QColor, QPainterPath


class AudioWaveformWidget(QWidget):
    """Виджет отображения волновой формы звука."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(80)
        self.setMaximumHeight(120)
        self.audio_data: Optional[np.ndarray] = None
        self.playhead_pos = 0.0  # 0.0 - 1.0
        self.setStyleSheet("background-color: #1e1e1e; border: 1px solid #555;")

    def set_audio(self, audio: np.ndarray):
        """Устанавливает аудио-данные (1D массив)."""
        self.audio_data = audio
        self.playhead_pos = 0.0
        self.update()

    def set_playhead(self, pos: float):
        """Устанавливает позицию воспроизведения (0.0 - 1.0)."""
        self.playhead_pos = max(0.0, min(1.0, pos))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        mid_y = h // 2

        # Фон
        painter.fillRect(0, 0, w, h, QColor("#1e1e1e"))

        # Центральная линия
        painter.setPen(QPen(QColor("#444444"), 1))
        painter.drawLine(0, mid_y, w, mid_y)

        if self.audio_data is not None and len(self.audio_data) > 0:
            # Нормализуем для отображения
            data = self.audio_data
            if len(data) > w:
                # Даунсэмплинг для отображения
                step = len(data) // w
                data = data[::step]

            n = len(data)
            max_val = np.max(np.abs(data)) + 1e-8

            # Рисуем волну
            pen = QPen(QColor("#4a88c7"), 1.5)
            painter.setPen(pen)

            path = QPainterPath()
            for i in range(n):
                x = int(i * w / n)
                y = int(mid_y - (data[i] / max_val) * (h // 2 - 5))
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            painter.drawPath(path)

            # Позиция воспроизведения
            if self.playhead_pos > 0:
                px = int(self.playhead_pos * w)
                painter.setPen(QPen(QColor("#cc7832"), 2))
                painter.drawLine(px, 0, px, h)

        else:
            painter.setPen(QColor("#777777"))
            painter.drawText(
                0, 0, w, h, Qt.AlignCenter,
                "Нет аудио. Загрузи или сгенерируй звук."
            )


class AudioPlayerWidget(QWidget):
    """
    Виджет воспроизведения аудио.

    Сигналы:
        audio_loaded(np.ndarray, int) — загружен аудио-массив и sample_rate

    Использование:
        player = AudioPlayerWidget()
        player.load_from_array(np_array, sample_rate=16000)
        player.load_from_file("sound.wav")
    """

    audio_loaded = pyqtSignal(np.ndarray, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.audio_data: Optional[np.ndarray] = None
        self.sample_rate = 16000
        self.is_playing = False
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Заголовок
        title = QLabel("🔊 Аудио-плеер")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #cc7832;")
        layout.addWidget(title)

        # Волновая форма
        self.waveform = AudioWaveformWidget()
        layout.addWidget(self.waveform)

        # Управление
        controls_layout = QHBoxLayout()

        self.play_btn = QPushButton("▶️ Воспроизвести")
        self.play_btn.setToolTip("Проиграть загруженный звук")
        self.play_btn.clicked.connect(self._toggle_play)
        controls_layout.addWidget(self.play_btn)

        self.stop_btn = QPushButton("⏹ Стоп")
        self.stop_btn.setToolTip("Остановить воспроизведение")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        controls_layout.addWidget(self.stop_btn)

        controls_layout.addSpacing(20)

        # Громкость
        controls_layout.addWidget(QLabel("🔉"))
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)
        self.volume_slider.setToolTip("Громкость воспроизведения")
        self.volume_slider.setMaximumWidth(100)
        controls_layout.addWidget(self.volume_slider)
        controls_layout.addWidget(QLabel("🔊"))

        controls_layout.addStretch()

        # Информация
        self.info_label = QLabel("Аудио не загружено")
        self.info_label.setStyleSheet("color: #888; font-size: 11px;")
        controls_layout.addWidget(self.info_label)

        layout.addLayout(controls_layout)

        # Загрузка
        load_layout = QHBoxLayout()

        self.load_file_btn = QPushButton("📁 Загрузить .wav файл")
        self.load_file_btn.setToolTip("Загрузить звуковой файл с диска")
        self.load_file_btn.clicked.connect(self._load_file)
        load_layout.addWidget(self.load_file_btn)

        self.generate_btn = QPushButton("🎵 Сгенерировать тестовый тон")
        self.generate_btn.setToolTip("Создать синусоидальный звук для проверки")
        self.generate_btn.clicked.connect(self._generate_test_tone)
        load_layout.addWidget(self.generate_btn)

        load_layout.addStretch()
        layout.addLayout(load_layout)

        # Таймер для анимации воспроизведения
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self._update_playhead)
        self.play_position = 0.0

    def load_from_array(self, audio: np.ndarray, sample_rate: int = 16000):
        """Загружает аудио из numpy-массива."""
        self.audio_data = audio.copy()
        self.sample_rate = sample_rate
        self.waveform.set_audio(audio)

        duration = len(audio) / sample_rate
        self.info_label.setText(
            f"Длительность: {duration:.2f}с | Sample rate: {sample_rate} Гц | "
            f"Сэмплов: {len(audio):,}"
        )
        self.play_btn.setEnabled(True)
        self.audio_loaded.emit(audio, sample_rate)

    def load_from_file(self, path: str):
        """Загружает аудио из .wav файла."""
        try:
            import wave
            with wave.open(path, 'rb') as wf:
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                self.sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                raw_data = wf.readframes(n_frames)

            if sample_width == 2:
                audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 1:
                audio = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
            else:
                audio = np.frombuffer(raw_data, dtype=np.float32)

            if n_channels > 1:
                audio = audio.reshape(-1, n_channels)[:, 0]  # берём первый канал

            self.load_from_array(audio, self.sample_rate)
        except Exception as e:
            QMessageBox.warning(self, "Ошибка загрузки", f"Не удалось прочитать файл:\n{e}")

    def _load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Загрузить аудио", "",
            "Аудио файлы (*.wav);;Все файлы (*)"
        )
        if path:
            self.load_from_file(path)

    def _generate_test_tone(self):
        """Генерирует тестовый синусоидальный тон 440 Гц."""
        duration = 2.0
        freq = 440.0
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        audio = np.sin(2 * np.pi * freq * t) * 0.8
        self.load_from_array(audio, self.sample_rate)

    def _toggle_play(self):
        if self.audio_data is None:
            return
        if self.is_playing:
            self._stop()
        else:
            self.is_playing = True
            self.play_position = 0.0
            self.play_btn.setText("⏸ Пауза")
            self.stop_btn.setEnabled(True)
            # В реальной реализации здесь был бы QAudioOutput
            # Для образовательной цели — просто анимация
            self.play_timer.start(50)

    def _stop(self):
        self.is_playing = False
        self.play_timer.stop()
        self.play_position = 0.0
        self.waveform.set_playhead(0.0)
        self.play_btn.setText("▶️ Воспроизвести")
        self.stop_btn.setEnabled(False)

    def _update_playhead(self):
        if self.audio_data is None:
            return
        duration = len(self.audio_data) / self.sample_rate
        self.play_position += 0.05 / duration
        if self.play_position >= 1.0:
            self._stop()
            return
        self.waveform.set_playhead(self.play_position)

    def get_audio(self) -> Optional[np.ndarray]:
        """Возвращает загруженные аудио-данные."""
        return self.audio_data

    def get_sample_rate(self) -> int:
        return self.sample_rate