from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from PIL import Image
from pydub import AudioSegment

logger = logging.getLogger(__name__)


def preprocess_image(input_path: str, output_path: str, target_size: int = 768) -> tuple[int, int]:
    """Resize and center-pad image to target_size x target_size square."""
    img = Image.open(input_path).convert("RGB")

    # Resize maintaining aspect ratio, fitting within target_size
    img.thumbnail((target_size, target_size), Image.LANCZOS)

    # Center-pad to exact square
    padded = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    x_offset = (target_size - img.width) // 2
    y_offset = (target_size - img.height) // 2
    padded.paste(img, (x_offset, y_offset))

    padded.save(output_path, "PNG")
    logger.info("Preprocessed image: %s -> %dx%d", output_path, target_size, target_size)
    return target_size, target_size


def preprocess_audio(
    input_path: str,
    output_path: str,
    target_sample_rate: int = 16000,
    target_lufs: float = -14.0,
    max_duration_seconds: int = 120,
) -> tuple[float, int]:
    """Normalize audio to WAV 16kHz mono with volume normalization.

    Returns (duration_seconds, sample_rate).
    """
    audio = AudioSegment.from_file(input_path)

    # Convert to mono
    audio = audio.set_channels(1)

    # Trim to max duration
    max_ms = max_duration_seconds * 1000
    if len(audio) > max_ms:
        audio = audio[:max_ms]
        logger.info("Trimmed audio to %d seconds", max_duration_seconds)

    # Strip leading/trailing silence (threshold: -40dBFS, keep 100ms buffer)
    audio = _strip_silence(audio)

    # Normalize volume (simple peak normalization to -1dB headroom)
    change_db = -1.0 - audio.max_dBFS
    if abs(change_db) > 0.5:
        audio = audio.apply_gain(change_db)

    # Export as WAV with target sample rate
    audio = audio.set_frame_rate(target_sample_rate)
    audio.export(output_path, format="wav")

    duration = len(audio) / 1000.0
    logger.info("Preprocessed audio: %.1fs, %dHz mono -> %s", duration, target_sample_rate, output_path)
    return duration, target_sample_rate


def _strip_silence(audio: AudioSegment, silence_thresh_db: float = -40.0, buffer_ms: int = 100) -> AudioSegment:
    """Remove leading and trailing silence."""
    # Find first non-silent frame from start
    start = 0
    for i in range(0, len(audio), 10):
        chunk = audio[i : i + 10]
        if chunk.dBFS > silence_thresh_db:
            start = max(0, i - buffer_ms)
            break

    # Find first non-silent frame from end
    end = len(audio)
    for i in range(len(audio), 0, -10):
        chunk = audio[max(0, i - 10) : i]
        if chunk.dBFS > silence_thresh_db:
            end = min(len(audio), i + buffer_ms)
            break

    if start >= end:
        return audio

    return audio[start:end]


def get_audio_duration(file_path: str) -> float:
    """Get duration in seconds without full processing."""
    audio = AudioSegment.from_file(file_path)
    return len(audio) / 1000.0


def get_image_dimensions(file_path: str) -> tuple[int, int]:
    """Get image width and height."""
    with Image.open(file_path) as img:
        return img.width, img.height
