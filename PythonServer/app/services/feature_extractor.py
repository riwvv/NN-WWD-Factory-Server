import os
import numpy as np
import librosa
from .task_manager import update_task
from .model_trainer import SAMPLE_RATE, NUM_SAMPLES

def _pad_or_trim(waveform: np.ndarray, num_samples: int) -> np.ndarray:
    if len(waveform) < num_samples:
        return np.pad(waveform, (0, num_samples - len(waveform)))
    return waveform[:num_samples]

def extract_waveforms_from_folder(folder_path: str, sample_rate: int = SAMPLE_RATE, num_samples: int = NUM_SAMPLES):
    """Загружает все .wav из папки, ресемплирует в SAMPLE_RATE и приводит к
    фиксированной длине NUM_SAMPLES. Никакой Mel-спектрограммы тут больше не
    считается — это теперь часть самой модели (см. model_trainer.py), поэтому
    train и inference гарантированно используют одну и ту же математику."""
    waveforms = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)
            audio, sr = librosa.load(file_path, sr=sample_rate)
            audio = _pad_or_trim(audio.astype(np.float32), num_samples)
            waveforms.append(audio)

    return np.array(waveforms, dtype=np.float32)

async def prepare_features(dataset_dir: str, task_id: str = None):
    """Подготавливает waveform-датасет из сгенерированных .wav файлов."""
    positive_dir = os.path.join(dataset_dir, "positive")
    negative_dir = os.path.join(dataset_dir, "negative")

    if task_id:
        update_task(task_id, message="Подготовка положительных примеров...")

    positive_waveforms = extract_waveforms_from_folder(positive_dir)

    if task_id:
        update_task(task_id, message="Подготовка отрицательных примеров...")

    negative_waveforms = extract_waveforms_from_folder(negative_dir)

    features_dir = os.path.join(dataset_dir, "features")
    os.makedirs(features_dir, exist_ok=True)

    positive_path = os.path.join(features_dir, "positive_waveforms.npy")
    negative_path = os.path.join(features_dir, "negative_waveforms.npy")

    np.save(positive_path, positive_waveforms)
    np.save(negative_path, negative_waveforms)

    return positive_path, negative_path
