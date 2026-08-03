import os
import re
import numpy as np
import librosa
from .task_manager import update_task
from .embedding_extractor import SAMPLE_RATE, NUM_SAMPLES, waveform_to_embedding

def _pad_or_trim(waveform: np.ndarray, num_samples: int) -> np.ndarray:
    if len(waveform) < num_samples:
        return np.pad(waveform, (0, num_samples - len(waveform)))
    return waveform[:num_samples]

# copy_real_samples (pipeline_service.py) сохраняет аугментированные копии
# одной реальной записи как "real_XXX.wav" + "real_XXX_aug0.wav", "..._aug1.wav"
# и т.д. Обрезая этот суффикс, получаем group id: все копии одного и того же
# исходника попадают в одну группу, чтобы train/val split (см. model_trainer.py)
# не растащил почти-дубликаты по разным сторонам и не завысил val_acc.
_AUG_SUFFIX_RE = re.compile(r"_aug\d+$")

# Сколько случайно сдвинутых по времени копий добавлять на каждый файл, поверх
# оригинала. Инференс проверяет скользящее окно живого аудио (см.
# WakeWordListener в Yuki), и слово оказывается в окне на произвольной позиции
# -- а не только в начале, как получается при простом pad/trim. Без этого
# модель никогда не видит "не в начале" вариант при обучении.
TIME_SHIFT_COPIES = 2
TIME_SHIFT_MAX_FRACTION = 0.5


def _time_shift(waveform: np.ndarray, num_samples: int, max_shift_fraction: float = TIME_SHIFT_MAX_FRACTION) -> np.ndarray:
    """Двигает уже выровненный (pad/trim'нутый) клип влево/вправо в пределах
    окна, дозаполняя освободившийся край тишиной. Не wrap-around: то, что
    "уезжает" за край, отбрасывается, а не переезжает на другую сторону."""
    max_shift = int(num_samples * max_shift_fraction)
    if max_shift == 0:
        return waveform
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return waveform
    shifted = np.roll(waveform, shift)
    if shift > 0:
        shifted[:shift] = 0.0
    else:
        shifted[shift:] = 0.0
    return shifted


def extract_embeddings_from_folder(folder_path: str, sample_rate: int = SAMPLE_RATE, num_samples: int = NUM_SAMPLES):
    """Загружает все .wav из папки, приводит к фиксированной длине и считает
    эмбеддинги (оригинал + TIME_SHIFT_COPIES случайно сдвинутых копий на
    файл) через заморожённый бэкенд openWakeWord (см. embedding_extractor.py).

    Возвращает (embeddings, group_ids) -- group_ids используется для
    группового train/val split, чтобы копии (аугментированные и сдвинутые)
    одной записи не утекали между train и val."""
    embeddings = []
    group_ids = []
    for filename in os.listdir(folder_path):
        if not filename.endswith('.wav'):
            continue

        file_path = os.path.join(folder_path, filename)
        audio, sr = librosa.load(file_path, sr=sample_rate)
        audio = _pad_or_trim(audio.astype(np.float32), num_samples)
        group_id = _AUG_SUFFIX_RE.sub("", os.path.splitext(filename)[0])

        variants = [audio] + [_time_shift(audio, num_samples) for _ in range(TIME_SHIFT_COPIES)]
        for variant in variants:
            embeddings.append(waveform_to_embedding(variant))
            group_ids.append(group_id)

    return np.array(embeddings, dtype=np.float32), np.array(group_ids)

async def prepare_features(dataset_dir: str, task_id: str = None):
    """Подготавливает эмбеддинг-датасет из сгенерированных .wav файлов."""
    positive_dir = os.path.join(dataset_dir, "positive")
    negative_dir = os.path.join(dataset_dir, "negative")

    if task_id:
        update_task(task_id, message="Подготовка положительных примеров...")

    positive_embeddings, positive_groups = extract_embeddings_from_folder(positive_dir)

    if task_id:
        update_task(task_id, message="Подготовка отрицательных примеров...")

    negative_embeddings, negative_groups = extract_embeddings_from_folder(negative_dir)

    features_dir = os.path.join(dataset_dir, "features")
    os.makedirs(features_dir, exist_ok=True)

    positive_path = os.path.join(features_dir, "positive_embeddings.npy")
    negative_path = os.path.join(features_dir, "negative_embeddings.npy")
    positive_groups_path = os.path.join(features_dir, "positive_groups.npy")
    negative_groups_path = os.path.join(features_dir, "negative_groups.npy")

    np.save(positive_path, positive_embeddings)
    np.save(negative_path, negative_embeddings)
    np.save(positive_groups_path, positive_groups)
    np.save(negative_groups_path, negative_groups)

    return positive_path, negative_path, positive_groups_path, negative_groups_path
