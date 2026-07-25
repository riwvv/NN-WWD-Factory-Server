import numpy as np
import torch
import audiomentations as A

# --- Инициализация NumPy (для совместимости) ---
try:
    np._core._multiarray_umath.initialize()
except AttributeError:
    pass

# --- Загрузка модели Silero (один раз при запуске) ---
device = torch.device('cpu')
model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                      model='silero_tts',
                                      language='ru',
                                      speaker='v3_1_ru')
model.to(device)

# --- Аугментация через audiomentations ---
def create_augmentation_pipeline(sample_rate=24000):
    """Создаёт комплексный пайплайн аугментации с audiomentations."""
    return A.Compose([
        A.Gain(min_gain_db=-12, max_gain_db=6, p=0.5),
        A.PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        A.TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        A.TanhDistortion(p=0.2),
        A.SevenBandParametricEQ(p=0.3),
        A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),
    ])

_augment_pipeline = None

def get_augmentation_pipeline(sample_rate=24000):
    global _augment_pipeline
    if _augment_pipeline is None:
        _augment_pipeline = create_augmentation_pipeline(sample_rate)
    return _augment_pipeline

def augment_audio(audio, sr):
    """Применяет комплексную аугментацию к аудио-массиву."""
    pipeline = get_augmentation_pipeline(sr)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    augmented = pipeline(samples=audio, sample_rate=sr)
    return augmented

# --- Функция генерации аудио ---
def generate_audio_file(text: str, sample_rate: int = 24000, speaker: str = None):
    """Генерирует аудио с заданным или случайным голосом."""
    if speaker is None:
        speaker = 'random'
    
    audio = model.apply_tts(text,
                            speaker=speaker,
                            sample_rate=sample_rate)
    return audio, sample_rate, speaker