import numpy as np
import torch
import random
import os
import asyncio
import soundfile as sf
from .task_manager import update_task
from app.core.config import settings
import audiomentations as A
from typing import List
import Levenshtein  # pip install python-Levenshtein

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

# --- Списки для генерации отрицательных примеров ---
RUSSIAN_WORDS = [
    "привет", "пока", "компьютер", "музыка", "время", "работа", "дом", "день",
    "ночь", "окно", "стол", "стул", "вода", "огонь", "земля", "небо",
    "человек", "город", "улица", "машина", "книга", "фильм", "игра"
]

def generate_similar_words(word: str, max_distance: int = 2, 
                           alphabet: str = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя',
                           length_variation: int = 2) -> List[str]:
    """Генерирует слова, похожие на заданное, с вариациями длины."""
    similar = set()
    
    # 1. Замены букв (как было раньше)
    for i in range(len(word)):
        for letter in alphabet:
            if letter != word[i]:
                new_word = word[:i] + letter + word[i+1:]
                if Levenshtein.distance(word, new_word) <= max_distance:
                    similar.add(new_word)
    
    # 2. Добавление случайных букв (+1..+length_variation символов)
    for extra in range(1, length_variation + 1):
        for _ in range(10):
            insert_pos = random.randint(0, len(word))
            random_letter = random.choice(alphabet)
            new_word = word[:insert_pos] + random_letter + word[insert_pos:]
            if Levenshtein.distance(word, new_word) <= max_distance + extra:
                similar.add(new_word)
    
    # 3. Удаление букв (-1..-length_variation символов)
    for remove in range(1, length_variation + 1):
        for _ in range(10):
            if len(word) - remove > 0:
                remove_pos = random.randint(0, len(word) - remove)
                new_word = word[:remove_pos] + word[remove_pos + remove:]
                if Levenshtein.distance(word, new_word) <= max_distance + remove:
                    similar.add(new_word)
    
    return list(similar)

def generate_random_text() -> str:
    """Генерирует случайный текст для отрицательного примера."""
    return random.choice(RUSSIAN_WORDS)

def generate_noise(sample_rate: int = 24000, duration: float = 1.0) -> np.ndarray:
    """Генерирует белый шум."""
    return np.random.normal(0, 0.1, int(sample_rate * duration))

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

# --- Функция пакетной генерации положительных примеров ---
async def process_batch_generation(task_id: str, request):
    """Фоновая задача для пакетной генерации (положительные примеры)."""
    # ... (логика остаётся без изменений, как в прошлых версиях) ...
    pass

# --- Функция пакетной генерации отрицательных примеров ---
async def process_negative_generation(task_id: str, request):
    """Фоновая задача для генерации отрицательных примеров."""
    # ... (логика остаётся без изменений, как в прошлых версиях) ...
    pass