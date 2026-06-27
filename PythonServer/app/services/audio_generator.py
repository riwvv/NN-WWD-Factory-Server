import numpy as np
import torch
import random
import os
import asyncio
import soundfile as sf
from .task_manager import update_task
from app.core.config import settings
import audiomentations as A
import Levenshtein
from typing import List

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

# --- Список доступных голосов ---
# Полный список можно посмотреть в документации Silero TTS v3
AVAILABLE_SPEAKERS = ['aidar', 'baya', 'kseniya', 'xenia', 'eugene']

# Списки для генерации отрицательных примеров
RUSSIAN_WORDS = [
    "привет", "пока", "компьютер", "музыка", "время", "работа", "дом", "день",
    "ночь", "окно", "стол", "стул", "вода", "огонь", "земля", "небо",
    "человек", "город", "улица", "машина", "книга", "фильм", "игра"
]

def generate_similar_words(word: str, max_distance: int = 2, 
                           alphabet: str = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя',
                           length_variation: int = 2) -> List[str]:
    """
    Генерирует слова, похожие на заданное, с вариациями длины.
    
    Аргументы:
        word (str): Исходное слово.
        max_distance (int): Максимальное расстояние Левенштейна.
        alphabet (str): Алфавит для замены букв.
        length_variation (int): Максимальное отклонение длины слова (+-).
    """
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
        for _ in range(10):  # Генерируем 10 вариантов для каждой длины
            insert_pos = random.randint(0, len(word))
            random_letter = random.choice(alphabet)
            new_word = word[:insert_pos] + random_letter + word[insert_pos:]
            if Levenshtein.distance(word, new_word) <= max_distance + extra:
                similar.add(new_word)
    
    # 3. Удаление букв (-1..-length_variation символов)
    for remove in range(1, length_variation + 1):
        for _ in range(10):  # Генерируем 10 вариантов для каждой длины
            if len(word) - remove > 0:
                remove_pos = random.randint(0, len(word) - remove)
                new_word = word[:remove_pos] + word[remove_pos + remove:]
                if Levenshtein.distance(word, new_word) <= max_distance + remove:
                    similar.add(new_word)
    
    return list(similar)

# Пример использования
SIMILAR_WORDS = generate_similar_words("джарвис", length_variation=2)

def generate_random_text() -> str:
    """Генерирует случайный текст для отрицательного примера."""
    return random.choice(RUSSIAN_WORDS)

def generate_similar_text() -> str:
    """Генерирует похожее на wake word слово."""
    return random.choice(SIMILAR_WORDS)

def generate_noise(sample_rate: int = 24000, duration: float = 1.0) -> np.ndarray:
    """Генерирует белый шум."""
    return np.random.normal(0, 0.1, int(sample_rate * duration))

# --- Функция пакетной генерации отрицательных примеров ---
async def process_negative_generation(task_id: str, request):
    try:
        total_files = request.count
        generated_files = 0

        update_task(task_id,
            status="processing",
            message="Генерация отрицательных примеров начата...",
            progress=0,
            total_files=total_files,
            generated_files=0
        )

        task_folder = os.path.join(settings.AUDIO_OUTPUT_DIR, task_id)
        os.makedirs(task_folder, exist_ok=True)

        folders = {
            "random_words": os.path.join(task_folder, "random_words"),
            "similar_words": os.path.join(task_folder, "similar_words"),
            "noise": os.path.join(task_folder, "noise"),
        }
        for folder in folders.values():
            os.makedirs(folder, exist_ok=True)

        for i in range(total_files):
            example_type = random.choices(
                ["random_words", "similar_words", "noise"],
                weights=[0.45, 0.35, 0.2]
            )[0]

            if example_type == "noise":
                audio = generate_noise(request.sample_rate, duration=1.0)
                sr = request.sample_rate
                filename = f"noise_{i:04d}.wav"
                folder = folders["noise"]
            else:
                if example_type == "random_words":
                    text = generate_random_text()
                    folder = folders["random_words"]
                else:
                    text = generate_similar_text()
                    folder = folders["similar_words"]

                audio, sr, speaker = await asyncio.to_thread(
                    generate_audio_file,
                    text,
                    request.sample_rate,
                    None
                )

                if hasattr(audio, 'numpy'):
                    audio = audio.numpy()

                if request.sample_rate in [8000, 24000, 48000]:
                    audio = await asyncio.to_thread(augment_audio, audio, sr)

                filename = f"{text.replace(' ', '_')}_{i:04d}_{speaker}.wav"

            sample_rate_int = int(sr)
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            file_path = os.path.join(folder, filename)
            await asyncio.to_thread(sf.write, file_path, audio, sample_rate_int)

            generated_files += 1

            if generated_files % 10 == 0 or generated_files == total_files:
                update_task(task_id,
                    generated_files=generated_files,
                    progress=int((generated_files / total_files) * 100)
                )

        update_task(task_id,
            status="completed",
            message=f"Генерация {generated_files} отрицательных примеров завершена",
            progress=100,
            generated_files=generated_files,
            file_path=task_folder
        )

    except Exception as e:
        update_task(task_id,
            status="failed",
            message=f"Ошибка генерации отрицательных примеров: {str(e)}"
        )

# --- Создание пайплайна аугментации (один раз) ---
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

# Создаём пайплайн один раз
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

# --- Функция генерации одного файла с выбором случайного голоса ---
def generate_audio_file(text: str, sample_rate: int = 24000, speaker: str = None):
    """Генерирует аудио с заданным или случайным голосом."""
    if speaker is None:
        speaker = 'random'
    
    audio = model.apply_tts(text,
                            speaker=speaker,
                            sample_rate=sample_rate)
    return audio, sample_rate, speaker

# --- Функция пакетной генерации ---
async def process_batch_generation(task_id: str, request):
    try:
        total_words = len(request.texts)
        total_files_to_generate = total_words * request.count_per_text
        generated_files = 0

        update_task(task_id,
            status="processing",
            message="Генерация начата...",
            progress=0,
            total_files=total_files_to_generate,
            generated_files=0
        )

        task_folder = os.path.join(settings.AUDIO_OUTPUT_DIR, task_id)
        os.makedirs(task_folder, exist_ok=True)

        for word_idx, word in enumerate(request.texts):
            update_task(task_id,
                message=f"Генерация для слова '{word}' ({word_idx + 1}/{total_words})",
                current_word=word,
                current_word_index=word_idx + 1,
                progress=int((word_idx / total_words) * 100)
            )

            word_folder = os.path.join(task_folder, word.replace(" ", "_"))
            os.makedirs(word_folder, exist_ok=True)

            for i in range(request.count_per_text):
                # 1. Генерация аудио со случайным голосом
                audio, sr, speaker = await asyncio.to_thread(
                    generate_audio_file,
                    word,
                    request.sample_rate,
                    None  # None = случайный голос
                )

                # 2. Преобразование PyTorch Tensor → NumPy
                if hasattr(audio, 'numpy'):
                    audio = audio.numpy()

                # 3. Применение комплексной аугментации
                if request.sample_rate in [8000, 24000, 48000]:
                    audio = await asyncio.to_thread(augment_audio, audio, sr)

                # 4. Приведение типов для записи
                sample_rate_int = int(sr)
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)

                filename = f"{word}_{i:04d}_{speaker}.wav"
                file_path = os.path.join(word_folder, filename)

                # 5. Сохранение
                await asyncio.to_thread(sf.write, file_path, audio, sample_rate_int)

                generated_files += 1

                if generated_files % 10 == 0 or generated_files == total_files_to_generate:
                    update_task(task_id,
                        generated_files=generated_files,
                        progress=int((generated_files / total_files_to_generate) * 100)
                    )

        update_task(task_id,
            status="completed",
            message=f"Генерация {generated_files} файлов завершена успешно",
            progress=100,
            generated_files=generated_files,
            current_word="",
            current_word_index=total_words,
            file_path=task_folder
        )

    except Exception as e:
        update_task(task_id,
            status="failed",
            message=f"Ошибка генерации пакета: {str(e)}"
        )