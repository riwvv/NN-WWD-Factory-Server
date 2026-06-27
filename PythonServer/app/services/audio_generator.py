import torch
import numpy as np
import random
import os
import asyncio
import soundfile as sf
from .task_manager import tasks_status, update_task
from app.core.config import settings

# Загрузка модели Silero (один раз при запуске)
device = torch.device('cpu')
model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                      model='silero_tts',
                                      language='ru',
                                      speaker='v3_1_ru')
model.to(device)

# --- Функции аугментации ---
def change_speed(audio, speed_factor):
    """Изменяет скорость аудио (растягивает/сжимает время)."""
    if speed_factor <= 0:
        return audio
    # Простое изменение скорости через изменение длины
    indices = np.round(np.arange(0, len(audio), speed_factor))
    indices = indices[indices < len(audio)].astype(int)
    return audio[indices]

def change_volume(audio, volume_factor):
    """Изменяет громкость аудио."""
    return audio * volume_factor

def add_noise(audio, noise_level=0.005):
    """Добавляет белый шум к аудио."""
    noise = np.random.normal(0, noise_level, len(audio))
    return audio + noise

def augment_audio(audio, sr, speed_range=(0.8, 1.2), volume_range=(0.7, 1.3), noise_level=0.005):
    """Применяет случайные аугментации к аудио."""
    # Случайное изменение скорости
    speed = random.uniform(*speed_range)
    if speed != 1.0:
        audio = change_speed(audio, 1.0 / speed)  # Обратная зависимость для скорости
    
    # Случайное изменение громкости
    volume = random.uniform(*volume_range)
    if volume != 1.0:
        audio = change_volume(audio, volume)
    
    # Случайное добавление шума
    if random.random() < 0.5:  # 50% шанс
        audio = add_noise(audio, noise_level)
    
    return audio

# --- Функция генерации одного файла ---
def generate_audio_file(text: str, speaker: str = 'xenia', sample_rate: int = 24000):
    """Генерирует аудио, применяет аугментации и возвращает массив."""
    audio = model.apply_tts(text,
                            speaker=speaker,
                            sample_rate=sample_rate)
    return audio, sample_rate

# --- Функция пакетной генерации ---
async def process_batch_generation(task_id: str, request):
    """Фоновая задача для пакетной генерации (с детальным прогрессом)."""
    try:
        total_words = len(request.texts)
        total_files_to_generate = total_words * request.count_per_text
        generated_files = 0
        
        # Обновляем начальный статус
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
            # Обновляем статус для каждого слова
            update_task(task_id,
                message=f"Генерация для слова '{word}' ({word_idx + 1}/{total_words})",
                current_word=word,
                current_word_index=word_idx + 1,
                progress=int((word_idx / total_words) * 100)
            )
            
            word_folder = os.path.join(task_folder, word.replace(" ", "_"))
            os.makedirs(word_folder, exist_ok=True)
            
            for i in range(request.count_per_text):
                # Генерация аудио (в отдельном потоке)
                audio, sr = await asyncio.to_thread(
                    generate_audio_file, 
                    word, 
                    request.speaker, 
                    request.sample_rate
                )
                
                # Применяем аугментацию
                if request.sample_rate in [8000, 24000, 48000]:
                    audio = await asyncio.to_thread(augment_audio, audio, sr)
                
                filename = f"{word}_{i:04d}.wav"
                file_path = os.path.join(word_folder, filename)
                
                await asyncio.to_thread(sf.write, file_path, audio, sr)
                
                generated_files += 1
                
                # Обновляем прогресс после каждого файла (каждые 10 файлов для снижения нагрузки)
                if generated_files % 10 == 0 or generated_files == total_files_to_generate:
                    update_task(task_id,
                        generated_files=generated_files,
                        progress=int((generated_files / total_files_to_generate) * 100)
                    )
        
        # Завершаем задачу
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
        # Обработка ошибок
        update_task(task_id,
            status="failed",
            message=f"Ошибка генерации пакета: {str(e)}"
        )