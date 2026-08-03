import os
import asyncio
import random
import numpy as np
import soundfile as sf
from app.services.audio_generator import generate_audio_file, augment_audio
from app.utils.text_utils import (
    generate_noise,
    generate_similar_words,
    RUSSIAN_WORDS,
)
from app.services.feature_extractor import prepare_features
from app.services.model_trainer import train_model
from app.services.task_manager import tasks_status, update_task
from app.core.config import settings

# ========================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (определены ДО использования)
# ========================================================

async def generate_positive_examples(wake_word, count, sample_rate, output_dir, task_id, total_files):
    """Генерирует положительные примеры (wake word)."""
    for i in range(count):
        audio, sr, speaker = await asyncio.to_thread(
            generate_audio_file,
            wake_word,
            sample_rate,
            None,
        )
        if hasattr(audio, "numpy"):
            audio = audio.numpy()
        if sample_rate in [8000, 24000, 48000]:
            audio = await asyncio.to_thread(augment_audio, audio, sr)

        filename = f"{wake_word}_{i:04d}_{speaker}.wav"
        file_path = os.path.join(output_dir, filename)
        sample_rate_int = int(sr)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        await asyncio.to_thread(sf.write, file_path, audio, sample_rate_int)

        generated = i + 1
        tasks_status[task_id]["sub_tasks"]["generation"]["generated_files"] = generated
        tasks_status[task_id]["sub_tasks"]["generation"]["progress"] = int((generated / total_files) * 100)
        tasks_status[task_id]["progress"] = tasks_status[task_id]["sub_tasks"]["generation"]["progress"] // 2

async def generate_negative_examples(wake_word, count, sample_rate, output_dir, task_id, total_files):
    """Генерирует отрицательные примеры (случайные слова, похожие слова, шум)."""
    similar_words = generate_similar_words(wake_word)
    random_count = int(count * 0.45)
    similar_count = int(count * 0.35)
    noise_count = count - random_count - similar_count

    negative_generated = 0

    # Случайные слова
    for i in range(random_count):
        text = random.choice(RUSSIAN_WORDS)
        audio, sr, speaker = await asyncio.to_thread(
            generate_audio_file,
            text,
            sample_rate,
            None,
        )
        if hasattr(audio, "numpy"):
            audio = audio.numpy()
        if sample_rate in [8000, 24000, 48000]:
            audio = await asyncio.to_thread(augment_audio, audio, sr)

        filename = f"random_{i:04d}_{speaker}.wav"
        file_path = os.path.join(output_dir, filename)
        sample_rate_int = int(sr)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        await asyncio.to_thread(sf.write, file_path, audio, sample_rate_int)
        negative_generated += 1

        total_generated = count + negative_generated
        tasks_status[task_id]["sub_tasks"]["generation"]["generated_files"] = total_generated
        tasks_status[task_id]["sub_tasks"]["generation"]["progress"] = int((total_generated / total_files) * 100)
        tasks_status[task_id]["progress"] = tasks_status[task_id]["sub_tasks"]["generation"]["progress"] // 2

    # Похожие слова
    for i in range(similar_count):
        text = random.choice(similar_words)
        audio, sr, speaker = await asyncio.to_thread(
            generate_audio_file,
            text,
            sample_rate,
            None,
        )
        if hasattr(audio, "numpy"):
            audio = audio.numpy()
        if sample_rate in [8000, 24000, 48000]:
            audio = await asyncio.to_thread(augment_audio, audio, sr)

        filename = f"similar_{i:04d}_{speaker}.wav"
        file_path = os.path.join(output_dir, filename)
        sample_rate_int = int(sr)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        await asyncio.to_thread(sf.write, file_path, audio, sample_rate_int)
        negative_generated += 1

        total_generated = count + negative_generated
        tasks_status[task_id]["sub_tasks"]["generation"]["generated_files"] = total_generated
        tasks_status[task_id]["sub_tasks"]["generation"]["progress"] = int((total_generated / total_files) * 100)
        tasks_status[task_id]["progress"] = tasks_status[task_id]["sub_tasks"]["generation"]["progress"] // 2

    # Шум
    for i in range(noise_count):
        audio = generate_noise(sample_rate, duration=1.0)
        sr = sample_rate
        filename = f"noise_{i:04d}.wav"
        file_path = os.path.join(output_dir, filename)
        sample_rate_int = int(sr)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        await asyncio.to_thread(sf.write, file_path, audio, sample_rate_int)
        negative_generated += 1

        total_generated = count + negative_generated
        tasks_status[task_id]["sub_tasks"]["generation"]["generated_files"] = total_generated
        tasks_status[task_id]["sub_tasks"]["generation"]["progress"] = int((total_generated / total_files) * 100)
        tasks_status[task_id]["progress"] = tasks_status[task_id]["sub_tasks"]["generation"]["progress"] // 2

# Сколько аугментированных копий генерировать на каждую реальную запись.
# Без этого модель рискует выучить "отпечаток" конкретного микрофона/комнаты
# вместо содержания звука (ровно то, что и произошло в первом прогоне).
# Поднято с 3 до 5 вместе со снижением count_per_text/negative_count —
# реальный голос тонул в синтетике (~15% датасета), из-за чего модель
# путала случайную речь с ключевым словом на held-out тесте.
REAL_SAMPLE_AUGMENTED_COPIES = 5


def copy_real_samples(source_dir: str, dest_dir: str) -> int:
    """Подмешивает реальные записи пользователя (не TTS-синтетику) к датасету.
    source_dir — settings.REAL_SAMPLES_DIR/positive или .../negative, куда
    пользователь вручную кладёт свои .wav (или любой формат, который читает
    soundfile/librosa — частота/формат не важны, ресемплируется на этапе
    извлечения признаков).

    Каждый файл добавляется как есть, плюс несколько аугментированных
    вариаций (тем же пайплайном, что и для синтетики — pitch/gain/шум/EQ) —
    так модель не сможет зацепиться за конкретный "отпечаток" записи
    (микрофон/фон комнаты), а вынуждена обобщать на сам звук.

    Возвращает количество итоговых файлов (оригиналы + аугментации)."""
    if not os.path.isdir(source_dir):
        return 0

    copied = 0
    for filename in os.listdir(source_dir):
        if not filename.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
            continue

        src = os.path.join(source_dir, filename)
        base_name = os.path.splitext(filename)[0]

        audio, sr = sf.read(src)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # стерео -> моно
        audio = audio.astype(np.float32)

        # Оригинал как есть
        dst = os.path.join(dest_dir, f"real_{base_name}.wav")
        sf.write(dst, audio, sr)
        copied += 1

        # Плюс аугментированные копии — каждый вызов даёт свою случайную
        # вариацию (pitch/gain/шум и т.д. применяются с вероятностью,
        # поэтому повторные вызовы на одном и том же входе не идентичны).
        for i in range(REAL_SAMPLE_AUGMENTED_COPIES):
            augmented = augment_audio(audio, sr).astype(np.float32)
            dst_aug = os.path.join(dest_dir, f"real_{base_name}_aug{i}.wav")
            sf.write(dst_aug, augmented, sr)
            copied += 1

    return copied

# ========================================================
# ОСНОВНАЯ ФУНКЦИЯ ПАЙПЛАЙНА
# ========================================================

async def run_full_pipeline(task_id: str, request):
    """Выполняет полный пайплайн: генерация данных -> извлечение признаков -> обучение."""
    try:
        generation_task_id = f"{task_id}_gen"
        training_task_id = f"{task_id}_train"

        # 1. Создаём структуру папок для датасета
        dataset_dir = os.path.join(settings.DATASETS_DIR, task_id)
        positive_dir = os.path.join(dataset_dir, "positive")
        negative_dir = os.path.join(dataset_dir, "negative")
        os.makedirs(positive_dir, exist_ok=True)
        os.makedirs(negative_dir, exist_ok=True)

        # 2. Инициализируем статус с подзадачами
        total_files = request.count_per_text + request.negative_count
        tasks_status[task_id]["sub_tasks"] = {
            "generation": {
                "task_id": generation_task_id,
                "status": "processing",
                "message": "Генерация датасета начата",
                "progress": 0,
                "total_files": total_files,
                "generated_files": 0,
            },
            "features": {
                "task_id": f"{task_id}_feat",
                "status": "pending",
                "message": "Ожидание извлечения признаков",
                "progress": 0,
            },
            "training": {
                "task_id": training_task_id,
                "status": "pending",
                "message": "Ожидание начала обучения",
                "progress": 0,
            },
        }
        tasks_status[task_id]["message"] = "Генерация датасета..."

        # 3. Генерация положительных примеров
        update_task(task_id, message=f"Генерация положительных примеров для '{request.wake_word}'...")
        await generate_positive_examples(
            wake_word=request.wake_word,
            count=request.count_per_text,
            sample_rate=request.sample_rate,
            output_dir=positive_dir,
            task_id=task_id,
            total_files=total_files,
        )

        # 4. Генерация отрицательных примеров
        update_task(task_id, message="Генерация отрицательных примеров...")
        await generate_negative_examples(
            wake_word=request.wake_word,
            count=request.negative_count,
            sample_rate=request.sample_rate,
            output_dir=negative_dir,
            task_id=task_id,
            total_files=total_files,
        )

        # 4.5. Подмешиваем реальные записи пользователя (если он их положил
        # в settings.REAL_SAMPLES_DIR/positive и /negative) — это то, чего
        # не может дать TTS: настоящий голос пользователя и настоящие
        # нечленораздельные звуки (кашель, кряхтение и т.п.).
        real_positive_count = copy_real_samples(
            os.path.join(settings.REAL_SAMPLES_DIR, "positive"), positive_dir
        )
        real_negative_count = copy_real_samples(
            os.path.join(settings.REAL_SAMPLES_DIR, "negative"), negative_dir
        )
        if real_positive_count or real_negative_count:
            update_task(
                task_id,
                message=f"Подмешано реальных записей: {real_positive_count} positive, {real_negative_count} negative",
            )

        # 5. Обновляем статус генерации
        tasks_status[task_id]["sub_tasks"]["generation"]["status"] = "completed"
        tasks_status[task_id]["sub_tasks"]["generation"]["message"] = "Датасет готов"
        tasks_status[task_id]["message"] = "Извлечение признаков..."
        tasks_status[task_id]["progress"] = 40

        # 6. Извлечение признаков
        tasks_status[task_id]["sub_tasks"]["features"]["status"] = "processing"
        tasks_status[task_id]["sub_tasks"]["features"]["message"] = "Извлечение Mel-спектрограмм..."

        positive_path, negative_path, positive_groups_path, negative_groups_path = await prepare_features(dataset_dir, task_id)

        tasks_status[task_id]["sub_tasks"]["features"]["status"] = "completed"
        tasks_status[task_id]["sub_tasks"]["features"]["message"] = "Признаки готовы"
        tasks_status[task_id]["message"] = "Обучение модели..."
        tasks_status[task_id]["progress"] = 60

        # 7. Запускаем обучение модели
        tasks_status[task_id]["sub_tasks"]["training"]["status"] = "processing"
        tasks_status[task_id]["sub_tasks"]["training"]["message"] = "Обучение начато..."

        # train_model теперь возвращает (model_path, config_path)
        model_path, config_path = train_model(
            positive_features_path=positive_path,
            negative_features_path=negative_path,
            positive_groups_path=positive_groups_path,
            negative_groups_path=negative_groups_path,
            task_id=training_task_id,
            epochs=request.epochs,
            wake_word=request.wake_word,
            sample_rate=request.sample_rate,
        )

        # 8. Финальный статус
        tasks_status[task_id]["sub_tasks"]["training"]["status"] = "completed"
        tasks_status[task_id]["sub_tasks"]["training"]["message"] = "Обучение завершено"
        tasks_status[task_id]["sub_tasks"]["training"]["progress"] = 100
        tasks_status[task_id]["status"] = "completed"
        tasks_status[task_id]["message"] = f"Модель для '{request.wake_word}' успешно обучена!"
        tasks_status[task_id]["progress"] = 100
        tasks_status[task_id]["file_path"] = model_path
        tasks_status[task_id]["config_path"] = config_path

    except Exception as e:
        tasks_status[task_id]["status"] = "failed"
        tasks_status[task_id]["message"] = f"Ошибка в пайплайне: {str(e)}"
        if "sub_tasks" in tasks_status[task_id]:
            for key in ["generation", "features", "training"]:
                if key in tasks_status[task_id]["sub_tasks"]:
                    if tasks_status[task_id]["sub_tasks"][key]["status"] == "processing":
                        tasks_status[task_id]["sub_tasks"][key]["status"] = "failed"
                        tasks_status[task_id]["sub_tasks"][key]["message"] = str(e)