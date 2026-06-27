from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from app.models.schemas import (
    GenerateBatchRequest, 
    GenerateResponse, 
    StatusResponse
)
from app.services.task_manager import tasks_status, create_task, update_task
from app.services.audio_generator import process_batch_generation
from app.core.config import settings
import os
import uuid
import asyncio

router = APIRouter()

@router.post("/generate-batch", response_model=GenerateResponse)
async def generate_batch(request: GenerateBatchRequest):
    """Генерирует пакет аудиофайлов для каждого слова из списка."""
    task_id = str(uuid.uuid4())
    total_files_to_generate = len(request.texts) * request.count_per_text
    create_task(task_id, len(request.texts), total_files_to_generate)
    asyncio.create_task(process_batch_generation(task_id, request))
    return GenerateResponse(
        task_id=task_id, 
        status="processing", 
        message="Задача на генерацию поставлена в очередь"
    )

@router.get("/generate-status/{task_id}", response_model=StatusResponse)
async def get_status(task_id: str):
    """Возвращает детальный статус выполнения задачи."""
    if task_id not in tasks_status:
        raise HTTPException(status_code=404, detail="Задача не найдена")

    status_info = tasks_status[task_id]
    
    return StatusResponse(
        task_id=task_id,
        status=status_info.get("status", "unknown"),
        message=status_info.get("message", ""),
        progress=status_info.get("progress", 0),
        total_files=status_info.get("total_files", 0),
        generated_files=status_info.get("generated_files", 0),
        current_word=status_info.get("current_word", ""),
        current_word_index=status_info.get("current_word_index", 0),
        total_words=status_info.get("total_words", 0),
        file_path=status_info.get("file_path", None)
    )

@router.get("/download/{task_id}")
async def download_file(task_id: str):
    """Скачивает сгенерированный файл или архив."""
    if task_id not in tasks_status:
        raise HTTPException(status_code=404, detail="Задача не найдена")

    status_info = tasks_status[task_id]
    if status_info.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Файл еще не готов")
    
    file_path = status_info.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Файл не найден на диске")

    # Если это папка — архивируем
    if os.path.isdir(file_path):
        import shutil
        archive_path = file_path + ".zip"
        shutil.make_archive(file_path, 'zip', file_path)
        return FileResponse(archive_path, filename=f"{task_id}.zip")
    
    return FileResponse(file_path, filename=os.path.basename(file_path))

# Можно добавить эндпоинт для одиночной генерации (для совместимости)
@router.post("/generate-audio", response_model=GenerateResponse)
async def generate_audio(request: GenerateBatchRequest):
    """Генерирует один аудиофайл (для обратной совместимости)."""
    # Просто используем пакетный метод с одним словом
    single_request = GenerateBatchRequest(
        texts=[request.texts[0]] if request.texts else ["test"],
        language=request.language,
        speaker=request.speaker,
        sample_rate=request.sample_rate,
        count_per_text=1
    )
    return await generate_batch(single_request)