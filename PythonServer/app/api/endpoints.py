# app/api/endpoints.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from app.core.config import settings
from app.models.schemas import (
    GenerateFullPipelineRequest,
    GenerateResponse,
    StatusResponse,
)
from app.services.task_manager import tasks_status, create_task
from app.services.pipeline_service import run_full_pipeline
import os
import uuid
import asyncio
import shutil
import signal

router = APIRouter()

@router.get("/generate-status/{task_id}", response_model=StatusResponse)
async def get_status(task_id: str):
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
        file_path=status_info.get("file_path", None),
        sub_tasks=status_info.get("sub_tasks", {}),
    )

@router.get("/download/{task_id}")
async def download_file(task_id: str):
    if task_id not in tasks_status:
        raise HTTPException(status_code=404, detail="Задача не найдена")

    status_info = tasks_status[task_id]
    if status_info.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Файл еще не готов")

    file_path = status_info.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Файл не найден на диске")

    if os.path.isdir(file_path):
        archive_path = file_path + ".zip"
        shutil.make_archive(file_path, "zip", file_path)
        return FileResponse(archive_path, filename=f"{task_id}.zip")

    return FileResponse(file_path, filename=os.path.basename(file_path))

@router.post("/train-full-pipeline", response_model=GenerateResponse)
async def train_full_pipeline(request: GenerateFullPipelineRequest):
    """Генерирует датасет и запускает обучение модели через VoxPulse."""
    task_id = str(uuid.uuid4())
    total_files_estimate = request.count_per_text + request.negative_count
    create_task(task_id, 2, total_files_estimate)

    asyncio.create_task(run_full_pipeline(task_id, request))

    return GenerateResponse(
        task_id=task_id,
        status="processing",
        message="Полный пайплайн запущен: генерация датасета и обучение модели.",
    )

@router.post("/shutdown")
async def shutdown():
    """Останавливает сервер."""
    # Отправляем сигнал завершения процессу
    os.kill(os.getpid(), signal.SIGTERM)
    return {"message": "Server shutting down..."}

@router.get("/download-package/{task_id}")
async def download_package(task_id: str):
    """
    Скачивает ZIP-архив с моделью (.pth) и конфигом (.json).
    """
    if task_id not in tasks_status:
        raise HTTPException(status_code=404, detail="Задача не найдена")

    status_info = tasks_status[task_id]
    if status_info.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Модель еще не готова")
    
    model_path = status_info.get("file_path")
    config_path = status_info.get("config_path")
    
    if not model_path or not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Файл модели не найден")
    
    if not config_path or not os.path.exists(config_path):
        raise HTTPException(status_code=404, detail="Файл конфига не найден")
    
    # Получаем имя для пакета из конфига (или используем task_id)
    try:
        import json
        with open(config_path, "r") as f:
            config = json.load(f)
        package_name = config.get("model_name", task_id)
    except Exception:
        package_name = task_id
    
    # Создаём временную папку для пакета
    package_dir = os.path.join(settings.PACKAGE_DIR, task_id)
    os.makedirs(package_dir, exist_ok=True)
    
    # Копируем файлы в пакет
    shutil.copy(model_path, package_dir)
    shutil.copy(config_path, package_dir)
    
    # Создаём ZIP-архив
    zip_filename = f"{package_name}_package"
    zip_path = os.path.join(settings.PACKAGE_DIR, zip_filename)
    shutil.make_archive(zip_path, 'zip', package_dir)
    
    # Очищаем временную папку
    shutil.rmtree(package_dir)
    
    return FileResponse(
        f"{zip_path}.zip", 
        filename=f"{zip_filename}.zip",
        media_type="application/zip"
    )