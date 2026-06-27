# app/api/endpoints.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from app.models.schemas import (
    GenerateBatchRequest,
    GenerateNegativeRequest,
    GenerateFullPipelineRequest,
    GenerateResponse,
    StatusResponse,
)
from app.services.task_manager import tasks_status, create_task
from app.services.audio_generator import process_batch_generation, process_negative_generation
from app.services.pipeline_service import run_full_pipeline
import os
import uuid
import asyncio
import shutil

router = APIRouter()

@router.post("/generate-batch", response_model=GenerateResponse)
async def generate_batch(request: GenerateBatchRequest):
    task_id = str(uuid.uuid4())
    total_files_to_generate = len(request.texts) * request.count_per_text
    create_task(task_id, len(request.texts), total_files_to_generate)
    asyncio.create_task(process_batch_generation(task_id, request))
    return GenerateResponse(
        task_id=task_id,
        status="processing",
        message="Задача на генерацию поставлена в очередь",
    )

@router.post("/generate-negative", response_model=GenerateResponse)
async def generate_negative(request: GenerateNegativeRequest):
    task_id = str(uuid.uuid4())
    total_files_to_generate = request.count
    create_task(task_id, 1, total_files_to_generate)
    asyncio.create_task(process_negative_generation(task_id, request))
    return GenerateResponse(
        task_id=task_id,
        status="processing",
        message="Задача на генерацию отрицательных примеров поставлена в очередь",
    )

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