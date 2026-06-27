# app/utils/file_utils.py
import os
import shutil
from pathlib import Path

def ensure_directory_exists(path: str) -> None:
    """Создаёт папку, если её нет."""
    os.makedirs(path, exist_ok=True)

def get_file_size(file_path: str) -> int:
    """Возвращает размер файла в байтах."""
    return os.path.getsize(file_path)

def clean_directory(path: str) -> None:
    """Удаляет все файлы и папки внутри указанной папки, но не саму папку."""
    if not os.path.exists(path):
        return
    
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)