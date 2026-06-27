from pydantic import BaseModel
from typing import List, Optional, Dict

class GenerateRequest(BaseModel):
    text: str
    language: str = "ru"
    speaker: str = "xenia"
    sample_rate: int = 24000

class GenerateBatchRequest(BaseModel):
    texts: List[str]
    language: str = "ru"
    speaker: str = "xenia"
    sample_rate: int = 24000
    count_per_text: int = 10

class GenerateNegativeRequest(BaseModel):
    count: int = 100
    language: str = "ru"
    sample_rate: int = 24000
    include_similar_words: bool = True
    include_random_words: bool = True
    include_noise: bool = True

# Новая модель для полного пайплайна
class GenerateFullPipelineRequest(BaseModel):
    wake_word: str
    sample_rate: int = 24000
    count_per_text: int = 1000
    negative_count: int = 2000
    epochs: int = 20

class GenerateResponse(BaseModel):
    task_id: str
    status: str
    file_path: Optional[str] = None
    count: Optional[int] = None
    message: Optional[str] = None

class SubTaskStatus(BaseModel):
    task_id: str
    status: str
    message: str
    progress: int = 0
    total_files: int = 0
    generated_files: int = 0

class StatusResponse(BaseModel):
    task_id: str
    status: str
    message: str
    progress: int = 0
    total_files: int = 0
    generated_files: int = 0
    current_word: str = ""
    current_word_index: int = 0
    total_words: int = 0
    file_path: Optional[str] = None
    sub_tasks: Optional[Dict[str, SubTaskStatus]] = None