from pydantic import BaseModel
from typing import List, Optional

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

class GenerateResponse(BaseModel):
    task_id: str
    status: str
    file_path: Optional[str] = None
    count: Optional[int] = None
    message: Optional[str] = None

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