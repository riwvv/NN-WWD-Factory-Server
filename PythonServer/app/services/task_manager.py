tasks_status = {}

def create_task(task_id: str, total_words: int, total_files_to_generate: int):
    tasks_status[task_id] = {
        "status": "processing",
        "message": "Генерация начата...",
        "progress": 0,
        "total_files": total_files_to_generate,
        "generated_files": 0,
        "current_word": "",
        "current_word_index": 0,
        "total_words": total_words,
        "file_path": None
    }
    return tasks_status[task_id]

def update_task(task_id: str, **kwargs):
    if task_id in tasks_status:
        tasks_status[task_id].update(kwargs)