from app.services.task_manager import create_task, update_task, tasks_status


def test_create_task_initializes_expected_fields():
    task = create_task("task-1", total_words=3, total_files_to_generate=30)
    assert task["status"] == "processing"
    assert task["total_files"] == 30
    assert task["total_words"] == 3
    assert task["generated_files"] == 0
    assert tasks_status["task-1"] is task


def test_update_task_merges_fields_for_existing_task():
    create_task("task-2", total_words=1, total_files_to_generate=10)
    update_task("task-2", status="completed", progress=100)
    assert tasks_status["task-2"]["status"] == "completed"
    assert tasks_status["task-2"]["progress"] == 100


def test_update_task_is_noop_for_unknown_task():
    update_task("does-not-exist", status="completed")
    assert "does-not-exist" not in tasks_status
