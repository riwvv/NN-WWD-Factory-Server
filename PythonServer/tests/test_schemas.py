import pytest
from pydantic import ValidationError

from app.models.schemas import GenerateFullPipelineRequest, StatusResponse


def test_generate_full_pipeline_request_defaults():
    request = GenerateFullPipelineRequest(wake_word="джарвис")
    assert request.sample_rate == 24000
    assert request.count_per_text == 1000
    assert request.negative_count == 2000
    assert request.epochs == 20


def test_generate_full_pipeline_request_requires_wake_word():
    with pytest.raises(ValidationError):
        GenerateFullPipelineRequest()


def test_status_response_defaults():
    status = StatusResponse(task_id="abc", status="processing", message="в процессе")
    assert status.progress == 0
    assert status.sub_tasks is None
    assert status.file_path is None
