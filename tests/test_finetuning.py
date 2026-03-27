"""Tests for LLM Fine-Tuning Platform."""
import pytest
from app.core.config import settings
from app.services.finetuning_service import FineTuningService


def test_settings():
    assert settings.LORA_R == 16
    assert settings.LORA_ALPHA == 32
    assert settings.NUM_EPOCHS == 3
    assert settings.USE_4BIT is True


def test_create_training_config():
    svc = FineTuningService()
    config = svc.create_training_config("mistralai/Mistral-7B-v0.1", "alpaca")
    assert config["base_model"] == "mistralai/Mistral-7B-v0.1"
    assert config["dataset_name"] == "alpaca"
    assert config["lora_config"]["r"] == settings.LORA_R
    assert config["lora_config"]["lora_alpha"] == settings.LORA_ALPHA
    assert config["training_config"]["num_train_epochs"] == settings.NUM_EPOCHS
    assert config["quantization"]["load_in_4bit"] is True


def test_create_config_overrides():
    svc = FineTuningService()
    config = svc.create_training_config(
        "meta-llama/Meta-Llama-3-8B", "my-dataset",
        lora_r=32, learning_rate=1e-4, num_epochs=5, use_4bit=False,
    )
    assert config["lora_config"]["r"] == 32
    assert config["training_config"]["learning_rate"] == 1e-4
    assert config["training_config"]["num_train_epochs"] == 5
    assert config["quantization"]["load_in_4bit"] is False


def test_list_jobs_empty():
    svc = FineTuningService()
    assert svc.list_jobs() == []


def test_get_nonexistent_job():
    svc = FineTuningService()
    job = svc.get_job("nonexistent-id")
    assert job is None


def test_finetuning_job_dict():
    from app.services.finetuning_service import FineTuningJob
    job = FineTuningJob("test-id", {"base_model": "test", "dataset_name": "data"})
    d = job.to_dict()
    assert d["job_id"] == "test-id"
    assert d["status"] == "queued"
    assert d["model_path"] is None


@pytest.mark.asyncio
async def test_api_health():
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    resp = client.get("/api/v1/finetune/health")
    assert resp.status_code == 200

@pytest.mark.asyncio
async def test_api_list_jobs():
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    resp = client.get("/api/v1/finetune/jobs")
    assert resp.status_code == 200
    assert "jobs" in resp.json()
