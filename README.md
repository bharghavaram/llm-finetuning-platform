> **📅 Project Period:** Jan 2025 – Feb 2025 &nbsp;|&nbsp; **Status:** Completed &nbsp;|&nbsp; **Author:** [Bharghava Ram Vemuri](https://github.com/bharghavaram)

# LLM Fine-Tuning Platform

> Production-grade PEFT/LoRA fine-tuning pipeline with QLoRA (4-bit) support and MLflow tracking

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-orange)](https://pytorch.org)
[![PEFT](https://img.shields.io/badge/PEFT-0.13-purple)](https://huggingface.co/docs/peft)
[![MLflow](https://img.shields.io/badge/MLflow-2.16-blue)](https://mlflow.org)

## Overview

A production-ready fine-tuning platform that enables efficient adaptation of large language models (Mistral-7B, Llama-3-8B, Gemma-7B) using **Parameter-Efficient Fine-Tuning (PEFT)** with **LoRA** and **QLoRA (4-bit quantization)**, reducing GPU memory requirements by 75%.

## Architecture

```
Training Request
      ↓
Config Builder (LoRA/QLoRA params)
      ↓
BitsAndBytes Quantization (4-bit NF4)
      ↓
LoRA Adapter Injection → PEFT Model
      ↓
SFT Trainer (TRL) → Training Loop
      ↓
MLflow Experiment Tracking
      ↓
Adapter Saved → Inference Ready
```

## Key Features

- **QLoRA (4-bit)** – fine-tune 7B models on 16GB VRAM
- **LoRA** – inject trainable adapters into attention layers (q_proj, k_proj, v_proj, o_proj)
- **MLflow tracking** – experiment comparison, metric logging, artifact storage
- **REST API** – submit jobs, poll status, run inference
- **Async job management** – submit multiple training runs simultaneously
- **Simulation mode** – runs without GPU for API testing

## Quick Start

```bash
git clone https://github.com/bharghavaram/llm-finetuning-platform
cd llm-finetuning-platform
pip install -r requirements.txt
cp .env.example .env    # Add HF_TOKEN
uvicorn main:app --reload
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/finetune/start` | Start a training job |
| GET | `/api/v1/finetune/jobs` | List all jobs |
| GET | `/api/v1/finetune/jobs/{id}` | Job status & metrics |
| POST | `/api/v1/finetune/evaluate` | Evaluate fine-tuned model |
| POST | `/api/v1/finetune/inference` | Run inference on fine-tuned model |

### Example: Start Fine-Tuning

```bash
curl -X POST "http://localhost:8000/api/v1/finetune/start" \
  -H "Content-Type: application/json" \
  -d '{
    "base_model": "mistralai/Mistral-7B-v0.1",
    "dataset_name": "tatsu-lab/alpaca",
    "lora_r": 16,
    "num_epochs": 3,
    "use_4bit": true
  }'
```

## MLflow Dashboard

```bash
mlflow ui --backend-store-uri mlruns
# Open http://localhost:5000
```

## LoRA Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `lora_r` | LoRA rank (adapter size) | 16 |
| `lora_alpha` | LoRA scaling factor | 32 |
| `lora_dropout` | Dropout probability | 0.05 |
| `target_modules` | Which layers to adapt | q,k,v,o projections |

## Docker (GPU)

```bash
docker-compose up --build  # Requires NVIDIA GPU + Docker
```
