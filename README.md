---
title: FarmRL OpenEnv Submission
emoji: 🌾
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# FarmRL OpenEnv Submission

A reinforcement learning environment and agent for optimizing farm management decisions using the OpenEnv framework.

## Overview

FarmRL trains an intelligent agent to manage crop farming decisions by controlling irrigation, fertilizer application, and pesticide use. The agent learns from a real agricultural dataset and aims to maximize crop yield while maintaining sustainability.

**Key Features:**
- OpenEnv-compliant REST API for environment interaction
- LLM-based inference via OpenAI-compatible endpoints
- Tabular RL training with preprocessing pipeline
- Grading and evaluation framework

## Quick Start

### Prerequisites
- Python 3.11+
- `uv` package manager (recommended) or pip
- Environment variables configured (see Configuration section)

### Installation

```bash
uv pip install -e .
```

Or with pip:
```bash
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root with required credentials:

```env
# API Configuration (defaults provided for base URL and model)
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
TASK_NAME=farm-yield-optimization
BENCHMARK=farmrl
PORT=7860

# Required API Credentials (no defaults)
# Submission/Judge environments inject API_KEY.
API_KEY=your_api_key_here
# Optional local compatibility fallback:
# OPENAI_API_KEY=your_api_key_here
```

**Environment Variables:**
- `API_BASE_URL`: LLM endpoint URL (default: `https://api.openai.com/v1`)
- `MODEL_NAME`: Model identifier (default: `gpt-4o-mini`)
- `TASK_NAME`: Task identifier (default: `farm-yield-optimization`)
- `BENCHMARK`: Benchmark name (default: `farmrl`)
- `PORT`: Server port (default: `7860`)
- `API_KEY`: Primary API key for submission validation (required, no default)
- `OPENAI_API_KEY`: Optional local compatibility fallback when `API_KEY` is not set

**Hackathon submission note:**
- Use only the injected `API_BASE_URL` and `API_KEY` values provided by the validator.
- Do not hardcode API keys, alternate providers, or alternate base URLs.
- Inference is configured to fail fast if proxy-compatible LLM settings are missing.

### Running the API Server

Start the OpenEnv API server on your configured port:

```bash
uv run python -m server.app
```

The server will be available at `http://localhost:7860`

### Running Inference

Execute the full inference pipeline:

```bash
uv run python inference.py
```

This runs the agent against the environment, logging results to stdout in the standard format:
- `[START]` - Episode initialization
- `[STEP]` - Individual step results
- `[END]` - Episode completion with score

## Project Structure

```
.
├── api/              # REST API endpoints
├── env/              # FarmRL environment implementation
├── server/           # API server setup
├── tasks/            # Grading and evaluation
├── scripts/          # Data preprocessing utilities
├── reference-material/  # Documentation and examples
├── inference.py      # Main inference script
├── openenv.yaml      # Environment schema definition
├── requirements.txt  # Python dependencies
└── farmer_advisor_dataset.csv  # Agricultural training data
```

## Environment Specification

The environment state and action spaces are defined in `openenv.yaml`:

**Observations:**
- `soil_moisture`: Soil water availability (0-100%)
- `soil_ph`: Soil acidity (4-9)
- `temperature`: Environmental temperature
- `rainfall`: Precipitation amount (mm)
- `crop_stage`: Current crop growth stage
- `day`: Days since planting

**Actions:**
- `water`: Irrigation amount (0-50 mm)
- `fertilizer`: Fertilizer application
- `pesticide`: Pesticide application

## API Endpoints

- `POST /reset` - Reset environment (optional seed parameter)
- `POST /step` - Execute action and get next state
- `GET /state` - Get current environment state
- `GET /health` - Health check

## Data Pipeline

The project includes a preprocessing script (`scripts/add_water_variable.py`) that:
1. Adds a `Water_mm` column representing agent-controllable irrigation
2. Adjusts `Rainfall_mm` to maintain water-balance invariance

Run preprocessing:
```bash
uv run python scripts/add_water_variable.py farmer_advisor_dataset.csv
```

## Evaluation

The grading system (`tasks/graders.py`) evaluates agent performance based on:
- Crop yield optimization
- Sustainability metrics
- Action validity

## Docker

The project includes a Dockerfile for containerized deployment:

```bash
docker build -t farmrl-openenv .
docker run -p 7860:7860 farmrl-openenv
```

## References

- [OpenEnv Framework](https://github.com/openenv-ai/openenv)
- [FarmGym Simulation](https://github.com/farm-gym)
- Dataset: Real agricultural data with crop yield observations
