# FarmRL OpenEnv Environment

## Overview

FarmRL simulates real-world crop operations where an agent chooses daily irrigation, fertilizer, and pesticide actions to improve productivity while preserving sustainability. This is a practical farm-operations decision problem, not a game task.

The environment is OpenEnv-compatible and exposes typed state/action/reward models with API endpoints for reset, step, and state retrieval.

## Motivation

Farm operations require balancing short-term yield goals with long-term soil and resource health. This environment provides a deterministic, reproducible benchmark for evaluating whether an LLM can make adaptive control decisions under changing weather and soil conditions.

## Observation Space

Observation is represented by `FarmState`:

- `soil_moisture` (0-100)
- `soil_ph` (4-9)
- `temperature` (float)
- `rainfall` (>=0)
- `crop_stage` (int, >=0)
- `day` (int, >=0)

## Action Space

Action is represented by `FarmAction`:

- `water` in [0, 50]
- `fertilizer` in [0, 20]
- `pesticide` in [0, 10]

## Reward Design

Reward is provided at every step and includes:

- Positive yield progress (`yield_score`)
- Sustainability encouragement (`sustainability_bonus`)
- Resource overuse penalty (`resource_penalty`)
- Explicit penalties for excessive chemical usage (`overuse_penalty`)
- Explicit loop/stall penalty (`loop_penalty`)

This gives dense trajectory feedback and discourages destructive/repetitive behavior.

## Tasks and Difficulty

Three deterministic grader tasks are provided:

1. `task_easy_yield` (easy): maximize normalized total reward.
2. `task_medium_chemical_efficiency` (medium): minimize aggregate fertilizer + pesticide usage.
3. `task_hard_sustainability_balance` (hard): optimize yield-to-chemical-use ratio.

Each grader returns a score in [0.0, 1.0].

## OpenEnv Interface

API endpoints:

- `POST /reset`
- `POST /step`
- `GET /state`

`step(action)` returns `observation`, `reward`, `done`, `info`.

OpenEnv metadata is declared in `openenv.yaml`.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure `.env`:

- `API_BASE_URL=https://api.openai.com/v1`
- `MODEL_NAME=gpt-4o-mini`
- `OPENAI_API_KEY=<your_key>`

## Usage

Run baseline inference:

```bash
python inference.py
```

Run API server:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 7860
```

## Baseline Scores

Typical baseline output includes an `[END]` line with score and rewards. Example from a recent run:

- `overall score`: 0.564
- `steps`: 60

Task-level baseline scores are reported by `tasks/graders.py` and constrained to [0.0, 1.0].

## Container and Deployment

Build container:

```bash
docker build -t farmrl-space-check:latest .
```

Run container:

```bash
docker run --rm -p 7860:7860 farmrl-space-check:latest
```

This image is suitable for Hugging Face Space deployment.
